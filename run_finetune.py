# coding=utf-8

import argparse
import json
import os
import numpy as np
import tensorflow.compat.v1 as tf
import time
import tqdm
from copy import copy
from encode_bpe import BPEEncoder_ja
import model

if int(tf.__version__[0]) > 1:
    from model import HParams as HParams
else:
    from tensorflow.contrib.training import HParams

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'

parser = argparse.ArgumentParser(
    description='Pretraining GPT2-JA on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input npz file')
parser.add_argument('--base_model', type=str, default='gpt2ja-small', help='a path to a model file')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
parser.add_argument('--optim', type=str, default='adam', help='"adam", "adagrad", or "sgd" to use optimizer')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=5e-5, help='Learning rate for optimizer')
parser.add_argument('--warmup_steps', metavar='WR', type=int, default=0, help='Learning rate warming up steps')

parser.add_argument('--run_name', type=str, default='gpt2ja_finetune', help='Run id. Name of subdirectory in checkpoint/')
parser.add_argument('--save_every', metavar='N', type=int, default=1000, help='Write a checkpoint every N steps')

parser.add_argument('--gpu', default='0', help='visible gpu number.')

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

with open('ja-bpe.txt', encoding='utf-8') as f:
    bpe = f.read().split('\n')

with open('emoji.json', encoding='utf-8') as f:
    emoji = json.loads(f.read())

enc = BPEEncoder_ja(bpe, emoji)
n_vocab = len(enc)

def main():
    args = parser.parse_args()

    if os.path.isfile(args.base_model+'/hparams.json'):
        with open(args.base_model+'/hparams.json', encoding='utf-8') as f:
            params = json.loads(f.read())
            hparams = HParams(**params)
    elif 'small' in args.base_model:
        hparams = HParams(**{
          "n_vocab": n_vocab,
          "n_ctx": 1024,
          "n_embd": 768,
          "n_head": 12,
          "n_layer": 12
        })
    elif 'medium' in args.base_model:
        hparams = HParams(**{
          "n_vocab": n_vocab,
          "n_ctx": 1024,
          "n_embd": 1024,
          "n_head": 16,
          "n_layer": 24
        })
    elif 'large' in args.base_model:
        hparams = HParams(**{
          "n_vocab": n_vocab,
          "n_ctx": 1024,
          "n_embd": 1280,
          "n_head": 20,
          "n_layer": 36
        })
    else:
        raise ValueError('invalid model name.')

    config = tf.ConfigProto()
    if int(args.gpu) >= 0:
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = args.gpu
    with tf.Session(config=config,graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [None, None])
        output = model.model(hparams=hparams, X=context, past=None, reuse=tf.AUTO_REUSE)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(args.base_model)
        saver.restore(sess, ckpt)

        train_vars = tf.trainable_variables()

        global_step = tf.Variable(0, trainable=False)
        if args.warmup_steps > 0:
            learning_rate = tf.compat.v1.train.polynomial_decay(
                    learning_rate=1e-10,
                    end_learning_rate=args.learning_rate,
                    global_step=global_step,
                    decay_steps=args.warmup_steps
                )
        else:
            learning_rate = args.learning_rate

        if args.optim=='adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.9,
                                           beta2=0.98,
                                           epsilon=1e-7)
        elif args.optim=='adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif args.optim=='sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            raise ValueError('invalid optimizer name.')

        train_vars = tf.trainable_variables()
        opt_grads = tf.gradients(loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)

        summaries = tf.summary.scalar('loss', loss)
        summary_log = tf.summary.FileWriter(
            os.path.join(CHECKPOINT_DIR, args.run_name))

        saver = tf.train.Saver(
            var_list=train_vars,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=2)
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.latest_checkpoint(args.base_model)
        saver.restore(sess, ckpt)
        print('Loading checkpoint', ckpt)

        print('Loading dataset...')
        global_chunks = []
        with np.load(args.dataset) as npz:
            for inditem, item in enumerate(npz.files):
                token_chunk = npz[item]
                current_token = []
                for ind in range(0,len(token_chunk)):
                    current_token.append(np.uint16(token_chunk[ind]))
                    if len(current_token) == hparams.n_ctx or current_token[-1] == n_vocab-1:
                        global_chunks.append(current_token)
                        current_token = []
                if len(current_token) > 1:
                    global_chunks.append(current_token)
                    current_token = []
        global_chunk_index = np.random.permutation(len(global_chunks))
        global_chunk_step = 0
        print('Training...')

        def sample_feature():
            nonlocal global_chunks,global_chunk_index,global_chunk_step
            p_input_ids = []

            for b in range(args.batch_size): # FULL-SENTENCES
                idx = global_chunk_index[global_chunk_step]
                global_chunk_step += 1
                if global_chunk_step >= len(global_chunk_index):
                    global_chunk_step = 0
                    global_chunk_index = np.random.permutation(len(global_chunks))
                sampled_token = global_chunks[idx]
                # Make Sequence
                ids = copy(global_chunks[idx])
                p_input_ids.append(ids)

            return {context:p_input_ids}

        counter = 1
        counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
        hparams_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'hparams.json')
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r', encoding='utf-8') as fp:
                counter = int(fp.read()) + 1

        maketree(os.path.join(CHECKPOINT_DIR, args.run_name))

        def save():
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
            print(
                'Saving',
                os.path.join(CHECKPOINT_DIR, args.run_name,
                             'model-{}').format(counter))
            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, args.run_name, 'model'),
                global_step=counter)
            with open(counter_path, 'w', encoding='utf-8') as fp:
                fp.write(str(counter) + '\n')
            with open(hparams_path, 'w', encoding='utf-8') as fp:
                fp.write(json.dumps({
                      "n_vocab": int(hparams.n_vocab),
                      "n_ctx": int(hparams.n_ctx),
                      "n_embd": int(hparams.n_embd),
                      "n_head": int(hparams.n_head),
                      "n_layer": int(hparams.n_layer),
                }))

        avg_loss = (0.0, 0.0)
        start_time = time.time()

        try:
            while True:
                if counter % args.save_every == 0:
                    save()

                (_, v_loss, v_summary) = sess.run(
                    (opt_apply, loss, summaries),
                    feed_dict=sample_feature())

                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

                counter = counter+1
                if args.warmup_steps > 0:
                    global_step = global_step+1
        except KeyboardInterrupt:
            print('interrupted')
            save()


if __name__ == '__main__':
    main()
