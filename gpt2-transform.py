import json
import os
import numpy as np
import tensorflow.compat.v1 as tf
import argparse
from tqdm import tqdm
import model
from encode_bpe import BPEEncoder_ja

if int(tf.__version__[0]) > 1:
    from model import HParams as HParams
else:
    from tensorflow.contrib.training import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2ja-medium')
parser.add_argument('--context', type=str, required=True)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

with open('ja-bpe.txt', encoding='utf-8') as f:
    bpe = f.read().split('\n')

with open('emoji.json', encoding='utf-8') as f:
    emoji = json.loads(f.read())

enc = BPEEncoder_ja(bpe, emoji)
n_vocab = len(enc)

if os.path.isfile(args.model+'/hparams.json'):
    with open(args.model+'/hparams.json', encoding='utf-8') as f:
        params = json.loads(f.read())
        hparams = HParams(**params)
elif 'small' in args.model:
    hparams = HParams(**{
      "n_vocab": n_vocab,
      "n_ctx": 1024,
      "n_embd": 768,
      "n_head": 12,
      "n_layer": 12
    })
elif 'medium' in args.model:
    hparams = HParams(**{
      "n_vocab": n_vocab,
      "n_ctx": 1024,
      "n_embd": 1024,
      "n_head": 16,
      "n_layer": 24
    })
elif 'large' in args.model:
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
    context = tf.placeholder(tf.int32, [1, None])
    output = model.model(hparams=hparams, X=context, past=None, reuse=tf.AUTO_REUSE)

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(args.model)
    saver.restore(sess, ckpt)

    context_tokens = enc.encode(args.context)
    out = sess.run(output, feed_dict={
        context: [context_tokens]
    })
    output = out['h_flat'][-1]
    print(output.tolist())
