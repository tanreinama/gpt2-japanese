import json
import os
import numpy as np
import tensorflow as tf
import requests
import argparse
from tqdm import tqdm
import sentencepiece as spm
from encoder import Encoder
from model import default_hparams
from sampling import sample_sequence

parser = argparse.ArgumentParser()
parser.add_argument('--context', type=str, default='<|endoftext|>')
parser.add_argument('--num_generate', type=int, default=5)
args = parser.parse_args()

sp = spm.SentencePieceProcessor()
sp.Load("stm-model/stm.model")

for filename in ['encoder.json', 'vocab.bpe', 'hparams.json']:
    if not os.path.isfile(filename):
        r = requests.get("https://storage.googleapis.com/gpt-2/models/117M/" + filename, stream=True)
        with open(filename, 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)

def get_encoder():
    with open('encoder.json', 'r') as f:
        encoder = json.load(f)
    with open('vocab.bpe', 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )

batch_size=1
length=None
temperature=1
top_k=0
top_p=0.0

enc = get_encoder()
hparams = default_hparams()
with open('hparams.json') as f:
    hparams.override_from_dict(json.load(f))

if length is None:
    length = hparams.n_ctx // 2
elif length > hparams.n_ctx:
    raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

with tf.Session(graph=tf.Graph()) as sess:
    context = tf.placeholder(tf.int32, [batch_size, None])
    output = sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k, top_p=top_p
    )

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint('ja-117M/')
    saver.restore(sess, ckpt)

    generated = 0
    while True:
        raw_text = sp.EncodeAsPieces(args.context) if args.context!= '<|endoftext|>' else '<|endoftext|>'
        raw_text = ' '.join([r for r in raw_text if r!='▁'])
        text = ''
        while True:
            context_tokens = enc.encode(raw_text)
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            splitted = enc.decode(out[0]).split('<|endoftext|>')

            text += splitted[0]

            if len(splitted) > 1:
                break
            else:
                raw_text = splitted[0][-256:]
            print(splitted[0])

        generated += 1
        if args.num_generate <= generated:
            break
        print("="*15)
