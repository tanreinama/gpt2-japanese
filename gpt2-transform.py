import json
import os
import numpy as np
import tensorflow as tf
import requests
import argparse
from tqdm import tqdm
import sentencepiece as spm
from encoder import Encoder
import model
from model import default_hparams

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ja-117M')
parser.add_argument('--context', type=str, required=True)
args = parser.parse_args()

sp = spm.SentencePieceProcessor()
sp.Load(args.model+"/stm.model")

model_params = '117M'
if '-' in args.model:
    model_params = args.model.split('-')[1]
    if '_' in model_params:
        model_params = model_params.split('_')[0]

for filename in ['encoder.json', 'vocab.bpe', 'hparams.json']:
    if not os.path.isfile(args.model+'/'+filename):
        r = requests.get("https://storage.googleapis.com/gpt-2/models/" + model_params + "/" + filename, stream=True)
        with open(args.model+'/'+filename, 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)

def get_encoder():
    with open(args.model+'/'+'encoder.json', 'r') as f:
        encoder = json.load(f)
    with open(args.model+'/'+'vocab.bpe', 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )

batch_size=1

enc = get_encoder()
hparams = default_hparams()
with open(args.model+'/'+'hparams.json') as f:
    hparams.override_from_dict(json.load(f))

with tf.Session(graph=tf.Graph()) as sess:
    context = tf.placeholder(tf.int32, [batch_size, None])
    output = model.model(hparams=hparams, X=context[:, :-1], past=None, reuse=tf.AUTO_REUSE)

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(args.model)
    saver.restore(sess, ckpt)

    raw_text = sp.EncodeAsPieces(args.context)
    raw_text = ' '.join([r for r in raw_text if r!='▁'])
    context_tokens = enc.encode(raw_text)
    out = sess.run(output, feed_dict={
        context: [context_tokens for _ in range(batch_size)]
    })
    output = out['h_flat'][-1]
    print(output)
