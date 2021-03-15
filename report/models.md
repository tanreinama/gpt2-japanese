# gpt2-japanese pretrain models


Japanese GPT2 Generation Pretrained Model



## 学習済みGPT2日本語モデル



smallモデル→→→[ダウンロード](https://www.nama.ne.jp/models/gpt2ja-small.tar.bz2) （[予備URL](http://ailab.nama.ne.jp/models/gpt2ja-small.tar.bz2)）←←←

mediumモデル→→→[ダウンロード](https://www.nama.ne.jp/models/gpt2ja-medium.tar.bz2) （[予備URL](http://ailab.nama.ne.jp/models/gpt2ja-medium.tar.bz2)）←←←

largeモデル→→→[ダウンロード](https://www.nama.ne.jp/models/gpt2ja-large.tar.bz2) （[予備URL](http://ailab.nama.ne.jp/models/gpt2ja-large.tar.bz2)）←←←



### モデル毎の違いについて

すべてのモデルは、[Japanese-BPEEncoder](https://github.com/tanreinama/Japanese-BPEEncoder)でエンコードされるトークンを使用します。また、入力トークン数は1024で固定です。

GPT-2では、出力の際に、トークンを一つ一つ予測しながら、出力数の分だけループでtransformerを実行します。そのため、GPT-2の文章生成タスクは、学習よりも実行時間がかかります。実行速度が遅い場合は、最大出力文字数を少なくすると、ループの回る回数が減るので、その分早く実行されます。

smallモデルとmediumモデルはadamアルゴリズムで、largeモデルのみ、GPUメモリの都合からadagradアルゴリズムで（LRを手動調整しながら）学習されました。そのため、largeモデルは、実行時間が増えた割に本来の（20head36layerの）性能に到達していない可能性があります。

ファインチューニングのベースとして利用するのでは無く、そのまま素の文章生成モデルとして使用する場合は、引き続きmediumモデルの使用を推奨します。



### コーパスと学習回数について

学習させたデータは、[コーパス2020](report/corpus.md)の混合コーパスです。

このコーパスをエンコードすると、約5.3Gトークンとなります。学習回数は、10Mイテレーションを超える程度を目安にしました。

1バッチで取り出すトークンは1024個なので、backward iterationsは約2エポック分に相当します。



## 公開している学習済みモデル



| モデル名      | 総パラメーター数 | 学習Optimizer | レイヤー数       | URL                                                          |
| ------------- | ---------------- | ------------- | ---------------- | ------------------------------------------------------------ |
| gpt2ja-large  | 736034560        | adam          | 20heads,36layers | https://www.nama.ne.jp/models/gpt2ja-large.tar.bz2<br />（予備：http://ailab.nama.ne.jp/models/gpt2ja-large.tar.bz2 ） |
| gpt2ja-medium | 324426752        | adam          | 16heads,24layers | https://www.nama.ne.jp/models/gpt2ja-medium.tar.bz2<br />（予備：http://ailab.nama.ne.jp/models/gpt2ja-medium.tar.bz2 ） |
| gpt2ja-small  | 101642496        | adagrad       | 12heads,12layers | https://www.nama.ne.jp/models/gpt2ja-small.tar.bz2<br />（予備：http://ailab.nama.ne.jp/models/gpt2ja-small.tar.bz2 ） |





### top_kとtop_pの設定



オリジナルのGPT-2の論文によると、top_k=40が一番良い結果と報告されています。

現在、gpt2-generate.pyのデフォルトは、top_k=40、top_p=0となっています。

top_kとtop_pの値は排他で、片方を>0に設定したら片方を0にしなければなりません。

日本語BPEEncoderでの語彙数は、オリジナルのものに比べて半分少々なので、top_k=20程度の値にした方が良いのかもしれません（未検証）。