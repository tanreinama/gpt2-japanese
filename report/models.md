# gpt2-japanese pretrain models


Japanese GPT2 Generation Pretrained Model



## 学習済みGPT2日本語モデル



mediumモデル→→→[ダウンロード](https://www.nama.ne.jp/models/gpt2ja-medium.tar.bz2) （[予備URL](http://ailab.nama.ne.jp/models/gpt2ja-medium.tar.bz2)）←←←

### 以前のモデルとの違いについて

[日本語版BPEEncoder](https://github.com/tanreinama/Japanese-BPEEncoder)でエンコードされたデータで学習したモデルをもって、正式のモデルとします。

以前のモデルはexperimentalなモデルという扱いになります。

語彙数が変わった結果、出力層のlogitsのパラメーター数が変わり、総パラメーター数も変化したので、モデルの名称も、以前のような（117M/345M/774M）という名前ではなく、（small/medium/large）という名前になります。

### コーパスと学習回数について

学習させたデータは、[コーパス2020](report/corpus.md)の混合コーパスです。

このコーパスをエンコードすると、約5.3Gトークンとなります。学習回数は、10Mイテレーションを超える程度を目安にしました。

1バッチで取り出すトークンは1024個なので、backward iterationsは約2エポック分に相当します（ランダムに取りだしているのでエポックではない）。



## 公開している学習済みモデル



| モデル名      | 総パラメーター数 | レイヤー数       | URL                                                          |
| ------------- | ---------------- | ---------------- | ------------------------------------------------------------ |
| gpt2ja-medium | 324426752        | 16heads,24layers | https://www.nama.ne.jp/models/gpt2ja-medium.tar.bz2<br />（予備：http://ailab.nama.ne.jp/models/gpt2ja-medium.tar.bz2 ） |



以前のモデル（experimentalモデル）は、性能がイマイチなので基本的に使いません。



### top_kとtop_pの設定



オリジナルのGPT-2の論文によると、top_k=40が一番良い結果と報告されています。

現在、gpt2-generate.pyのデフォルトは、top_k=40、top_p=0となっています。

top_kとtop_pの値は排他で、片方を>0に設定したら片方を0にしなければなりません。