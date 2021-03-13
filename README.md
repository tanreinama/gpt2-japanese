# gpt2-japanese
Japanese GPT2 Generation Model

このプロジェクトは、[スポンサーを募集しています](report/sponsor.md)。[camp-fireの活動報告](https://camp-fire.jp/projects/320938/activities#menu)。[支援者一覧](special_thanks.txt)。

* かねてより、camp-fireの募集期間が終了した後でも、支援出来ないかというお問い合わせを頂いていましたが、[GitHub Sponsors](https://github.com/sponsors/tanreinama)でのスポンサーを募集することにしました。リターンも、camp-fireの時と変わりません（さらに、camp-fireでは規約上不可能だったコーパスの提供も含めています）。

# GPT2日本語モデル

***<font color='red'>New</font>***

- [デモンストレーションサイトを作成しました](http://ailab.nama.ne.jp/#gpt2ja)
- [GitHub Sponserによるスポンサーシップを開始しました](https://github.com/sponsors/tanreinama)
- [ファインチューニング用のコードを公開しました](run_finetune.py)
- [Largeモデルを公開しました](report/models.md)

  

## GPT2に日本語コーパスを学習させたものです

### 学習済みモデルについて

[report/models.md](report/models.md)

### 学習させたコーパスについて

[report/corpus.md](report/corpus.md)

### スポンサーシップについて

[report/sponsor.md](report/sponsor.md)

### デモンストレーション

[Sakamoto's AI lab](http://ailab.nama.ne.jp/#gpt2ja)

### 関連プロジェクト

[text2text-japanese](https://github.com/tanreinama/text2text-japanese)

### TODO

✓大規模コーパスの作成（2020/8/20）<br>✓日本語版BPEEncoder作成（2020/9/15）<br>
✓mediumモデルの公開（2020/11/07）<br>✓smallモデルの公開（2020/12/24）<br>✓largeモデルの公開（2021/3/13）<br>

## 使い方



GitHubからコードをクローンします

```sh
$ git clone https://github.com/tanreinama/gpt2-japanese
$ cd gpt2-japanese
```

モデルファイルをダウンロードして展開します

```sh
$ wget https://www.nama.ne.jp/models/gpt2ja-medium.tar.bz2
$ tar xvfj gpt2ja-medium.tar.bz2
```

モデルを指定して実行します。Tensorflow 1.x/2.x両方で動作します

```sh
$ python3 gpt2-generate.py --model gpt2ja-medium --num_generate 1
```

### サンプル

```sh
$ python3 gpt2-generate.py
2017年、東京と大阪が初めて合併したことをきっかけに日本文化を深く理解するとともに豊かさを体感出来る場を作りたい。2017年の日本は“豊か”な時代になっているのか。そして、どの時間帯にどこへ行こうか、今自分が何をしているかが大切であると考えるようになった。今回のセミナーでは、私たち自身が今こうして今の時代を生きていることの大きな魅力とは何かを、多くの皆様と共にさらに深く知っていただきたいと思っています。 ●会場               東京都港区新橋

（・・・略）
```

- 以前のWikipediaのみ学習させたモデルよりも、自然な文章が生成されるようになりました
- ブログ記事、ニュースリリース、百科事典サイトなどの要素を含んだ汎用的な文章生成が出来ます

```sh
$ python3 gpt2-generate.py --context="『GPT-2』は、入力された文章の続きを作成するAIです。このデモンストレーションでは、このエリアに入力された文章の続きとなる文章を生成します。このエリアに、続きを生成したい文章を入力して、ページの下にある「Generate Japanese Text\!」ボタンをクリックしてください。"
これで、ページの下にある「Generate Japanese Text\!」ボタンが押されたら、完了です。
このように画面内に作りたい文章やその作り方が表示されます。ボタンの部分をクリックして、キーボード上のボタンを押してページを閉じると、そのページを作り始めることができます。作ったものを保存し、ページを閉じようとしています。

1つ目の画面
AIではこのような画面が作れます。

2つ目の画面
ボタン上の「Generate Japanese Text\!」をクリックしてページの下にある「Generate Japanese Text\!」を押すと、次のような画面が出てきます。

今回はページの下にある「Generate Japanese Text\!」を押してページを作り始めます。
ボタンの部分をクリックすると、この画面が出ます。この画面を見てみると、次のような文章が沢山表示されています。

これを読んで「これが作りたい文章を言ってほしい文章だな」と思ったら、ページのページを閉じてください。

 （・・・略）
```

- contextを指定すると、それに続く文章が出てきます

- 長めのcontextを指定する方が、続く文章も自然な文章になる傾向があるようです


```sh
$ python3 gpt2-generate.py --context="俺の名前は坂本俊之。何処にでもいるサラリーマンだ。"
そして、その坂本と同時に、目の前に現れた人物に驚き固まる。
「なんでここに！
……なんでこんな……」
俺は目の前に現れた人を見ようと顔を上げたが、その人は俺の驚きを無視できないように、少し顔を近づけて俺を観察していた。
「……って、もしかして、お前、俺と同じ学年だな？」
俺の表情を読み取ったのか、坂本がそう言い、目の前の人物に問いかけた。その姿を見れば、なんとなく察してしまった。
「おいおい、驚いたなぁ〜。本当に、君のお兄さんは？」
その人物の方も驚いた様子で、目の前の人物の顔を凝視し始める。
「はぁ、あんた誰？いきなり現れて驚いたんだよ。まぁ、こんな人を見なかったことにしとくよ。俺はただのクラスの人で、こんなんがクラスに居たら目立つからね」
この人が坂本先輩。俺がここに来た時、突然現れた謎の人物である先輩。
「おっ、お兄さん、本当に、やっぱり僕のクラスの人っすね。こんな偶然ってありますかねぇ？」
「おいおい、あんたはクラスの人からそんなこと言われてもねぇ。クラスまで俺のクラスだと思うなんて酷いじゃないか。まぁ、これでも一応高校生だったんだけどさ〜」
「そ、そういう意味じゃ無かったんで

（・・・略）
```

- ウェブ小説っぽい文章も生成出来ます。

  

```sh
$ python3 gpt2-generate.py --min_length 512 --max_length 1024
```

- 「min_length」と「max_length」で生成する文章の長さを指定出来ます。「min_length」は1024までです。

  

### 文章ベクトル生成

```sh
$ python3 gpt2-transform.py --model gpt2ja-medium --context="江戸時代のインターネット回線"
[0.9214873313903809, 0.015924066305160522, 1.3564162254333496, 5.106584548950195, 2.991609573364258, 4.2116875648498535, 2.169468641281128, -55.102230072021484, -0.41729745268821716,

 （・・・略）
```

- 文章のベクトル化を行います

### 文章の採点

```sh
$ printf "現在も行方不明のまま。\n現在も行方不明のママ。" | python3 gpt2-score.py --exclude-end -
現在も行方不明のまま。	-25.579
現在も行方不明のママ。	-33.87
```

- 言語モデルにより文章の確率の対数が表示されます。

```sh
$ echo 完全の域に達することは難い。 | python3 gpt2-score.py --tokens --exclude-end -
完全の域に達することは難い。	-44.858
完全	-8.5745
の	-6.352
域	-9.6646
に	-1.1741
達	-0.6282
する	-1.3645
こと	-3.2514
は	-2.0751
難	-2.9429
い	-8.4598
。	-0.37045
```

- `--tokens`を指定すると、それぞれのトークンの確率の対数が表示されます。



## ファインチューニング

[コーパス2020](https://github.com/tanreinama/gpt2-japanese/blob/master/report/corpus.md)でプレトレーニングしたモデルは公開しています。ここでの手順は、独自のデータでモデルをさらにファインチューニングする方法です。

### エンコード

[Japanese-BPEEncoder](https://github.com/tanreinama/Japanese-BPEEncoder)を使用して、学習させたい独自のデータをエンコードします。

```sh
$ git clone https://github.com/tanreinama/Japanese-BPEEncoder.git
$ cd Japanese-BPEEncoder
$ python encode_bpe.py --src_dir <content file path> --dst_file finetune
$ mv finetune.npz ../
$ cd ..
```

### 学習

「--base_model」に元のプレトレーニング済みモデルを「--dataset 」にエンコードしたファイルを指定して、「run_finetune.py」を起動します。

```sh
$ python run_finetune.py --base_model gpt2ja-medium --dataset finetune.npz --run_name gpr2ja-finetune_run1
```

学習したモデルは、「checkpoint」以下の「--run_name」で指定したディレクトリ内に保存されます。

