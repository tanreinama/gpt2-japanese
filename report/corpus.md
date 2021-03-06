# コーパス2020



## 概要



日本語の大規模なコーパスとしては、[BCCWJコーパス](https://pj.ninjal.ac.jp/corpus_center/bccwj/)等があり、優秀な均衡コーパスとして利用出来るものの、利用許諾料が必要であり、研究機関に属しない個人としては利用にハードルがありました。

そこで、2020年3月から7月にかけて、独自にウェブクロールを行い、今後の個人研究に利用出来るコーパスを構築することにしました。

コーパスのサイズは合計20GB程度を目標とし、SNS投稿やオンラインショップの商品説明などの解析に利用出来るように、個人ブログ・Web小説・ニュースリリース・オンライン辞書などのジャンルから混合的にスクレイピングしました。



## コーパス2020



スクレイピングしたコンテンツに、日本語Wikipedia全文コーパスを追加し、合計21GB程度のコンテンツを用意しました。

さらに、日付やURLアドレス等の要素を正規表現でタグに置換し、二つの大規模コーパスを作成しました。また、二つの混合コーパスでSentencePieceを学習させ（ワード数=50000）、分かち書きをしました。



| コーパス     | ジャンル                                                     | article数 | token数 |
| ------------ | ------------------------------------------------------------ | --------- | --------- |
| コーパスA    | 個人・技術ブログ 3.2GB<br/><br/>質問・まとめ・採点サイト 1.8GB<br/><br/>ウェブ辞書サイト（含むWikipedia） 3.4GB<br/><br/>ニュースリリース 2.1GB<br/><br/>ニュースサイト 0.3GB<br/> | 5079419   | 17億token   |
| コーパスB    | ウェブ小説サイト 13.1GB                                      | 1675927   | 21.6億token   |
| 混合コーパス | コーパスA＋コーパスB                                         | 6755346   | 38.6億token   |



このうち、コーパスAとコーパスBは、それぞれ117MパラメーターのGPT2モデルのトレーニングに使用し、コーパスのジャンル違いによる機械学習モデルの日本語生成能力への影響を比較調査できるようにします。

また、より大きなパラメーター数を持つモデルは、混合コーパスでトレーニングされます。



## 公開について



スクレイピングによって作成したコーパスは、著作権等の特性により一般に公開することが出来ません。

このコーパスを使用して特定のモデルを作成したい、等の場合は、ご連絡頂ければ何らかの対応を考慮します。

