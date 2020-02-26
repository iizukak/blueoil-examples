# custom\_network

## 概要

このサンプルでは、Blueoil を使って自作のネットワークを定義し、学習を行います。

## 利用方法

```
mynetwork.py : ネットワークが定義されているファイルです
config.py : 学習用の設定ファイルです。 simple_classification のサンプルとほぼ同様です
```

`mynetwork.py` に含まれる `MyNetworkQuantize` が今回学習に用いるネットワーク定義クラスです。
Blueoil を用いて学習する場合は、 [base クラス]() を継承したネットワークが必要です。

## 学習

以下のようなコマンドで学習を開始できます。定数部分については適宜変換します。

```
$ CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. blueoil train -c config.py
```

## 推論

推論のための変換のコマンドは以下のようになります。

```
$ CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. blueoil convert -e config_20200226121537
```

推論の実行に関しては `simple_classification` と同様です。
