# blueoil-examples

[![CircleCI](https://circleci.com/gh/iizukak/blueoil-examples.svg?style=svg)](https://circleci.com/gh/iizukak/blueoil-examples)

## 概要

このリポジトリは、低ビット量子化ディープラーニングフレームワーク 
[Blueoil](https://github.com/blue-oil/blueoil) を利用するサンプルプログラム集です。
実験的なプログラムも含みます。

## Examples

### simple\_classification

Blueoil に組み込まれたシンプルな畳み込みニューラルネットワーク (CNN) を学習します。

### custom\_network

独自の量子化ニューラルネットを定義し、Blueoil を用いて学習を行います。

### custom\_dataset\_loader

Blueoil には [様々な種類のデータセットローダ](https://github.com/blue-oil/blueoil/tree/master/blueoil/datasets) 
が定義されていますが、多くの場合、自分が作成したデータセットのデータローダは自前で書く必要があります。
このサンプルでは、独自のデータセットに対してデータローダを定義します。

### custom\_training

Blueoil はいわゆる学習ループを提供します。
しかし、複雑なネットワークを定義する場合には学習ループを自分で定義する必要がある場合があり、
このサンプルでは学習ループを実際に定義します。

### inference\_on\_fpga

FPGA 上で、アクセラレータを用いて推論を実行します。

## インストール

Blueoil は pip でインストールできますが、PyPI には登録されていないため、
リポジトリを clone してインストールします。

```
$ pip install -U pip setuptools
$ pip install -r requirements.txt
$ pip install tensorflow-gpu==1.15.2 # GPU がある場合
$ which blueoil
```

`which blueoil` して、パスが表示されればインストールは完了しています。

## ユニットテスト

pytest を用いてユニットテストを実行します。

```
$ pytest tests/
```

## 動作を確認した環境

- Ubuntu 16.04.6 LTS
- macOS Mojave 10.14.6
- Python 3.6
- cudnn 7.6.2.24-1+cuda10.0

