# 環境構築方法

## TensorFlowのインストール

※ 現時点のTensorFlowの最新版はr0.11ですが、サンプルをr0.10で検証していたので、
r0.10のインストール方法になります.

(多分r0.11でも動くので、入れてしまっていたらそちらで問題ないです)

以下、MacOSXに、VirtualEnv環境でTensorFlowをインストールする方法になります。

pipとVirtualEnvが入っていなければインストール

    $ sudo easy_install pip
    $ sudo pip install --upgrade virtualenv

VirtualEnv環境を ~/tensorflow に構築

    $ virtualenv --system-site-packages ~/tensorflow

    $ source ~/tensorflow/bin/activate

これでVirtualEnv環境に入り、抜ける時は、

    $ deactivate

でVirtualEnv環境を抜けます。

    $ source ~/tensorflow/bin/activate

にて、VirtualEnv環境に入った状態にして、pip にて、TensorFlowをインストールしています。

TensorFlow (Mac OSX CPU only, Python 2.7版) をインストール

    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py2-none-any.whl
    $ sudo pip install --upgrade $TF_BINARY_URL

TensorFlowがインストールできているかどうか確認

    $ python
    ...
    >>> import tensorflow as tf
    >>> quit()

エラーがでなければインストールできている.

## 
