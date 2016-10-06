# -*- coding: utf-8 -*-

# ~/tensorflow/bin/frameworkpython check_image.py で実行

import matplotlib.pyplot as plt

# MNISTデータロード用のクラスをimport
from tensorflow.examples.tutorials.mnist import input_data

# MNISTデータセットをダウンロード
# このmnistオブジェクトから学習用データ(画像と正解の数字のラベル)、および検証用データを取り出す.
mnist = input_data.read_data_sets("./tmp/data/", one_hot=True)

# テスト画像の最初の10個を取り出してくる
test_images = mnist.test.images[0:10]

# それらの画像を.pngで保存
for i in range(0,10):
  file_name = "image{0}.png".format(i)
  plt.gray()
  plt.imsave(file_name,
             test_images[i].reshape(28, 28))
  print("file creaetd: {0}".format(file_name))
