# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# MNISTデータロード用のクラスをimport
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def main(_):
  # MNISTデータセットをダウンロード
  # このmnistオブジェクトから学習用データ(画像と正解の数字のラベル)、および検証用データを取り出す.
  # (最初だけダウンロードに時間がかかる)
  mnist = input_data.read_data_sets("./tmp/data/", one_hot=True)

  # [グラフ定義]

  # 入力用の値のいれ物(=PlaceHolder)を作成する. ここには画像データをテンソルにしたものが入ってくる.
  # Noneは"数が未定"を表す. 学習時はここが100になり、確認時は10000になる.
  # なので、学習時は(100x784)のテンソル、確認時は(10000x784)のテンソルになる.
  x = tf.placeholder(tf.float32, [None, 784])

  # 784x10個の重み. 学習により変化していく.
  W = tf.Variable(tf.zeros([784, 10]))

  # 10個のBias値. 学習により変化していく.
  b = tf.Variable(tf.zeros([10]))

  # (x * W + b)の結果をsoftmax関数に入れ、その結果をyとする.
  # yは学習時は(100x10)のテンソル. 確認時は(10000x10)のテンソルになる.
  y = tf.nn.softmax(tf.matmul(x, W) + b)

  # 損失関数(正解とどれくらいずれているかを表すもの)を定義していく
  
  # y_ は正解データを入れる入れもの.
  # Noneとなっているが、学習時にはここが100になり、
  # y_は(100, 10)のテンソルとなる.
  y_ = tf.placeholder(tf.float32, [None, 10])

  # ニューラルネットの出した10個の値と正解の10個の値(正解部分だけが1の配列)を
  # もちいて、どれくらいずれていたか、を出す.
  # 小さければ小さいほど正解に近かった事を表す値. (合計をひとつのスカラー値として集めたもの)
  # 100個分の合計値を求める.
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  
  
  tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
                                reduction_indices=[1]))

  # 上記のずれを小さくする様に学習させるOptimizerを用意する.
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

  # Sessionを生成
  sess = tf.InteractiveSession()

  # [初期化]

  # 変数Wとbを初期化する. 初期化を走らせると、tf.zeros()が実行され、
  # Wとbの中の値がすべて0.0で初期化される.
  tf.initialize_all_variables().run()


  # [学習実行]

  # 学習開始. 上記のずれを小さくする学習を1000回繰り返す.
  for _ in range(1000):
    # 画像データと、正解データをそれぞれ学習用データセット55000個の中から
    # ランダムで100個ずつ集めてくる.
    # batch_xsが(100x784)のテンソル. batch_ysが(100x10)のテンソル.
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={ x:  batch_xs,
                                     y_: batch_ys })

  # 学習が終わると、Variable であるWとbの重みが正解を出力できる様に更新されていることになる.

  # [学習結果確認]    

  # 学習が終わったので、結果を確認してみる
  # ニューラルネットが出力した10個の値の中で最大だった値のindex(0〜9)と
  # 正解データの10個の値の中でどれが最大だったかのindex(0〜9)を比較し
  # 一致していれば1(正解)、違っていれば0(不正解)が、correct_predictionに入る
  target_index  = tf.argmax(y, 1)
  correct_index = tf.argmax(y_, 1)
  correct_prediction = tf.equal(target_index, correct_index)

  # 上記の 0or1 の値を、10000個の全テストデータに関して求めて平均を出す.
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # テストデータ10000個を入力して実行する.
  # 出てきた結果が正解率. 0.9くらいの値となり、90%の正解率となる.
  print(sess.run(accuracy, feed_dict={ x:  mnist.test.images,
                                       y_: mnist.test.labels }))

  # 試しにテストデータの最初の10個に関して結果を見てみる.
  print(sess.run(target_index, feed_dict={x: mnist.test.images[0:10]}))
  

if __name__ == '__main__':
  tf.app.run()
