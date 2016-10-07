# -*- coding: utf-8 -*-

# ~/tensorflow/bin/frameworkpython.py で実行

from __future__ import absolute_import
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import codecs

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: データのロード

file_name = 'wonderland.txt'
#file_name = 'rap.txt'
#file_name = 'rap_ja.txt'

# 入力データをスペース区切りで単語配列にする
def read_data():
  f = codecs.open(file_name, 'r', 'utf-8')
  data = f.read().split() # スペースで単語ごとにわける
  # TODO: 本当は小文字化, ピリオドなどの除去をしないといけない  
  f.close()
  return data

words = read_data()

print('Data size', len(words))

# 各単語を出現頻度ごとにソートし、vocabulary_size 以上の低頻度後をすべて UNK というワードとして
# まとめてしまう.
#vocabulary_size = 30000 # rap_ja.txtの場合
vocabulary_size = 5000 # wonderland.txtの場合


# Step 2: 辞書を構築し低頻度語を'UNK'に置き換え

def build_dataset(words):
  count = [['UNK', -1]]
  
  counter = collections.Counter(words)
  print( ">>> original vocabrary size =", len(counter.most_common() ) )
  count.extend(counter.most_common(vocabulary_size - 1))
  
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # 元の文章は破棄してメモリ解放

# data:       元の文章をindex列にしたもの
# count:      各ワードが何回出現するか
# dictionary:          単語からindexへの変換辞書
# reverse_dictionary:  indexから単語への変換辞書

print('vocabrary size=', len(count));

# 再頻出ワードとその出現数を上位5個表示
print('Most common words (+UNK)', count[:5])

# 元の文章をindex列にしたものの最初の10個を表示
print('Sample data', data[:10])

data_index = 0

# Step 3: skip-gramモデル学習用のバッチ生成関数
def generate_batch(batch_size, num_skips, skip_window):
  # batch_size  = 128
  # num_skips   = 2  # 前後1個ずつの合計2個がターゲットという意味
  # skip_window = 1
  
  global data_index
  
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  
  batch  = np.ndarray(shape=(batch_size),    dtype=np.int32)         # (128)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)         # (128, 1)
  span   = 2 * skip_window + 1 # [ skip_window target skip_window ]  # 3
  
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)

  # buffer には3つの連続するワードのindexが入った状態

  # (中央の)1単語のにつき、前後の各ワードをtargetとする.
  # なので1バッチが128個処理するとすると、64ワードのそれらの前後ワードをとってくることになる.
  for i in range(batch_size // num_skips): # 0 ~ 63
    # targetはbufferの3つのindexの中央 (pos=1)
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips): # 0 ~ 1
      while target in targets_to_avoid:
        target = random.randint(0, span - 1) # 0,1,2
      
      targets_to_avoid.append(target)
      batch[ i * num_skips + j   ] = buffer[skip_window] # 中央の入力ワード
      labels[i * num_skips + j, 0] = buffer[target]      # 前後のターゲットワード
      
    buffer.append(data[data_index]) # buffer内容を次の3ワードに更新
    data_index = (data_index + 1) % len(data)
    
  return batch, labels

# バッチ生成確認の為に、batchサイズ=8で試しに生成
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

for i in range(8):
  print(batch[i], '->', labels[i, 0])
  print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])

# Step 4: skip-gramモデルを生成して学習

batch_size     = 128  # バッチ処理のサイズ
embedding_size = 128  # embeddingベクトルの次元数
skip_window    = 1    # 左右何個の単語まで考慮に入れるか
num_skips      = 2    # How many times to reuse an input to generate a label.

# 単語類似度検証用のデータ
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size     = 16   # Random set of words to evaluate similarity on.
valid_window   = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))

# 類似度検査用のID=0~100のあいだの単語を16個選んでおく (IDが低いので出現頻度が高い単語)

num_sampled    = 64   # softmaxを高速化する為のnegative samplingのサンプリング数

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])      # 単語ID入力
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])   # ターゲット単語入力
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)      # 類似度検証入力ワード入力

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # embed = (128, 128) = (batch, embedding_size)

    # Construct the variables for the NCE loss
    # (30000, 128)のweight と(30000)のbias
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                     num_sampled, vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  # コサイン類似度を計算
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm # 各単語ベクトルを長さで割って単位ベクトルに  
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  # 検証ワードの単語ベクトルを引いてきて、全単語ベクトルと掛け合わせる
  # (結果が類似度となる)
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
  
# Step 5: 学習開始
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  tf.initialize_all_variables().run()
  print("Initialized")

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
    
    feed_dict = {train_inputs : batch_inputs,
                 train_labels : batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    
    # ここが学習の部分
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # 一定ステップ毎に学習内容を確認する
    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      # コサイン類似度を計算
      sim = similarity.eval()
      
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log_str = "Nearest to %s:" % valid_word
        
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  
  # 最終的に得られる埋め込みベクトルを全単語分集めた行列
  final_embeddings = normalized_embeddings.eval()

# Step 6: embeddingの可視化
# 128次元のembeddingベクトルをtSNEという手法で2次元上に圧縮して表示する.

from matplotlib.font_manager import FontProperties

# (元の文章が日本語である場合にも対応できる様に、日本語フォントロード)
fp = FontProperties(fname=r'./ipaexg.ttf', size=14)

def plot_with_labels(low_dim_embs, labels, fp, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom',
                 fontproperties=fp) #...

  plt.savefig(filename)

# tSNEで2次元上に圧縮
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500 # 500個だけ表示
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [reverse_dictionary[i] for i in xrange(plot_only)]
plot_with_labels(low_dim_embs, labels, fp)

print("file generated: tsne.png")
