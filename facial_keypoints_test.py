# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:56:57 2017

@author: KAI
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

TRAIN_FILE = './facial data/training.csv'
TEST_FILE = './facial data/test.csv'
SAVE_PATH = "./facial data/save_CNN.ckpt"
Examples_to_show = 6
VALIDATION_SIZE = 2000  # 验证集大小


def input_data(test=False):
    file_name = TEST_FILE if test else TRAIN_FILE
    df = pd.read_csv(file_name, nrows=VALIDATION_SIZE)
    cols = df.columns[:-1]
    # dropna()是丢弃有缺失数据的样本，这样最后7000多个样本只剩2140个可用的。
    df = df.dropna()
    df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' ') / 255.0)
    X = np.vstack(df['Image'])
    X = X.reshape((-1, 96, 96, 1))  # (batch height,width,channel)
    if test:
        y = None
    else:
        y = df[cols].values / 96.0  # 将y值缩放到[0,1]区间 30维
    return X, y


def CNN(tf_x, tf_y):
    cnn1 = tf.layers.conv2d(
        inputs=tf_x,
        filters=16,  # shape(96,96,16)
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        cnn1,
        pool_size=2,
        strides=2)  # 池化=压缩->shape(48,48,16)
    cnn2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)  # (48,48,32)
    pool2 = tf.layers.max_pooling2d(cnn2, 2, 2)  # (24,24,32)
    cnn3 = tf.layers.conv2d(pool2, 64, 3, 1, 'same', activation=tf.nn.relu)  # (24,24,64)
    pool3 = tf.layers.max_pooling2d(cnn3, 2, 2)  # (12,12,64)
    cnn4 = tf.layers.conv2d(pool3, 128, 3, 1, 'same', activation=tf.nn.relu)  # (12,12,128)
    pool4 = tf.layers.max_pooling2d(cnn4, 2, 2)  # (6,6,128)
    output_previous = tf.layers.dense(tf.reshape(pool4, [-1, 6 * 6 * 128]), 500)
    output = tf.layers.dense(output_previous, 30)

    loss = tf.losses.mean_squared_error(tf_y, output)
    # loss=tf.losses.sparse_softmax_cross_entropy(labels=tf_y,logits=output)
    # accuracy=tf.metrics.accuracy(labels=tf_y,predictions=tf.argmax(output,axis=1),)[1]

    return output, loss


if __name__ == '__main__':
    X, y = input_data(test=True)

    tf_x = tf.placeholder(tf.float32, [None, 96, 96, 1])
    tf_y = tf.placeholder(tf.float32, [None, 30])
    output, loss = CNN(tf_x, tf_y)

    sess_ = tf.Session()
    saver_ = tf.train.Saver()
    saver_.restore(sess_, './facial data/save_CNN.ckpt')

    pred_test = sess_.run(output, feed_dict={tf_x: X})
    print(pred_test)
    i = np.random.choice(pred_test.shape[0], size=Examples_to_show)
    x_img = pred_test[i, ::2] * 96.0
    y_img = pred_test[i, 1::2] * 96.0
    _, a = plt.subplots(1, Examples_to_show, figsize=(Examples_to_show, 1))
    for j in range(Examples_to_show):
        a[j].imshow(np.reshape(X[i][:][j], (96, 96)))
        a[j].scatter(x_img[j], y_img[j], c='r')
    plt.show()
