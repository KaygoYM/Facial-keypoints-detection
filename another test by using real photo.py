# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:15:13 2017

@author: KAI
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
    # output_second=tf.layers.dense(output_previous,500)
    output = tf.layers.dense(output_previous, 30)

    loss = tf.losses.mean_squared_error(tf_y, output)
    # loss=tf.losses.sparse_softmax_cross_entropy(labels=tf_y,logits=output)
    # accuracy=tf.metrics.accuracy(labels=tf_y,predictions=tf.argmax(output,axis=1),)[1]
    return output, loss
if __name__=='__main__':
    X=np.zeros((4,96,96,1))
    a=plt.imread('IMG_3526.png')
    X[0,:,:,:]=a[:,:,1,np.newaxis]
    a=plt.imread('IMG_3572.png')
    X[1,:,:,:]=a[:,:,1,np.newaxis]
    a=plt.imread('IMG_3927.png')
    X[2,:,:,:]=a[:,:,1,np.newaxis]
    a=plt.imread('IMG_3948.png')
    X[3,:,:,:]=a[:,:,1,np.newaxis]

    tf_x = tf.placeholder(tf.float32, [None, 96, 96, 1])
    tf_y = tf.placeholder(tf.float32, [None, 30])
    output, loss = CNN(tf_x, tf_y)
    
    sess_ = tf.Session()
    saver_ = tf.train.Saver()
    saver_.restore(sess_, './facial data/save_CNN.ckpt')
    
    pred_test = sess_.run(output, feed_dict={tf_x: X})
    x_img = pred_test[:, ::2] * 96.0
    y_img = pred_test[:, 1::2] * 96.0
    
    _, p = plt.subplots(1, 4, figsize=(4, 1))
    for j in range(4):
        p[j].imshow(np.squeeze(X[j,:,:,:]))
        p[j].scatter(x_img[j], y_img[j], c='r')
    plt.show()