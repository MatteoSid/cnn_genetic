# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from tensorflow.examples.tutorials.mnist import input_data


# # IMPORTS
# import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
# import os
# import sys
# from tqdm import tqdm

def cnn_model_fn(X, MODE, log=False):

    # INPUT LAYER
    with tf.name_scope('input_layer') as scope:
        input_layer = tf.reshape(X, [-1, 1000, 48, 1])

    # CONVOLUTIONAL LAYER #1
    with tf.name_scope('Conv1') as scope:
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=4,
            kernel_size=[10, 10],
            strides=(2, 2),
            padding="valid",
        )
        if log==True:
            print('[LOG:conv1]: ' + str(conv1.shape))
        # input('Premi invio per continuare')
        # APPLICO LA FUNZIONE RELU
        conv1_relu = tf.nn.relu(conv1)
        if log==True:
            print('[LOG:conv1_relu]: ' + str(conv1_relu.shape))
        # input('Premi invio per continuare')

    # POOLING LAYER #1
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1_relu,
            pool_size=[2, 2],
            strides=2
        )
        if log==True:
            print('[LOG:pool1]: ' + str(pool1.shape))
        # input('Premi invio per continuare')

    # CONVOLUTIONAL LAYER #2
    with tf.name_scope('Conv2'):
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
        )
        if log==True:
            print('[LOG:conv2]: ' + str(conv2.shape))
        # input('Premi invio per continuare')
        # APPLICO LA FUNZIONE RELU
        conv2_relu = tf.nn.relu(conv2)
        if log==True:
            print('[LOG:conv2_relu]: ' + str(conv2_relu.shape))
        # input('Premi invio per continuare')

    # POOLING LAYER #2
    with tf.name_scope('Pool2'):
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2_relu,
            pool_size=[2, 2],
            strides=2
        )
        if log==True:
            print('[LOG:pool2]: ' + str(pool2.shape))

        # creo una variabile con le dimensioni di pool2 perché mi servono per calcolare le dimensioni di pool2_flat
        x = tf.TensorShape.as_list(pool2.shape)

    # RIDIMENSIONO POOL2 PER RIDURRE IL CARICO COMPUTAZIONALE
    with tf.name_scope('Reshape'):
        pool2_flat = tf.reshape(pool2, [-1, x[1] * x[2] * x[3]])
        if log==True:
            print('[LOG:pool2_flat]: ' + str(pool2_flat.shape))
        # input('Premi invio per continuare')

    # DENSE LAYER
    with tf.name_scope('Dense_layer'):
        dense = tf.layers.dense(
            inputs=pool2_flat,
            units=1024,
        )
        if log==True:
            print('[LOG:dense]: ' + str(dense.shape))
        # input('Premi invio per continuare')

        # APPLICO LA FUNZIONE RELU
        dense_relu = tf.nn.relu(dense)
        if log==True:
            print('[LOG:dense_relu]: ' + str(dense_relu.shape))
        # input('Premi invio per continuare')

    # AGGIUNGO L'OPERAZIONE DI DROPOUT
    with tf.name_scope('Dropout'):
        dropout = tf.layers.dropout(
            inputs=dense_relu,
            rate=0.4,
            training=MODE == tf.estimator.ModeKeys.TRAIN
        )
        if log==True:
            print('[LOG:dropout]: ' + str(dropout.shape))
        # input('Premi invio per continuare')

    # LOGIT LAYER
    with tf.name_scope('Logit_layer'):
        logits = tf.layers.dense(
            inputs=dropout,
            units=2
        )
        if log==True:
            print('[LOG:logits]: ' + str(logits.shape))
        # input('Premi invio per continuare')

    return logits
