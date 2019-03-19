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

def cnn_model_fn(X, MODE):

    # INPUT LAYER
    with tf.name_scope('input_layer') as scope:
        input_layer = tf.reshape(X, [-1, 28, 28, 1])

    # CONVOLUTIONAL LAYER #1
    with tf.name_scope('Conv1') as scope:
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
        )
        # APPLICO LA FUNZIONE RELU
        conv1_relu = tf.nn.relu(conv1)

    # POOLING LAYER #1
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1_relu,
            pool_size=[2, 2],
            strides=2
        )

    # CONVOLUTIONAL LAYER #2
    with tf.name_scope('Conv2'):
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
        )

        # APPLICO LA FUNZIONE RELU
        conv2_relu = tf.nn.relu(conv2)

    # POOLING LAYER #2
    with tf.name_scope('Pool2'):
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2_relu,
            pool_size=[2, 2],
            strides=2
        )

    # RIDIMENSIONO POOL2 PER RIDURRE IL CARICO COMPUTAZIONALE
    with tf.name_scope('Reshape'):
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # DENSE LAYER
    with tf.name_scope('Dense_layer'):
        dense = tf.layers.dense(
            inputs=pool2_flat,
            units=1024,
        )

        # APPLICO LA FUNZIONE RELU
        dense_relu = tf.nn.relu(dense)

    # AGGIUNGO L'OPERAZIONE DI DROPOUT
    with tf.name_scope('Dropout'):
        dropout = tf.layers.dropout(
            inputs=dense_relu,
            rate=0.4,
            training=MODE == tf.estimator.ModeKeys.TRAIN
        )

    # LOGIT LAYER
    with tf.name_scope('Logit_layer'):
        logits = tf.layers.dense(
            inputs=dropout,
            units=10
        )

    return logits
