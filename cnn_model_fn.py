from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# IMPORTS
import numpy as np
import tensorflow as tf

#####################################
#    FUNZIONE PER IL MODELLO CNN    #
#####################################
def cnn_model_fn(features, labels, mode):

    # INPUT LAYER
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # CONVOLUTIONAL LAYER #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
        
    # POOLING LAYER #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    # CONVOLUTIONAL LAYER 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # POLLING LAYER #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    # RIDIMENSIONO POOL2 PER RIDURRE IL CARICO COMPUTAZIONALE
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # DENSE LAYER
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu
    )

    # AGGIUNGO L'OPERAZIONE DI DROPOUT
    dropout = tf.layers.dropout(
        inputs=dense, 
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # LOGIT LAYER
    logits = tf.layers.dense(
        inputs=dropout,
        units=10
    )

    # GENERARE PREVISIONI
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # CALCOLARE L'ERRORE
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, 
        logits=logits
    )

    
    # CONFIGURO LE OPERAZIONI DI TRAINING
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    

    # AGGIUNGO METRICHE DI VALUTAZIONE
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )