"""
# Create a dataset tensor from the images and the labels
dataset = tf.data.Dataset.from_tensor_slices(
    (mnist.train.images, mnist.train.labels))
# Automatically refill the data queue when empty
dataset = dataset.repeat()
# Create batches of data
dataset = dataset.batch(batch_size)
# Prefetch data for faster consumption
dataset = dataset.prefetch(batch_size)

# Create an iterator over the dataset
iterator = dataset.make_initializable_iterator()
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data

# IMPORTS
import numpy as np
import tensorflow as tf
import os

os.system('clear')

MODE='TRAIN'
#MODE = 'TEST'

#USER = 'Matteo'
USER='Giacomo'

if USER == 'Matteo':
    MNIST = input_data.read_data_sets("Users/matteo/desktop/cnn_save/data/mnist/", one_hot=True)
    save_path = "/Users/matteo/.TensorFlow_Data/model.ckpt"
elif USER == 'Giacomo':
    MNIST = input_data.read_data_sets("/data/mnist/", one_hot=True)
    save_path = "C:\\Users\\Giacomo\\Documents\\Checkpoints\\model.ckpt"

# Training Parameters
learning_rate = 0.00001
batch_size = 128
epochs = 1000

# HyperParameters
inputs = 784
classes = 10
dropout = 0.2

dataset = tf.data.Dataset.from_tensor_slices(
    (MNIST.train.images, MNIST.train.labels))
# Automatically refill the data queue when empty
dataset = dataset.repeat()
# Create batches of data
dataset = dataset.batch(batch_size)

# Prefetch data for faster consumption
# divido il dataset in batch_size
dataset = dataset.prefetch(batch_size)

# Create an iterator over the dataset
# prende il dataset
iterator = dataset.make_initializable_iterator()


def cnn_model_fn(x):
    # INPUT LAYER
    with tf.name_scope('Model'):
        with tf.name_scope('input_layer'):
            input_layer = tf.reshape(x, [-1, 28, 28, 1])

        # CONVOLUTIONAL LAYER #1
        with tf.name_scope('Conv1'):
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


# metto in X e Y batch_size elementi con le rispettive lagles
X, Y = iterator.get_next()

# RICHIAMO LA FUNZIONE
logits = cnn_model_fn(X)

# DICHIARO LE ETICHETTE CHE AVREI APPLICATO AD OGNI IMMAGINE
prediction = tf.nn.softmax(logits)
with tf.name_scope("Loss"):
    # TASSO DI ERRORE
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    tf.summary.scalar("loss2",loss)

# DICHIARO UN OPTIMIZER CHE MODIFICA I PESI IN BASE AL LEARNING RATE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# APPLICO L'OTTIMIZZAZIONE MINIMIZZANDO GLI ERRORI
train_op = optimizer.minimize(loss)

# CONFRONTO LE MIE PREVISIONI CON QUELLE CORRETTE DEL TRAIN TEST
correct_predict = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
with tf.name_scope("Accuracy"):
    # CONTROLLO L'ACCURACY CHE HO AVUTO
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    tf.summary.scalar("Accuracy2",accuracy)
init = tf.global_variables_initializer()
best_acc = 0
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    sess.run(init)
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("C:\\Users\\Giacomo\\Documents\\Tensorboard\\logs", sess.graph)
    if MODE == 'TRAIN':
        os.system('clear')
        print("TRAINING MODE")

        for step in range(1, epochs + 1):
            check = '[ ]'

            sess.run(train_op)
            los, acc = sess.run([loss, accuracy])


            result=sess.run(merged)
            writer.add_summary(result,step)
            if acc >= best_acc:
                check = '[X]'
                best_acc = acc
                print(str(step) + '\t' + '%.4f' % acc + '\t\t' + check)
                #saver.save(sess, save_path)
            elif step % 20 == 0:
                print(str(step) + '\t' + '%.4f' % acc + '\t\t' + check)

        #writer = tf.summary.FileWriter("/Users/matteo/Documents/TensorBoard", sess.graph)

    elif MODE == 'TEST':
        os.system('clear')
        print("TESTING MODE")
        saver.restore(sess, save_path)
        print("Initialization Complete")

        # test the model
        print("Testing Accuracy:" + str(sess.run(accuracy)))
        print("Testing finished")

    else:
        os.system('clear')
        print('\nLa modalità inserita non è corretta.\n')

sess.close()