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

from tqdm import tqdm


print("-------------------------------------------------------------------------")
print("Example of input")
print("Insert minimum dropout rate: 0.1")
print("Insert maximum dropout rate: 1")
print("Insert dropout increasing rate: 0.05\n")
print("Insert minimum learning rate: 0.0001")
print("Insert maximum learning rate: 0.01")
print("Insert learning rate decreasing rate: 0.0001\n")
print("Insert number of steps: 1000")
print("-------------------------------------------------------------------------")


dropoutmin=float(input("Insert minimum dropout rate: "))
dropoutmax=float(input("Insert maximum dropout rate: "))
dropoutincreaserate=float(input("Insert dropout increasing rate: "))

learning_ratemin=float(input("Insert minimum learning rate: "))
learning_ratemax=float(input("Insert maximum learning rate: "))
learning_rateincreaserate=float(input("Insert learning rate decreasing rate: "))

steps=int(input("Insert number of steps: "))
total=0
for dropout_rate in np.arange(dropoutmin, dropoutmax, dropoutincreaserate):
    for learning_rate in (np.arange(learning_ratemax, learning_ratemin, -learning_rateincreaserate)):
            total+=1
print("Total iteration requested: "+str(total)+" each of: "+str(steps) + " epochs.")
if(input("Would you like to continue(y,[n]): ")=='y'):

    MODE='TRAIN'
    #os.system('cls')
    print("BULA")
    MNIST = input_data.read_data_sets("/home/mdonato/MNIST", one_hot=True)
    save_path = "/home/mdonato/save/model.ckpt"

    # Training Parameters
    #learning_rate = 0.1
    batch_size = 128

    # HyperParameters
    inputs = 784
    classes = 10
    #dropout = 0.5

    def datasetinit():
        dataset = tf.data.Dataset.from_tensor_slices((MNIST.train.images, MNIST.train.labels))
        # Automatically refill the data queue when empty
        dataset = dataset.repeat()
        # Create batches of data
        dataset = dataset.batch(batch_size)

        # Prefetch data for faster consumption
        # divido il dataset in batch_size
        dataset = dataset.prefetch(batch_size)

        # Create an iterator over the dataset
        # prende il dataset
        return dataset.make_initializable_iterator()


    def cnn_model_fn(x,dropoutrate):
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
                    rate=dropout_rate,
                    training= 1
                )

            # LOGIT LAYER
            with tf.name_scope('Logit_layer'):
                logits = tf.layers.dense(
                    inputs=dropout,
                    units=classes
                )

        return logits


    os.system('cls')
    # metto in X e Y batch_size elementi con le rispettive lagles



    def evluationmethod(l_rate, dropout, steps):
        tf.reset_default_graph()
        with tf.Session() as sess:
            # RICHIAMO LA FUNZIONE
            iterator = datasetinit()
            X, Y = iterator.get_next()
            logits = cnn_model_fn(X,dropout)

            # DICHIARO LE ETICHETTE CHE AVREI APPLICATO AD OGNI IMMAGINE
            prediction = tf.nn.softmax(logits)

            with tf.name_scope("Loss"):
                # TASSO DI ERRORE
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
                tf.summary.scalar("loss2", loss)

            # DICHIARO UN OPTIMIZER CHE MODIFICA I PESI IN BASE AL LEARNING RATE
            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)

            # APPLICO L'OTTIMIZZAZIONE MINIMIZZANDO GLI ERRORI
            train_op = optimizer.minimize(loss)

            # CONFRONTO LE MIE PREVISIONI CON QUELLE CORRETTE DEL TRAIN TEST
            correct_predict = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
            with tf.name_scope("Accuracy"):
                # CONTROLLO L'ACCURACY CHE HO AVUTO
                accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
                tf.summary.scalar("Accuracy2", accuracy)
            init = tf.global_variables_initializer()
            best_acc = 0
            merged = tf.summary.merge_all()
            sess.run(iterator.initializer)
            sess.run(init)
            saver = tf.train.Saver()
            writer = tf.summary.FileWriter("/home/mdonato/Tensorboard/logs" + "lr_%.0E,%s,%s" % (l_rate, dropout, steps), sess.graph)
            for step in tqdm(range(1, steps + 1)):
                sess.run(train_op)
                los, acc = sess.run([loss, accuracy])
                #print("STEP: "+str(step))
                result = sess.run(merged)
                writer.add_summary(result, step)
            sess.close()


    #------------------------------------------------------------------------------------------------------------------------------------------------------------

    count = 0
        #for learning_rate in np.arange(0.1, 0.01, -0.01):
    for dropout_rate in np.arange(dropoutmin, dropoutmax, dropoutincreaserate):
        for learning_rate in (np.arange(learning_ratemax, learning_ratemin, -learning_rateincreaserate)):
            evluationmethod(learning_rate, dropout_rate, steps)
            print("Count: " + str(count))
            count+=1
    print("Finished.")
 #---------------------------------------------------------------------------------------------------------------------------------------------------------------------


else:
    print("Aborting.")

print("Goodbye.")


