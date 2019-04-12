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
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

import cnn_model_fn_v2
cnn_model_fn = cnn_model_fn_v2.cnn_model_fn

os.system('clear')

# Uso path diversi in basa alla piattaforme in cui eseguo il programma
if sys.platform == 'linux':
    print('Programma avviato su Cluster:\n')
    MNIST_path = '/home/mdonato/MNIST'
    save_path = '/home/mdonato/Checkpoints/model.ckpt'
    TensorBoard_path = "/home/mdonato/TensorBoard"
elif sys.platform == 'darwin':
    print('Programma avviato su Mac:\n')
    MNIST_path = '/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/MNIST_data'
    save_path = "/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/.TensorFlow_Data/model.ckpt"
    TensorBoard_path = "/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/TensorBoard"

#Training Parameters
learning_rate = 0.001
batch_size = 128
epochs = 100

#HyperParameters
inputs = 784
classes = 10
dropout = 0.75

# Chiedo la modalità di esecuzione in un ciclo che esce solo in caso di risposta corretta
while True:
    MODE = str(input('Seleziona in che modo vuoi eseguire il modello (TRAIN, TEST, BOTH):\n'))
    if MODE=='TRAIN':
        break
    if MODE=='TEST':
        break
    if MODE=='BOTH':
        break

epochs = int(input('Inserisci il numero di epoche da eseguire: '))
print('\n')

MNIST = input_data.read_data_sets(MNIST_path, one_hot=True)

dataset = tf.data.Dataset.from_tensor_slices(
    (MNIST.train.images, MNIST.train.labels)
    )
#print(MNIST.train.labels)
#print(MNIST.train.images)

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

# metto in X e Y batch_size elementi con le rispettive labels
X, Y = iterator.get_next()

# RICHIAMO LA FUNZIONE
logits = cnn_model_fn(X, MODE)

# DICHIARO LE ETICHETTE CHE AVREI APPLICATO AD OGNI IMMAGINE
prediction = tf.nn.softmax(logits)

# TASSO DI ERRORE
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))

# DICHIARO UN OPTIMIZER CHE MODIFICA I PESI IN BASE AL LEARNING RATE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# APPLICO L'OTTIMIZZAZIONE MINIMIZZANDO GLI ERRORI
train_op = optimizer.minimize(loss)

# CONFRONTO LE MIE PREVISIONI CON QUELLE CORRETTE DEL TRAIN TEST
correct_predict = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))

# CONTROLLO L'ACCURACY CHE HO AVUTO
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

init = tf.global_variables_initializer()
best_acc=0

with tf.Session() as sess:

    sess.run(iterator.initializer)
    sess.run(init)
    saver = tf.train.Saver()

    if MODE == 'TRAIN':
        os.system('clear')
        print("TRAINING MODE")

        for step in range(1,epochs+1):
            check = '[ ]'

            sess.run(train_op)
            los, acc= sess.run([loss, accuracy])

            if step % 20 == 0  or acc >= 0.90:
                if acc >= best_acc:
                    check = '[X]'
                    best_acc = acc
                    print(str(step) + '\t' + '%.4f' % acc + '\t\t' + check)
                    saver.save(sess,save_path)
                elif step %20 == 0:
                    print(str(step) + '\t' + '%.4f' % acc + '\t\t' + check)

        writer = tf.summary.FileWriter(TensorBoard_path, sess.graph)

    elif MODE=='TEST':
        os.system('clear')
        print("TESTING MODE")
        saver.restore(sess, save_path)
        print("Initialization Complete")

        # test the model
        print("Testing Accuracy:"+str(sess.run(accuracy)))
        print("Testing finished")

    else:
        os.system('clear')
        print('\nLa modalità inserita non è corretta.\n')

sess.close()
