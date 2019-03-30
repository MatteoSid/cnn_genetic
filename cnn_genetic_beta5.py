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


# IMPORTS
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import load_dataset
import datetime
import time
get_images = load_dataset.get_images
next_batch = load_dataset.next_batch

import cnn_model_fn_v2
cnn_model_fn = cnn_model_fn_v2.cnn_model_fn

now = datetime.datetime.now()
now = str(now.strftime("%Y-%m-%d|%H-%M"))
log_acc = open('log_acc_' + now + '.txt', 'w')
log_acc = open('log_acc_' + now + '.txt', 'a')

os.system('clear')
local_path = os.getcwd()
# Uso path diversi in basa alla piattaforme in cui eseguo il programma
if sys.platform == 'linux':
    print('Programma avviato su Cluster:\n')
    save_path = '/home/mdonato/Checkpoints/model.ckpt'
    TensorBoard_path = "/home/mdonato/TensorBoard"
    dataset_path = local_path + '/DATASET/'
elif sys.platform == 'darwin':
    print('Programma avviato su Mac:\n')
    save_path = "/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/.TensorFlow_Data/model.ckpt"
    TensorBoard_path = "/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/TensorBoard"
    dataset_path = '/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/DATASET/'

#Training Parameters
learning_rate = 0.001
batch_size = 5
epochs = 100

#HyperParameters
inputs = 784
classes = 10
dropout = 0.75

# Chiedo la modalità di esecuzione in un ciclo che esce solo in caso di risposta corretta
# while True:
#     MODE = str(input('Slezionare in che modo si vuole eseguire il modello (TRAIN, TEST, BOTH): '))
#     if MODE=='TRAIN':
#         break
#     if MODE=='TEST':
#         break
#     if MODE=='BOTH':
#         break

MODE = 'TRAIN'
batch_size = int(input('Inserire un valore per il batch_size: ' ))
epochs = int(input('Inserire il numero di epoche da eseguire: '))
print('\n')

log_acc.write('PARAMETRI:\n - learning_rate: ' + str(learning_rate) + '\n - batch_size: ' + str(batch_size) + '\n - epochs: ' + str(epochs) + '\n\n')


len_X, X, Y = get_images(
    files_path=dataset_path,
    img_size_h=1000,
    img_size_w=48,
    mode='TRAIN',
    randomize=True
)

print('[LOG: len_X]: ' + str(len_X))
print('[LOG:len_X/batch_size]: ' + str(len_X) + '/' + str(batch_size) + ' = ' + str(int(len_X/batch_size)))
log_acc.write('DATASET:\n - numero di immagini caricate: ' + str(len_X) + '\n - formato X: ' + str(X.shape) + '\n - formato Y: ' + str(Y.shape) + '\n - con un batch_size di ' + str(batch_size) + ' verranno eseguite ' + str(len_X) + '/(' + str(batch_size) + ')+1 = ' + str(int(len_X/batch_size)) + ' iterazioni per epoca\n\n')
log_acc.close()

X_batch, Y_X_batch = next_batch(
    total=len_X,
    images=X,
    labels=Y,
    batch_size=batch_size,
    index=0
)

# RICHIAMO LA FUNZIONE
logits = cnn_model_fn(X_batch, MODE, log=True)

# DICHIARO LE ETICHETTE CHE AVREI APPLICATO AD OGNI IMMAGINE
prediction = tf.nn.softmax(logits)

# TASSO DI ERRORE
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_X_batch))

# DICHIARO UN OPTIMIZER CHE MODIFICA I PESI IN BASE AL LEARNING RATE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# APPLICO L'OTTIMIZZAZIONE MINIMIZZANDO GLI ERRORI
train_op = optimizer.minimize(loss)

# CONFRONTO LE MIE PREVISIONI CON QUELLE CORRETTE DEL TRAIN TEST
correct_predict = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y_X_batch, 1))

# CONTROLLO L'ACCURACY CHE HO AVUTO
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

init = tf.global_variables_initializer()
best_acc=0

with tf.Session() as sess:

    sess.run(init)
    saver = tf.train.Saver()

    if MODE == 'TRAIN':
        # os.system('clear')
        print("TRAINING MODE")

        for step in range(1,epochs+1):
            for i in range(0, int(len_X/batch_size)+1):
                t0 = time.time()

                if i > 0:
                    X_batch, Y_X_batch = next_batch(
                        total=len_X,
                        images=X,
                        labels=Y,
                        batch_size=batch_size,
                        index=i
                    )


                sess.run(train_op)
                los, acc= sess.run([loss, accuracy])

                t1 = time.time()
                t = t1-t0

                ######## LOG ########
                #if step % 20 == 0  or acc >= 0.90:
                check = '[ ]'
                if acc >= best_acc:
                    check = '[X]'
                    best_acc = acc
                    print('[e:' + str(step) + ', i:' + str(i) + ']\t\t' + '%.4f' % acc + '\t\t' + check + '\t\t' + '%.3f' % t + 's')
                    log_acc = open('log_acc_' + now + '.txt', 'a')
                    log_acc.write('[e:' + str(step) + ', i:' + str(i) + ']\t' +'%.4f' % acc + '\t' + '%.3f' % t + '\t' + check + '\n')
                    log_acc.close()
                    saver.save(sess,save_path)
                # elif step %20 == 0:
                else:
                    print('[e:' + str(step) + ', i:' + str(i) + ']\t\t' + '%.4f' % acc + '\t\t' + check + '\t\t' + '%.3f' % t + 's')
                    log_acc = open('log_acc_' + now + '.txt', 'a')
                    log_acc.write('[e:' + str(step) + ', i:' + str(i) + ']\t' + '%.4f' % acc + '\t' + '%.3f' % t + '\t' + check + '\n')
                    log_acc.close()

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
