from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# IMPORTS
import os
import sys
import time
import datetime
import numpy as np
from os import listdir
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from os.path import isfile, join
from PIL import Image, ImageChops

import load_dataset
get_images = load_dataset.get_images
next_batch = load_dataset.next_batch

import cnn_model_fn
cnn_model_fn = cnn_model_fn.cnn_model_fn

os.system('clear')
local_path = os.getcwd()
# Uso path diversi in basa alla piattaforme in cui eseguo il programma
save_path = ''
TensorBoard_path = ''
dataset_path = ''
reorder_path = ''

if sys.platform == 'linux':
    print('Programma avviato su Cluster:\n')
    save_path = '/home/mdonato/Checkpoints/model.ckpt'
    TensorBoard_path = "/home/mdonato/TensorBoard"
    dataset_path = local_path + '/DATASET/'
    reorder_path = '/home/mdonato/Reorder'
elif sys.platform == 'darwin':
    print('Programma avviato su Mac:\n')
    save_path = "/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/.TensorFlow_Data/model.ckpt"
    TensorBoard_path = "/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/TensorBoard"
    dataset_path = '/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/DATASET/'
    reorder_path = '/Users/matteo/Desktop/Reorder/'

#Training Parameters
learning_rate = 0.001
batch_size = 512
epochs = 5

#HyperParameters
img_size_h = 1000
img_size_w = 48
n_input = img_size_h*img_size_w
n_classes = 2
load = True

# Variabili per l'acc
best_acc = 0
best_acc_eval = 0

# Chiedo la modalità di esecuzione in un ciclo che esce solo in caso di risposta corretta
while True:
    MODE = str(input('Slezionare in che modo si vuole eseguire il modello (TRAIN/TEST): '))
    if MODE=='TRAIN':
        batch_size = int(input('Inserire un valore per il batch_size: ' ))
        epochs = int(input('Inserire il numero di epoche da eseguire: '))
        print('\n')

        # Creo due nuovi file di log ad ogni allenamento, un file txt che replica l'output su console e un file csv con solo i valori delle accuracy per ogni esecuzione
        now = datetime.datetime.now()
        now = str(now.strftime("%Y-%m-%d_%H-%M"))
        log_acc = open('log_acc_' + now + '.txt', 'w')
        log_acc = open('log_acc_' + now + '.txt', 'a')
        log_acc.write('PARAMETRI:\n - learning_rate: ' + str(learning_rate) + '\n - batch_size: ' + str(batch_size) + '\n - epochs: ' + str(epochs) + '\n\n')
        log_acc.close()
        log_csv = open('log_csv_' + now + '.csv', 'w')
        log_csv.write('Iterazione;Accuracy\n')
        log_csv.close()
        break
    if MODE=='TEST':
        break

x = tf.placeholder(tf.float32, [None, img_size_h, img_size_w])
y = tf.placeholder(tf.float32, [None, n_classes])

# RICHIAMO LA FUNZIONE
logits = cnn_model_fn(x, MODE, log=False)
# DICHIARO LE ETICHETTE CHE AVREI APPLICATO AD OGNI IMMAGINE
prediction = tf.nn.softmax(logits)
# TASSO DI ERRORE
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
# DICHIARO UN OPTIMIZER CHE MODIFICA I PESI IN BASE AL LEARNING RATE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# APPLICO L'OTTIMIZZAZIONE MINIMIZZANDO GLI ERRORI
train_op = optimizer.minimize(loss)
# CONFRONTO LE MIE PREVISIONI CON QUELLE CORRETTE DEL TRAIN TEST
correct_predict = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
# CONTROLLO L'ACCURACY CHE HO AVUTO
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    saver = tf.train.Saver()
                
    if MODE == 'TRAIN':
        print("\nTRAINING MODE")

        len_X, X, Y= get_images(
            files_path=dataset_path,
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            mode='TRAIN',
            randomize=True
        )
        print('\n')
          
        len_X_eval, X_eval, Y_eval = get_images(
            files_path=dataset_path,
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            mode='EVAL',
            randomize=True
        )

        # print('[LOG:type x]: ' + str(type(X)))
        log_acc = open('log_acc_' + now + '.txt', 'a')
        print('[LOG: len_X]: ' + str(len_X))
        # print(X)
        print('[LOG:(len_X/batch_size)+1]: iterazioni: (' + str(len_X) + '/' + str(batch_size) + ')+1 = ' + str(int(len_X/batch_size)+1))
        log_acc.write('DATASET:\n - numero di immagini caricate: ' + str(len_X) + '\n - formato X: ' + str(X.shape) + '\n - formato Y: ' + str(Y.shape) + '\n - con un batch_size di ' + str(batch_size) + ' verranno eseguite ' + str(len_X) + '/(' + str(batch_size) + ')+1 = ' + str(int(len_X/batch_size)) + ' iterazioni per epoca\n\n')
        log_acc.close()
        
        # Carico il modello addestrato in precedenza
        if load == True:
            sav = Path('/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/.TensorFlow_Data/checkpoint')
            if sav.is_file():
                saver.restore(sess, save_path)

        # epoche
        for step in range(1,epochs+1):
            
            # iterazioni
            for i in range(0, int(len_X/batch_size)+1):
                t0 = time.time()

                X_batch, Y_batch = next_batch(
                    total=len_X,
                    images=X,
                    labels=Y,
                    batch_size=batch_size,
                    index=i
                )

                train_feed = {x: X_batch, y: Y_batch}

                sess.run(train_op, feed_dict = train_feed)
                los, acc= sess.run([loss, accuracy], feed_dict = train_feed)

                t1 = time.time()
                t = t1-t0

                ######## LOG ########
                log_acc = open('log_acc_' + now + '.txt', 'a')
                if acc >= best_acc:
                    best_acc = acc
                    print('[e:' + str(step) + ', i:' + str(i) + ']\t\t' + '%.4f' % acc + '\t\t[X]\t\t' + '%.3f' % t + 's')#
                    log_acc.write('[e:' + str(step) + ', i:' + str(i) + ']\t' +'%.4f' % acc + '\t' + '%.3f' % t + '\t[X]\n')
                    saver.save(sess, save_path)
                else:
                    print('[e:' + str(step) + ', i:' + str(i) + ']\t\t' + '%.4f' % acc + '\t\t[ ]\t\t' + '%.3f' % t + 's')
                    log_acc.write('[e:' + str(step) + ', i:' + str(i) + ']\t' +'%.4f' % acc + '\t' + '%.3f' % t + '\t[ ]\n')

                log_acc.close()
                log_csv = open('log_csv_' + now + '.csv', 'a')
                log_csv.write(str(step) + ',' + str(i) + ';' + '%.4f' % acc + '\n')
                log_csv.close()
            else: 
                # ESEGUO UN TEST SULL'EVALUATION SET ALLA FINE DI OGNI EPOCA
                train_feed = {x: X_eval, y: Y_eval}
                acc_eval = sess.run(accuracy, feed_dict=train_feed)

                if acc_eval > best_acc_eval:
                    best_acc_eval = acc_eval
                
                print('Evaluation test: ' + str(acc_eval) + '\n')
                log_acc = open('log_acc_' + now + '.txt', 'a')
                log_acc.write('Evaluation test: ' + str(acc_eval) + '\n\n')
                log_acc.close()

                log_csv = open('log_csv_' + now + '.csv', 'a')
                log_csv.write(str(step) + ',eval;;' + '%.4f' % acc_eval + '\n')
                log_csv.close()
            

        log_csv = open('log_csv_' + now + '.csv', 'a')
        log_csv.write('\n\n\n')
        log_csv.close()

        writer = tf.summary.FileWriter(TensorBoard_path, sess.graph)

    elif MODE=='TEST':

        os.system('clear')
        print("TESTING MODE")
        
        sav = Path('/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/.TensorFlow_Data/checkpoint')
        if sav.is_file():

            saver.restore(sess, save_path)

            len_X_test, X_test, Y_test = get_images(
                files_path=dataset_path,
                img_size_h=img_size_h,
                img_size_w=img_size_w,
                mode='TEST',
                randomize=True
            )

            train_feed = {x: X_test, y: Y_test}
            # test the model
            print("Testing Accuracy:"+str(sess.run(accuracy, feed_dict=train_feed)))
            print("Testing finished")
        else:
            print('Nessun modello allenato')

sess.close()
