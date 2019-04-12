import tensorflow as tf
import random
# from __future__ import print_function
import numpy
import numpy as np
from PIL import Image
from PIL import ImageChops
from os import listdir
from os.path import isfile, join
import os

import load_dataset
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
get_images = load_dataset.get_images
next_batch = load_dataset.next_batch

os.system('clear')

# NOTA: non ho cambiato rete perche' questa rete va molto bene.
# Raggiunge velocemente la training accuracy al 99%.
# Il problema e' l'overfitting che non fa crescere molto la test accuracy.
# Sto facendo molti tentativi con i dropout e la data augmentation pero' il massimo ottenuto e' 90%
# Secondo me un problema sono le immagini molto simili fra di loro e con bassa risoluzione.

learning_rate = 0.005
training_iters = 100000
display_step = 20
minibatch = 16             # Dimensione del batch, in questo caso uso il batch intero

# Con queste variabili cambio il dropout di ogni singolo layer convoluzionale e il dropout di ogni
# singolo layer fully connected.
# Le variabili di scaling servono a ridurre le dimensioni della rete Alexnet completa.
# (Se li metti entrambi a 1 puoi guardare la shape e vedere che e' la stessa del modello alexnet)
# Ho ridotto la grandezza a 0.125 (un ottavo) perche' abbiamo poche immagini e non e' necessario averlo completo.
# Con questa configurazione e senza data augmentation ho raggiunto 90% test accuracy in 8 minuti
dout        = [0.99,0.99,1,1,0.99]
fc_dout     = [0.5,0.5]
scaling     = 0.125
scalingFc   = 0.125

# Con queste variabili faccio partire il training, il test, la data augmentation e la stampa delle shape
toggle_train                = True
toggle_test                 = True
toggle_data_augmentation    = False
toggle_print_shape          = True

n_input =48000 # h*w*c
n_classes = 2
std_dev = 1.0
img_size_h = 1000
img_size_w = 48

local_path = os.getcwd()
# Qui puoi mettere le tue directory
model_path = local_path + '/Model/'
dataset_path = local_path + '/DATASET/'

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Ho cambiato relu con elu perche' ho letto in alcuni paper che porta a migliore performance e accuracy
def conv2d(name, l_input, w, b, s):
        return tf.nn.elu( tf.nn.bias_add( tf.nn.conv2d( l_input, w, strides=[1, s, s, 1], padding='VALID'), b ) )

def max_pool(name, l_input, k, s):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID')

# Ho sostituito lrn con l2_normalize per lo stesso modivo di Relu ed elu
def norm(name, l_input, lsize):
        return tf.nn.l2_normalize(l_input,3)

def alex_net_model(_X, _weights, _biases):

        _X = tf.reshape(_X, shape=[-1, img_size_w, img_size_h])
        
        # Questi sono i placeholder con i quali passo il dropout ai diversi layer
        dropout = tf.placeholder(tf.float32)
        fc_dropout = tf.placeholder(tf.float32)

        # NOTA: ho usato tf.pad per mettere lo stesso padding usato da Alexnet, il primo layer ha padding 3, il secondo 2
        # e gli altri 1 ecc...

        print('_X.shape: ' + str(_X.shape))
        conv1 = conv2d('conv1', tf.pad(_X, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT"), _weights['wc1'], _biases['bc1'], 4)

        pool1 = max_pool('pool1', conv1, k=3, s=2)
        norm1 = norm('norm1', pool1, lsize=4)
        dropout1 = tf.nn.dropout(norm1, dropout[0])

        conv2 = conv2d('conv2', tf.pad(dropout1, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT"), _weights['wc2'], _biases['bc2'], s=1)
        pool2 = max_pool('pool2', conv2, k=3, s=2)
        norm2 = norm('norm2', pool2, lsize=4)
        dropout2 = tf.nn.dropout(norm2, dropout[1])
        
        # Il layer 3 e 4 di alexnet non hanno il pooling quindi lo ho tolto

        conv3 = conv2d('conv3', tf.pad(dropout2, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['wc3'], _biases['bc3'], s=1)
        norm3 = norm('norm3', conv3, lsize=4)
        dropout3 = tf.nn.dropout(norm3, dropout[2])

        conv4 = conv2d('conv4', tf.pad(dropout3, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['wc4'], _biases['bc4'], s=1)
        norm4 = norm('norm4', conv4, lsize=4)
        dropout4 = tf.nn.dropout(norm4, dropout[3])

        # Ho aggiunto la normalizzazione al layer 5

        conv5 = conv2d('conv5', tf.pad(dropout4, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT"), _weights['wc5'], _biases['bc5'], s=1)
        norm5 = norm('norm5', conv5, lsize=4)
        pool5 = max_pool('pool5', norm5, k=3, s=2)
        dropout5 = tf.nn.dropout(pool5, dropout[4])

        pool5_shape = pool5.get_shape().as_list()
        dense = tf.reshape(dropout5, [-1, pool5_shape[1] * pool5_shape[2] * pool5_shape[3]])

        # Ho aggiunto il dropout per gli strati completamente connessi.
        # Ho letto molti articoli online dove si dice che e' meglio mettere il dropout agli strati Fully Connected.
        # Generalmente si mette un dropout di 0.8/0.9 agli strati convoluzionali e 0.5 a quelli fully connected

        fc1 = tf.nn.elu(tf.matmul(dense, _weights['wd']) + _biases['bd'])
        dropfc1 = tf.nn.dropout(fc1, fc_dropout[0])

        fc2 = tf.nn.elu(tf.matmul(dropfc1, _weights['wfc']) + _biases['bfc'])
        dropfc2 = tf.nn.dropout(fc2, fc_dropout[1])

        out = tf.matmul(dropfc2, _weights['out']) + _biases['out']
        softmax_l = tf.nn.softmax(out)

        # Se metti a true puoi mostrare le dimensioni della rete
        if toggle_print_shape:
            print ("conv1.shape: ", conv1.get_shape())
            print ("pool1.shape:", pool1.get_shape())
            print ("norm1.shape:", norm1.get_shape())
            print ("conv2.shape:", conv2.get_shape())
            print ("pool2.shape:", pool2.get_shape())
            print ("norm2.shape:", norm2.get_shape())
            print ("conv3.shape:", conv3.get_shape())
            print ("conv4.shape:", conv4.get_shape())
            print ("conv5.shape:", conv5.get_shape())
            print ("pool5.shape:", pool5.get_shape())
            print ("fc1.shape:", fc1.get_shape())
            print ("fc2.shape:", fc2.get_shape())

        return out, dropout, fc_dropout

# Pesi e bias sono stati modificati per rispettare le dimensioni di Alexnet
# Con le variabili scaling e scalingFc puoi modificare la dimensione dei convolutional layer e dei Fully connected layer
weights = {
    'wc1':  tf.Variable(tf.random_normal([11, 11, 3, int(96*scaling)],                  stddev=std_dev)),
    'wc2':  tf.Variable(tf.random_normal([5, 5, int(96*scaling), int(256*scaling)],     stddev=std_dev)),
    'wc3':  tf.Variable(tf.random_normal([3, 3, int(256*scaling), int(384*scaling)],    stddev=std_dev)),
    'wc4':  tf.Variable(tf.random_normal([3, 3, int(384*scaling), int(384*scaling)],    stddev=std_dev)),
    'wc5':  tf.Variable(tf.random_normal([3, 3, int(384*scaling), int(256*scaling)],    stddev=std_dev)),
    'wd':   tf.Variable(tf.random_normal([int(9216*scaling), int(4096*scalingFc)],      stddev=std_dev)),
    'wfc':  tf.Variable(tf.random_normal([int(4096*scalingFc), int(4096*scalingFc)],    stddev=std_dev)),
    'out':  tf.Variable(tf.random_normal([int(4096*scalingFc), n_classes],              stddev=std_dev))
}

biases = {
    'bc1':  tf.Variable(tf.random_normal([int(96*scaling)])),
    'bc2':  tf.Variable(tf.random_normal([int(256*scaling)])),
    'bc3':  tf.Variable(tf.random_normal([int(384*scaling)])),
    'bc4':  tf.Variable(tf.random_normal([int(384*scaling)])),
    'bc5':  tf.Variable(tf.random_normal([int(256*scaling)])),
    'bd':   tf.Variable(tf.random_normal([int(4096*scalingFc)])),
    'bfc':  tf.Variable(tf.random_normal([int(4096*scalingFc)])),
    'out':  tf.Variable(tf.random_normal([n_classes]))
}


pred, dropout, fc_dropout = alex_net_model(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

# Con True comincia il training
if toggle_train:

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Uso sia le immagini di train che le immagini di test cosi' a ogni step posso vedere se migliora
        # la test accuracy
        l, batch_xs_train, batch_ys_train = get_images(
            files_path=dataset_path,
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            mode='TRAIN',
            randomize=True
        )

        l_test, batch_xs_test, batch_ys_test = get_images(
            files_path=dataset_path,
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            mode='TEST',
            randomize=True
        )

        print("Starting traing...")
        print("Number of training images: %d" % l)
        print("Number of test images: %d" % l_test)

        step = 0
        max_test_acc = 0
        max_test_acc_step = 0

        while step  < training_iters:

            # Prendo i batch della dimensione scelta
            batch_xs,batch_ys = next_batch(l,batch_xs_train,batch_ys_train,minibatch,step)

            train_feed = {x: batch_xs, y: batch_ys, dropout: dout, fc_dropout: fc_dout}
            test_feed = {x: batch_xs_test, y: batch_ys_test, dropout: [1,1,1,1,1], fc_dropout: [1,1]}

            sess.run(optimizer, feed_dict = train_feed)
            acc = sess.run(accuracy, feed_dict = train_feed)

            if step % display_step == 0:
                
                # Calcolo accuracy del anche del test
                test_acc = sess.run(accuracy, feed_dict = test_feed)
                
                # Memorizzo quale e' stata la test accuracy maggiore
                if test_acc>=max_test_acc:
                    max_test_acc=test_acc
                    max_test_acc_step = step

                # Stampo iterazione, training accuracy e test accuracy
                print ("It: " + str(step) + ", Training Accuracy= " + "{:.5f}".format(acc) + ", Test Accuracy = " + "{:.5f}".format(test_acc))

            # Salvo il modello quando arrivo al 99%
            if acc >= 0.99 :
                test_acc = sess.run(accuracy, feed_dict = test_feed)
                print ("It: " + str(step) + ", Training Accuracy= " + "{:.5f}".format(acc) + ", Test Accuracy = " + "{:.5f}".format(test_acc))
                step=training_iters+1
                save_path = saver.save(sess, model_path+ 'model.ckpt')
                print("Model saved in file: %s" % model_path)

            step += 1

        print("Optimization Finished!")
        print("Max test accuracy: %f" % max_test_acc)
        print("Max test accuracy step: %f" % max_test_acc_step)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict = test_feed)) 

# Con True fa il test
if toggle_test:
    with tf.Session() as sess:
        saver.restore(sess, model_path+'model.ckpt')
        l_test, batch_xs_test, batch_ys_test = get_images(
            files_path=dataset_path,
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            mode='TEST',
            randomize=True
        )
        fdict = {x: batch_xs_test, y: batch_ys_test, dropout: [1,1,1,1,1], fc_dropout: [1,1]}
        print ("Testing Accuracy: ", "{:.5f}".format(sess.run(accuracy, feed_dict=fdict)))
