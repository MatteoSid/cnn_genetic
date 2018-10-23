from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# IMPORTS
import numpy as np
import tensorflow as tf
import os
import os.path

# CNN_MODEL_FN IMPORT
import cnn_model_fn
cnn_model_fn =cnn_model_fn.cnn_model_fn

tf.logging.set_verbosity(tf.logging.INFO)

MODE = 'BOTH'
model_path = "/Users/matteo/Desktop/cnn_mnist/data/"
epochs = 1 #1
steps=100 #20000

os.system('clear')

print("*******************")
print("*    CNN_MNIST    *")
print("*******************\n")

##############
#    MAIN    #
##############
def main(unused_argv):

    # CARICO I DATI DI TRAINING E VALUTAZIONE
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    # CARICO I DATI DI TRAINING
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    # CARICO I DATI DI TEST
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # CREO L'ESTIMATORE
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir= "/Users/matteo/Desktop/cnn_mnist/data"
    )

    # SETTO UN LOGGING HOOK
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50
    )

    # ADDESTRo IL MODELLO
    if MODE == 'TRAIN' or MODE == 'BOTH':
        os.system('clear')
        print("Programma avviato in modalità TRAINING\n")
        
        with tf.Session() as sess:

            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": train_data},
                y=train_labels,
                batch_size=100,
                num_epochs=None,
                shuffle=True
            )

            mnist_classifier.train(
                input_fn=train_input_fn,
                steps=steps
                #hooks=[logging_hook]
            )

    # TESTO IL MODELLO
    if MODE == 'TEST' or MODE == 'BOTH':
        os.system('clear')
        print("Programma avviato in modalità TEST")
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=epochs,
            shuffle=False)
        
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        
        print(eval_results)

if __name__ == "__main__":
    tf.app.run()