from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# IMPORTS
import numpy as np
import tensorflow as tf
import os

# CNN_MODEL_FN IMPORT
#import cnn_model_fn
#cnn_model_fn =cnn_model_fn.cnn_model_fn

tf.logging.set_verbosity(tf.logging.INFO)

MODE = 'TRAIN'
model_path = "/Users/matteo/Desktop/cnn_mnist/data/"

print("\n\n*******************")
print("*    CNN_MNIST    *")
print("*******************\n")

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

def serving_input_receiver_fn():
        serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensors')
        receiver_tensors      = {"predictor_inputs": serialized_tf_example}
        feature_spec          = {"words": tf.FixedLenFeature([25],tf.int64)}
        features              = tf.parse_example(serialized_tf_example, feature_spec)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def estimator_spec_for_softmax_classification(logits, labels, mode):
    predicted_classes = tf.argmax(logits, 1)
    if (mode == tf.estimator.ModeKeys.PREDICT):
        export_outputs = {'predict_output': tf.estimator.export.PredictOutput({"pred_output_classes": predicted_classes, 'probabilities': tf.nn.softmax(logits)})}
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'class': predicted_classes, 'prob': tf.nn.softmax(logits)}, export_outputs=export_outputs) # IMPORTANT!!!
    onehot_labels = tf.one_hot(labels, 31, 1, 0)
    loss          = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    if (mode == tf.estimator.ModeKeys.TRAIN):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op  = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=predicted_classes)}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def model_custom(features, labels, mode):
    bow_column           = tf.feature_column.categorical_column_with_identity("words", num_buckets=1000)
    bow_embedding_column = tf.feature_column.embedding_column(bow_column, dimension=50)   
    bow                  = tf.feature_column.input_layer(features, feature_columns=[bow_embedding_column])
    logits               = tf.layers.dense(bow, 31, activation=None)
    return estimator_spec_for_softmax_classification(logits=logits, labels=labels, mode=mode)

##############
#    MAIN    #
##############
def main(unused_argv):
    
    # CARICO I DATI DI TRAINING E VALUTAZION
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    # CARICO I DATI DI TRAINING
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    # CARICO I DATI DI TEST
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # CREO L'ESTIMATORE
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_custom,
        model_dir= "/Users/matteo/Desktop/cnn_mnist/data"
    )

    # SETTO UN LOGGING HOOK
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50
    )

    # ADDESTRo IL MODELLO
    if MODE == 'TRAIN':
        print("Programma avviato in modalità TRAINING")
        
        with tf.Session() as sess:
            
            #sess.run(init)

            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"words": features_train_set},
                y=labels_train_set,
                batch_size=batch_size_param,
                num_epochs=None,
                shuffle=True
            )


            classifier.train(
                input_fn=train_input_fn,
                steps=100
            )

            full_model_dir = classifier.export_savedmodel(
                export_dir_base="/Users/matteo/Tensorflow_Data/cnn_genetic",
                serving_input_receiver_fn=serving_input_receiver_fn
            )

    # TESTO IL MODELLO
    elif MODE == 'TEST':
        print("Programma avviato in modalità TEST")
        with tf.Session() as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], full_model_dir)
            predictor   = tf.contrib.predictor.from_saved_model(full_model_dir)
            model_input = tf.train.Example(features=tf.train.Features( feature={"words": tf.train.Feature(int64_list=tf.train.Int64List(value=features_test_set)) })) 
            model_input = model_input.SerializeToString()
            output_dict = predictor({"predictor_inputs":[model_input]})
            y_predicted = output_dict["pred_output_classes"][0]
    else:
        print("La modalità inserita non è valida")


if __name__ == "__main__":
    tf.app.run() 