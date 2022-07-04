'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-07-03 20:27:59
LastEditors: ZhangHongYu
LastEditTime: 2022-07-04 17:04:55
'''
# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib import learn

import data_helper

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# File paths
tf.flags.DEFINE_string('test_data_file', None, '''Test data file path''')
tf.flags.DEFINE_string('prediction_dir', None, '''Predictions directory''')
tf.flags.DEFINE_string('model_dir', None, '''Restore the model from this run''')
tf.flags.DEFINE_string('params_dir', None, '''Restore the model from this run''')
# tf.flags.DEFINE_string('checkpoint', 'clf', '''Restore the graph from this checkpoint''')
# Test batch size
tf.flags.DEFINE_integer('batch_size', 64, 'Test batch size')
FLAGS = tf.flags.FLAGS
def main(_):
    #Restore parameters
    with open(os.path.join(FLAGS.params_dir,'params.pkl'), 'rb') as f:
        params = pkl.load(f, encoding='bytes')

    # Restore vocabulary processor
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join(FLAGS.params_dir, 'vocab'))

    # Load test data
    data, labels, lengths, _ = data_helper.load_data(file_path=FLAGS.test_data_file,
                                                    sw_path=params['stop_word_file'],
                                                    min_frequency=params['min_frequency'],
                                                    max_length=params['max_length'],
                                                    vocab_processor=vocab_processor,
                                                    shuffle=False)
    # Restore graph
    graph = tf.Graph()
    with tf.Session(graph=tf.Graph()) as sess:
        sess = tf.Session()


        tf.saved_model.loader.load(sess, ['serve'], FLAGS.model_dir)
        graph = tf.get_default_graph()


        # Get tensors
        input_x = graph.get_tensor_by_name('input_x:0')
        input_y = graph.get_tensor_by_name('input_y:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        predictions = graph.get_tensor_by_name('softmax/predictions:0')
        accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')

        # Generate batches
        batches = data_helper.batch_iter(data, labels, lengths, FLAGS.batch_size, 1)

        num_batches = int(len(data)/FLAGS.batch_size)
        all_predictions = []
        sum_accuracy = 0
        # Test
        for batch in batches:
            x_test, y_test, x_lengths = batch
            if params['clf'] == 'cnn':
                feed_dict = {input_x: x_test, input_y: y_test, keep_prob: 1.0}
                batch_predictions, batch_accuracy = sess.run([predictions, accuracy], feed_dict)
            else:
                batch_size = graph.get_tensor_by_name('batch_size:0')
                sequence_length = graph.get_tensor_by_name('sequence_length:0')
                feed_dict = {input_x: x_test, input_y: y_test, batch_size: FLAGS.batch_size, sequence_length: x_lengths, keep_prob: 1.0}

                batch_predictions, batch_accuracy = sess.run([predictions, accuracy], feed_dict)
            sum_accuracy += batch_accuracy
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        final_accuracy = sum_accuracy / num_batches

    # Print test accuracy
    print('Test accuracy: {}'.format(final_accuracy))

    if not os.path.exists(FLAGS.prediction_dir):
        os.makedirs(FLAGS.prediction_dir)

    # Save all predictions
    prediction_path = os.path.join(FLAGS.prediction_dir, "prediction.csv")
    with open(prediction_path, 'w', encoding='utf-8', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['True class', 'Prediction'])
        for i in range(len(all_predictions)):
            csvwriter.writerow([labels[i], all_predictions[i]])
        print('Predictions saved to {}'.format(prediction_path))
if __name__ =="__main__":
    tf.app.run()
