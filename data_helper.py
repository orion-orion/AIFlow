# -*- coding: utf-8 -*-
import re
import os
import sys
import csv
import time
import json
import collections

import numpy as np
from tensorflow.contrib import learn
import pandas as pd

def load_data(file_path, sw_path=None, min_frequency=0, max_length=0, vocab_processor=None, shuffle=True):
    """
    Build dataset for mini-batch iterator
    :param file_path: Data file path
    :param sw_path: Stop word file path
    :param min_frequency: the minimal frequency of words to keep
    :param max_length: the max document length
    :param vocab_processor: the predefined vocabulary processor
    :param shuffle: whether to shuffle the data
    :return data, labels, lengths, vocabulary processor
    """

    data=pd.read_csv(file_path,encoding="utf-8")

    print('Building dataset ...')
    labels=data['label'].tolist()
    urls=data['content'].tolist() 

    start=time.time()

    new_urls=[]
    new_labels=[]
    if sw_path is not None:
        sw = _stop_words(sw_path)
    else:
        sw = None

    for url,label in zip(urls,labels):
        url=url.strip()
        url = url.lower()
        url = _clean_data(url, sw)  # Remove stop words and special characters
        if len(url) < 1:
            continue
        new_urls.append(url)
        if int(label) < 0:
            new_labels.append(2)
        else:
            new_labels.append(int(label))
    new_labels = np.array(new_labels)
    # print(new_urls)
    # Real lengths
    lengths = np.array(list(map(len, [url.strip().split(' ') for url in new_urls])))
    # print(lengths.　)
    if max_length == 0:
        max_length = max(lengths)

    # Extract vocabulary from sentences and map words to indices
    if vocab_processor is None:
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, min_frequency=min_frequency)
        data = np.array(list(vocab_processor.fit_transform(new_urls)))
    else:
        data = np.array(list(vocab_processor.transform(new_urls)))

    data_size = len(data)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        data = data[shuffle_indices]
        new_labels = new_labels[shuffle_indices]
        lengths = lengths[shuffle_indices]

    end = time.time()

    print('Dataset has been built successfully.')
    print('Run time: {}'.format(end - start))
    print('Number of sentences: {}'.format(len(data)))
    print('Vocabulary size: {}'.format(len(vocab_processor.vocabulary_._mapping)))
    print('Max document length: {}\n'.format(vocab_processor.max_document_length))
    return data, new_labels, lengths, vocab_processor


def batch_iter(data, labels, lengths, batch_size, num_epochs):
    """
    A mini-batch iterator to generate mini-batches for training neural network
    :param data: a list of sentences. each sentence is a vector of integers
    :param labels: a list of labels
    :param batch_size: the size of mini-batch
    :param num_epochs: number of epochs
    :return: a mini-batch iterator
    """
    assert len(data) == len(labels) == len(lengths)

    data_size = len(data)
    epoch_length = data_size // batch_size

    for _ in range(num_epochs):
        for i in range(epoch_length):
            start_index = i * batch_size
            end_index = start_index + batch_size

            xdata = data[start_index: end_index]
            ydata = labels[start_index: end_index]
            sequence_length = lengths[start_index: end_index]

            yield xdata, ydata, sequence_length

# --------------- Private Methods ---------------


def _stop_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        sw = list()
        for line in f:
            sw.append(line.strip())

    return set(sw)


def _clean_data(sent, sw):
    """ Remove special characters and stop words """
    sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
    sent = re.sub(r"\'s", " \'s", sent)
    sent = re.sub(r"\'ve", " \'ve", sent)
    sent = re.sub(r"n\'t", " n\'t", sent)
    sent = re.sub(r"\'re", " \'re", sent)
    sent = re.sub(r"\'d", " \'d", sent)
    sent = re.sub(r"\'ll", " \'ll", sent)
    sent = re.sub(r",", " , ", sent)
    sent = re.sub(r"!", " ! ", sent)
    sent = re.sub(r"\(", " \( ", sent)
    sent = re.sub(r"\)", " \) ", sent)
    sent = re.sub(r"\?", " \? ", sent)
    sent = re.sub(r"\s{2,}", " ", sent)
    if sw is not None:
        sent = "".join([word for word in sent if word not in sw])

    return sent
