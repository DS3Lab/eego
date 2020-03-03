import matplotlib. pyplot as plt
import os
import numpy as np
from transformers import *
import config
import tensorflow as tf
import tensorflow_datasets

import tensorflow_hub as hub
import bert



def plot_label_distribution(y):

    print(len(set(y)))

    plt.hist(y, bins=len(set(y)), alpha=0.5)
    plt.xticks(rotation=90, fontsize=7)
    plt.show()


def load_glove_embeddings(vocab_size, word_index, EMBEDDING_DIM):

    print('Indexing word vectors.')

    embeddings_index = {}
    with open(os.path.join(
            config.base_dir + 'eego/feature_extraction/embeddings/glove.6B.'+str(EMBEDDING_DIM)+'d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    # prepare embedding matrix
    num_words = min(vocab_size, len(word_index) + 1)

    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_bert_embeddings():

    # todo: use this tutorial for Bert with TF; https://github.com/huggingface/transformers#quick-tour


    # from https://colab.research.google.com/drive/1IubZ3T7gqD09ZIVmJapiB5MXUnVGlzwH#scrollTo=lyRTv9GNzdJt
    bert_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
    bert_layer = hub.KerasLayer(bert_path, trainable=True)

    vocab_file1 = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    bert_tokenizer_tfhub = bert.bert_tokenization.FullTokenizer(vocab_file1, do_lower_case=True)

    bert_inputs = _get_inputs(df=train.head(), tokenizer=bert_tokenizer_tfhub, _maxlen=100)
    print(bert_inputs)