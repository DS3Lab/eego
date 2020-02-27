import os
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import LSTM, Embedding
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.python.keras.utils import np_utils, to_categorical
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.initializers import Constant
import sklearn.metrics
from sklearn.model_selection import KFold
import ml_helpers

# Machine learning model for TERNARY sentiment classification

seed_value = 42
os.environ['KERAS_BACKEND'] = 'tensorflow'
np.random.seed(seed_value)
import tensorflow as tf
tf.set_random_seed(seed_value)
import os
os.environ['PYTHONHASHSEED']=str(seed_value)


def lstm_classifier(features, labels, embedding_type, param_dict):

    X = list(features.keys())
    y = list(labels.values())

    # plot sample distribution
    # ml_helpers.plot_label_distribution(y)

    # convert class labels to one hot vectors
    y = np_utils.to_categorical(y)

    vocab_size = 100000

    # prepare text samples
    print('Processing text dataset')

    print('Found %s texts.' % len(X))

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    max_length = max([len(s) for s in sequences])
    print("max: ", max_length)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X_data = pad_sequences(sequences, maxlen=max_length)

    print('Shape of data tensor:', X_data.shape)
    print('Shape of label tensor:', y.shape)

    if embedding_type is 'glove':
        "Loading Glove embeddings..."
        embedding_dim = 300
        num_words = min(vocab_size, len(word_index) + 1)
        embedding_matrix = ml_helpers.load_glove_embeddings(vocab_size, word_index, embedding_dim)

    if embedding_type is 'bert':
        "Loading Bert embeddings..."
        bert_pretrained = ml_helpers.load_bert_embeddings()

    # split data into train/test
    kf = KFold(n_splits=3, random_state=seed_value, shuffle=False)

    fold = 0
    fold_results = {}

    for train_index, test_index in kf.split(X_data):

        print("FOLD: ", fold)

        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = np.array(X_data)[train_index], np.array(X_data)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        #print(X_train.shape)
        #print(X_test.shape)  # test samples
        print(y_train.shape)
        print(y_test.shape)  # test labels

        # reset model
        K.clear_session()

        lstm_dim = param_dict['lstm_dim']
        dense_dim = param_dict['dense_dim']
        dropout = param_dict['dropout']
        batch_size = param_dict['batch_size']
        epochs = param_dict['epochs']
        lr = param_dict['lr']

        fold_results['params'] = [lstm_dim, dense_dim, dropout, batch_size, epochs, lr, embedding_type]

        model = Sequential()

        if embedding_type is 'none':
            embedding_layer = Embedding(vocab_size, 32, input_length=max_length)
        elif embedding_type is 'glove':
            # load pre-trained word embeddings into an Embedding layer
            # note that we set trainable = False so as to keep the embeddings fixed
            embedding_layer = Embedding(num_words,
                                        embedding_dim,
                                        embeddings_initializer=Constant(embedding_matrix),
                                        input_length=max_length,
                                        trainable=False)
        elif embedding_type is 'bert':
            embedding_layer = bert_pretrained

        model.add(embedding_layer)
        model.add(LSTM(lstm_dim))
        model.add(Dense(dense_dim, activation='relu'))
        model.add(Dropout(rate=dropout))
        model.add(Dense(y_train.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(lr=lr),
                      metrics=['accuracy'])

        model.summary()

        # train model
        model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size)

        # evaluate model
        scores = model.evaluate(X_test, y_test, verbose=0)
        predictions = model.predict(X_test)

        rounded_predictions = [np.argmax(p) for p in predictions]
        rounded_labels = np.argmax(y_test, axis=1)
        p, r, f, support = sklearn.metrics.precision_recall_fscore_support(rounded_labels, rounded_predictions, average='macro')
        print(p, r, f)
        conf_matrix = sklearn.metrics.confusion_matrix(rounded_labels, rounded_predictions)
        print(conf_matrix)

        if fold == 0:
            fold_results['loss'] = [scores[0]]
            fold_results['test-accuracy'] = [scores[1]]
            fold_results['precision'] = [p]
            fold_results['recall'] = [r]
            fold_results['fscore'] = [f]
        else:
            fold_results['loss'].append(scores[0])
            fold_results['test-accuracy'].append(scores[1])
            fold_results['precision'].append(p)
            fold_results['recall'].append(r)
            fold_results['fscore'].append(f)

        fold += 1

    return fold_results



