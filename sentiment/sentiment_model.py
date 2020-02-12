import os
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import LSTM, Embedding
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import one_hot
from tensorflow.python.keras.utils import np_utils

from tensorflow.python.keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics
from tensorflow.python.keras.utils import to_categorical
import ml_helpers
from sklearn.model_selection import KFold
import ml_helpers


os.environ['KERAS_BACKEND'] = 'tensorflow'
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)


def lstm_classfier(features, labels):

    X = list(features.keys())
    y = list(labels.values())
    print(y)

    # plot sample distribution
    # ml_helpers.plot_label_distribution(y)

    # convert class labels to one hot vectors
    y = np_utils.to_categorical(y)
    print(y)

    # integer encode the documents
    # todo: add lookup for embeddings here
    vocab_size = 100000
    encoded_X = [one_hot(d, vocab_size) for d in X]
    #print(encoded_X)

    # todo: pad sequences
    max_length = len(np.amax(encoded_X, axis=0))
    print(max_length)
    padded_X = pad_sequences(encoded_X, maxlen=max_length, padding='post')
    #print(padded_X)

    # split data into train/test
    kf = KFold(n_splits=3, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = np.array(padded_X)[train_index], np.array(padded_X)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        print(X_train.shape)
        print(X_test.shape)  # test samples
        print(y_train.shape)
        print(y_test.shape)  # test labels

        # reset model
        K.clear_session()

        lstm_dim = 128
        dense_dim = 64
        dropout = 0.5
        batch_size = 20
        epochs = 5

        model = Sequential()
        model.add(Embedding(vocab_size, 32, input_length=max_length))
        model.add(LSTM(lstm_dim))
        model.add(Dense(dense_dim, activation='relu'))
        model.add(Dropout(rate=dropout))
        model.add(Dense(y_train.shape[1], activation='sigmoid'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(lr=0.1),
                      metrics=['accuracy'])

        model.summary()

        # train model
        model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size)

        # evaluate model
        scores = model.evaluate(X_test, y_test, verbose=0)
        predictions = model.predict(X_test)
        #print(predictions)
        #rounded_labels = np.argmax(predictions, axis=0)
        #print(rounded_labels)

        p, r, f, support = sklearn.metrics.precision_recall_fscore_support(y_test, binary_pred, average='macro')
        print(p, r, f)

        # neg = 0 and pos = 1
        #conf_matrix = sklearn.metrics.confusion_matrix(y_test, binary_pred)
        #print(conf_matrix)

        return scores[0], scores[1], predictions

