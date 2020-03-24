import os
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.initializers import Constant
import tensorflow.python.keras.backend as K
import sklearn.metrics
from sklearn.model_selection import KFold
import ml_helpers
import config
import time
from datetime import timedelta
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, concatenate, Embedding, LSTM, Bidirectional, Flatten, Dropout


os.environ['KERAS_BACKEND'] = 'tensorflow'


# Machine learning model for sentiment classification (binary and ternary)
# Learning on EEG data only!

def lstm_classifier(labels, gaze, embedding_type, param_dict, random_seed_value):

    # set random seed
    np.random.seed(random_seed_value)
    tf.random.set_seed(random_seed_value)
    os.environ['PYTHONHASHSEED'] = str(random_seed_value)

    start = time.time()

    y = list(labels.values())

    # plot sample distribution
    # ml_helpers.plot_label_distribution(y)

    # convert class labels to one hot vectors
    y = np_utils.to_categorical(y)

    # prepare EEG data
    gaze_X = []
    max_len = 0
    gaze_feats_file = open('gaze_feats_file.py', 'w')
    print(gaze, file=gaze_feats_file)
    for s in gaze.values():
        # average over all subjects
        sent_feats = []
        max_len = len(s) if len(s) > max_len else max_len
        for w, fts in s.items():
            #print(len(fts))
            #print(fts)
            subj_mean_word_feats = np.nanmean(fts, axis=0)
            print(w, subj_mean_word_feats)
            sent_feats.append(subj_mean_word_feats)
        print(len(sent_feats))
        gaze_X.append(sent_feats)
    print(len(gaze_X))
    print(max_len)

    for s in gaze_X:
        print(len(s))
        while len(s) < max_len:
            s.append(np.zeros(5))
        print(len(s))

    X_data_gaze = np.array(gaze_X)
    print(X_data_gaze.shape)

    max_length_gaze = max_len

    # todo: pad gaze sequences

    X_data = X_data_gaze
    max_length = max_length_gaze

    # split data into train/test
    kf = KFold(n_splits=config.folds, random_state=random_seed_value, shuffle=True)

    fold = 0
    fold_results = {}

    for train_index, test_index in kf.split(X_data):

        print("FOLD: ", fold)
        # print("TRAIN:", train_index, "TEST:", test_index)
        print("splitting train and test data...")
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X_data[train_index], X_data[test_index]

        print(y_train.shape)
        print(y_test.shape)
        print(X_train.shape)
        print(X_test.shape)

        # reset model
        K.clear_session()

        lstm_dim = param_dict['lstm_dim']
        lstm_layers = param_dict['lstm_layers']
        dense_dim = param_dict['dense_dim']
        dropout = param_dict['dropout']
        batch_size = param_dict['batch_size']
        epochs = param_dict['epochs']
        lr = param_dict['lr']

        fold_results['params'] = [lstm_dim, lstm_layers, dense_dim, dropout, batch_size, epochs, lr, embedding_type,
                                  random_seed_value]

        # define model
        print("Preparing model...")
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Input(shape=(max_length,), dtype='int32', name='input_eeg'))

        text_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(text_model)
        text_model = Flatten()(text_model)
        text_model = Dense(dense_dim, activation="relu")(text_model)
        text_model = Dropout(dropout)(text_model)
        text_model = Dense(y_train.shape[1], activation="softmax")(text_model)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=lr),
                      metrics=['accuracy'])

        model.summary()

        # train model
        history = model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size)

        # evaluate model
        scores = model.evaluate(X_test, y_test, verbose=0)
        predictions = model.predict(X_test)

        rounded_predictions = [np.argmax(p) for p in predictions]
        rounded_labels = np.argmax(y_test, axis=1)
        p, r, f, support = sklearn.metrics.precision_recall_fscore_support(rounded_labels, rounded_predictions,
                                                                           average='macro')
        print(p, r, f)
        # conf_matrix = sklearn.metrics.confusion_matrix(rounded_labels, rounded_predictions)
        # print(conf_matrix)

        if fold == 0:
            fold_results['train-loss'] = [history.history['loss']]
            fold_results['train-accuracy'] = [history.history['accuracy']]
            fold_results['val-loss'] = [history.history['val_loss']]
            fold_results['val-accuracy'] = [history.history['val_accuracy']]
            fold_results['test-loss'] = [scores[0]]
            fold_results['test-accuracy'] = [scores[1]]
            fold_results['precision'] = [p]
            fold_results['recall'] = [r]
            fold_results['fscore'] = [f]
        else:
            fold_results['train-loss'].append(history.history['loss'])
            fold_results['train-accuracy'].append(history.history['accuracy'])
            fold_results['val-loss'].append(history.history['val_loss'])
            fold_results['val-accuracy'].append(history.history['val_accuracy'])
            fold_results['test-loss'].append(scores[0])
            fold_results['test-accuracy'].append(scores[1])
            fold_results['precision'].append(p)
            fold_results['recall'].append(r)
            fold_results['fscore'].append(f)

        fold += 1

    elapsed = (time.time() - start)
    print("Training time (all folds):", str(timedelta(seconds=elapsed)))
    fold_results['training_time'] = elapsed

    return fold_results
