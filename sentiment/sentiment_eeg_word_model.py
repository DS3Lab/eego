import os
import sys
import numpy as np
import time
from datetime import timedelta
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, LSTM, Bidirectional, Flatten, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import np_utils
import tensorflow.python.keras.backend as K
import sklearn.metrics
from sklearn.model_selection import KFold
import config
import json

os.environ['KERAS_BACKEND'] = 'tensorflow'


# Machine learning model for sentiment classification (binary and ternary)
# Learning on EEG data only!

def lstm_classifier(labels, eeg, embedding_type, param_dict, random_seed_value):



    # set random seed
    np.random.seed(random_seed_value)
    tf.random.set_seed(random_seed_value)
    os.environ['PYTHONHASHSEED'] = str(random_seed_value)

    y = list(labels.values())

    # check order of sentences in labels and features dicts
    sents_y = list(labels.keys())
    sents_eeg = list(eeg.keys())
    if sents_y[0] != sents_eeg[0]:
        sys.exit("STOP! Order of sentences in labels and features dicts not the same!")

    # convert class labels to one hot vectors
    y = np_utils.to_categorical(y)

    start = time.time()

    # prepare EEG data
    eeg_X = []
    max_len = 0

    # save gaze feats
    # gaze_feats_file = open('gaze_feats_file.json', 'w')
    # json.dump(gaze, gaze_feats_file)

    # average EEG features over all subjects
    print(eeg[0])
    for s in eeg.values():
        sent_feats = []
        max_len = len(s) if len(s) > max_len else max_len
        for w, fts in s.items():
            print(fts)
            subj_mean_word_feats = np.nanmean(fts, axis=0)
            #subj_mean_word_feats[np.isnan(subj_mean_word_feats)] = 0.0
            print(subj_mean_word_feats)
            sent_feats.append(subj_mean_word_feats)
        eeg_X.append(sent_feats)

    # pad gaze sequences
    for s in eeg_X:
        while len(s) < max_len:
            s.append(np.zeros(5))

    X_data_gaze = np.array(eeg_X)
    print(X_data_gaze.shape)

    max_length_gaze = max_len



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
        input_text = Input(shape=(X_train.shape[1], X_train.shape[2]), name='gaze_input_tensor')
        # todo: change type of all layers to tf.float64?
        text_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(input_text)
        for _ in list(range(lstm_layers-1)):
            text_model = Bidirectional(LSTM(lstm_dim, recurrent_dropout=0.2, dropout=0.2, return_sequences=True))(text_model)
        text_model = Flatten()(text_model)
        text_model = Dense(dense_dim, activation="relu")(text_model)
        text_model = Dropout(dropout)(text_model)
        text_model = Dense(y_train.shape[1], activation="softmax")(text_model)

        model = Model(inputs=input_text, outputs=text_model)

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
