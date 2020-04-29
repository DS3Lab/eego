import os
import numpy as np
from tensorflow.python.keras.utils import np_utils
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Flatten, Dropout
from tensorflow.python.keras.layers.merge import concatenate, add, subtract, dot, maximum
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn.metrics
from sklearn.model_selection import KFold
import ml_helpers
import config
import time
from datetime import timedelta
import tensorflow as tf
import sys
import datetime

d = datetime.datetime.now()

os.environ['KERAS_BACKEND'] = 'tensorflow'


# Machine learning model for sentiment classification (binary and ternary)
# Jointly learning from cognitive word-level features (EEG and eye-tracking)


def lstm_classifier(labels, eeg, gaze, embedding_type, param_dict, random_seed_value):

    # set random seeds
    np.random.seed(random_seed_value)
    tf.random.set_seed(random_seed_value)
    os.environ['PYTHONHASHSEED'] = str(random_seed_value)

    start = time.time()

    # check order of sentences in labels and features dicts
    sents_y = list(labels.keys())
    sents_eeg = list(eeg.keys())
    sents_gaze = list(gaze.keys())
    if sents_y[0] != sents_gaze[0] != sents_eeg[0]:
        sys.exit("STOP! Order of sentences in labels and features dicts not the same!")

    y = list(labels.values())

    # plot sample distribution
    # ml_helpers.plot_label_distribution(y)

    # convert class labels to one hot vectors
    y = np_utils.to_categorical(y)

    # prepare EEG data
    eeg_X, max_length_cogni = ml_helpers.prepare_cogni_seqs(eeg)
    eeg_X = ml_helpers.scale_feature_values(eeg_X)
    X_data_eeg = ml_helpers.pad_cognitive_feature_seqs(eeg_X, max_length_cogni)

    # prepare eye-tracking data
    gaze_X, max_length_cogni = ml_helpers.prepare_cogni_seqs(gaze)
    gaze_X = ml_helpers.scale_feature_values(gaze_X)
    X_data_gaze = ml_helpers.pad_cognitive_feature_seqs(gaze_X, max_length_cogni)

    # split data into train/test
    kf = KFold(n_splits=config.folds, random_state=random_seed_value, shuffle=True)

    fold = 0
    fold_results = {}
    all_labels = []
    all_predictions = []

    for train_index, test_index in kf.split(X_data_eeg):

        print("FOLD: ", fold)
        print("splitting train and test data...")
        y_train, y_test = y[train_index], y[test_index]
        X_train_eeg, X_test_eeg = X_data_eeg[train_index], X_data_eeg[test_index]
        X_train_gaze, X_test_gaze = X_data_gaze[train_index], X_data_gaze[test_index]

        print(y_train.shape)
        print(y_test.shape)
        print(X_train_eeg.shape)
        print(X_test_eeg.shape)
        print(X_train_gaze.shape)
        print(X_test_gaze.shape)

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

        # define two sets of inputs
        input_eeg = Input(shape=(X_train_eeg.shape[1], X_train_eeg.shape[2]), name='eeg_input_tensor')
        input_gaze = Input(shape=(X_train_gaze.shape[1], X_train_gaze.shape[2]), name='gaze_input_tensor')

        # the first branch operates on the second input (EEG data)
        eeg_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(input_eeg)
        eeg_model = Flatten()(eeg_model)
        eeg_model = Dense(dense_dim, activation="relu")(eeg_model)
        eeg_model = Dropout(dropout)(eeg_model)
        eeg_model = Dense(16, activation="relu")(eeg_model)
        eeg_model_model = Model(inputs=input_eeg, outputs=eeg_model)

        eeg_model_model.summary()

        # the second branch operates on the second input (gaze data)
        gaze_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(input_gaze)
        gaze_model = Flatten()(gaze_model)
        gaze_model = Dense(dense_dim, activation="relu")(gaze_model)
        gaze_model = Dropout(dropout)(gaze_model)
        gaze_model = Dense(16, activation="relu")(gaze_model)
        gaze_model_model = Model(inputs=input_gaze, outputs=gaze_model)

        gaze_model_model.summary()

        # combine the output of the three branches
        combined = concatenate([eeg_model_model.output, gaze_model_model.output])
        # softmax prediction on the combined outputs
        combi_model = Dense(y_train.shape[1], activation="softmax")(combined)

        model = Model(inputs=[eeg_model_model.input, gaze_model_model.input], outputs=combi_model)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=lr),
                      metrics=['accuracy'])

        model.summary()

        # callbacks for early stopping and saving the best model
        early_stop, model_save, model_name = ml_helpers.callbacks(fold, random_seed_value)

        # train model
        history = model.fit([X_train_eeg, X_train_gaze], y_train, validation_split=config.validation_split, epochs=epochs, batch_size=batch_size, callbacks=[early_stop, model_save])
        print("Best epoch:",len(history.history['loss'])-config.patience)

        # evaluate model
        # load the best saved model
        model.load_weights(model_name)

        scores = model.evaluate([X_test_eeg, X_test_gaze], y_test, verbose=0)
        predictions = model.predict([X_test_eeg, X_test_gaze])

        rounded_predictions = [np.argmax(p) for p in predictions]
        rounded_labels = np.argmax(y_test, axis=1)
        p, r, f, support = sklearn.metrics.precision_recall_fscore_support(rounded_labels, rounded_predictions,
                                                                           average='macro')
        print(p, r, f)
        print(sklearn.metrics.classification_report(rounded_labels, rounded_predictions))
        #print(sklearn.metrics.classification_report(rounded_labels, rounded_predictions, output_dict=True))

        all_labels += list(rounded_labels)
        all_predictions += list(rounded_predictions)

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
            fold_results['model'] = [model_name]
            fold_results['best-e'] = [len(history.history['loss']) - config.patience]
            fold_results['patience'] = config.patience
            fold_results['min_delta'] = config.min_delta
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
            fold_results['model'].append(model_name)
            fold_results['best-e'].append(len(history.history['loss']) - config.patience)

        fold += 1

    elapsed = (time.time() - start)
    print("Training time (all folds):", str(timedelta(seconds=elapsed)))
    fold_results['training_time'] = elapsed

    print(sklearn.metrics.classification_report(all_labels, all_predictions))
    conf_matrix = sklearn.metrics.confusion_matrix(all_labels, all_predictions)
    print(conf_matrix)
    ml_helpers.plot_confusion_matrix(conf_matrix)
    ml_helpers.plot_prediction_distribution(all_labels, all_predictions)

    return fold_results
