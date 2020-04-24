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
import datetime

d = datetime.datetime.today()

os.environ['KERAS_BACKEND'] = 'tensorflow'


# Machine learning model for sentiment classification (binary and ternary)
# Jointly learning from text and cognitive word-level features (EEG pr eye-tracking)


def lstm_classifier(labels, eeg, gaze, embedding_type, param_dict, random_seed_value):
    # set random seed
    np.random.seed(random_seed_value)
    tf.random.set_seed(random_seed_value)
    os.environ['PYTHONHASHSEED'] = str(random_seed_value)

    y = list(labels.values())

    # plot sample distribution
    # ml_helpers.plot_label_distribution(y)

    # check order of sentences in labels and features dicts
    """
    sents_y = list(labels.keys())
    sents_text = list(features.keys())
    sents_gaze = list(gaze.keys())
    if sents_y[0] != sents_gaze[0] != sents_text[0]:
        sys.exit("STOP! Order of sentences in labels and features dicts not the same!")
    """

    # convert class labels to one hot vectors
    y = np_utils.to_categorical(y)

    start = time.time()


    # prepare EEG data
    print('Processing EEG data...')
    # load saved features
    max_len = 0
    eeg_X = []

    # average gaze features over all subjects
    for s in eeg.values():
        sent_feats = []
        max_len = len(s) if len(s) > max_len else max_len
        for w, fts in s.items():
            subj_mean_word_feats = np.nanmean(fts, axis=0)
            subj_mean_word_feats[np.isnan(subj_mean_word_feats)] = 0.0
            sent_feats.append(subj_mean_word_feats)
        eeg_X.append(sent_feats)
    print(len(eeg_X))

    # scale features
    eeg_X = ml_helpers.scale_feature_values(eeg_X)

    # pad EEG sequences
    for idx, s in enumerate(eeg_X):
        while len(s) < max_len:
            s.append(np.zeros(105))

    X_data_eeg = np.array(eeg_X)
    print(X_data_eeg.shape)


    print('Processing gaze data...')
    # prepare eye-tracking data
    gaze_X = []
    max_len = 0

    # average gaze features over all subjects
    for s in gaze.values():
        sent_feats = []
        max_len = len(s) if len(s) > max_len else max_len
        for w, fts in s.items():
            subj_mean_word_feats = np.nanmean(fts, axis=0)
            subj_mean_word_feats[np.isnan(subj_mean_word_feats)] = 0.0
            sent_feats.append(subj_mean_word_feats)
        gaze_X.append(sent_feats)
    print(len(gaze_X))

    # scale feature values
    gaze_X = ml_helpers.scale_feature_values(gaze_X)

    # pad gaze sequences
    for s in gaze_X:
        while len(s) < max_len:
            s.append(np.zeros(5))

    X_data_gaze = np.array(gaze_X)
    print(X_data_gaze.shape)

    # split data into train/test
    kf = KFold(n_splits=config.folds, random_state=random_seed_value, shuffle=True)

    fold = 0
    fold_results = {}
    all_labels = []
    all_predictions = []

    for train_index, test_index in kf.split(X_data_eeg):

        print("FOLD: ", fold)
        # print("TRAIN:", train_index, "TEST:", test_index)
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

        # define three sets of inputs
        input_eeg = Input(shape=(X_train_eeg.shape[1], X_train_eeg.shape[2]), name='eeg_input_tensor')
        input_gaze = Input(shape=(X_train_gaze.shape[1], X_train_gaze.shape[2]), name='gaze_input_tensor')


        # the second branch operates on the second input (EEG data)
        eeg_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(input_eeg)
        eeg_model = Flatten()(eeg_model)
        eeg_model = Dense(dense_dim, activation="relu")(eeg_model)
        eeg_model = Dropout(dropout)(eeg_model)
        # # todo: also train this dense latent dim?
        eeg_model = Dense(16, activation="relu")(eeg_model)
        eeg_model_model = Model(inputs=input_eeg, outputs=eeg_model)

        eeg_model_model.summary()

        # the third branch operates on the second input (gaze data)
        gaze_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(input_gaze)
        gaze_model = Flatten()(gaze_model)
        gaze_model = Dense(dense_dim, activation="relu")(gaze_model)
        gaze_model = Dropout(dropout)(gaze_model)
        # # todo: also train this dense latent dim?
        gaze_model = Dense(16, activation="relu")(gaze_model)
        gaze_model_model = Model(inputs=input_gaze, outputs=gaze_model)

        gaze_model_model.summary()

        # combine the output of the three branches
        combined = concatenate([eeg_model_model.output, gaze_model_model.output])
        # apply another dense layer and then a softmax prediction on the combined outputs
        #combined = Dense(8, activation="relu", name="final_dense")(combined)
        combi_model = Dense(y_train.shape[1], activation="softmax")(combined)

        model = Model(inputs=[eeg_model_model.input, gaze_model_model.input], outputs=combi_model)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=lr),
                      metrics=['accuracy'])

        model.summary()

        # callbacks for early stopping and saving the best model
        patience = 5
        es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.05, patience=patience)
        model_name = '../models/' + str(random_seed_value) + '_fold' + str(fold) + '_' + config.class_task + '_' + config.feature_set[0] + '_' + d.strftime(
            '%d-%m-%Y') + '.h5'
        mc = ModelCheckpoint(model_name, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

        # train model
        history = model.fit([X_train_eeg, X_train_gaze], y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[es,mc])
        print("Best epoch:",len(history.history['loss'])-patience)

        # evaluate model
        # load the best saved model
        saved_model = load_model(model_name)

        scores = saved_model.evaluate([X_test_eeg, X_test_gaze], y_test, verbose=0)
        predictions = saved_model.predict([X_test_eeg, X_test_gaze])

        rounded_predictions = [np.argmax(p) for p in predictions]
        rounded_labels = np.argmax(y_test, axis=1)
        p, r, f, support = sklearn.metrics.precision_recall_fscore_support(rounded_labels, rounded_predictions,
                                                                           average='macro')
        print(p, r, f)
        print(sklearn.metrics.classification_report(rounded_labels, rounded_predictions))
        print(sklearn.metrics.classification_report(rounded_labels, rounded_predictions, output_dict=True))

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
            fold_results['best-e'] = [len(history.history['loss']) - patience]
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
            fold_results['best-e'] = [len(history.history['loss']) - patience]

        fold += 1

    elapsed = (time.time() - start)
    print("Training time (all folds):", str(timedelta(seconds=elapsed)))
    fold_results['training_time'] = elapsed

    print(sklearn.metrics.classification_report(all_labels, all_predictions))
    conf_matrix = sklearn.metrics.confusion_matrix(all_labels, all_predictions)  # todo: add labels
    print(conf_matrix)
    ml_helpers.plot_confusion_matrix(conf_matrix)
    ml_helpers.plot_prediction_distribution(all_labels, all_predictions)

    return fold_results
