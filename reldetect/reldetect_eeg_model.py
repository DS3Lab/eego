import os
import numpy as np
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Flatten, Dropout
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


def lstm_classifier(labels, eeg, embedding_type, param_dict, random_seed_value, threshold):
    # set random seed
    np.random.seed(random_seed_value)
    tf.random.set_seed(random_seed_value)
    os.environ['PYTHONHASHSEED'] = str(random_seed_value)

    y = list(labels.values())

    # check order of sentences in labels and features dicts
    """
    sents_y = list(labels.keys())
    sents_text = list(features.keys())
    sents_gaze = list(gaze.keys())
    if sents_y[0] != sents_gaze[0] != sents_text[0]:
        sys.exit("STOP! Order of sentences in labels and features dicts not the same!")
    """

    # these are already one hot categorical encodings
    y = np.asarray(y)

    start = time.time()

    # prepare EEG data
    print('Processing EEG data...')
    # prepare eye-tracking data
    eeg_X = []
    max_length_cogni = 0
    # average cognitive features over all subjects
    for s in eeg.values():
        sent_feats = []
        max_length_cogni = max(len(s),max_length_cogni)
        for w, fts in s.items():
            subj_mean_word_feats = np.nanmean(fts, axis=0)
            subj_mean_word_feats[np.isnan(subj_mean_word_feats)] = 0.0
            sent_feats.append(subj_mean_word_feats)
        eeg_X.append(sent_feats)

    # scale feature values
    eeg_X = ml_helpers.scale_feature_values(eeg_X)

    # pad EEG sequences
    for s in eeg_X:
        while len(s) < max_length_cogni:
            # 105 = number of EEG electrodes
            s.append(np.zeros(105))

    X_data = np.array(eeg_X)
    print(X_data.shape)

    # split data into train/test
    kf = KFold(n_splits=config.folds, random_state=random_seed_value, shuffle=True)

    fold = 0
    fold_results = {}
    all_labels = []
    all_predictions = []

    for train_index, test_index in kf.split(X_data):

        print("FOLD: ", fold)
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

        print("Preparing model...")
        # define model
        print("Preparing model...")
        input_eeg = Input(shape=(X_train.shape[1], X_train.shape[2]), name='eeg_input_tensor')
        eeg_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(input_eeg)
        for _ in list(range(lstm_layers - 1)):
            eeg_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(eeg_model)
        eeg_model = Flatten()(eeg_model)
        eeg_model = Dense(dense_dim, activation="relu")(eeg_model)
        eeg_model = Dropout(dropout)(eeg_model)
        eeg_model = Dense(y_train.shape[1], activation="sigmoid")(eeg_model)

        model = Model(inputs=input_eeg, outputs=eeg_model)

        model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=lr),
                      metrics=['accuracy'])

        model.summary()

        # callbacks for early stopping and saving the best model
        es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=config.min_delta, patience=config.patience)
        model_name = '../models/' + str(random_seed_value) + '_fold' + str(fold) + '_' + config.class_task + '_' + \
                     config.feature_set[0] + '_' + d.strftime(
            '%d-%m-%Y') + '.h5'
        mc = ModelCheckpoint(model_name, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

        # train model
        history = model.fit(X_train, y_train, validation_split=config.validation_split, epochs=epochs, batch_size=batch_size, callbacks=[es, mc])
        print("Best epoch:", len(history.history['loss']) - config.patience)

        # evaluate model
        # load the best saved model
        saved_model = load_model(model_name)
        scores = saved_model.evaluate(X_test, y_test, verbose=0)
        predictions = saved_model.predict(X_test)

        print("For threshold:", threshold)
        pred = predictions.copy()
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0

        p, r, f, support = sklearn.metrics.precision_recall_fscore_support(y_test, pred,
                                                                           average='micro')
        print(p, r, f)

        label_names = ["Visited", "Founder", "Nationality", "Wife", "PoliticalAffiliation", "JobTitle", "Education",
                       "Employer", "Awarded", "BirthPlace", "DeathPlace"]
        print(sklearn.metrics.classification_report(y_test, pred, target_names=label_names))

        all_labels += list(y_test)
        all_predictions += list(pred)

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
            fold_results['threshold'] = threshold
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
    conf_matrix = sklearn.metrics.confusion_matrix(all_labels, all_predictions)  # todo: add labels
    print(conf_matrix)
    ml_helpers.plot_confusion_matrix(conf_matrix)
    ml_helpers.plot_prediction_distribution(all_labels, all_predictions)

    return fold_results
