import os
import numpy as np
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn.metrics
from sklearn.model_selection import KFold
import ml_helpers
import config
import time
from datetime import timedelta
import tensorflow as tf
import datetime
import sys
from ml_models import create_inception_cognitive_model_single, create_lstm_cognitive_model_single

d = datetime.datetime.now()

os.environ['KERAS_BACKEND'] = 'tensorflow'


# Machine learning model for sentiment classification , (binary and ternary)
# Jointly learning from text and cognitive word-level features (EEG pr eye-tracking)


def classifier(labels, gaze, embedding_type, param_dict, random_seed_value, threshold):

    # set random seeds
    np.random.seed(random_seed_value)
    tf.random.set_seed(random_seed_value)
    os.environ['PYTHONHASHSEED'] = str(random_seed_value)

    start = time.time()

    # check order of sentences in labels and features dicts
    sents_y = list(labels.keys())
    sents_gaze = list(gaze.keys())
    if sents_y[0] != sents_gaze[0]:
        sys.exit("STOP! Order of sentences in labels and features dicts not the same!")

    y = list(labels.values())
    # these are already one hot categorical encodings
    y = np.asarray(y)

    # prepare gaze data
    gaze_X, max_length_cogni = ml_helpers.prepare_cogni_seqs(gaze)
    gaze_X = ml_helpers.scale_feature_values(gaze_X)
    X_data = ml_helpers.pad_cognitive_feature_seqs(gaze_X, max_length_cogni, "eye_tracking")

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
        inception_filters = param_dict['inception_filters']
        inception_kernel_sizes = param_dict['inception_kernel_sizes']
        inception_pool_size = param_dict['inception_pool_size']
        inception_dense_dim = param_dict['inception_dense_dim']

        fold_results['params'] = [lstm_dim, lstm_layers, dense_dim, dropout, batch_size, epochs, lr, embedding_type,
                                  random_seed_value]

        # define model
        print("Preparing model...")
        if config.model is 'lstm':
            model = create_lstm_cognitive_model_single(param_dict, (X_train.shape[1], X_train.shape[2]), y_train.shape[1], 'gaze_input_tensor', random_seed_value)
        elif config.model is 'cnn':
            model = create_inception_cognitive_model_single(param_dict, (X_train.shape[1], X_train.shape[2]), y_train.shape[1], 'gaze_input_tensor', random_seed_value)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=lr),
                      metrics=['accuracy'])

        model.summary()

        # callbacks for early stopping and saving the best model
        early_stop, model_save, model_name = ml_helpers.callbacks(fold, random_seed_value)

        # train model
        history = model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[early_stop, model_save])
        print("Best epoch:", len(history.history['loss']) - config.patience)

        # evaluate model
        # load the best saved model
        model.load_weights(model_name)

        scores = model.evaluate(X_test, y_test, verbose=0)
        predictions = model.predict(X_test)

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
            fold_results['best-e'] = [len(history.history['loss'])-config.patience]
            fold_results['patience'] = config.patience
            fold_results['min_delta'] = config.min_delta
            fold_results['data_percentage'] = config.data_percentage
            fold_results['model_type'] = config.model

            if config.model is 'cnn':
                fold_results['inception_filters'] = param_dict['inception_filters'] 
                fold_results['inception_kernel_sizes'] = param_dict['inception_kernel_sizes']
                fold_results['inception_pool_size'] = param_dict['inception_pool_size']
                fold_results['inception_dense_dim'] = param_dict['inception_dense_dim']
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
            fold_results['best-e'].append(len(history.history['loss'])-config.patience)

        fold += 1

    elapsed = (time.time() - start)
    print("Training time (all folds):", str(timedelta(seconds=elapsed)))
    fold_results['training_time'] = elapsed

    #print(sklearn.metrics.classification_report(all_labels, all_predictions))
    #conf_matrix = sklearn.metrics.confusion_matrix(all_labels, all_predictions)  # todo: add labels
    #print(conf_matrix)
    #ml_helpers.plot_confusion_matrix(conf_matrix)
    #ml_helpers.plot_prediction_distribution(all_labels, all_predictions)

    return fold_results
