import os
import numpy as np
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.initializers import Constant
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Flatten, Dropout, Conv1D, MaxPooling1D
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
import sys
from ml_models import create_lstm_cognitive_model, create_inception_cognitive_model, create_lstm_word_model_combi

d = datetime.datetime.now()

os.environ['KERAS_BACKEND'] = 'tensorflow'


# Machine learning model for sentiment classification (binary and ternary)
# Jointly learning from text and cognitive word-level features (EEG theta + alpha + beta + gamma)


def classifier(features, labels, eeg_theta, eeg_alpha, eeg_beta, eeg_gamma, embedding_type, param_dict, random_seed_value):

    # set random seeds
    np.random.seed(random_seed_value)
    tf.random.set_seed(random_seed_value)
    os.environ['PYTHONHASHSEED'] = str(random_seed_value)

    start = time.time()

    X_text = list(features.keys())
    y = list(labels.values())

    # plot sample distribution
    # ml_helpers.plot_label_distribution(y)

    # check order of sentences in labels and features dicts
    if list(labels.keys())[0] != list(eeg_alpha.keys())[0] != list(features.keys())[0] != list(eeg_beta.keys())[0] != list(eeg_gamma.keys())[0] != list(eeg_theta.keys())[0]:
        sys.exit("STOP! Order of sentences in labels and features dicts not the same!")

    # convert class labels to one hot vectors
    y = np_utils.to_categorical(y)

    # prepare text samples
    X_data_text, num_words, text_feats = ml_helpers.prepare_text(X_text, embedding_type, random_seed_value)

    # prepare EEG data
    theta_X, max_length_cogni = ml_helpers.prepare_cogni_seqs(eeg_theta)
    theta_X = ml_helpers.scale_feature_values(theta_X)
    X_data_theta = ml_helpers.pad_cognitive_feature_seqs(theta_X, max_length_cogni, "eeg")

    alpha_X, max_length_cogni = ml_helpers.prepare_cogni_seqs(eeg_alpha)
    alpha_X = ml_helpers.scale_feature_values(alpha_X)
    X_data_alpha = ml_helpers.pad_cognitive_feature_seqs(alpha_X, max_length_cogni, "eeg")

    beta_X, max_length_cogni = ml_helpers.prepare_cogni_seqs(eeg_beta)
    beta_X = ml_helpers.scale_feature_values(beta_X)
    X_data_beta = ml_helpers.pad_cognitive_feature_seqs(beta_X, max_length_cogni, "eeg")

    gamma_X, max_length_cogni = ml_helpers.prepare_cogni_seqs(eeg_gamma)
    gamma_X = ml_helpers.scale_feature_values(gamma_X)
    X_data_gamma = ml_helpers.pad_cognitive_feature_seqs(gamma_X, max_length_cogni, "eeg")

    # split data into train/test
    kf = KFold(n_splits=config.folds, random_state=random_seed_value, shuffle=True)

    fold = 0
    fold_results = {}
    all_labels = []
    all_predictions = []

    for train_index, test_index in kf.split(X_data_text):

        print("FOLD: ", fold)
        print("splitting train and test data...")
        y_train, y_test = y[train_index], y[test_index]
        X_train_text, X_test_text = X_data_text[train_index], X_data_text[test_index]
        if embedding_type is 'bert':
            X_train_masks, X_test_masks = text_feats[train_index], text_feats[test_index]
        X_train_alpha, X_test_alpha = X_data_alpha[train_index], X_data_alpha[test_index]
        X_train_beta, X_test_beta = X_data_beta[train_index], X_data_beta[test_index]
        X_train_gamma, X_test_gamma = X_data_gamma[train_index], X_data_gamma[test_index]
        X_train_theta, X_test_theta = X_data_theta[train_index], X_data_theta[test_index]

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
        text_model_model = create_lstm_word_model_combi(param_dict, embedding_type, X_train_text.shape[1], num_words, text_feats, random_seed_value)

        if config.model is 'lstm':
            theta_model_model = create_lstm_cognitive_model(param_dict, (X_train_theta.shape[1], X_train_theta.shape[2]), 't_input_tensor', random_seed_value)
            alpha_model_model = create_lstm_cognitive_model(param_dict, (X_train_alpha.shape[1], X_train_alpha.shape[2]), 'a_input_tensor', random_seed_value)
            beta_model_model = create_lstm_cognitive_model(param_dict, (X_train_beta.shape[1], X_train_beta.shape[2]), 'b_input_tensor', random_seed_value)
            gamma_model_model = create_lstm_cognitive_model(param_dict, (X_train_gamma.shape[1], X_train_gamma.shape[2]), 'g_input_tensor', random_seed_value)
        elif config.model is 'cnn':
            theta_model_model = create_inception_cognitive_model(param_dict, (X_train_theta.shape[1], X_train_theta.shape[2]), 't_input_tensor', random_seed_value)
            alpha_model_model = create_inception_cognitive_model(param_dict, (X_train_alpha.shape[1], X_train_alpha.shape[2]), 'a_input_tensor', random_seed_value)
            beta_model_model = create_inception_cognitive_model(param_dict, (X_train_beta.shape[1], X_train_beta.shape[2]), 'b_input_tensor', random_seed_value)
            gamma_model_model = create_inception_cognitive_model(param_dict, (X_train_gamma.shape[1], X_train_gamma.shape[2]), 'g_input_tensor', random_seed_value)


        # combine the output of the two branches
        combined = concatenate([text_model_model.output, alpha_model_model.output, beta_model_model.output, gamma_model_model.output, theta_model_model.output])
        # apply another dense layer and then a softmax prediction on the combined outputs
        combi_model = Dense(y_train.shape[1], activation="softmax")(combined)

        model = Model(inputs=[text_model_model.input, alpha_model_model.input, beta_model_model.input, gamma_model_model.input, theta_model_model.input], outputs=combi_model)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=lr),
                      metrics=['accuracy'])

        model.summary()

        # callbacks for early stopping and saving the best model
        early_stop, model_save, model_name = ml_helpers.callbacks(fold, random_seed_value)

        # train model
        history = model.fit([X_train_text, X_train_theta, X_train_alpha, X_train_beta, X_train_gamma] if embedding_type is not 'bert' else [X_train_text, X_train_masks, X_train_theta, X_train_alpha, X_train_beta, X_train_gamma], y_train,
                            validation_split=0.1, epochs=epochs, batch_size=batch_size,callbacks=[early_stop, model_save])
        print("Best epoch:", len(history.history['loss']) - config.patience)

        # load the best saved model
        model.load_weights(model_name)

        # evaluate model
        scores = model.evaluate([X_test_text, X_test_theta, X_test_alpha, X_test_beta, X_test_gamma] if embedding_type is not 'bert' else [X_test_text, X_test_masks, X_test_theta, X_test_alpha, X_test_beta, X_test_gamma], y_test,
                                verbose=0)
        predictions = model.predict([X_test_text, X_test_theta, X_test_alpha, X_test_beta, X_test_gamma] if embedding_type is not 'bert' else [X_test_text, X_test_masks, X_test_theta, X_test_alpha, X_test_beta, X_test_gamma])

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
            fold_results['best-e'] = [len(history.history['loss']) - config.patience]
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
