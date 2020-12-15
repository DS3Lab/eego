import os
import numpy as np
from tensorflow.python.keras.initializers import Constant
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Flatten, Dropout
from tensorflow.python.keras.layers.merge import concatenate
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

d = datetime.datetime.now()

os.environ['KERAS_BACKEND'] = 'tensorflow'


# Machine learning model for sentiment classification (binary and ternary)
# Jointly learning from text and cognitive word-level features (EEG pr eye-tracking)

def create_lstm_word_model(param_dict, embedding_type, X_train_shape, num_words, text_feats): # X_train_shape = X_train_text.shape[1]
    lstm_dim = param_dict['lstm_dim']
    dense_dim = param_dict['dense_dim']
    dropout = param_dict['dropout']

    input_text = Input(shape=(X_train_shape,), name='text_input_tensor') if embedding_type is not 'bert' else Input(
            shape=(X_train_shape,), dtype=tf.int32, name='text_input_tensor')
    input_text_list = [input_text]
    
    if embedding_type is 'none':
            text_model = Embedding(num_words, 32, input_length=X_train_shape,
                  name='none_input_embeddings')(input_text)
    elif embedding_type is 'glove':
        text_model = Embedding(num_words,
                  300, # glove embedding dim
                  embeddings_initializer=Constant(text_feats),
                  input_length=X_train_shape,
                  trainable=False,
                  name='glove_input_embeddings')(input_text)
    elif embedding_type is 'bert':
        input_mask = tf.keras.layers.Input((X_train_shape,), dtype=tf.int32, name='input_mask')
        input_text_list.append(input_mask)
        text_model = ml_helpers.create_new_bert_layer()(input_text, attention_mask=input_mask)[0]
    
    text_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(text_model)
    text_model = Flatten()(text_model)
    text_model = Dense(dense_dim, activation="relu")(text_model)
    text_model = Dropout(dropout)(text_model)
    text_model = Dense(16, activation="relu")(text_model)
    text_model_model = Model(inputs=input_text_list, outputs=text_model)
    return text_model_model



def create_lstm_cognitive_model(param_dict, X_train_eeg_shape, input_tensor_name): # X_train_eeg_shape = (X_train_eeg.shape[1], X_train_eeg.shape[2])
    lstm_dim = param_dict['lstm_dim']
    dense_dim = param_dict['dense_dim']
    dropout = param_dict['dropout']

    input_eeg = Input(shape=X_train_eeg_shape, name=input_tensor_name)
    cognitive_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(input_eeg)
    cognitive_model = Flatten()(cognitive_model)
    cognitive_model = Dense(dense_dim, activation="relu")(cognitive_model)
    cognitive_model = Dropout(dropout)(cognitive_model)
    cognitive_model = Dense(16, activation="relu")(cognitive_model)    
    
    cognitive_model_model = Model(inputs=input_eeg, outputs=cognitive_model)
    return cognitive_model_model



def create_inception_cognitive_model(param_dict, X_train_eeg_shape, input_tensor_name):  # X_train_eeg_shape = (X_train_eeg.shape[1], X_train_eeg.shape[2])
    inception_filters = param_dict['inception_filters']
    inception_kernel_sizes = param_dict['inception_kernel_sizes']
    inception_pool_size = param_dict['inception_pool_size']    
    inception_dense_dim = param_dict['inception_dense_dim']
    dropout = param_dict['dropout']

    input_eeg = Input(shape=X_train_eeg_shape, name=input_tensor_name) # eeg_input_tensor / gaze_input_tensor

    conv_1 = Conv1D(filters=inception_filters, kernel_size=inception_kernel_sizes[0], activation='elu', strides=1, use_bias=False, padding='same')(input_eeg)

    conv_3 = Conv1D(filters=inception_filters, kernel_size=inception_kernel_sizes[0], activation='elu', strides=1, use_bias=False, padding='same')(input_eeg)
    conv_3 = Conv1D(filters=inception_filters, kernel_size=inception_kernel_sizes[1], activation='elu', strides=1, use_bias=False, padding='same')(conv_3)

    conv_5 = Conv1D(filters=inception_filters, kernel_size=inception_kernel_sizes[0], activation='elu', strides=1, use_bias=False, padding='same')(input_eeg)
    conv_5 = Conv1D(filters=inception_filters, kernel_size=inception_kernel_sizes[2], activation='elu', strides=1, use_bias=False, padding='same')(conv_5)

    pool_proj = MaxPooling1D(pool_size=inception_pool_size, strides=1, padding='same')(input_eeg)
    pool_proj = Conv1D(filters=inception_filters, kernel_size=inception_kernel_sizes[0], activation='elu', strides=1, use_bias=False, padding='same')(pool_proj)

    cognitive_model = concatenate([conv_1, conv_3, conv_5, pool_proj])
    cognitive_model = Flatten()(cognitive_model)
    cognitive_model = Dense(inception_dense_dim[0], activation='elu')(cognitive_model)

    cognitive_model = Dropout(dropout)(cognitive_model)
    cognitive_model = Dense(inception_dense_dim[1], activation='elu')(cognitive_model)

    cognitive_model_model = Model(inputs=input_eeg, outputs=cognitive_model)
    return cognitive_model_model



def classifier(features, labels, gaze, embedding_type, param_dict, random_seed_value, threshold):

    # set random seeds
    np.random.seed(random_seed_value)
    tf.random.set_seed(random_seed_value)
    os.environ['PYTHONHASHSEED'] = str(random_seed_value)

    start = time.time()

    # check order of sentences in labels and features dicts
    sents_y = list(labels.keys())
    sents_text = list(features.keys())
    sents_gaze = list(gaze.keys())
    if sents_y[0] != sents_gaze[0] != sents_text[0]:
        sys.exit("STOP! Order of sentences in labels and features dicts not the same!")

    X_text = list(features.keys())
    y = list(labels.values())
    # these are already one hot categorical encodings
    y = np.asarray(y)

    # prepare text samples
    X_data_text, num_words, text_feats = ml_helpers.prepare_text(X_text, embedding_type, random_seed_value)

    # prepare gaze data
    gaze_X, max_length_cogni = ml_helpers.prepare_cogni_seqs(gaze)
    gaze_X = ml_helpers.scale_feature_values(gaze_X)
    X_data_gaze = ml_helpers.pad_cognitive_feature_seqs(gaze_X, max_length_cogni, "eye_tracking")

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
        X_train_gaze, X_test_gaze = X_data_gaze[train_index], X_data_gaze[test_index]

        print(y_train.shape)
        print(y_test.shape)
        print(X_train_text.shape)
        print(X_test_text.shape)
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

        inception_filters = param_dict['inception_filters']
        inception_kernel_sizes = param_dict['inception_kernel_sizes']
        inception_pool_size = param_dict['inception_pool_size']
        inception_dense_dim = param_dict['inception_dense_dim']

        fold_results['params'] = [lstm_dim, lstm_layers, dense_dim, dropout, batch_size, epochs, lr, embedding_type,
                                  random_seed_value]

        print("Preparing model...")

        # the first branch operates on the first input (word embeddings)
        text_model_model = create_lstm_word_model(param_dict, embedding_type, X_train_text.shape[1], num_words, text_feats)
        text_model_model.summary()

        # the second branch operates on the second input (gaze data)
        if config.model is 'lstm':
            cognitive_model_model = create_lstm_cognitive_model(param_dict, (X_train_gaze.shape[1], X_train_gaze.shape[2]), 'gaze_input_tensor')
        elif config.model is 'cnn':
            cognitive_model_model = create_inception_cognitive_model(param_dict, (X_train_gaze.shape[1], X_train_gaze.shape[2]), 'gaze_input_tensor')

        cognitive_model_model.summary()

        # combine the output of the two branches
        combined = concatenate([text_model_model.output, cognitive_model_model.output])
        # apply another dense layer and then a softmax prediction on the combined outputs
        combi_model = Dense(y_train.shape[1], activation="softmax")(combined)

        model = Model(inputs=[text_model_model.input, cognitive_model_model.input], outputs=combi_model)

        model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=lr),
                      metrics=[tf.keras.metrics.Precision(), 'accuracy'])

        model.summary()

        # callbacks for early stopping and saving the best model
        early_stop, model_save, model_name = ml_helpers.callbacks(fold, random_seed_value)

        # train model
        history = model.fit([X_train_text, X_train_gaze] if embedding_type is not 'bert' else [X_train_text, X_train_masks, X_train_gaze], y_train,
                            validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[early_stop, model_save])
        print("Best epoch:", len(history.history['loss']) - config.patience)

        # evaluate model
        # load the best saved model
        model.load_weights(model_name)

        scores = model.evaluate([X_test_text, X_test_gaze] if embedding_type is not 'bert' else [X_test_text, X_test_masks, X_test_gaze], y_test,
                                verbose=0)
        predictions = model.predict([X_test_text, X_test_gaze] if embedding_type is not 'bert' else [X_test_text, X_test_masks, X_test_gaze])

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
        per_class_results = sklearn.metrics.classification_report(y_test, pred, target_names=label_names,
                                                                  output_dict=True)

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

            fold_results['model_type'] = config.model

            if config.model is 'cnn':
                fold_results['inception_filters'] = inception_filters
                fold_results['inception_kernel_sizes'] = inception_kernel_sizes
                fold_results['inception_pool_size'] = inception_pool_size
                fold_results['inception_dense_dim'] = inception_dense_dim
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
        #conf_matrix = sklearn.metrics.confusion_matrix(all_labels, all_predictions)  # todo: add labels
        #print(conf_matrix)
        #ml_helpers.plot_confusion_matrix(conf_matrix)
        #ml_helpers.plot_prediction_distribution(all_labels, all_predictions)

    return fold_results
