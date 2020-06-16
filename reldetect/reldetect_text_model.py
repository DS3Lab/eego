import os
import numpy as np
from tensorflow.python.keras.initializers import Constant
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Input, Dense, concatenate, Embedding, LSTM, Bidirectional, Flatten, Dropout
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


# Machine learning model for relation detection (11 classes)
# Only learning from text


def lstm_classifier(features, labels, embedding_type, param_dict, random_seed_value, threshold):
    # set random seed
    np.random.seed(random_seed_value)
    tf.random.set_seed(random_seed_value)
    os.environ['PYTHONHASHSEED'] = str(random_seed_value)

    start = time.time()

    # check order of sentences in labels and features dicts
    sents_y = list(labels.keys())
    sents_feats = list(features.keys())
    if sents_y[0] != sents_feats[0]:
        sys.exit("STOP! Order of sentences in labels and features dicts not the same!")

    X = list(features.keys())
    y = list(labels.values())

    # plot sample distribution
    #ml_helpers.plot_label_distribution(y)
    print("Label distribution:")
    for cl in range(len(y[0])):
        class_count = [1 if int(n[cl]) == 1 else 0 for n in y]
        class_count = sum(class_count)
        print(cl, class_count)

    ml_helpers.drop_classes(y)

    print("Label distribution:")
    for cl in range(len(y[0])):
        class_count = [1 if int(n[cl]) == 1 else 0 for n in y]
        class_count = sum(class_count)
        print(cl, class_count)

    # these are already one hot categorical encodings
    y = np.asarray(y)

    # prepare text samples
    X_data_text, num_words, text_feats = ml_helpers.prepare_text(X, embedding_type, random_seed_value)

    # split data into train/test
    kf = KFold(n_splits=config.folds, random_state=random_seed_value, shuffle=True)

    fold = 0
    fold_results = {}
    all_labels = []
    all_predictions = []

    for train_index, test_index in kf.split(X_data_text):

        print("FOLD: ", fold)
        # print("TRAIN:", train_index, "TEST:", test_index)
        print("splitting train and test data...")
        y_train, y_test = y[train_index], y[test_index]
        X_train_text, X_test_text = X_data_text[train_index], X_data_text[test_index]

        print(y_train.shape)
        print(y_test.shape)
        print(X_train_text.shape)
        print(X_test_text.shape)

        if embedding_type is 'bert':
            X_train_masks, X_test_masks = text_feats[train_index], text_feats[test_index]
            if config.data_percentage > 0:
                X_train_text, X_train_masks, y_train = ml_helpers.drop_train_sents([X_train_text, X_train_masks, y_train])
        else:
            if config.data_percentage > 0:
                X_train_text, y_train = ml_helpers.drop_train_sents([X_train_text, y_train])


        print(y_train.shape)
        print(y_test.shape)
        print(X_train_text.shape)
        print(X_test_text.shape)

        print("Label distribution:")
        for cl in range(len(y_train[0])):
            class_count = [1 if int(n[cl]) == 1 else 0 for n in y_train]
            class_count = sum(class_count)
            print(cl, class_count)

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
        input_text = Input(shape=(X_train_text.shape[1],)) if embedding_type is not 'bert' else Input(
            shape=(X_train_text.shape[1],), dtype=tf.int32)
        input_list = [input_text]

        # the first branch operates on the first input (word embeddings)
        if embedding_type is 'none':
            text_model = Embedding(num_words, 32, input_length=X_train_text.shape[1],
                                   name='none_input_embeddings')(input_text)
        elif embedding_type is 'glove':
            text_model = Embedding(num_words,
                                   300,  # glove embedding dim
                                   embeddings_initializer=Constant(text_feats),
                                   input_length=X_train_text.shape[1],
                                   trainable=False,
                                   name='glove_input_embeddings')(input_text)
        elif embedding_type is 'bert':
            input_mask = tf.keras.layers.Input((X_train_masks.shape[1],), dtype=tf.int32)
            input_list.append(input_mask)
            text_model = ml_helpers.create_new_bert_layer()(input_text, attention_mask=input_mask)[0]

        for l in list(range(lstm_layers)):
            text_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(text_model)
        text_model = Flatten()(text_model)
        text_model = Dense(dense_dim, activation="relu")(text_model)
        text_model = Dropout(dropout)(text_model)

        # https://towardsdatascience.com/multi-label-image-classification-with-neural-network-keras-ddc1ab1afede
        text_model = Dense(y_train.shape[1], activation="sigmoid")(text_model)

        model = Model(inputs=input_list, outputs=text_model)

        model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=lr),
                      metrics=[tf.keras.metrics.Precision(), 'accuracy'])

        model.summary()

        # callbacks for early stopping and saving the best model
        early_stop, model_save, model_name = ml_helpers.callbacks(fold, random_seed_value)

        # train model
        history = model.fit([X_train_text] if embedding_type is not 'bert' else [X_train_text, X_train_masks], y_train,
                            validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[early_stop, model_save])
        print("Best epoch:", len(history.history['loss']) - config.patience)

        # evaluate model
        # load the best saved model
        model.load_weights(model_name)

        scores = model.evaluate([X_test_text] if embedding_type is not 'bert' else [X_test_text, X_test_masks], y_test,
                                verbose=0)
        predictions = model.predict([X_test_text] if embedding_type is not 'bert' else [X_test_text, X_test_masks])

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
        per_class_results = sklearn.metrics.classification_report(y_test, pred, target_names=label_names, output_dict=True)

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

    print(sklearn.metrics.classification_report(all_labels, all_predictions))
    #conf_matrix = sklearn.metrics.confusion_matrix(all_labels, all_predictions)  # todo: add labels
    #print(conf_matrix)
    #ml_helpers.plot_confusion_matrix(conf_matrix)
    #ml_helpers.plot_prediction_distribution(all_labels, all_predictions)

    return fold_results
