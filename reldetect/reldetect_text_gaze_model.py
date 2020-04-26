import os
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
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

d = datetime.datetime.today()

os.environ['KERAS_BACKEND'] = 'tensorflow'


# Machine learning model for sentiment classification (binary and ternary)
# Jointly learning from text and cognitive word-level features (EEG pr eye-tracking)


def lstm_classifier(features, labels, gaze, embedding_type, param_dict, random_seed_value, threshold):

    # set random seeds
    np.random.seed(random_seed_value)
    tf.random.set_seed(random_seed_value)
    os.environ['PYTHONHASHSEED'] = str(random_seed_value)

    X_text = list(features.keys())
    y = list(labels.values())

    # check order of sentences in labels and features dicts
    sents_y = list(labels.keys())
    sents_text = list(features.keys())
    sents_gaze = list(gaze.keys())
    if sents_y[0] != sents_gaze[0] != sents_text[0]:
        sys.exit("STOP! Order of sentences in labels and features dicts not the same!")

    # these are already one hot categorical encodings
    y = np.asarray(y)

    start = time.time()

    vocab_size = 100000

    # prepare text samples
    print('Processing text dataset...')

    print('Found %s sentences.' % len(X_text))

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_text)
    sequences = tokenizer.texts_to_sequences(X_text)
    max_length_text = max([len(s) for s in sequences])
    print("Maximum sentence length: ", max_length_text)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    num_words = min(vocab_size, len(word_index) + 1)

    if embedding_type is 'none':
        X_data_text = pad_sequences(sequences, maxlen=max_length_text, padding='post', truncating='post')
        print('Shape of data tensor:', X_data_text.shape)
        print('Shape of label tensor:', y.shape)

    if embedding_type is 'glove':
        X_data_text = pad_sequences(sequences, maxlen=max_length_text, padding='post', truncating='post')
        print('Shape of data tensor:', X_data_text.shape)
        print('Shape of label tensor:', y.shape)

        print("Loading Glove embeddings...")
        embedding_dim = 300
        embedding_matrix = ml_helpers.load_glove_embeddings(vocab_size, word_index, embedding_dim)

    if embedding_type is 'bert':
        print("Prepare sequences for Bert ...")
        max_length = ml_helpers.get_bert_max_len(X_text)
        X_data_text, X_data_masks = ml_helpers.prepare_sequences_for_bert_with_mask(X_text, max_length)

        print('Shape of data tensor:', X_data_text.shape)
        print('Shape of data (masks) tensor:', X_data_masks.shape)
        print('Shape of label tensor:', y.shape)

    # prepare gaze data
    print('Processing gaze data...')
    # prepare eye-tracking data
    gaze_X = []
    max_length_cogni = 0
    # average cognitive features over all subjects
    for s in gaze.values():
        sent_feats = []
        max_length_cogni = max(len(s),max_length_cogni)
        for w, fts in s.items():
            subj_mean_word_feats = np.nanmean(fts, axis=0)
            subj_mean_word_feats[np.isnan(subj_mean_word_feats)] = 0.0
            sent_feats.append(subj_mean_word_feats)
        gaze_X.append(sent_feats)

    # scale feature values
    gaze_X = ml_helpers.scale_feature_values(gaze_X)

    # pad gaze sequences
    for s in gaze_X:
        while len(s) < max_length_cogni:
            # 5 = number of gaze features
            s.append(np.zeros(5))

    X_data_gaze = np.array(gaze_X)
    print(X_data_gaze.shape)

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
            X_train_masks, X_test_masks = X_data_masks[train_index], X_data_masks[test_index]
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

        fold_results['params'] = [lstm_dim, lstm_layers, dense_dim, dropout, batch_size, epochs, lr, embedding_type,
                                  random_seed_value]

        print("Preparing model...")

        # define both sets of inputs
        input_text = Input(shape=(X_train_text.shape[1],), name='text_input_tensor') if embedding_type is not 'bert' else Input(
            shape=(X_train_text.shape[1],), dtype=tf.int32, name='text_input_tensor')
        input_text_list = [input_text]
        input_gaze = Input(shape=(X_train_gaze.shape[1], X_train_gaze.shape[2]), name='gaze_input_tensor')

        # the first branch operates on the first input (word embeddings)
        if embedding_type is 'none':
            text_model = Embedding(num_words, 32, input_length=X_train_text.shape[1],
                  name='none_input_embeddings')(input_text)
        elif embedding_type is 'glove':
            text_model = Embedding(num_words,
                      embedding_dim,
                      embeddings_initializer=Constant(embedding_matrix),
                      input_length=X_train_text.shape[1],
                      trainable=False,
                      name='glove_input_embeddings')(input_text)
        elif embedding_type is 'bert':
            input_mask = tf.keras.layers.Input((X_train_masks.shape[1],), dtype=tf.int32, name='input_mask')
            input_text_list.append(input_mask)
            text_model = ml_helpers.create_new_bert_layer()(input_text, attention_mask=input_mask)[0]
        text_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(text_model)
        text_model = Flatten()(text_model)
        text_model = Dense(dense_dim, activation="relu")(text_model)
        text_model = Dropout(dropout)(text_model)
        text_model = Dense(16, activation="relu")(text_model)
        text_model_model = Model(inputs=input_text_list, outputs=text_model)

        text_model_model.summary()

        # the second branch operates on the second input (gaze data)
        cognitive_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(input_gaze)
        cognitive_model = Flatten()(cognitive_model)
        cognitive_model = Dense(dense_dim, activation="relu")(cognitive_model)
        cognitive_model = Dropout(dropout)(cognitive_model)
        cognitive_model = Dense(16, activation="relu")(cognitive_model)
        cognitive_model_model = Model(inputs=input_gaze, outputs=cognitive_model)

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
        es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=config.min_delta, patience=config.patience)
        model_name = '../models/' + str(random_seed_value) + '_fold' + str(fold) + '_' + config.class_task + '_' + \
                     config.feature_set[0] + '_' + config.embeddings[0] + '_' + d.strftime(
            '%d-%m-%Y') + '.h5'
        mc = ModelCheckpoint(model_name, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

        # train model
        history = model.fit([X_train_text, X_train_gaze] if embedding_type is not 'bert' else [X_train_text, X_train_masks, X_train_gaze], y_train,
                            validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[es, mc])
        print("Best epoch:", len(history.history['loss']) - config.patience)

        # evaluate model
        # load the best saved model
        saved_model = load_model(model_name)
        scores = saved_model.evaluate([X_test_text, X_test_gaze] if embedding_type is not 'bert' else [X_test_text, X_test_masks, X_test_gaze], y_test,
                                verbose=0)
        predictions = saved_model.predict([X_test_text, X_test_gaze] if embedding_type is not 'bert' else [X_test_text, X_test_masks, X_test_gaze])

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
