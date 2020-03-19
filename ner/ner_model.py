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
import bert
import tensorflow as tf

os.environ['KERAS_BACKEND'] = 'tensorflow'

# Machine learning model for named entity recognition


def lstm_classifier(features, labels, embedding_type, param_dict, random_seed_value):

    # set random seed
    np.random.seed(random_seed_value)
    tf.random.set_seed(random_seed_value)
    os.environ['PYTHONHASHSEED'] = str(random_seed_value)

    start = time.time()

    X = list(features.keys())
    y = list(labels.values())

    label_names = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}

    # plot sample distribution
    # ml_helpers.plot_label_distribution(y)

    vocab_size = 100000

    # prepare text samples
    print('Processing text dataset...')

    print('Found %s sentences.' % len(X))

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    max_length = max([len(s) for s in sequences])
    print("Maximum sentence length: ", max_length)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    num_words = min(vocab_size, len(word_index) + 1)

    # pad label sequences too
    y_padded = pad_sequences(y, maxlen=max_length, value=0, padding='post', truncating='post')

    if embedding_type is 'none':
        X_data = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        print('Shape of data tensor:', X_data.shape)
        print('Shape of label tensor:', y_padded.shape)

    if embedding_type is 'glove':
        X_data = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        print('Shape of data tensor:', X_data.shape)
        print('Shape of label tensor:', y_padded.shape)

        print("Loading Glove embeddings...")
        embedding_dim = 300
        embedding_matrix = ml_helpers.load_glove_embeddings(vocab_size, word_index, embedding_dim)

    if embedding_type is 'bert':
        print("Prepare sequences for Bert ...")
        X_data_bert = ml_helpers.prepare_sequences_for_bert(X)
        embedding_dim = 768

        X_data = pad_sequences(X_data_bert, maxlen=max_length, padding='post', truncating='post')

        print('Shape of data tensor:', X_data.shape)
        print('Shape of label tensor:', y_padded.shape)

    # split data into train/test
    kf = KFold(n_splits=config.folds, random_state=random_seed_value, shuffle=True)

    fold = 0
    fold_results = {}

    for train_index, test_index in kf.split(X_data):

        print("FOLD: ", fold)
        # print("TRAIN:", train_index, "TEST:", test_index)
        print("splitting train and test data...")
        y_train, y_test = y_padded[train_index], y_padded[test_index]
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

        fold_results['params'] = [lstm_dim, lstm_layers, dense_dim, dropout, batch_size, epochs, lr, embedding_type, random_seed_value]

        # define model
        print("Preparing model...")
        model = tf.keras.Sequential()

        if embedding_type is 'none':
            # todo: tune embedding dim?
            embedding_layer = tf.keras.layers.Embedding(num_words, 32, input_length=max_length, name='none_input_embeddings')
            model.add(embedding_layer)

        elif embedding_type is 'glove':
            # load pre-trained word embeddings into an Embedding layer
            # note that we set trainable = False so as to keep the embeddings fixed
            embedding_layer = tf.keras.layers.Embedding(num_words,
                                        embedding_dim,
                                        embeddings_initializer=Constant(embedding_matrix),
                                        input_length=max_length,
                                        trainable=False,
                                        name='glove_input_embeddings')
            model.add(embedding_layer)

        elif embedding_type is 'bert':
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(shape=(max_length,), dtype='int32', name='input_ids'))
            bert_layer = ml_helpers.createBertLayer()
            model.add(bert_layer)

        model.summary()

        for _ in list(range(lstm_layers)):
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim, return_sequences=True)))

        # todo: try without time dist.
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(label_names), activation='softmax')))

        # todo: try diffrent loss/metric --> invalid shape error
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=lr),
                      metrics=['accuracy'])

        model.summary()

        # train model
        history = model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size)

        # evaluate model
        scores = model.evaluate(X_test, y_test, verbose=0)
        predictions = model.predict(X_test)

        # get predictions
        # remove padded tokens at end of sentence
        out_pred = []
        out_test = []
        for pred_i, test_i, sent_i in zip(predictions, y_test, X_test):
            x_cut = [x for x in sent_i if x != 0]
            original_sent_length = len(x_cut)
            out_i_pred = []
            out_i_test = []
            for p in pred_i[:original_sent_length]:
                p_i = np.argmax(p)
                out_i_pred.append(label_names[p_i])
            for t in test_i[:original_sent_length]:
                out_i_test.append(label_names[t])

            out_pred += out_i_pred
            out_test += out_i_test

        print("Accuracy with padded:", scores[1])
        print("Accuracy without padded:", sklearn.metrics.accuracy_score(out_test, out_pred))
        p, r, f, support = sklearn.metrics.precision_recall_fscore_support(out_test, out_pred, average='macro')
        print(sklearn.metrics.classification_report(out_test, out_pred))
        print(p, r, f)

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



