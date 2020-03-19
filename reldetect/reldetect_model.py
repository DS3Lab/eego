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

# Machine learning model for sentiment classification (binary and ternary)


def lstm_classifier(features, labels, embedding_type, param_dict, random_seed_value):

    # set random seed
    np.random.seed(random_seed_value)
    tf.random.set_seed(random_seed_value)
    os.environ['PYTHONHASHSEED'] = str(random_seed_value)

    start = time.time()

    X = list(features.keys())
    y = list(labels.values())
    y = np.asarray(y)
    #print(y.shape)

    # todo: check number of rels per sentence
    # plot sample distribution
    # ml_helpers.plot_label_distribution(y)

    # convert class labels to one hot vectors
    #y = np_utils.to_categorical(y)

    vocab_size = 100000

    # prepare text samples
    print('Processing text dataset')

    print('Found %s texts.' % len(X))

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    print(type(sequences))
    max_length = max([len(s) for s in sequences])
    print("max: ", max_length)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    num_words = min(vocab_size, len(word_index) + 1)

    if embedding_type is 'none':

        X_data = pad_sequences(sequences, maxlen=max_length)
        print('Shape of data tensor:', X_data.shape)
        #print('Shape of label tensor:', y.shape)

    if embedding_type is 'glove':

        X_data = pad_sequences(sequences, maxlen=max_length)
        print('Shape of data tensor:', X_data.shape)
        #print('Shape of label tensor:', y.shape)

        print("Loading Glove embeddings...")
        embedding_dim = 300
        embedding_matrix = ml_helpers.load_glove_embeddings(vocab_size, word_index, embedding_dim)

    if embedding_type is 'bert':
        print("Prepare sequences for Bert ...")
        X_data_bert = ml_helpers.prepare_sequences_for_bert(X)
        embedding_dim = 768

        X_data = pad_sequences(X_data_bert, maxlen=max_length)

        print('Shape of data tensor:', X_data.shape)
        #print('Shape of label tensor:', y.shape)

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

        print(y_train)

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

        """
        elif embedding_type is 'bert':

            ml_helpers.createBertLayer()

            def createModel():
                global model

                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(max_length,), dtype='int32', name='input_ids'),
                    bert_layer,
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(y_train.shape[1], activation=tf.nn.softmax)
                ])

                model.build(input_shape=(None, max_length))

                model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001),
                              metrics=['accuracy'])

                print(model.summary())

            createModel()

        """

        model.summary()

        for l in list(range(lstm_layers)):
            if l < lstm_layers-1:
                model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim, return_sequences=True)))
            else:
                model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim)))
        model.add(tf.keras.layers.Dense(dense_dim, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=dropout))
        model.add(tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=lr),
                      metrics=['accuracy'])

        model.summary()

        # train model
        history = model.fit(X_train, y_train, class_weight='balanced', validation_split=0.1, epochs=epochs, batch_size=batch_size)

        # evaluate model
        scores = model.evaluate(X_test, y_test, verbose=0)
        predictions = model.predict(X_test)

        rounded_predictions = [np.argmax(p) for p in predictions]
        rounded_labels = np.argmax(y_test, axis=1)

        # todo: micro or macro for multulabel?
        p, r, f, support = sklearn.metrics.precision_recall_fscore_support(rounded_labels, rounded_predictions, average='macro')

        # todo: add f1-score threshold
        # https://medium.com/towards-artificial-intelligence/keras-for-multi-label-text-classification-86d194311d0e
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for val in thresholds:
            print("For threshold:", val)
            pred = predictions.copy()

            pred[pred >= val] = 1
            pred[pred < val] = 0

            precision = sklearn.metrics.precision_score(y_test, pred, average='micro')
            recall = sklearn.metrics.recall_score(y_test, pred, average='micro')
            f1 = sklearn.metrics.f1_score(y_test, pred, average='micro')

            print("Micro-average quality numbers")
            print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
            print("-----")

        # save results
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
    fold_results['training_time'] = [elapsed]

    return fold_results



