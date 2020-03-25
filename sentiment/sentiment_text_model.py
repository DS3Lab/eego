import os
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.initializers import Constant
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Flatten, Dropout
from tensorflow.python.keras.models import Model
import sklearn.metrics
from sklearn.model_selection import KFold
import ml_helpers
import config
import time
from datetime import timedelta
import tensorflow as tf

os.environ['KERAS_BACKEND'] = 'tensorflow'


# Machine learning model for sentiment classification (binary and ternary)
# Only learning from text 


def lstm_classifier(features, labels, embedding_type, param_dict, random_seed_value):
    # set random seed
    np.random.seed(random_seed_value)
    tf.random.set_seed(random_seed_value)
    os.environ['PYTHONHASHSEED'] = str(random_seed_value)

    start = time.time()

    X = list(features.keys())
    y = list(labels.values())

    # plot sample distribution
    # ml_helpers.plot_label_distribution(y)

    # convert class labels to one hot vectors
    y = np_utils.to_categorical(y)

    vocab_size = 100000

    # prepare text samples
    print('Processing text dataset...')

    print('Found %s sentences.' % len(X))

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    max_length = max([len(s) for s in sequences])

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    num_words = min(vocab_size, len(word_index) + 1)

    if embedding_type is 'none':
        X_data_text = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        print('Shape of data tensor:', X_data_text.shape)
        print('Shape of label tensor:', y.shape)

    if embedding_type is 'glove':
        X_data_text = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        print('Shape of data tensor:', X_data_text.shape)
        print('Shape of label tensor:', y.shape)

        print("Loading Glove embeddings...")
        embedding_dim = 300
        embedding_matrix = ml_helpers.load_glove_embeddings(vocab_size, word_index, embedding_dim)

    if embedding_type is 'bert':
        print("Prepare sequences for Bert ...")

        max_length = ml_helpers.get_bert_max_len(X)
        X_data_text, X_data_masks = ml_helpers.prepare_sequences_for_bert_with_mask(X, max_length)

        print('Shape of data tensor:', X_data_text.shape)
        print('Shape of data (masks) tensor:', X_data_masks.shape)
        print('Shape of label tensor:', y.shape)

    print("Maximum sentence length: ", max_length)

    # split data into train/test
    kf = KFold(n_splits=config.folds, random_state=random_seed_value, shuffle=True)

    fold = 0
    fold_results = {}

    for train_index, test_index in kf.split(X_data_text):

        print("FOLD: ", fold)
        print("splitting train and test data...")
        y_train, y_test = y[train_index], y[test_index]
        X_train_text, X_test_text = X_data_text[train_index], X_data_text[test_index]
    
        if embedding_type is 'bert':
            X_train_masks, X_test_masks = X_data_masks[train_index], X_data_masks[test_index]

        print(y_train.shape)
        print(y_test.shape)
        print(X_train_text.shape)
        print(X_test_text.shape)

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
        input_text = Input(shape=(X_train_text.shape[1],)) if embedding_type is not 'bert' else Input(shape=(X_train_text.shape[1],), dtype=tf.int32)
        input_list = [input_text]

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
            input_mask = tf.keras.layers.Input((X_train_masks.shape[1],), dtype=tf.int32)
            input_list.append(input_mask)
            text_model = ml_helpers.create_new_bert_layer()(input_text, attention_mask=input_mask)[0]

        for l in list(range(lstm_layers)):
            print(l)
            text_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(text_model)
        text_model = Flatten()(text_model)
        text_model = Dense(dense_dim, activation="relu")(text_model)
        text_model = Dropout(dropout)(text_model)
        text_model = Dense(y_train.shape[1], activation="softmax")(text_model)

        model = Model(inputs=input_list, outputs=text_model)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=lr),
                      metrics=['accuracy'])

        model.summary()

        # train model
        history = model.fit([X_train_text] if embedding_type is not 'bert' else [X_train_text, X_train_masks], y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size)

        # evaluate model
        scores = model.evaluate([X_test_text] if embedding_type is not 'bert' else [X_test_text, X_test_masks], y_test, verbose=0)
        predictions = model.predict([X_test_text] if embedding_type is not 'bert' else [X_test_text, X_test_masks])

        rounded_predictions = [np.argmax(p) for p in predictions]
        rounded_labels = np.argmax(y_test, axis=1)
        p, r, f, support = sklearn.metrics.precision_recall_fscore_support(rounded_labels, rounded_predictions,
                                                                           average='macro')
        print(p, r, f)
        # conf_matrix = sklearn.metrics.confusion_matrix(rounded_labels, rounded_predictions)
        # print(conf_matrix)

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
