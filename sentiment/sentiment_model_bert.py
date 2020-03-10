import os
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.initializers import Constant
import sklearn.metrics
from sklearn.model_selection import KFold
import ml_helpers
import config
import time
from datetime import timedelta, date
import bert


# Machine learning model for TERNARY sentiment classification

seed_value = 42
os.environ['KERAS_BACKEND'] = 'tensorflow'
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

modelBertDir = "/mnt/ds3lab-scratch/noraho/embeddings/bert"


def createBertLayer():
    global bert_layer

    bertDir = os.path.join(modelBertDir, "multi_cased_L-12_H-768_A-12")

    bert_params = bert.params_from_pretrained_ckpt(bertDir)

    bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert_layer")

    bert_layer.apply_adapter_freeze()

    print(bert_layer)

    print("Bert layer created")

def loadBertCheckpoint():
    modelsFolder = os.path.join(modelBertDir, "multi_cased_L-12_H-768_A-12")
    checkpointName = os.path.join(modelsFolder, "bert_model.ckpt")

    bert.load_stock_weights(bert_layer, checkpointName)


def lstm_classifier(features, labels, embedding_type, param_dict):

    start = time.time()

    X = list(features.keys())
    y = list(labels.values())

    # plot sample distribution
    # ml_helpers.plot_label_distribution(y)

    # convert class labels to one hot vectors
    y = np_utils.to_categorical(y)

    vocab_size = 100000

    # prepare text samples
    print('Processing text dataset')

    print('Found %s texts.' % len(X))

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    max_length = max([len(s) for s in sequences])
    print("max: ", max_length)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    num_words = min(vocab_size, len(word_index) + 1)

    if embedding_type is 'none':

        X_data = pad_sequences(sequences, maxlen=max_length)
        print('Shape of data tensor:', X_data.shape)
        print('Shape of label tensor:', y.shape)

    if embedding_type is 'glove':

        X_data = pad_sequences(sequences, maxlen=max_length)
        print('Shape of data tensor:', X_data.shape)
        print('Shape of label tensor:', y.shape)

        print("Loading Glove embeddings...")
        embedding_dim = 300
        embedding_matrix = ml_helpers.load_glove_embeddings(vocab_size, word_index, embedding_dim)
        print(embedding_matrix)
        print(embedding_matrix.shape)

    if embedding_type is 'bert':
        print("Prepare sequences for Bert ...")
        X_data = ml_helpers.prepare_sequences_for_bert(X, max_length)
        print("Bert sequences ready")
        print(X_data.shape)

        print('Shape of data tensor:', X_data.shape)
        print('Shape of label tensor:', y.shape)


    # split data into train/test
    kf = KFold(n_splits=config.folds, random_state=seed_value, shuffle=True)

    fold = 0
    fold_results = {}

    for train_index, test_index in kf.split(X_data):

        print("FOLD: ", fold)

        #print(np.array(X_data))

        #print(np.array(X_data)[train_index])

        # print("TRAIN:", train_index, "TEST:", test_index)
        # todo: why does this take so much time for BErt??
        print("splitting X")
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X_data[train_index], X_data[test_index]
        print("splitting y")


        print(X_train.shape)
        #print(X_train[0].shape)
        print(X_test.shape)  # test samples
        print(y_train.shape)
        print(y_test.shape)  # test labels

        print(X_train[0])
        print(type(X_train[0]))

        # reset model
        #K.clear_session()

        lstm_dim = param_dict['lstm_dim']
        dense_dim = param_dict['dense_dim']
        dropout = param_dict['dropout']
        batch_size = param_dict['batch_size']
        epochs = param_dict['epochs']
        lr = param_dict['lr']

        fold_results['params'] = [lstm_dim, dense_dim, dropout, batch_size, epochs, lr, embedding_type]

        print("Preparing model...")

        #model = Sequential()

        if embedding_type is 'none':
            embedding_layer = Embedding(num_words, 32, input_length=max_length, name='none_input_embeddings')

        elif embedding_type is 'glove':
            # load pre-trained word embeddings into an Embedding layer
            # note that we set trainable = False so as to keep the embeddings fixed
            embedding_layer = Embedding(num_words,
                                        embedding_dim,
                                        embeddings_initializer=Constant(embedding_matrix),
                                        input_length=max_length,
                                        trainable=False,
                                        name='glove_input_embeddings')
        elif embedding_type is 'bert':

            createBertLayer()

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

        # train model
        history = model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size)
        print(history)
        # todo: add final validation accuracy + loss to fold results

        # evaluate model
        scores = model.evaluate(X_test, y_test, verbose=0)
        predictions = model.predict(X_test)

        rounded_predictions = [np.argmax(p) for p in predictions]
        rounded_labels = np.argmax(y_test, axis=1)
        p, r, f, support = sklearn.metrics.precision_recall_fscore_support(rounded_labels, rounded_predictions, average='macro')
        print(p, r, f)
        conf_matrix = sklearn.metrics.confusion_matrix(rounded_labels, rounded_predictions)
        print(conf_matrix)

        if fold == 0:
            fold_results['loss'] = [scores[0]]
            fold_results['test-accuracy'] = [scores[1]]
            fold_results['precision'] = [p]
            fold_results['recall'] = [r]
            fold_results['fscore'] = [f]
        else:
            fold_results['loss'].append(scores[0])
            fold_results['test-accuracy'].append(scores[1])
            fold_results['precision'].append(p)
            fold_results['recall'].append(r)
            fold_results['fscore'].append(f)

        fold += 1

    elapsed = (time.time() - start)
    print("Training time (all folds):", str(timedelta(seconds=elapsed)))
    fold_results['training_time'] = [elapsed]

    return fold_results



