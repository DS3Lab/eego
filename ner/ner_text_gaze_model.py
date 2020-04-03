from tensorflow.python.keras.preprocessing.text import hashing_trick
import sklearn.metrics
import os
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.initializers import Constant
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Flatten, Dropout, TimeDistributed
from tensorflow.python.keras.layers.merge import concatenate
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


def lstm_classifier(features, labels, gaze, embedding_type, param_dict, random_seed_value):
    # set random seed
    np.random.seed(random_seed_value)
    tf.random.set_seed(random_seed_value)
    os.environ['PYTHONHASHSEED'] = str(random_seed_value)

    start = time.time()

    y = list(labels.values())
    X_tokenized = list(features.values())

    # check order of sentences in labels and features dicts
    """
    sents_y = list(labels.keys())
    sents_gaze = list(gaze.keys())
    if sents_y[0] != sents_gaze[0]:
        sys.exit("STOP! Order of sentences in labels and features dicts not the same!")
    """
    vocab_size = 4633
    sequences = []
    word_index = {}
    for sent in X_tokenized:
        seq = hashing_trick(' '.join(sent), vocab_size, hash_function=None, filters='', lower=False, split=' ')
        for token, number in zip(sent, seq):
            if token not in word_index:
                word_index[token] = number
        sequences.append(seq)

    label_names = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}

    # plot sample distribution
    # ml_helpers.plot_label_distribution(y)

    # prepare text samples
    print('Processing text dataset...')

    print('Found %s sentences.' % len(sequences))

    max_length = max([len(s) for s in sequences])

    print('Found %s unique tokens.' % len(word_index))
    num_words = min(vocab_size, len(word_index) + 1)

    if embedding_type is 'none':
        X_data_text = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        print('Shape of data tensor:', X_data_text.shape)

    if embedding_type is 'glove':
        X_data_text = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        print('Shape of data tensor:', X_data_text.shape)

        print("Loading Glove matrix...")
        embedding_dim = 300
        embedding_matrix = ml_helpers.load_glove_embeddings(vocab_size, word_index, embedding_dim)

    if embedding_type is 'bert':
        # todo: try cased model
        print("Prepare sequences for Bert ...")

        max_length = ml_helpers.get_bert_max_len(X)
        X_data_text, X_data_masks = ml_helpers.prepare_sequences_for_bert_with_mask(X, max_length)

        print('Shape of data tensor:', X_data_text.shape)
        print('Shape of data (masks) tensor:', X_data_masks.shape)

    print("Maximum sentence length: ", max_length)
    # pad label sequences too
    y_padded = pad_sequences(y, maxlen=max_length, value=0, padding='post', truncating='post')
    print('Shape of label tensor:', y_padded.shape)

    # prepare gaze data
    print('Processing gaze data...')
    # prepare eye-tracking data
    gaze_X = []
    max_length_cogni = 0
    # average cognitive features over all subjects
    for s in gaze.values():
        sent_feats = []
        max_length_cogni = len(s) if len(s) > max_length_cogni else max_length_cogni
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

    for train_index, test_index in kf.split(X_data_text):

        print("FOLD: ", fold)
        print("splitting train and test data...")
        y_train, y_test = y_padded[train_index], y_padded[test_index]
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

        # define model
        print("Preparing model...")

        # define two sets of inputs
        input_text = Input(shape=(X_train_text.shape[1],)) if embedding_type is not 'bert' else Input(shape=(X_train_text.shape[1],), dtype=tf.int32)
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
            input_mask = tf.keras.layers.Input((X_train_masks.shape[1],), dtype=tf.int32)
            input_text_list.append(input_mask)
            text_model = ml_helpers.create_new_bert_layer()(input_text, attention_mask=input_mask)[0]

        text_model = Dropout(0.2)(text_model)
        for _ in list(range(lstm_layers)):
            text_model = Bidirectional(LSTM(lstm_dim, recurrent_dropout=0.2, dropout=0.2, return_sequences=True))(text_model)

        #text_model = TimeDistributed(Dense(len(label_names), activation='softmax'))(text_model)
        text_model = Flatten()(text_model)
        text_model = Dense(dense_dim, activation="relu")(text_model)
        text_model = Dropout(dropout)(text_model)
        # # todo: also train this dense latent dim?
        text_model = Dense(len(label_names), activation="relu")(text_model)

        text_model_model = Model(inputs=input_text_list, outputs=text_model)

        text_model_model.summary()

        # the second branch operates on the second input (EEG data)
        cognitive_model = Bidirectional(LSTM(lstm_dim, recurrent_dropout=0.2, dropout=0.2, return_sequences=True))(input_gaze)
        #text_model = TimeDistributed(Dense(len(label_names), activation='softmax'))(text_model)
        cognitive_model = Flatten()(cognitive_model)
        cognitive_model = Dense(dense_dim, activation="relu")(cognitive_model)
        cognitive_model = Dropout(dropout)(cognitive_model)
        # # todo: also train this dense latent dim?
        cognitive_model = Dense(len(label_names), activation="relu")(cognitive_model)
        cognitive_model_model = Model(inputs=input_gaze, outputs=cognitive_model)

        cognitive_model_model.summary()

        # combine the output of the two branches
        # todo: try add, substract, average and dot product in addition to concat
        combined = concatenate([text_model_model.output, cognitive_model_model.output])
        # apply another dense layer and then a softmax prediction on the combined outputs
        # todo: does this layer help?
        # combi_model = Dense(2, activation="relu")(combined)
        combi_model = Dense(len(label_names), activation="softmax")(combined)

        model = Model(inputs=[text_model_model.input, cognitive_model_model.input], outputs=combi_model)

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=lr),
                      metrics=['accuracy'])

        # train model
        history = model.fit(
            [X_train_text, X_train_gaze] if embedding_type is not 'bert' else [X_train_text, X_train_masks,
                                                                               X_train_gaze], y_train,
            validation_split=0.1, epochs=epochs, batch_size=batch_size)

        # evaluate model
        scores = model.evaluate(
            [X_test_text, X_test_gaze] if embedding_type is not 'bert' else [X_test_text, X_test_masks, X_test_gaze],
            y_test,
            verbose=0)
        predictions = model.predict(
            [X_test_text, X_test_gaze] if embedding_type is not 'bert' else [X_test_text, X_test_masks, X_test_gaze])

        # get predictions
        # remove padded tokens at end of sentence
        out_pred = []
        out_test = []
        for pred_i, test_i, sent_i in zip(predictions, y_test, X_test_text):
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
