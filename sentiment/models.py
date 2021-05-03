from tensorflow.python.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.python.keras.layers.merge import concatenate, add, subtract, dot, maximum
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.initializers import Constant
import tensorflow as tf
import ml_helpers



def create_lstm_word_model_combi(param_dict, embedding_type, X_train_shape, num_words,
                           text_feats, seed_value):  # X_train_shape = X_train_text.shape[1]
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
                               300,  # glove embedding dim
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
    text_model = Dropout(dropout, seed=seed_value)(text_model)
    text_model = Dense(16, activation="relu")(text_model)
    text_model_model = Model(inputs=input_text_list, outputs=text_model)
    return text_model_model


def create_lstm_cognitive_model(param_dict, X_train_eeg_shape, input_tensor_name, seed_value): # X_train_eeg_shape = (X_train_eeg.shape[1], X_train_eeg.shape[2])
    lstm_dim = param_dict['lstm_dim']
    dense_dim = param_dict['dense_dim']
    dropout = param_dict['dropout']

    input_eeg = Input(shape=X_train_eeg_shape, name=input_tensor_name)
    cognitive_model = Bidirectional(LSTM(lstm_dim, return_sequences=True))(input_eeg)
    cognitive_model = Flatten()(cognitive_model)
    cognitive_model = Dense(dense_dim, activation="relu")(cognitive_model)
    cognitive_model = Dropout(dropout, seed=seed_value)(cognitive_model)
    cognitive_model = Dense(16, activation="relu")(cognitive_model)

    cognitive_model_model = Model(inputs=input_eeg, outputs=cognitive_model)
    return cognitive_model_model


def create_inception_cognitive_model(param_dict, X_train_eeg_shape, input_tensor_name, seed_value):  # X_train_eeg_shape = (X_train_eeg.shape[1], X_train_eeg.shape[2])
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

    cognitive_model = Dropout(dropout, seed=seed_value)(cognitive_model)
    cognitive_model = Dense(inception_dense_dim[1], activation='elu')(cognitive_model)

    cognitive_model_model = Model(inputs=input_eeg, outputs=cognitive_model)
    return cognitive_model_model
