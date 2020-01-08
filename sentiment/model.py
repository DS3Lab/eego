import os
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import LSTM, Bidirectional
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics
from tensorflow.python.keras.utils import to_categorical


os.environ['KERAS_BACKEND'] = 'tensorflow'


def keras_lstm_binary_classfier(train_X, test_X, train_y, test_y):

    # reset model
    K.clear_session()

    # create scaler
    scaler = MinMaxScaler()
    # fit and transform in one step
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.fit_transform(test_X)

    max_input_length = train_X.shape[1]
    print(max_input_length)

    # reshape for LSTM input shape
    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

    # define model: 1-layer biLSTM, 2 dense layers with dropout before the last layer

    # 4+5
    lstm_dim = 128
    dense_dim = 64
    input_dropout = 0.38
    recurrent_dropout = 0.2
    dropout = 0.5
    learning_rate = 0.18
    optimizer = 'adam'
    batch_size = 80
    epochs = 200

    print('lstm_dim:', lstm_dim)
    print('dense_dim:', dense_dim)
    print('input_dropout:', input_dropout)
    print('recurrent_dropout:', recurrent_dropout)
    print('dropout:', dropout)
    print('learning rate:', learning_rate)
    print('optimizer:', optimizer)
    print('batch size:', batch_size)
    print('epochs:', epochs)

    max_input_length = train_X.shape[2]
    print(train_X.shape)
    print(max_input_length)

    model = Sequential()
    model.add(
        LSTM(lstm_dim, dropout=input_dropout, recurrent_dropout=recurrent_dropout, input_shape=(1, max_input_length)))
    model.add(Dense(dense_dim, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=0.1),
                  metrics=['accuracy'])

    model.summary()

    # train model
    model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size)

    # evaluate model
    scores = model.evaluate(test_X, test_y, verbose=0)
    predictions = model.predict(test_X)
    print(predictions)

    binary_pred = np.round(predictions, 0)
    print(binary_pred)

    p, r, f, support = sklearn.metrics.precision_recall_fscore_support(test_y, binary_pred, average='macro')
    print(p, r, f)

    # neg = 0 and pos = 1
    conf_matrix = sklearn.metrics.confusion_matrix(test_y, binary_pred)
    print(conf_matrix)

    return scores[0], scores[1], binary_pred, conf_matrix
