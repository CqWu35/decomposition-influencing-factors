import numpy as np
from keras.models import Sequential
from keras.layers import GRU, LSTM, SimpleRNN, Dense, Conv1D, MaxPooling1D, Flatten, Bidirectional, TimeDistributed, Reshape
from sklearn.metrics import mean_squared_error, mean_absolute_error

def build_and_train_model(model, train_X, train_y, validation_X, validation_y, epochs=15, batch_size=32):
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(
        train_X, train_y,
        epochs=epochs, batch_size=batch_size,
        validation_data=(validation_X, validation_y),
        verbose=1
    )
    return model, history

def define_models(input_shape, input_shape_cnnlstm):
    models = {
        'GRU': Sequential([
            GRU(50, input_shape=input_shape),
            Dense(1)
        ]),
        'LSTM': Sequential([
            LSTM(50, input_shape=input_shape),
            Dense(1)
        ]),
        'CNN-LSTM': Sequential([
            TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=input_shape_cnnlstm),
            TimeDistributed(MaxPooling1D(pool_size=2)),
            TimeDistributed(Flatten()),
            LSTM(50),
            Dense(1)
        ]),
        'BP': Sequential([
            Reshape((input_shape[0] * input_shape[1],), input_shape=input_shape),
            Dense(50, activation='relu'),
            Dense(1)
        ]),
        'RNN': Sequential([
            SimpleRNN(50, input_shape=input_shape),
            Dense(1)
        ]),
        'CNN': Sequential([
            Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1)
        ]),
        'BiLSTM': Sequential([
            Bidirectional(LSTM(50, input_shape=input_shape)),
            Dense(1)
        ]),
        'CNN-BiLSTM': Sequential([
            TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=input_shape_cnnlstm),
            TimeDistributed(MaxPooling1D(pool_size=2)),
            TimeDistributed(Flatten()),
            Bidirectional(LSTM(50)),
            Dense(1)
        ])
    }
    return models

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred.flatten())
    mae = mean_absolute_error(y_true, y_pred.flatten())
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred.flatten()) / y_true)) * 100
    return mse, rmse, mae, mape
