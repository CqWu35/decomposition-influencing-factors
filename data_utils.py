import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path, target_column):
    data = pd.read_excel(file_path)
    data.set_index('日期', inplace=True)

    train_size = int(len(data) * 0.7)
    validation_size = int(len(data) * 0.15)
    train = data.iloc[:train_size]
    validation = data.iloc[train_size:train_size + validation_size]
    out_of_sample = data.iloc[train_size + validation_size:]
    return data, train, validation, out_of_sample

def scale_data(train, validation, out_of_sample):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    validation_scaled = scaler.transform(validation)
    out_of_sample_scaled = scaler.transform(out_of_sample)
    return scaler, train_scaled, validation_scaled, out_of_sample_scaled

def inverse_transform(scaler, data, feature_index):
    import numpy as np
    dummy = np.zeros((len(data), scaler.data_max_.shape[0]))
    dummy[:, feature_index] = data
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, feature_index]
