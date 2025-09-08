import pandas as pd
from data_utils import load_data, scale_data, inverse_transform
from vmd_utils import prepare_data_vmd, reshape_for_cnn_lstm
from model_utils import define_models, build_and_train_model, evaluate_model
from plot_utils import plot_metrics

def main():
    file_path = 'D:\\OneDrive\\桌面\\分1\\数据\\0\\ND.xlsx'
    output_dir = 'D:\\OneDrive\\桌面\\新代码'
    target_column = 'INE'

    vmd_params = {
        'CSE': (48.8542, 0.141861, 6),
        'IR': (57.5798, 0.0209903, 2),
        'ER': (110.15, 0.102665, 4),
        'AEPU': (1998.79, 2.85638e-07, 10),
        'SC': (17.1305, 0.0101648, 2),
        'GR': (970.854, 0.0408602, 7),
        'CND': (27.9509, 0.0503917, 2),
        'SCC': (452.577, 0.0339066, 10),
        'COS': (340.0708, 0.000794, 9),
        'NG': (82.5026, 0.006307, 2),
        'EWS': (292.2933, 0.115726, 4),
        'WTI': (55.2906, 0.148730, 2),
        'BRE': (49.1517, 0.149995, 3)
    }

    data, train, validation, out_of_sample = load_data(file_path, target_column)
    features = data.columns.drop(target_column)
    scaler, train_scaled, validation_scaled, out_of_sample_scaled = scale_data(train, validation, out_of_sample)

    n_features = 10
    train_X, train_y = prepare_data_vmd(train_scaled, n_features, features, vmd_params)
    validation_X, validation_y = prepare_data_vmd(validation_scaled, n_features, features, vmd_params)
    out_of_sample_X, out_of_sample_y = prepare_data_vmd(out_of_sample_scaled, n_features, features, vmd_params)
    train_X_cnnlstm = reshape_for_cnn_lstm(train_X)
    validation_X_cnnlstm = reshape_for_cnn_lstm(validation_X)
    out_of_sample_X_cnnlstm = reshape_for_cnn_lstm(out_of_sample_X)

    models = define_models(
        input_shape=(train_X.shape[1], train_X.shape[2]),
        input_shape_cnnlstm=(train_X.shape[1], train_X.shape[2], 1)
    )

    histories, predictions, metrics = {}, {}, {}
    for name, model in models.items():
        print(f"Training {name} model...")
        if 'CNN-LSTM' in name or 'CNN-BiLSTM' in name:
            model, history = build_and_train_model(model, train_X_cnnlstm, train_y, validation_X_cnnlstm, validation_y)
            predictions[name] = model.predict(out_of_sample_X_cnnlstm)
            metrics[name] = {
                'validation': evaluate_model(validation_y, model.predict(validation_X_cnnlstm)),
                'out_of_sample': evaluate_model(out_of_sample_y, predictions[name])
            }
        else:
            model, history = build_and_train_model(model, train_X, train_y, validation_X, validation_y)
            predictions[name] = model.predict(out_of_sample_X)
            metrics[name] = {
                'validation': evaluate_model(validation_y, model.predict(validation_X)),
                'out_of_sample': evaluate_model(out_of_sample_y, predictions[name])
            }
        histories[name] = history

    inverse_predictions = {name: inverse_transform(scaler, pred.flatten(), 0) for name, pred in predictions.items()}
    out_of_sample_original = inverse_transform(scaler, out_of_sample_y, 0)
    time_index = data.index[len(train) + len(validation) + n_features:]
    results_df = pd.DataFrame({'Date': time_index, 'Original': out_of_sample_original})
    for name, pred in inverse_predictions.items():
        results_df[name + '_Prediction'] = pred
    results_df.to_excel(output_dir + '\\pre.xlsx', index=False)

    plot_metrics(models, metrics, ['MSE', 'RMSE', 'MAE', 'MAPE'])


if __name__ == "__main__":
    main()
