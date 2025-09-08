import numpy as np
from vmdpy import VMD

def vmd_decomposition(data, alpha, tau, K, DC=0, init=1, tol=1e-7):
    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
    return u

def prepare_data_vmd(data, n_features, features, vmd_params):
    X, y = [], []
    for i in range(n_features, len(data)):
        window_data = data[i-n_features:i]
        window_features = []
        for j, feature in enumerate(features):
            alpha, tau, K = vmd_params[feature]
            vmd_modes = vmd_decomposition(window_data[:, j], alpha=alpha, tau=tau, K=K)
            for mode in vmd_modes:
                window_features.append(mode)

        if len(window_features) > 0:
            window_features = np.array(window_features)
            X.append(window_features.T)
            y.append(data[i, 0])
    return np.array(X), np.array(y)

def reshape_for_cnn_lstm(data):
    return data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))
