import numpy as np


EPS = 1e-8


def RSE(pred, true):
    numerator = np.sqrt(np.sum((true - pred) ** 2))
    denominator = np.sqrt(np.sum((true - true.mean()) ** 2))
    return numerator / (denominator + EPS)


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / (d + EPS)).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

# Troubleshooting for PEMS Nov 8
# def MAPE(pred, true):
#     return np.mean(np.abs((pred - true) / true))
def MAPE(pred, true):
    mape = np.abs((pred - true) / (true + EPS))
    mape = np.where(mape > 5, 0, mape)
    return np.mean(mape)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / (true + EPS)))


def SMAPE(pred, true):
    denominator = np.abs(pred) + np.abs(true) + EPS
    return np.mean(2.0 * np.abs(pred - true) / denominator)


def WAPE(pred, true):
    return np.sum(np.abs(pred - true)) / (np.sum(np.abs(true)) + EPS)


def R2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1.0 - (ss_res / (ss_tot + EPS))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def metric_extended(pred, true):
    mae, mse, rmse, mape, mspe = metric(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    smape = SMAPE(pred, true)
    wape = WAPE(pred, true)
    r2 = R2(pred, true)

    if isinstance(corr, np.ndarray):
        corr = float(np.mean(corr))

    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'mspe': float(mspe),
        'rse': float(rse),
        'corr': float(corr),
        'smape': float(smape),
        'wape': float(wape),
        'r2': float(r2),
    }
