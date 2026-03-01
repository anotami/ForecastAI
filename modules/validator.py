import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    """Calcula el error del pronóstico"""
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # El Bias nos dice si estamos sobre-pronosticando o sub-pronosticando
    bias = np.sum(y_pred - y_true) / np.sum(y_true)
    
    return {
        "MAPE": f"{mape:.2%}",
        "RMSE": round(rmse, 2),
        "BIAS": f"{bias:.2%}"
    }
