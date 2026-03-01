import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    """Calcula el error del pronóstico"""
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.sum(y_pred - y_true) / np.sum(y_true)
    
    return {
        "MAPE": f"{mape:.2%}",
        "RMSE": round(rmse, 2),
        "BIAS": f"{bias:.2%}"
    }

def get_error_heatmap(df_result):
    """Genera un mapa de calor basado en el error absoluto"""
    df_result['ds'] = pd.to_datetime(df_result['ds'])
    df_result['hora'] = df_result['ds'].dt.hour
    df_result['dia_semana'] = df_result['ds'].dt.day_name()
    df_result['error_abs'] = abs(df_result['yhat'] - df_result['yhat']) # En producción usar y_real
    
    pivot = df_result.groupby(['dia_semana', 'hora'])['error_abs'].mean().unstack()
    dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(dias)
    
    fig = px.imshow(pivot, labels=dict(x="Hora", y="Día", color="Error Abs"),
                    title="Distribución Horaria del Error")
    return fig
