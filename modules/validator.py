import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_absolute_percentage_error

def calculate_metrics(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"MAPE": f"{mape:.2%}"}

def get_error_heatmap(df_result):
    df_result['ds'] = pd.to_datetime(df_result['ds'])
    df_result['hora'] = df_result['ds'].dt.hour
    df_result['dia_semana'] = df_result['ds'].dt.day_name()
    
    # Usamos yhat para el ejemplo si no hay y_real
    pivot = df_result.groupby(['dia_semana', 'hora'])['yhat'].mean().unstack()
    dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(dias)
    
    fig = px.imshow(pivot, labels=dict(x="Hora", y="Día", color="Volumen"),
                    title="Distribución Horaria de Llamadas")
    return fig
