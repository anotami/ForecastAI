import pandas as pd
import plotly.express as px

def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_percentage_error
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"MAPE": f"{mape:.2%}"}

def get_error_heatmap(df_result):
    df_result['ds'] = pd.to_datetime(df_result['ds'])
    df_result['hora'] = df_result['ds'].dt.hour
    df_result['dia_semana'] = df_result['ds'].dt.day_name()
    
    pivot = df_result.groupby(['dia_semana', 'hora'])['yhat'].mean().unstack()
    dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(dias)
    
    fig = px.imshow(pivot, labels=dict(x="Hora", y="Día", color="Llamadas"),
                    title="Distribución Horaria de la Demanda")
    return fig
