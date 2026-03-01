import pandas as pd
import plotly.express as px

def get_error_heatmap(df_result):
    # df_result debe tener 'ds', 'y_real', 'yhat'
    df_result['hora'] = df_result['ds'].dt.hour
    df_result['dia_semana'] = df_result['ds'].dt.day_name()
    df_result['error_abs'] = abs(df_result['yhat'] - df_result['y_real'])
    
    # Pivot para el heatmap
    pivot = df_result.groupby(['dia_semana', 'hora'])['error_abs'].mean().unstack()
    # Ordenar días
    dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(dias)
    
    fig = px.imshow(pivot, labels=dict(x="Hora del Día", y="Día", color="Error Absoluto"),
                    title="Mapa de Calor: ¿Dónde falla más el modelo?")
    return fig
