import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

def run_prophet(df, periods):
    """
    Ejecuta el pronóstico usando Prophet de Meta.
    df: DataFrame con columnas ['ds', 'y']
    periods: Cantidad de intervalos de 30 min a predecir.
    """
    # Configuramos Prophet para detectar ciclos diarios (48 intervalos) y semanales
    model = Prophet(
        daily_seasonality=True, 
        weekly_seasonality=True, 
        yearly_seasonality=False,
        interval_width=0.95  # Genera intervalos de confianza al 95%
    )
    
    # Añadimos feriados de Perú si es necesario (opcional)
    # model.add_country_holidays(country_name='PE') 
    
    model.fit(df[['ds', 'y']])
    
    # Creamos el dataframe futuro con frecuencia de 30 minutos ('30T')
    future = model.make_future_dataframe(periods=periods, freq='30min')
    forecast = model.predict(future)
    
    # Retornamos solo las columnas necesarias para la app
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def run_sarima(df, periods):
    """
    Ejecuta el pronóstico usando SARIMA.
    df: DataFrame con columnas ['ds', 'y']
    periods: Cantidad de intervalos de 30 min a predecir.
    """
    # Establecemos el índice temporal para statsmodels
    series = df.set_index('ds')['y']
    
    # Configuración SARIMA (p,d,q)x(P,D,Q)s
    # s=48 porque hay 48 intervalos de 30 min en un día
    try:
        model = SARIMAX(
            series, 
            order=(1, 1, 1), 
            seasonal_order=(1, 1, 0, 48),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        
        # Generar predicción
        forecast_res = results.get_forecast(steps=periods)
        mean_forecast = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()
        
        # Crear DataFrame de salida
        future_dates = pd.date_range(start=series.index[-1], periods=periods + 1, freq='30min')[1:]
        
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': mean_forecast.values,
            'yhat_lower': conf_int.iloc[:, 0].values,
            'yhat_upper': conf_int.iloc[:, 1].values
        })
    except Exception as e:
        # En caso de error por falta de datos o convergencia, devolsi
