import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_prophet(df, periods):
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.fit(df[['ds', 'y']])
    future = m.make_future_dataframe(periods=periods, freq='30min')
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def run_sarima(df, periods):
    series = df.set_index('ds')['y']
    try:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 0, 48))
        results = model.fit(disp=False)
        forecast_values = results.get_forecast(steps=periods)
        idx = pd.date_range(start=df['ds'].iloc[-1], periods=periods+1, freq='30min')[1:]
        
        return pd.DataFrame({
            'ds': idx,
            'yhat': forecast_values.predicted_mean.values,
            'yhat_lower': forecast_values.conf_int().iloc[:, 0].values,
            'yhat_upper': forecast_values.conf_int().iloc[:, 1].values
        })
    except Exception as e:
        # Bloque con sangría correcta para evitar IndentationError
        print(f"Error en SARIMA: {e}")
        return pd.DataFrame()
