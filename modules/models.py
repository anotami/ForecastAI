import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_prophet(df, periods):
    # Aumentamos changepoint_prior_scale para que sea muy flexible con los cambios
    m = Prophet(
        daily_seasonality=True, 
        weekly_seasonality=True,
        changepoint_prior_scale=0.5, 
        seasonality_prior_scale=10.0,
        seasonality_mode='additive'
    )
    m.fit(df[['ds', 'y']])
    future = m.make_future_dataframe(periods=periods, freq='30min')
    forecast = m.predict(future)
    return forecast[['ds', 'yhat']]

def run_sarima(df, periods):
    # SARIMA es más sensible a variaciones cíclicas cortas
    # order=(1,1,1), seasonal_order=(1,1,1,48) -> 48 son intervalos de 30min en un día
    model = SARIMAX(df['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 48))
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=periods)
    
    last_date = df['ds'].max()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=30), periods=periods, freq='30min')
    return pd.DataFrame({'ds': forecast_dates, 'yhat': forecast.predicted_mean})
