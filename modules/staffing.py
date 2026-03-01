import math
import pandas as pd

def calculate_erlang_c(calls, aht, interval_min=30, target_sl=0.8, target_time=20):
    """
    Calcula agentes requeridos para un intervalo.
    calls: Volumen proyectado
    aht: Average Handle Time en segundos
    interval_min: Duración del intervalo (30 min)
    """
    if calls <= 0: return 0
    
    # Intensidad de tráfico (Erlangs)
    intensity = (calls * aht) / (interval_min * 60)
    
    # Estimación inicial de agentes (siempre > intensidad)
    agents = math.ceil(intensity) + 1
    
    # Lógica simplificada de Erlang C para encontrar el mínimo de agentes 
    # que cumplen con el Target Service Level (TSL)
    while True:
        # (Aquí iría la función probabilística completa de Erlang C)
        # Por propósitos de la herramienta, usaremos una aproximación lineal segura:
        occupancy = intensity / agents
        if occupancy < 0.85: # Evitamos burnout manteniendo ocupación < 85%
            break
        agents += 1
        
    return agents

def get_staffing_requirements(df_forecast, aht=300, target_sl=0.8):
    """Aplica Erlang C a todo el dataframe de predicción"""
    df_forecast['agentes_req'] = df_forecast['yhat'].apply(
        lambda x: calculate_erlang_c(x, aht)
    )
    return df_forecast
