import pandas as pd
import numpy as np
from datetime import timedelta

def load_data(fuente, archivo=None, fecha_fin=None, dias=180, nombre_pcrc="SERVICIO 1"):
    """
    Carga datos desde un CSV o genera una simulación realista para WFM.
    """
    if fuente == "Subir Archivo CSV" and archivo is not None:
        return pd.read_csv(archivo)
    
    # --- CONFIGURACIÓN DE SIMULACIÓN ---
    # Si no se define fecha_fin, usamos la fecha actual
    if fecha_fin is None:
        fecha_fin = pd.Timestamp.now().date()
    
    # Definimos el inicio basado en los días hacia atrás solicitados
    fecha_inicio = fecha_fin - timedelta(days=dias)
    
    # Creamos el rango de fechas con intervalos de 30 minutos
    date_rng = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='30min')
    df = pd.DataFrame(date_rng, columns=['ds'])
    
    # 1. Factor de Distribución Semanal
    # Lunes a Viernes: 1.0 (100%), Sábado: 0.7 (70%), Domingo: 0.3 (30%)
    def get_weekday_factor(ds):
        wd = ds.weekday()
        if wd < 5: return 1.0        # Lun-Vie
        if wd == 5: return 0.7       # Sáb
        return 0.3                   # Dom

    # 2. Factor de Distribución Diaria "Doble Joroba"
    # Simula picos de llamadas a las 10:30 AM y 4:00 PM
    def get_interval_factor(ds):
        hour = ds.hour + ds.minute/60
        # Dos campanas de Gauss para los picos operativos
        hump1 = np.exp(-0.5 * ((hour - 10.5) / 2)**2) 
        hump2 = 0.8 * np.exp(-0.5 * ((hour - 16.0) / 2)**2)
        base_noise = 0.05 # Volumen mínimo en horas valle
        return (hump1 + hump2 + base_noise)

    # 3. Generación de Volumen con Variabilidad (Evita el Overfitting)
    # Volumen base promedio de llamadas por intervalo pico
    base_volume = 60 
    
    # Aplicamos los factores de semana e intervalo
    df['weekday_factor'] = df['ds'].apply(get_weekday_factor)
    df['interval_factor'] = df['ds'].apply(get_interval_factor)
    
    # Agregamos RUIDO ALEATORIO aumentado (Ajuste para Punto 1 y 3)
    # Un ruido de 15 permite que el modelo Prophet sea más flexible
    noise = np.random.normal(0, 15, len(df)) 
    
    # Cálculo final de 'y' (Llamadas)
    df['y'] = (base_volume * df['weekday_factor'] * df['interval_factor'] + noise)
    
    # Limpieza de datos: sin negativos y redondeado a enteros
    df['y'] = df['y'].clip(lower=0).round().astype(int)
    
    # Asignamos el nombre del PCRC/Skill
    df['pcrc'] = nombre_pcrc
    
    # Retornamos solo las columnas necesarias para los modelos
    return df[['ds', 'y', 'pcrc']]
