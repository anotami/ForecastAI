import pandas as pd
import numpy as np

def load_data(fuente, archivo=None):
    if fuente == "Subir Archivo" and archivo is not None:
        return pd.read_csv(archivo)
    
    # --- GENERACIÓN DE DATOS SINTÉTICOS REALISTAS ---
    # Generamos 180 días de datos en intervalos de 30 min
    date_rng = pd.date_range(start='2025-09-01', periods=180*48, freq='30min')
    df = pd.DataFrame(date_rng, columns=['ds'])
    
    # 1. Distribución Semanal (Lunes=0, Domingo=6)
    # Lunes a Viernes: 100%, Sábado: 70%, Domingo: 30%
    def get_weekday_factor(ds):
        wd = ds.weekday()
        if wd < 5: return 1.0        # Lunes-Viernes
        if wd == 5: return 0.7       # Sábado
        return 0.3                   # Domingo

    # 2. Distribución Diaria "Doble Joroba" (Double Hump)
    # Picos habituales: 10:00 - 12:00 y 15:00 - 17:00
    def get_interval_factor(ds):
        hour = ds.hour + ds.minute/60
        # Mezcla de dos funciones gaussianas para simular las jorobas
        hump1 = np.exp(-0.5 * ((hour - 10.5) / 2)**2) # Pico mañana
        hump2 = np.exp(-0.5 * ((hour - 16.0) / 2)**2) # Pico tarde
        base_noise = 0.05
        return (hump1 + 0.8 * hump2 + base_noise)

    # Aplicar factores
    df['weekday_factor'] = df['ds'].apply(get_weekday_factor)
    df['interval_factor'] = df['ds'].apply(get_interval_factor)
    
    # Generar volumen base (ej. promedio de 60 llamadas por intervalo pico)
    base_volume = 60
    noise = np.random.normal(0, 5, len(df))
    
    df['y'] = (base_volume * df['weekday_factor'] * df['interval_factor'] + noise)
    df['y'] = df['y'].clip(lower=0).round().astype(int)
    
    # Añadimos columna de PCRC para segmentación
    df['pcrc'] = 'Atencion_Cliente_Peru'
    
    return df[['ds', 'y', 'pcrc']]
