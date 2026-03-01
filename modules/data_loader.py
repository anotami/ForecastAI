import pandas as pd
import numpy as np
from datetime import timedelta

def load_data(fuente, archivo=None, fecha_fin=None, dias=180, nombre_pcrc="SERVICIO 1"):
    if fuente == "Subir Archivo CSV" and archivo is not None:
        return pd.read_csv(archivo)
    
    # Generación de Simulación
    if fecha_fin is None:
        fecha_fin = pd.Timestamp.now().date()
    
    fecha_inicio = fecha_fin - timedelta(days=dias)
    date_rng = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='30min')
    
    df = pd.DataFrame(date_rng, columns=['ds'])
    
    # Factores de distribución (Doble Joroba)
    def get_factors(ds):
        wd = ds.weekday()
        w_factor = 1.0 if wd < 5 else (0.7 if wd == 5 else 0.3)
        hour = ds.hour + ds.minute/60
        h_factor = np.exp(-0.5 * ((hour - 10.5) / 2)**2) + 0.8 * np.exp(-0.5 * ((hour - 16.0) / 2)**2) + 0.05
        return w_factor * h_factor

    df['y'] = df['ds'].apply(get_factors) * 60 + np.random.normal(0, 5, len(df))
    df['y'] = df['y'].clip(lower=0).round().astype(int)
    df['pcrc'] = nombre_pcrc
    
    return df
