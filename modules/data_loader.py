import pandas as pd
import numpy as np
import streamlit as st

def load_data(source_type, uploaded_file=None):
    if source_type == "Simulación Aleatoria":
        pcrcs = ['Ventas', 'Soporte', 'Retenciones']
        # 180 días, frecuencia 30 min
        date_rng = pd.date_range(start='2024-01-01', periods=180*48, freq='30min')
        data = []
        for p in pcrcs:
            base = np.random.randint(20, 100)
            # Estacionalidad diaria + ruido
            calls = base + 20*np.sin(2*np.pi*np.arange(len(date_rng))/48) + np.random.normal(0, 5, len(date_rng))
            temp_df = pd.DataFrame({'ds': date_rng, 'y': calls.clip(0), 'pcrc': p})
            data.append(temp_df)
        return pd.concat(data)
    
    elif source_type == "Subir Archivo (CSV/Excel)":
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            # Normalización de columnas
            df.columns = ['ds', 'y', 'pcrc']
            df['ds'] = pd.to_datetime(df['ds'])
            return df
    return None
