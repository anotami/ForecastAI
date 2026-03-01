import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.data_loader import load_data
    from modules.models import run_prophet, run_sarima
    from modules.staffing import get_staffing_requirements
    from modules.validator import calculate_metrics, get_error_heatmap
except Exception as e:
    st.error(f"Error de módulos: {e}")
    st.stop()

st.set_page_config(layout="wide")

# SIDEBAR
with st.sidebar:
    st.title("⚙️ WFM Panel")
    fuente = st.radio("Datos", ["Simulación Aleatoria", "Subir Archivo"])
    archivo = st.file_uploader("CSV", type=['csv']) if fuente == "Subir Archivo" else None
    st.markdown("---")
    aht = st.number_input("AHT (Seg)", value=300)
    sl = st.slider("Target SL", 0.7, 0.99, 0.8)
    shrinkage = st.slider("Shrinkage", 0.0, 0.5, 0.3)

# MAIN
data = load_data(fuente, archivo)
if data is not None:
    pcrc = st.selectbox("PCRC", data['pcrc'].unique())
    df_p = data[data['pcrc'] == pcrc].copy()
    
    st.subheader("1. Histórico")
    st.line_chart(df_p.set_index('ds')['y'])

    st.subheader("2. Pronóstico")
    col1, col2 = st.columns(2)
    f_ini = col1.date_input("Inicio")
    f_fin = col2.date_input("Fin")

    if st.button("Procesar Pronóstico y Staffing"):
        periodos = ((f_fin - f_ini).days + 1) * 48
        forecast = run_prophet(df_p, periodos)
        forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)
        
        # LLAMADA CORREGIDA PARA EVITAR TYPEERROR
        res_wfm = get_staffing_requirements(forecast, aht, sl, shrinkage)
        
        t1, t2, t3 = st.tabs(["Forecast", "Staffing", "Heatmap"])
        t1.line_chart(res_wfm.set_index('ds')['yhat'])
        t2.line_chart(res_wfm.set_index('ds')[['agentes_netos', 'agentes_nominales']])
        t3.plotly_chart(get_error_heatmap(res_wfm))
