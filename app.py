import streamlit as st
import pandas as pd
import sys
import os

# Solución para rutas en Streamlit Cloud
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.data_loader import load_data
    from modules.models import run_prophet, run_sarima
    from modules.staffing import get_staffing_requirements
    from modules.validator import calculate_metrics, get_error_heatmap
except ImportError as e:
    st.error(f"Error de importación: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="WFM Professional Suite")

# --- SIDEBAR PERMANENTE ---
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/data-report-concept-illustration_114360-883.jpg")
    st.title("⚙️ Configuración WFM")
    fuente = st.radio("Fuente de Datos", ["Simulación Aleatoria", "Subir Archivo"])
    archivo = st.file_uploader("Cargar Datos", type=['csv']) if fuente == "Subir Archivo" else None
    st.markdown("---")
    st.header("Parámetros Staffing")
    aht = st.number_input("AHT (Seg)", value=300)
    sl = st.slider("Target SL (%)", 0.5, 0.95, 0.8)
    shrinkage = st.slider("Shrinkage (%)", 0.0, 0.5, 0.3)

# --- FLUJO PRINCIPAL ---
st.title("🚀 Planificación de Demanda y Personal")
data = load_data(fuente, archivo)

if data is not None:
    pcrc = st.selectbox("Selecciona PCRC", data['pcrc'].unique())
    df_pcrc = data[data['pcrc'] == pcrc].copy()
    df_pcrc['y'] = df_pcrc['y'].round().astype(int)

    st.subheader("1. Análisis Histórico")
    st.line_chart(df_pcrc.set_index('ds')['y'])

    st.subheader("2. Ejecución de Pronóstico")
    col1, col2 = st.columns(2)
    f_inicio = col1.date_input("Inicio", df_pcrc['ds'].max() + pd.Timedelta(days=1))
    f_fin = col2.date_input("Fin", df_pcrc['ds'].max() + pd.Timedelta(days=7))

    if st.button("Procesar Pronóstico y Staffing"):
        periodos = ((f_fin - f_inicio).days + 1) * 48
        
        with st.spinner("Calculando dimensionamiento masivo..."):
            forecast = run_prophet(df_pcrc, periodos)
            forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)
            
            res_wfm = get_staffing_requirements(forecast, aht, sl, shrinkage)
            
            t1, t2, t3 = st.tabs(["📈 Forecast", "👥 Staffing", "🔥 Heatmap"])
            with t1:
                st.line_chart(res_wfm.set_index('ds')['yhat'])
            with t2:
                st.line_chart(res_wfm.set_index('ds')[['agentes_netos', 'agentes_nominales']])
            with t3:
                st.plotly_chart(get_error_heatmap(res_wfm))
