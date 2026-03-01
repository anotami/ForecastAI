import streamlit as st
import pandas as pd
import sys
import os

# Fuerza a Python a encontrar la carpeta 'modules'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.data_loader import load_data
    from modules.models import run_prophet, run_sarima
    from modules.staffing import get_staffing_requirements
    from modules.validator import get_error_heatmap
except ImportError as e:
    st.error(f"Error al importar módulos: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="WFM Engine Pro")

# --- SIDEBAR PERMANENTE ---
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/data-report-concept-illustration_114360-883.jpg", caption="WFM Tool")
    st.title("⚙️ Configuración")
    fuente = st.radio("Fuente de Datos:", ["Simulación Aleatoria", "Subir Archivo"])
    archivo = st.file_uploader("Cargar CSV", type=['csv']) if fuente == "Subir Archivo" else None
    st.markdown("---")
    st.header("Parámetros WFM")
    aht = st.number_input("AHT (Segundos)", value=300)
    sl = st.slider("Target SL (%)", 0.5, 0.99, 0.8)
    shrinkage = st.slider("Shrinkage (%)", 0.0, 0.5, 0.3)

# --- FLUJO PRINCIPAL ---
st.title("🚀 Planificación de Demanda y Personal")
data = load_data(fuente, archivo)

if data is not None:
    pcrc = st.selectbox("Selecciona PCRC", data['pcrc'].unique())
    df_p = data[data['pcrc'] == pcrc].copy()
    df_p['y'] = df_p['y'].round().astype(int)

    st.subheader("1. Análisis Histórico")
    st.line_chart(df_p.set_index('ds')['y'])

    st.subheader("2. Configuración de Pronóstico")
    col1, col2 = st.columns(2)
    f_ini = col1.date_input("Fecha Inicio", df_p['ds'].max() + pd.Timedelta(days=1))
    f_fin = col2.date_input("Fecha Fin", df_p['ds'].max() + pd.Timedelta(days=7))

    if st.button("Procesar Pronóstico y Staffing"):
        periodos = ((f_fin - f_ini).days + 1) * 48
        
        with st.spinner("Calculando..."):
            # Ejecución del Forecast (Prophet por defecto)
            forecast = run_prophet(df_p, periodos)
            forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)
            
            # Cálculo de Staffing (Erlang C)
            res_wfm = get_staffing_requirements(forecast, aht, sl, shrinkage)
            
            tab1, tab2, tab3 = st.tabs(["📈 Forecast", "👥 Staffing", "🔥 Heatmap"])
            
            with tab1:
                # Zoom a los primeros 7 días para evitar el efecto de "mancha azul"
                st.line_chart(res_wfm.head(336).set_index('ds')['yhat'])
                st.dataframe(res_wfm[['ds', 'yhat']])
                
            with tab2:
                st.line_chart(res_wfm.head(336).set_index('ds')[['agentes_netos', 'agentes_nominales']])
                st.metric("Pico de Agentes Requeridos", int(res_wfm['agentes_netos'].max()))
                
            with tab3:
                st.plotly_chart(get_error_heatmap(res_wfm))
else:
    st.info("Configura la fuente de datos en la barra lateral.")
