import streamlit as st
import pandas as pd
import sys
import os
import math

# 1. Configuración de Rutas para evitar ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. Importación Segura de Módulos
try:
    from modules.data_loader import load_data
    from modules.models import run_prophet, run_sarima
    from modules.staffing import get_staffing_requirements
except ImportError as e:
    st.error(f"⚠️ Error al importar módulos internos: {e}")
    st.info("Asegúrate de tener la carpeta 'modules' con __init__.py y los archivos .py correspondientes.")
    st.stop()

# 3. Configuración de la Página
st.set_page_config(layout="wide", page_title="WFM Forecast & Staffing Engine")

st.title("📊 WFM Engine: Forecast & Dimensionamiento")
st.markdown("---")

# --- SIDEBAR: Configuración Global ---
with st.sidebar:
    st.header("1. Fuente de Datos")
    fuente = st.radio("Selecciona origen:", ["Simulación Aleatoria", "Subir Archivo (CSV/Excel)"])
    
    archivo = None
    if fuente == "Subir Archivo (CSV/Excel)":
        archivo = st.file_uploader("Formato: [fecha, llamadas, pcrc]", type=['csv', 'xlsx'])
        st.caption("Nota: La fecha debe estar en intervalos de 30 min.")

    st.header("2. Parámetros Operativos")
    modelo_choice = st.selectbox("Algoritmo de Forecast", ["Prophet", "SARIMA"])
    horizonte = st.selectbox("Horizonte de tiempo", ["1 Día", "1 Semana", "1 Mes"])
    
    st.header("3. Configuración Erlang C")
    aht_input = st.number_input("AHT (Segundos)", value=300, step=10)
    target_sl = st.slider("Target Service Level (%)", 0.5, 0.99, 0.8)
    shrinkage_input = st.slider("Shrinkage Total (%)", 0.0, 0.5, 0.3)

# Mapeo de intervalos (48 intervalos de 30min = 1 día)
mapeo_periodos = {"1 Día": 48, "1 Semana": 336, "1 Mes": 1440}
n_periodos = mapeo_periodos[horizonte]

# --- PROCESAMIENTO ---
data = load_data(fuente, archivo)

if data is not None:
    # Selección de PCRC (Cola)
    lista_pcrc = data['pcrc'].unique()
    pcrc_selected = st.selectbox("Selecciona el PCRC / Cola a analizar", lista_pcrc)
    
    df_pcrc = data[data['pcrc'] == pcrc_selected].copy()
    
    # Pestañas de la Aplicación
    tab_forecast, tab_staffing = st.tabs(["📈 Pronóstico de Llamadas", "👥 Dimensionamiento de Personal"])

    if st.button("🚀 Ejecutar Análisis Masivo"):
        with st.spinner('Procesando datos y entrenando modelos...'):
            # 1. Ejecución del Forecast
            if modelo_choice == "Prophet":
                forecast_res = run_prophet(df_pcrc, n_periodos)
            else:
                forecast_res = run_sarima(df_pcrc, n_periodos)
            
            with tab_forecast:
                st.subheader(f"Predicción de Volumen: {pcrc_selected}")
                # Gráfico interactivo
                st.line_chart(forecast_res.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])
                
                with st.expander("Ver tabla de datos proyectados"):
                    st.write(forecast_res)

            with tab_staffing:
                st.subheader("Requerimiento de Agentes (Erlang C)")
                
                # 2. Cálculo de Staffing
                res_wfm = get_staffing_requirements(
                    forecast_res, 
                    aht=aht_input, 
                    target_sl=target_sl, 
                    shrinkage=shrinkage_input
                )
                
                # Gráfico de Staffing
                st.line_chart(res_wfm.set_index('ds')[['agentes_netos', 'agentes_nominales']])
                
                # Resumen de métricas
                c1, c2, c3 = st.columns(3)
                c1.metric("Máximo de Agentes en Línea", int(res_wfm['agentes_netos'].max()))
                c2.metric("Promedio de Agentes", int(res_wfm['agentes_netos'].mean()))
                c3.metric("Plantilla Total (Nominal)", int(res_wfm['agentes_nominales'].max()))

                # Botón de descarga
                csv = res_wfm.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Descargar Plan de Staffing",
                    csv,
                    f"staffing_{pcrc_selected}_{horizonte}.csv",
                    "text/csv"
                )
else:
    st.info("👋 Bienvenido. Selecciona una fuente de datos en el panel izquierdo para comenzar.")
    if fuente == "Subir Archivo (CSV/Excel)":
        st.warning("Esperando archivo... Asegúrate de que tenga las columnas: [ds, y, pcrc]")
