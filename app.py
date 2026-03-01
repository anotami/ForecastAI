import streamlit as st
import pandas as pd
import sys
import os

# 1. Configuración de Rutas y Módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.data_loader import load_data
    from modules.models import run_prophet, run_sarima
    from modules.staffing import get_staffing_requirements
    from modules.validator import calculate_metrics, get_error_heatmap
except ImportError as e:
    st.error(f"Error al importar módulos: {e}")
    st.stop()

# 2. Configuración de la página
st.set_page_config(layout="wide", page_title="WFM Professional Suite")

# --- BARRA LATERAL (SIDEBAR) SIEMPRE PRESENTE ---
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/data-report-concept-illustration_114360-883.jpg", caption="WFM Forecasting Tool")
    st.title("⚙️ Panel de Control")
    st.markdown("---")
    
    # Módulo 1: Ingesta
    st.header("1. Ingesta de Datos")
    fuente = st.radio("Fuente:", ["Simulación Aleatoria", "Subir Archivo"])
    archivo = st.file_uploader("Cargar histórico", type=['csv', 'xlsx']) if fuente == "Subir Archivo" else None
    
    st.markdown("---")
    
    # Módulo 2: Parámetros de Forecast
    st.header("2. Configuración Forecast")
    modelo_choice = st.selectbox("Algoritmo", ["Prophet", "SARIMA"])
    
    st.markdown("---")
    
    # Módulo 3: Parámetros WFM (Staffing)
    st.header("3. Parámetros de Staffing")
    aht = st.number_input("AHT (Segundos)", value=300)
    sl = st.slider("Target SL (%)", 0.5, 0.99, 0.8)
    shrinkage = st.slider("Shrinkage (%)", 0.0, 0.5, 0.3)

# --- CUERPO PRINCIPAL: FLUJO PASO A PASO ---
st.title("🚀 Flujo de Planificación de Call Center")

# PASO 1: Carga y Visualización del Histórico
data = load_data(fuente, archivo)

if data is not None:
    pcrc_selected = st.selectbox("Selecciona PCRC para analizar", data['pcrc'].unique())
    df_pcrc = data[data['pcrc'] == pcrc_selected].copy()
    df_pcrc['y'] = df_pcrc['y'].round().astype(int)

    st.subheader("📊 Paso 1: Análisis del Histórico")
    st.line_chart(df_pcrc.set_index('ds')['y'])
    
    st.markdown("---")

    # PASO 2: Selección de Rango y Ejecución de Pronóstico
    st.subheader("🎯 Paso 2: Ejecución del Pronóstico")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f_inicio = st.date_input("Inicio Forecast", df_pcrc['ds'].max() + pd.Timedelta(days=1))
    with col_f2:
        f_fin = st.date_input("Fin Forecast", df_pcrc['ds'].max() + pd.Timedelta(days=7))

    if st.button("Generar Pronóstico"):
        # Cálculo de periodos
        dias = (f_fin - f_inicio).days + 1
        periodos = dias * 48 # Intervalos de 30 min

        with st.spinner('Entrenando modelos y calculando dimensionamiento...'):
            # Ejecutar Forecast
            if modelo_choice == "Prophet":
                forecast = run_prophet(df_pcrc, periodos)
            else:
                forecast = run_sarima(df_pcrc, periodos)
            
            forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)
            
            # PASO 3: Staffing (Automático tras el forecast)
            res_wfm = get_staffing_requirements(forecast, aht, sl, shrinkage)

            # --- RESULTADOS FINALES ---
            st.markdown("---")
            st.subheader("✅ Paso 3: Resultados del Dimensionamiento")
            
            tab1, tab2, tab3 = st.tabs(["📈 Volumen Proyectado", "👥 Staffing Requerido", "🔥 Calidad/Heatmap"])
            
            with tab1:
                st.line_chart(res_wfm.set_index('ds')['yhat'])
                st.dataframe(res_wfm[['ds', 'yhat']])

            with tab2:
                st.line_chart(res_wfm.set_index('ds')[['agentes_netos', 'agentes_nominales']])
                c1, c2 = st.columns(2)
                c1.metric("Pico de Agentes Requeridos", int(res_wfm['agentes_netos'].max()))
                c2.metric("Plantilla Total Sugerida", int(res_wfm['agentes_nominales'].max()))

            with tab3:
                # Simulamos datos reales para el heatmap (usando los últimos días del histórico)
                # En producción aquí compararías forecast vs actuals
                st.plotly_chart(get_error_heatmap(res_wfm))

else:
    st.info("Utiliza la barra lateral izquierda para cargar datos y comenzar el proceso.")
