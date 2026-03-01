import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.data_loader import load_data
    from modules.models import run_prophet
    from modules.staffing import get_staffing_requirements
except ImportError as e:
    st.error(f"Error de módulos: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="WFM Cascada Engine")

# --- SIDEBAR PERSISTENTE ---
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/data-report-concept-illustration_114360-883.jpg")
    st.title("⚙️ Configuración WFM")
    fuente = st.radio("Origen de Datos", ["Simulación Aleatoria", "Subir Archivo"])
    st.markdown("---")
    st.header("Parámetros Staffing")
    aht = st.number_input("AHT (Seg)", value=300)
    sl = st.slider("Target SL (%)", 0.5, 0.99, 0.8)
    shrinkage = st.slider("Shrinkage (%)", 0.0, 0.5, 0.3)

st.title("🚀 Planificación WFM en Cascada")

# CARGA INICIAL
data = load_data(fuente) # Asumimos que carga el dataframe base

if data is not None:
    pcrc = st.selectbox("Selecciona PCRC / Skill", data['pcrc'].unique())
    df_pcrc = data[data['pcrc'] == pcrc].copy()
    df_pcrc['y'] = df_pcrc['y'].round().astype(int)

    # --- PASO 1: FORECAST MENSUAL (MACRO) ---
    st.header("Paso 1: Pronóstico Mensual (Presupuesto)")
    with st.expander("Ver Análisis Mensual", expanded=True):
        df_monthly = df_pcrc.set_index('ds').resample('M').sum().reset_index()
        st.line_chart(df_monthly.set_index('ds')['y'])
        if st.button("Validar Meses y Continuar"):
            st.success("Volumen mensual validado.")

    # --- PASO 2: FORECAST DIARIO (DISTRIBUCIÓN) ---
    st.header("Paso 2: Pronóstico Diario (Day-of-Week)")
    with st.expander("Ver Análisis Diario"):
        df_daily = df_pcrc.set_index('ds').resample('D').sum().reset_index()
        st.bar_chart(df_daily.tail(30).set_index('ds')['y'])
        st.info("Aquí se observan los picos de lunes y bajas de fin de semana.")
        if st.button("Validar Días y Continuar"):
            st.success("Distribución diaria validada.")

    # --- PASO 3: FORECAST POR INTERVALO (MICRO) ---
    st.header("Paso 3: Pronóstico por Intervalo (30 min)")
    with st.expander("Ejecutar Forecast de Detalle"):
        col1, col2 = st.columns(2)
        f_ini = col1.date_input("Desde", df_pcrc['ds'].max().date())
        f_fin = col2.date_input("Hasta", df_pcrc['ds'].max().date() + pd.Timedelta(days=7))
        
        if st.button("Ejecutar Forecast Detallado"):
            periodos = ((f_fin - f_ini).days + 1) * 48
            with st.spinner("Calculando curvas de llegada..."):
                forecast = run_prophet(df_pcrc, periodos)
                forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)
                st.session_state.current_forecast = forecast
                st.line_chart(forecast.head(336).set_index('ds')['yhat'])

    # --- PASO 4: STAFFING (FINAL) ---
    st.header("Paso 4: Dimensionamiento de Personal")
    if 'current_forecast' in st.session_state:
        with st.expander("Calcular Agentes Requeridos", expanded=True):
            res_wfm = get_staffing_requirements(st.session_state.current_forecast, aht, sl, shrinkage)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Llamadas Totales", f"{res_wfm['yhat'].sum():,}")
            c2.metric("Agentes Requeridos (Pico)", int(res_wfm['agentes_netos'].max()))
            c3.metric("Plantilla Nominal", int(res_wfm['agentes_nominales'].max()))
            
            st.subheader("Curva de Agentes vs Llamadas")
            st.line_chart(res_wfm.head(336).set_index('ds')[['agentes_netos', 'agentes_nominales']])
            
            csv = res_wfm.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar Plan Maestro (CSV)", csv, "plan_wfm.csv", "text/csv")
    else:
        st.warning("Completa el Paso 3 para habilitar el cálculo de agentes.")

else:
    st.info("Por favor, selecciona una fuente de datos en el panel izquierdo.")
