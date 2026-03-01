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

# --- PASO 2: DIARIO ---
st.header("Paso 2: Pronóstico Diario (Day-of-Week)")
with st.expander("Ver Análisis Diario"):
    # Resumen por día de la semana para validar 100/70/30
    df_daily = df_pcrc.copy()
    df_daily['dia_nombre'] = df_daily['ds'].dt.day_name()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    resumen_semanal = df_daily.groupby('dia_nombre')['y'].mean().reindex(order)
    st.bar_chart(resumen_semanal)
    st.caption("Validación: Lunes-Viernes (Pico), Sábado (70%), Domingo (30%)")

# --- PASO 3: INTERVALO ---
st.header("Paso 3: Pronóstico por Intervalo (30 min)")
with st.expander("Ejecutar Forecast de Detalle"):
    # ... (código anterior de Prophet) ...
    if 'current_forecast' in st.session_state:
        # Mostrar un solo día para ver la "Doble Joroba" claramente
        st.subheader("Curva de Llegada (Doble Joroba)")
        un_dia = st.session_state.current_forecast.head(48)
        st.line_chart(un_dia.set_index('ds')['yhat'])

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
