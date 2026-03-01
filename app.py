import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error

# Configuración de rutas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.data_loader import load_data
    from modules.models import run_prophet
    from modules.staffing import get_staffing_requirements
except ImportError as e:
    st.error(f"Error de módulos: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="WFM Unified Engine Pro")

# --- ESTADOS DE NAVEGACIÓN Y VALORES BASE ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'data' not in st.session_state: st.session_state.data = None
if 'aht_val' not in st.session_state: st.session_state.aht_val = 550.0
if 'shr_val' not in st.session_state: st.session_state.shr_val = 0.25

# --- SIDEBAR: PARÁMETROS OPERATIVOS ---
with st.sidebar:
    st.title("⚙️ Parámetros BPO")
    
    st.header("Ajuste de TMO (Seg)")
    c1, c2 = st.columns(2)
    if c1.button("-10% TMO"): st.session_state.aht_val *= 0.9
    if c2.button("+10% TMO"): st.session_state.aht_val *= 1.1
    aht = st.number_input("Valor TMO", value=float(st.session_state.aht_val), key="aht_input")
    
    st.markdown("---")
    st.header("Ajuste de Shrinkage (%)")
    c3, c4 = st.columns(2)
    if c3.button("-10% Shr"): st.session_state.shr_val *= 0.9
    if c4.button("+10% Shr"): st.session_state.shr_val = min(0.9, st.session_state.shr_val * 1.1)
    shrinkage = st.slider("Valor Shrinkage", 0.0, 0.9, float(st.session_state.shr_val), step=0.01)
    
    st.markdown("---")
    sl = st.slider("Target SL (%)", 0.5, 0.99, 0.8)
    
    if st.button("🔄 Reiniciar Proceso"):
        st.session_state.step = 1
        st.session_state.data = None
        st.rerun()

st.title("🚀 Planificación BPO en Cascada")

# --- PASO 1: INGESTA CON CONFIGURACIÓN DE SIMULACIÓN ---
if st.session_state.step == 1:
    st.header("1️⃣ Ingesta de Información")
    fuente = st.radio("Origen de datos:", ["Simulación Aleatoria", "Subir Archivo CSV"])
    
    if fuente == "Simulación Aleatoria":
        st.info("Configura el periodo de tiempo histórico que deseas simular.")
        dias_sim = st.number_input("Días de histórico a generar:", min_value=30, max_value=730, value=180)
        # Nota: La función load_data en modules/data_loader debe aceptar este parámetro
        if st.button("Generar Datos de Simulación ➡️"):
            with st.spinner("Creando universo de datos..."):
                # Pasamos los días elegidos a la función de carga
                st.session_state.data = load_data(fuente, dias_hist=dias_sim)
                st.session_state.step = 2
                st.rerun()
    
    else:
        archivo = st.file_uploader("Cargar histórico CSV", type=['csv'])
        if archivo is not None:
            if st.button("Procesar Archivo ➡️"):
                st.session_state.data = load_data(fuente, archivo=archivo)
                st.session_state.step = 2
                st.rerun()

# --- PASO 2: FORECAST & BACKTESTING ---
elif st.session_state.step == 2:
    st.header("2️⃣ Pronóstico e Indicadores de Precisión")
    df = st.session_state.data
    df['ds'] = pd.to_datetime(df['ds'])
    
    col_f1, col_f2 = st.columns(2)
    f_ini = col_f1.date_input("Inicio del Pronóstico", df['ds'].max().date())
    f_fin = col_f2.date_input("Fin del Pronóstico", df['ds'].max().date() + pd.Timedelta(days=7))
    
    if st.button("🚀 Generar Pronóstico"):
        dias_forecast = (f_fin - f_ini).days + 1
        with st.spinner("Entrenando modelos..."):
            forecast = run_prophet(df, dias_forecast * 48)
            forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)
            st.session_state.current_forecast = forecast
            st.session_state.f_range = (pd.to_datetime(f_ini), pd.to_datetime(f_fin))

    if 'current_forecast' in st.session_state:
        forecast = st.session_state.current_forecast
        f_start, f_end = st.session_state.f_range
        
        # Lógica de MAPE para Backtesting
        real_overlap = df[(df['ds'] >= f_start) & (df['ds'] <= f_end)]
        if not real_overlap.empty:
            eval_df = real_overlap.merge(forecast, on='ds')
            if not eval_df.empty:
                mape = mean_absolute_percentage_error(eval_df['y'], eval_df['yhat'])
                st.metric("🎯 Precisión: MAPE", f"{mape:.2%}")

        # Gráfico Unificado (Azul: Real, Naranja: Pronóstico)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Dato Real (Azul)', line=dict(color='#4682B4')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Pronóstico (Naranja)', line=dict(color='#FF8C00', dash='dot')))
        st.plotly_chart(fig, use_container_width=True)
        
        c_n1, c_n2 = st.columns(2)
        if c_n1.button("⬅️ Atrás"): st.session_state.step = 1; st.rerun()
        if c_n2.button("Calcular Staffing ➡️"): st.session_state.step = 3; st.rerun()

# --- PASO 3: STAFFING TOTAL ---
elif st.session_state.step == 3:
    st.header("3️⃣ Dimensionamiento Final (Staffing)")
    forecast = st.session_state.current_forecast
    df_hist = st.session_state.data
    
    res_wfm = get_staffing_requirements(forecast, aht, sl, shrinkage)
    
    # Gráfico Maestro
    fig_master = go.Figure()
    fig_master.add_trace(go.Scatter(x=df_hist['ds'], y=df_hist['y'], name='Pasado (Real)', line=dict(color='#4682B4', width=1)))
    fig_master.add_trace(go.Scatter(x=res_wfm['ds'], y=res_wfm['yhat'], name='Futuro (Llamadas)', line=dict(color='#FF8C00')))
    fig_master.add_trace(go.Scatter(x=res_wfm['ds'], y=res_wfm['agentes_nominales'], name='Staffing (Verde)', line=dict(color='#2E8B57', width=3)))
    
    st.plotly_chart(fig_master, use_container_width=True)
    
    # Exportación
    csv = res_wfm.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar Plan Maestro CSV", csv, "plan_wfm.csv", "text/csv")
    
    if st.button("⬅️ Volver a Pronóstico"): st.session_state.step = 2; st.rerun()
