import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_percentage_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import load_data
from modules.models import run_prophet, run_sarima
from modules.staffing import get_staffing_requirements

st.set_page_config(layout="wide", page_title="WFM Engine - Miraflores")

# --- ESTADOS Y SIDEBAR ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'data' not in st.session_state: st.session_state.data = None
if 'aht_val' not in st.session_state: st.session_state.aht_val = 550.0
if 'shr_val' not in st.session_state: st.session_state.shr_val = 0.25

with st.sidebar:
    st.title("⚙️ Parámetros BPO")
    st.session_state.aht_val = st.number_input("TMO (Seg)", value=float(st.session_state.aht_val))
    st.session_state.shr_val = st.slider("Shrinkage (%)", 0.0, 0.9, float(st.session_state.shr_val))
    sl = st.slider("Target SL (%)", 0.5, 0.99, 0.8)
    if st.button("🔄 Reiniciar"): 
        st.session_state.step = 1
        st.rerun()

# --- PASO 1: INGESTA (Fecha por defecto: 1 de enero 2026) ---
if st.session_state.step == 1:
    st.header("1️⃣ Configuración de Datos")
    f_ini_default = datetime(2026, 1, 1).date()
    f_fin_sim = st.date_input("Fecha final histórico:", datetime.now().date())
    dias_hist = max(30, (f_fin_sim - f_ini_default).days)
    
    if st.button("Generar Histórico ➡️"):
        st.session_state.data = load_data("Simulación Aleatoria", fecha_fin=f_fin_sim, dias=dias_hist)
        st.session_state.step = 2
        st.rerun()

# --- PASO 2: PRONÓSTICO (1/11/25 al 1/3/26) ---
elif st.session_state.step == 2:
    st.header("2️⃣ Comparativa: Prophet vs SARIMA")
    df = st.session_state.data
    df['ds'] = pd.to_datetime(df['ds'])
    
    f_ini = st.date_input("Inicio Forecast", datetime(2025, 11, 1).date())
    f_fin = st.date_input("Fin Forecast", datetime(2026, 3, 1).date())
    
    if st.button("🚀 Ejecutar Modelos"):
        dias_f = (pd.to_datetime(f_fin).date() - df['ds'].min().date()).days
        with st.spinner("Procesando dual..."):
            fp = run_prophet(df, dias_f * 48)
            fs = run_sarima(df, dias_f * 48)
            mask = (fp['ds'] >= pd.to_datetime(f_ini)) & (fp['ds'] <= pd.to_datetime(f_fin))
            st.session_state.forecast_p = fp.loc[mask].copy()
            st.session_state.forecast_s = fs.loc[mask].copy()
            st.session_state.f_range = (pd.to_datetime(f_ini), pd.to_datetime(f_fin))

    if 'forecast_p' in st.session_state:
        # Aquí va el gráfico y las métricas de MAPE comparadas
        st.metric("MAPE Prophet", "Calculando...") # Lógica de MAPE intersección
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Real', line=dict(color='#4682B4')))
        fig.add_trace(go.Scatter(x=st.session_state.forecast_p['ds'], y=st.session_state.forecast_p['yhat'], 
                                 name='Prophet', line=dict(color='#FF8C00', dash='dot')))
        st.plotly_chart(fig, use_container_width=True)
        if st.button("Siguiente ➡️"): 
            st.session_state.current_forecast = st.session_state.forecast_p
            st.session_state.step = 3
            st.rerun()

# --- PASO 3: STAFFING (Jerárquico) ---
elif st.session_state.step == 3:
    st.header("3️⃣ Staffing Final")
    vision = st.radio("Detalle:", ["Mensual", "Semanal", "Diario"], horizontal=True)
    res_wfm = get_staffing_requirements(st.session_state.current_forecast, st.session_state.aht_val, sl, st.session_state.shr_val)
    # Lógica de agrupación de tablas y gráficos por jerarquía...
    st.download_button("📥 Descargar Plan", res_wfm.to_csv().encode('utf-8'))
