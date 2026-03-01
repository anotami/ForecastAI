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

st.set_page_config(layout="wide", page_title="WFM Engine - Miraflores") # Eliminado width='stretch' problemático

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
    if st.button("🔄 Reiniciar Todo"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()

# --- PASO 1: INGESTA (1 de enero 2026) ---
if st.session_state.step == 1:
    st.header("1️⃣ Configuración de Datos")
    f_ini_default = datetime(2026, 1, 1).date()
    f_fin_sim = st.date_input("Fecha final histórico:", datetime.now().date())
    dias_hist = max(30, (f_fin_sim - f_ini_default).days)
    
    if st.button("Generar Histórico ➡️"):
        st.session_state.data = load_data("Simulación Aleatoria", fecha_fin=f_fin_sim, dias=dias_hist)
        st.session_state.step = 2
        st.rerun()

# --- PASO 2: PRONÓSTICO (Con Opción de Selección de Modelo) ---
elif st.session_state.step == 2:
    st.header("2️⃣ Pronóstico Inteligente")
    df = st.session_state.data
    df['ds'] = pd.to_datetime(df['ds'])
    
    col_f1, col_f2 = st.columns(2)
    f_ini = col_f1.date_input("Inicio Forecast", datetime(2025, 11, 1).date())
    f_fin = col_f2.date_input("Fin Forecast", datetime(2026, 3, 1).date())
    
    modo_ejecucion = st.radio("Modo de Procesamiento:", 
                              ["Solo Prophet (Rápido)", "Solo SARIMA", "Comparativa Dual (Lento)"], 
                              horizontal=True)
    
    if st.button("🚀 Ejecutar Modelos"):
        dias_f = (pd.to_datetime(f_fin).date() - df['ds'].min().date()).days
        with st.spinner(f"Procesando {modo_ejecucion}..."):
            if "Prophet" in modo_ejecucion or "Dual" in modo_ejecucion:
                st.session_state.fp = run_prophet(df, dias_f * 48)
            if "SARIMA" in modo_ejecucion or "Dual" in modo_ejecucion:
                st.session_state.fs = run_sarima(df, dias_f * 48)
            
            st.session_state.f_range = (pd.to_datetime(f_ini), pd.to_datetime(f_fin))

    if 'f_range' in st.session_state:
        f_start, f_end = st.session_state.f_range
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Histórico Real', line=dict(color='#4682B4')))
        
        # Lógica de visualización según selección
        if 'fp' in st.session_state:
            mask_p = (st.session_state.fp['ds'] >= f_start) & (st.session_state.fp['ds'] <= f_end)
            df_p = st.session_state.fp.loc[mask_p]
            fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['yhat'], name='Prophet', line=dict(color='#FF8C00', dash='dot')))
            
        if 'fs' in st.session_state:
            mask_s = (st.session_state.fs['ds'] >= f_start) & (st.session_state.fs['ds'] <= f_end)
            df_s = st.session_state.fs.loc[mask_s]
            fig.add_trace(go.Scatter(x=df_s['ds'], y=df_s['yhat'], name='SARIMA', line=dict(color='#2E8B57', dash='dash')))

        st.plotly_chart(fig, use_container_width=True)
        
        # Selección para Staffing
        opciones_staff = []
        if 'fp' in st.session_state: opciones_staff.append("Prophet")
        if 'fs' in st.session_state: opciones_staff.append("SARIMA")
        
        modelo_final = st.selectbox("Usar para Staffing:", opciones_staff)
        if st.button("Confirmar y Calcular Staffing ➡️"):
            st.session_state.current_forecast = st.session_state.fp if modelo_final == "Prophet" else st.session_state.fs
            st.session_state.step = 3
            st.rerun()

# --- PASO 3: STAFFING (Jerárquico) ---
elif st.session_state.step == 3:
    st.header("3️⃣ Staffing y Vistas Jerárquicas")
    vision = st.radio("Detalle:", ["Mensual", "Semanal", "Diario"], horizontal=True)
    
    res_wfm = get_staffing_requirements(st.session_state.current_forecast, st.session_state.aht_val, sl, st.session_state.shr_val)
    st.dataframe(res_wfm.head(100)) # Vista rápida
    st.download_button("📥 Descargar Plan Maestro", res_wfm.to_csv(index=False).encode('utf-8'), "plan_wfm.csv")
