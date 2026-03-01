import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.data_loader import load_data
    from modules.models import run_prophet
    from modules.staffing import get_staffing_requirements
    from modules.validator import calculate_metrics, get_error_heatmap
except Exception as e:
    st.error(f"Error al cargar módulos: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="WFM Unified Engine")

# --- ESTADOS ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'data' not in st.session_state: st.session_state.data = None
if 'aht_val' not in st.session_state: st.session_state.aht_val = 550.0
if 'shr_val' not in st.session_state: st.session_state.shr_val = 0.25

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Parámetros BPO")
    st.session_state.aht_val = st.number_input("TMO (Seg)", value=float(st.session_state.aht_val))
    st.session_state.shr_val = st.slider("Shrinkage (%)", 0.0, 0.9, float(st.session_state.shr_val))
    sl = st.slider("Target SL (%)", 0.5, 0.99, 0.8)
    if st.button("🔄 Reiniciar"):
        st.session_state.step = 1
        st.rerun()

# --- FLUJO PASO A PASO ---
if st.session_state.step == 1:
    st.header("1️⃣ Configuración de Datos")
    fuente = st.radio("Origen:", ["Simulación Aleatoria", "Subir Archivo CSV"])
    
    if fuente == "Simulación Aleatoria":
        nombre = st.text_input("Nombre PCRC:", value="SERVICIO 1")
        col1, col2 = st.columns(2)
        f_fin = col1.date_input("Fecha final:", datetime.now().date())
        dias = col2.number_input("Días atrás:", value=180)
        
        if st.button("Generar Datos ➡️"):
            st.session_state.data = load_data(fuente, fecha_fin=f_fin, dias=dias, nombre_pcrc=nombre)
            st.session_state.step = 2
            st.rerun()
    else:
        archivo = st.file_uploader("Subir CSV", type=['csv'])
        if archivo and st.button("Cargar ➡️"):
            st.session_state.data = load_data(fuente, archivo=archivo)
            st.session_state.step = 2
            st.rerun()

elif st.session_state.step == 2:
    st.header("2️⃣ Pronóstico")
    df = st.session_state.data
    df['ds'] = pd.to_datetime(df['ds'])
    
    col1, col2 = st.columns(2)
    f_ini = col1.date_input("Inicio Forecast", df['ds'].max().date())
    f_fin = col2.date_input("Fin Forecast", df['ds'].max().date() + timedelta(days=7))
    
    if st.button("🚀 Ejecutar Forecast"):
        periodos = ((f_fin - f_ini).days + 1) * 48
        forecast = run_prophet(df, periodos)
        forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)
        st.session_state.current_forecast = forecast
        st.session_state.f_range = (pd.to_datetime(f_ini), pd.to_datetime(f_fin))

    if 'current_forecast' in st.session_state:
        # Gráfico Unificado
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Histórico', line=dict(color='#4682B4')))
        fig.add_trace(go.Scatter(x=st.session_state.current_forecast['ds'], y=st.session_state.current_forecast['yhat'], 
                                 name='Forecast', line=dict(color='#FF8C00', dash='dot')))
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Calcular Staffing ➡️"):
            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 3:
    st.header("3️⃣ Staffing Final")
    # Asegúrate de que get_staffing_requirements reciba (df, aht, target_sl, shrinkage)
    res_wfm = get_staffing_requirements(st.session_state.current_forecast, 
                                        st.session_state.aht_val, 
                                        sl, 
                                        st.session_state.shr_val)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res_wfm['ds'], y=res_wfm['yhat'], name='Llamadas', line=dict(color='#FF8C00')))
    fig.add_trace(go.Scatter(x=res_wfm['ds'], y=res_wfm['agentes_nominales'], name='Staffing', line=dict(color='#2E8B57', width=3)))
    st.plotly_chart(fig, use_container_width=True)
    
    st.download_button("📥 Descargar CSV", res_wfm.to_csv(index=False).encode('utf-8'), "plan_wfm.csv")
