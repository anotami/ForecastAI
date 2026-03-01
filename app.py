import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_percentage_error

# 1. Configuración de rutas y carga de módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.data_loader import load_data
    from modules.models import run_prophet, run_sarima  # Corrección de NameError
    from modules.staffing import get_staffing_requirements
except Exception as e:
    st.error(f"Error crítico al cargar módulos: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="WFM Steps Engine - Miraflores", width="stretch")

# --- ESTADOS DE SESIÓN Y VALORES BASE ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'data' not in st.session_state: st.session_state.data = None
if 'aht_val' not in st.session_state: st.session_state.aht_val = 550.0  # TMO Base
if 'shr_val' not in st.session_state: st.session_state.shr_val = 0.25   # Shrinkage Base

# --- SIDEBAR: PARÁMETROS BPO (COMPLETO) ---
with st.sidebar:
    st.title("⚙️ Parámetros BPO")
    
    st.subheader("Configuración TMO (Seg)")
    c1, c2 = st.columns(2)
    if c1.button("-10% TMO"): st.session_state.aht_val *= 0.9
    if c2.button("+10% TMO"): st.session_state.aht_val *= 1.1
    aht = st.number_input("Valor Actual TMO", value=float(st.session_state.aht_val), key="aht_input")
    st.session_state.aht_val = aht

    st.markdown("---")
    st.subheader("Configuración Shrinkage (%)")
    c3, c4 = st.columns(2)
    if c3.button("-10% Shr"): st.session_state.shr_val *= 0.9
    if c4.button("+10% Shr"): st.session_state.shr_val = min(0.9, st.session_state.shr_val * 1.1)
    shrinkage = st.slider("Valor Actual Shrinkage", 0.0, 0.9, float(st.session_state.shr_val), step=0.01)
    st.session_state.shr_val = shrinkage

    st.markdown("---")
    sl = st.slider("Target Service Level (%)", 0.5, 0.99, 0.8)
    
    if st.button("🔄 Reiniciar Todo"):
        st.session_state.step = 1
        st.session_state.data = None
        for key in ['forecast_p', 'forecast_s', 'current_forecast']:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

st.title("🚀 Planificación BPO en Cascada")

# --- PASO 1: INGESTA DE DATOS ---
if st.session_state.step == 1:
    st.header("1️⃣ Configuración de Datos Históricos")
    fuente = st.radio("Origen de datos:", ["Simulación Aleatoria", "Subir Archivo CSV"])
    
    if fuente == "Simulación Aleatoria":
        nombre_pcrc = st.text_input("Nombre del PCRC:", value="SERVICIO 1")
        # Fecha de inicio solicitada: 1 de enero 2026
        f_inicio_default = datetime(2026, 1, 1).date()
        f_fin_sim = st.date_input("Fecha final del histórico:", datetime.now().date())
        
        dias_hist = (f_fin_sim - f_inicio_default).days
        if dias_hist < 30: dias_hist = 30 # Mínimo para Prophet
        
        if st.button("Generar Histórico Simulado ➡️"):
            with st.spinner("Construyendo universo de datos..."):
                st.session_state.data = load_data(fuente, fecha_fin=f_fin_sim, dias=dias_hist, nombre_pcrc=nombre_pcrc)
                st.session_state.step = 2
                st.rerun()
    else:
        archivo = st.file_uploader("Subir CSV", type=['csv'])
        if archivo and st.button("Cargar ➡️"):
            st.session_state.data = load_data(fuente, archivo=archivo)
            st.session_state.step = 2
            st.rerun()

# --- PASO 2: COMPARATIVA DE PRONÓSTICOS (PROPHET VS SARIMA) ---
elif st.session_state.step == 2:
    st.header("2️⃣ Comparativa de Pronósticos: Prophet vs SARIMA")
    df = st.session_state.data
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Rango solicitado por defecto: 01/11/2025 al 01/03/2026
    col_f1, col_f2 = st.columns(2)
    f_ini = col_f1.date_input("Inicio del Pronóstico", datetime(2025, 11, 1).date())
    f_fin = col_f2.date_input("Fin del Pronóstico", datetime(2026, 3, 1).date())
    
    if st.button("🚀 Ejecutar Modelos Duales"):
        fecha_ref = min(df['ds'].min().date(), f_ini)
        dias_forecast = (pd.to_datetime(f_fin).date() - fecha_ref).days
        
        with st.spinner("Procesando Prophet (AI) y SARIMA (Stats)..."):
            fp = run_prophet(df, dias_forecast * 48)
            fs = run_sarima(df, dias_forecast * 48)
            
            # Filtro estricto al rango solicitado
            mask_p = (fp['ds'] >= pd.to_datetime(f_ini)) & (fp['ds'] <= pd.to_datetime(f_fin))
            mask_s = (fs['ds'] >= pd.to_datetime(f_ini)) & (fs['ds'] <= pd.to_datetime(f_fin))
            
            st.session_state.forecast_p = fp.loc[mask_p].copy()
            st.session_state.forecast_s = fs.loc[mask_s].copy()
            st.session_state.f_range = (pd.to_datetime(f_ini), pd.to_datetime(f_fin))

    if 'forecast_p' in st.session_state:
        fp = st.session_state.forecast_p
        fs = st.session_state.forecast_s
        f_start, f_end = st.session_state.f_range
        
        # Auditoría de Precisión (MAPE)
        real_overlap = df[(df['ds'] >= f_start) & (df['ds'] <= f_end)]
        if not real_overlap.empty:
            eval_p = real_overlap.merge(fp, on='ds')
            eval_s = real_overlap.merge(fs, on='ds')
            
            c_m1, c_m2 = st.columns(2)
            c_m1.metric("MAPE Prophet (Naranja)", f"{mean_absolute_percentage_error(eval_p['y'], eval_p['yhat']):.2%}")
            c_m2.metric("MAPE SARIMA (Verde)", f"{mean_absolute_percentage_error(eval_s['y'], eval_s['yhat']):.2%}")

        # Gráfico Comparativo
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Histórico Real', line=dict(color='#4682B4', width=1)))
        fig.add_trace(go.Scatter(x=fp['ds'], y=fp['yhat'], name='Prophet', line=dict(color='#FF8C00', dash='dot')))
        fig.add_trace(go.Scatter(x=fs['ds'], y=fs['yhat'], name='SARIMA', line=dict(color='#2E8B57', dash='dash')))

        if not real_overlap.empty:
            fig.add_vrect(x0=real_overlap['ds'].min(), x1=real_overlap['ds'].max(), 
                         fillcolor="rgba(200, 200, 200, 0.2)", layer="below", annotation_text="Backtesting")

        fig.update_xaxes(range=[min(df['ds'].min(), f_start), f_end])
        st.plotly_chart(fig, use_container_width='stretch')
        
        mod_sel = st.radio("Selecciona modelo para Staffing:", ["Prophet", "SARIMA"], horizontal=True)
        if st.button("Confirmar y Continuar ➡️"):
            st.session_state.current_forecast = fp if mod_sel == "Prophet" else fs
            st.session_state.step = 3
            st.rerun()

# --- PASO 3: STAFFING Y VISTAS JERÁRQUICAS ---
elif st.session_state.step == 3:
    st.header("3️⃣ Staffing y Dimensionamiento")
    vision = st.radio("Ver reporte por:", ["Mensual (Semanas)", "Semanal (Días)", "Diario (Intervalos)"], horizontal=True)
    
    res_wfm = get_staffing_requirements(st.session_state.current_forecast, aht, sl, shrinkage)
    df_viz = res_wfm.copy()
    df_viz['ds'] = pd.to_datetime(df_viz['ds'])

    # Lógica de agrupación solicitada
    if "Mensual" in vision:
        df_plot = df_viz.set_index('ds').resample('W').agg({'yhat':'sum', 'agentes_nominales':'max'}).reset_index()
    elif "Semanal" in vision:
        df_plot = df_viz.set_index('ds').resample('D').agg({'yhat':'sum', 'agentes_nominales':'max'}).reset_index()
        df_plot['dia'] = df_plot['ds'].dt.day_name()
        dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_plot['dia'] = pd.Categorical(df_plot['dia'], categories=dias_orden, ordered=True)
        df_plot = df_plot.sort_values('ds')
    else:
        dia_sel = st.selectbox("Día a inspeccionar:", df_viz['ds'].dt.date.unique())
        df_plot = df_viz[df_viz['ds'].dt.date == dia_sel]

    # Gráfico Maestro
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=df_plot['ds'], y=df_plot['yhat'], name='Llamadas', line=dict(color='#FF8C00')))
    fig_s.add_trace(go.Scatter(x=df_plot['ds'], y=df_plot['agentes_nominales'], name='Staffing', line=dict(color='#2E8B57', width=3)))
    st.plotly_chart(fig_s, use_container_width='stretch')
    
    st.metric("Eficiencia (Ocupación)", f"{(res_wfm['yhat'] * aht).sum() / (res_wfm['agentes_netos'].sum() * 1800):.1%}")
    st.download_button(f"📥 Descargar {vision}", df_plot.to_csv(index=False).encode('utf-8'), f"WFM_{vision}.csv")
    if st.button("⬅️ Volver"): st.session_state.step = 2; st.rerun()
