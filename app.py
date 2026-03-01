import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_percentage_error

# 1. Configuración de rutas para módulos internos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.data_loader import load_data
    from modules.models import run_prophet
    from modules.staffing import get_staffing_requirements
except Exception as e:
    st.error(f"Error al cargar módulos: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="WFM Steps Engine - Miraflores")

# --- ESTADOS DE NAVEGACIÓN Y VALORES BASE ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'data' not in st.session_state: st.session_state.data = None
if 'aht_val' not in st.session_state: st.session_state.aht_val = 550.0
if 'shr_val' not in st.session_state: st.session_state.shr_val = 0.25

# --- SIDEBAR: PARÁMETROS BPO ---
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/data-report-concept-illustration_114360-883.jpg")
    st.title("⚙️ Parámetros BPO")
    
    st.subheader("Configuración TMO (Seg)")
    c1, c2 = st.columns(2)
    if c1.button("-10% TMO"): st.session_state.aht_val *= 0.9
    if c2.button("+10% TMO"): st.session_state.aht_val *= 1.1
    aht = st.number_input("Valor Actual", value=float(st.session_state.aht_val), key="aht_input")
    st.session_state.aht_val = aht

    st.markdown("---")
    st.subheader("Configuración Shrinkage (%)")
    c3, c4 = st.columns(2)
    if c3.button("-10% Shr"): st.session_state.shr_val *= 0.9
    if c4.button("+10% Shr"): st.session_state.shr_val = min(0.9, st.session_state.shr_val * 1.1)
    shrinkage = st.slider("Valor Actual", 0.0, 0.9, float(st.session_state.shr_val), step=0.01)
    st.session_state.shr_val = shrinkage

    st.markdown("---")
    sl = st.slider("Target Service Level (%)", 0.5, 0.99, 0.8)
    
    if st.button("🔄 Reiniciar Proceso"):
        st.session_state.step = 1
        st.session_state.data = None
        if 'current_forecast' in st.session_state: del st.session_state.current_forecast
        st.rerun()

st.title("🚀 Planificación BPO en Cascada")

# --- PASO 1: INGESTA DE DATOS ---
if st.session_state.step == 1:
    st.header("1️⃣ Configuración de Datos Históricos")
    fuente = st.radio("Origen de datos:", ["Simulación Aleatoria", "Subir Archivo CSV"])
    
    if fuente == "Simulación Aleatoria":
        nombre_pcrc = st.text_input("Nombre del PCRC / Skill:", value="SERVICIO 1")
        
        # FECHA POR DEFECTO: 1 de enero 2026
        f_inicio_default = datetime(2026, 1, 1).date()
        f_fin_sim = st.date_input("Fecha final del histórico:", datetime.now().date())
        
        # Calculamos cuántos días hay desde el 1 de enero 2026 hasta la fecha fin elegida
        dias_hist = (f_fin_sim - f_inicio_default).days
        if dias_hist < 30: dias_hist = 30 # Mínimo de seguridad para Prophet
        
        st.caption(f"Generando datos desde el **{f_inicio_default}** (por defecto) hasta hoy.")

        if st.button("Generar Histórico ➡️"):
            st.session_state.data = load_data(fuente, fecha_fin=f_fin_sim, dias=dias_hist, nombre_pcrc=nombre_pcrc)
            st.session_state.step = 2
            st.rerun()
    else:
        archivo = st.file_uploader("Subir CSV", type=['csv'])
        if archivo and st.button("Cargar ➡️"):
            st.session_state.data = load_data(fuente, archivo=archivo)
            st.session_state.step = 2
            st.rerun()

# --- PASO 2: FORECAST & COMPARATIVA ---
elif st.session_state.step == 2:
    st.header("2️⃣ Pronóstico: Prophet vs SARIMA")
    df = st.session_state.data
    df['ds'] = pd.to_datetime(df['ds'])
    
    col_f1, col_f2 = st.columns(2)
    f_ini = col_f1.date_input("Inicio del Pronóstico", datetime(2025, 11, 1).date())
    f_fin = col_f2.date_input("Fin del Pronóstico", datetime(2026, 3, 1).date())
    
    modelo_sel = st.selectbox("Selecciona Algoritmo:", ["Prophet (AI)", "SARIMA (Estadístico)"])
    
    if st.button("🚀 Ejecutar Modelo"):
        dias_forecast = (pd.to_datetime(f_fin).date() - df['ds'].min().date()).days
        with st.spinner(f"Calculando con {modelo_sel}..."):
            if modelo_sel == "Prophet (AI)":
                forecast_res = run_prophet(df, dias_forecast * 48)
            else:
                forecast_res = run_sarima(df, dias_forecast * 48)
            
            # Filtro estricto al rango pedido
            mask = (forecast_res['ds'] >= pd.to_datetime(f_ini)) & (forecast_res['ds'] <= pd.to_datetime(f_fin))
            st.session_state.current_forecast = forecast_res.loc[mask].copy()
            st.session_state.f_range = (pd.to_datetime(f_ini), pd.to_datetime(f_fin))

    if 'current_forecast' in st.session_state:
        forecast = st.session_state.current_forecast
        f_start, f_end = st.session_state.f_range
        
        # Auditoría de Precisión (MAPE)
        real_overlap = df[(df['ds'] >= f_start) & (df['ds'] <= f_end)]
        fig = go.Figure()
        
        if not real_overlap.empty:
            eval_df = real_overlap.merge(forecast, on='ds')
            if not eval_df.empty:
                from sklearn.metrics import mean_absolute_percentage_error
                mape = mean_absolute_percentage_error(eval_df['y'], eval_df['yhat'])
                st.metric(f"🎯 Precisión {modelo_sel}", f"{mape:.2%}")
                
                # Sombreado de Backtesting
                fig.add_vrect(x0=eval_df['ds'].min(), x1=eval_df['ds'].max(), 
                             fillcolor="rgba(255, 165, 0, 0.2)", layer="below", line_width=0)

        # Gráfico Unificado
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Real', line=dict(color='#4682B4')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name=f'Predicción {modelo_sel}', line=dict(color='#FF8C00', dash='dot')))
        fig.update_xaxes(range=[df['ds'].min(), f_end])
        st.plotly_chart(fig, use_container_width=True)

# --- PASO 3: STAFFING JERÁRQUICO ---
elif st.session_state.step == 3:
    st.header("3️⃣ Staffing y Vistas Jerárquicas")
    vision = st.radio("Detalle:", ["Mensual (Semanas)", "Semanal (Días)", "Diario (Intervalos)"], horizontal=True)
    
    res_wfm = get_staffing_requirements(st.session_state.current_forecast, aht, sl, shrinkage)
    df_viz = res_wfm.copy()
    df_viz['ds'] = pd.to_datetime(df_viz['ds'])

    # Lógica de Jerarquía (Mes -> Semanas, Semana -> Días, Día -> Intervalos)
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

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=df_plot['ds'], y=df_plot['yhat'], name='Llamadas', line=dict(color='#FF8C00')))
    fig_s.add_trace(go.Scatter(x=df_plot['ds'], y=df_plot['agentes_nominales'], name='Personal Requerido', line=dict(color='#2E8B57', width=3)))
    st.plotly_chart(fig_s, use_container_width=True)
    
    st.download_button("📥 Descargar Plan CSV", df_plot.to_csv(index=False).encode('utf-8'), f"WFM_{vision}.csv")
    if st.button("⬅️ Volver"): st.session_state.step = 2; st.rerun()
