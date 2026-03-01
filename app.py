import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuración de rutas para módulos internos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.data_loader import load_data
    from modules.models import run_prophet
    from modules.staffing import get_staffing_requirements
    from modules.validator import calculate_metrics, get_error_heatmap
except Exception as e:
    st.error(f"Error al cargar módulos: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="WFM Steps Engine - Miraflores")

# --- ESTADOS DE NAVEGACIÓN Y VALORES BASE ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'data' not in st.session_state: st.session_state.data = None
if 'aht_val' not in st.session_state: st.session_state.aht_val = 550.0 #
if 'shr_val' not in st.session_state: st.session_state.shr_val = 0.25  #

# --- SIDEBAR: PARÁMETROS BPO (SIEMPRE VISIBLE) ---
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/data-report-concept-illustration_114360-883.jpg")
    st.title("⚙️ Parámetros BPO")
    
    # Ajuste de TMO con botones +/- 10%
    st.subheader("Configuración TMO (Seg)")
    c1, c2 = st.columns(2)
    if c1.button("-10% TMO"): st.session_state.aht_val *= 0.9
    if c2.button("+10% TMO"): st.session_state.aht_val *= 1.1
    aht = st.number_input("Valor Actual", value=float(st.session_state.aht_val), key="aht_input")
    st.session_state.aht_val = aht

    st.markdown("---")
    
    # Ajuste de Shrinkage con botones +/- 10%
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

# --- PASO 1: INGESTA PERSONALIZADA ---
if st.session_state.step == 1:
    st.header("1️⃣ Configuración de Datos Históricos")
    fuente = st.radio("Origen de datos:", ["Simulación Aleatoria", "Subir Archivo CSV"])
    
    if fuente == "Simulación Aleatoria":
        nombre_pcrc = st.text_input("Nombre del PCRC / Skill:", value="SERVICIO 1") #
        col_s1, col_s2 = st.columns(2)
        f_fin_sim = col_s1.date_input("Fecha final del histórico:", datetime.now().date()) #
        dias_hist = col_s2.number_input("Días hacia atrás para generar:", min_value=30, max_value=730, value=180) #
        
        if st.button("Generar Histórico ➡️"):
            st.session_state.data = load_data(fuente, fecha_fin=f_fin_sim, dias=dias_hist, nombre_pcrc=nombre_pcrc) #
            st.session_state.step = 2
            st.rerun()
    else:
        archivo = st.file_uploader("Subir CSV", type=['csv'])
        if archivo and st.button("Cargar ➡️"):
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
    f_fin = col_f2.date_input("Fin del Pronóstico", df['ds'].max().date() + timedelta(days=7))
    
    if st.button("🚀 Ejecutar Pronóstico"):
        # Calculamos periodos necesarios para cubrir desde el inicio del histórico hasta f_fin
        dias_totales = (pd.to_datetime(f_fin).date() - df['ds'].min().date()).days
        with st.spinner("Entrenando modelos..."):
            forecast_full = run_prophet(df, dias_totales * 48)
            # Filtro estricto al rango solicitado
            mask = (forecast_full['ds'] >= pd.to_datetime(f_ini)) & (forecast_full['ds'] <= pd.to_datetime(f_fin))
            forecast_filtered = forecast_full.loc[mask].copy()
            forecast_filtered['yhat'] = forecast_filtered['yhat'].clip(lower=0).round().astype(int)
            st.session_state.current_forecast = forecast_filtered
            st.session_state.f_range = (pd.to_datetime(f_ini), pd.to_datetime(f_fin))

    if 'current_forecast' in st.session_state:
        forecast = st.session_state.current_forecast
        f_start, f_end = st.session_state.f_range
        
        # Métrica de Precisión y Sombreado
        real_overlap = df[(df['ds'] >= f_start) & (df['ds'] <= f_end)]
        fig = go.Figure()
        
        if not real_overlap.empty:
            from sklearn.metrics import mean_absolute_percentage_error
            eval_df = real_overlap.merge(forecast, on='ds')
            if not eval_df.empty:
                mape = mean_absolute_percentage_error(eval_df['y'], eval_df['yhat'])
                st.metric("🎯 Precisión (MAPE)", f"{mape:.2%}")
                fig.add_vrect(x0=eval_df['ds'].min(), x1=eval_df['ds'].max(), 
                             fillcolor="rgba(173, 216, 230, 0.3)", layer="below", line_width=0,
                             annotation_text="Zona de Validación") #

        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Histórico (Azul)', line=dict(color='#4682B4'))) #
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast (Naranja)', line=dict(color='#FF8C00', dash='dot'))) #
        fig.update_xaxes(range=[df['ds'].min(), f_end]) #
        st.plotly_chart(fig, use_container_width=True)
        
        c_n1, c_n2 = st.columns(2)
        if c_n1.button("⬅️ Atrás"): st.session_state.step = 1; st.rerun()
        if c_n2.button("Calcular Staffing ➡️"): st.session_state.step = 3; st.rerun()

# --- PASO 3: STAFFING JERÁRQUICO ---
elif st.session_state.step == 3:
    st.header("3️⃣ Staffing y Vistas Jerárquicas")
    
    vision = st.radio("Selecciona detalle:", ["Mensual (Semanas)", "Semanal (Días)", "Diario (Intervalos)"], horizontal=True) #
    
    res_wfm = get_staffing_requirements(st.session_state.current_forecast, aht, sl, shrinkage)
    df_viz = res_wfm.copy()
    df_viz['ds'] = pd.to_datetime(df_viz['ds'])

    # Lógica de Jerarquía
    if "Mensual" in vision:
        df_plot = df_viz.set_index('ds').resample('W').agg({'yhat':'sum', 'agentes_nominales':'max'}).reset_index()
    elif "Semanal" in vision:
        df_plot = df_viz.set_index('ds').resample('D').agg({'yhat':'sum', 'agentes_nominales':'max'}).reset_index()
        df_plot['dia'] = df_plot['ds'].dt.day_name()
        # Orden de Lun a Dom
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
    st.plotly_chart(fig_s, use_container_width=True)
    
    st.download_button(f"📥 Exportar {vision}", df_plot.to_csv(index=False).encode('utf-8'), f"WFM_{vision}.csv") #
    if st.button("⬅️ Volver al Pronóstico"): st.session_state.step = 2; st.rerun()
