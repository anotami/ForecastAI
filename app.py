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

# --- LÓGICA DE NAVEGACIÓN Y ESTADOS ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'data' not in st.session_state: st.session_state.data = None
if 'aht_val' not in st.session_state: st.session_state.aht_val = 550.0
if 'shr_val' not in st.session_state: st.session_state.shr_val = 0.25

# --- BARRA LATERAL (SIDEBAR): SIEMPRE PRESENTE ---
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/data-report-concept-illustration_114360-883.jpg")
    st.title("⚙️ Parámetros BPO")
    
    # Ajuste de TMO
    st.subheader("Configuración TMO (Seg)")
    c1, c2 = st.columns(2)
    if c1.button("-10% TMO"): st.session_state.aht_val *= 0.9
    if c2.button("+10% TMO"): st.session_state.aht_val *= 1.1
    aht = st.number_input("Valor Actual TMO", value=float(st.session_state.aht_val), key="aht_input")
    st.session_state.aht_val = aht

    st.markdown("---")
    
    # Ajuste de Shrinkage
    st.subheader("Configuración Shrinkage (%)")
    c3, c4 = st.columns(2)
    if c3.button("-10% Shr"): st.session_state.shr_val *= 0.9
    if c4.button("+10% Shr"): st.session_state.shr_val = min(0.9, st.session_state.shr_val * 1.1)
    shrinkage = st.slider("Valor Actual Shrinkage", 0.0, 0.9, float(st.session_state.shr_val), step=0.01)
    st.session_state.shr_val = shrinkage

    st.markdown("---")
    sl = st.slider("Target Service Level (%)", 0.5, 0.99, 0.8)
    
    if st.button("🔄 Reiniciar Proceso"):
        st.session_state.step = 1
        st.session_state.data = None
        st.rerun()

# --- CUERPO PRINCIPAL ---
st.title("🚀 Planificación de Demanda y Personal")

# PASO 1: INGESTA
if st.session_state.step == 1:
    st.header("1️⃣ Ingesta de Información")
    fuente = st.radio("Origen de datos:", ["Simulación Aleatoria", "Subir Archivo CSV"])
    
    if fuente == "Simulación Aleatoria":
        nombre = st.text_input("Nombre del PCRC:", value="SERVICIO 1")
        col_s1, col_s2 = st.columns(2)
        f_fin_sim = col_s1.date_input("Fecha final histórico:", datetime.now().date())
        dias_sim = col_s2.number_input("Días a generar:", value=180)
        
        if st.button("Generar Datos ➡️"):
            st.session_state.data = load_data(fuente, fecha_fin=f_fin_sim, dias=dias_sim, nombre_pcrc=nombre)
            st.session_state.step = 2
            st.rerun()
    else:
        archivo = st.file_uploader("Subir CSV", type=['csv'])
        if archivo and st.button("Cargar ➡️"):
            st.session_state.data = load_data(fuente, archivo=archivo)
            st.session_state.step = 2
            st.rerun()

# PASO 2: VISIÓN MENSUAL / SEMANAL (HISTÓRICO)
elif st.session_state.step == 2:
    st.header("2️⃣ Análisis del Histórico Real")
    df = st.session_state.data
    st.subheader(f"PCRC: {df['pcrc'].unique()[0]}")
    
    tab_m, tab_d = st.tabs(["Vista Mensual", "Día de la Semana"])
    
    with tab_m:
        df_m = df.set_index('ds').resample('M').sum().reset_index()
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=df_m['ds'], y=df_m['y'], name='Real', line=dict(color='#4682B4', width=3)))
        st.plotly_chart(fig_m, use_container_width=True)
    
    with tab_d:
        df['dia'] = df['ds'].dt.day_name()
        dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        resumen_w = df.groupby('dia')['y'].mean().reindex(dias_orden)
        st.bar_chart(resumen_w, color='#4682B4')

    c_b1, c_b2 = st.columns(2)
    if c_b1.button("⬅️ Atrás"): st.session_state.step = 1; st.rerun()
    if c_b2.button("Ir al Pronóstico ➡️"): st.session_state.step = 3; st.rerun()

# PASO 3: FORECAST (PROPHET)
elif st.session_state.step == 3:
    st.header("3️⃣ Pronóstico Detallado (Forecast AI)")
    df = st.session_state.data
    col_f1, col_f2 = st.columns(2)
    f_ini = col_f1.date_input("Inicio Forecast", df['ds'].max().date())
    f_fin = col_f2.date_input("Fin Forecast", df['ds'].max().date() + timedelta(days=7))
    
    if st.button("🚀 Ejecutar Prophet"):
        periodos = ((f_fin - f_ini).days + 1) * 48
        with st.spinner("Entrenando modelos de doble joroba..."):
            forecast = run_prophet(df, periodos)
            forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)
            st.session_state.current_forecast = forecast
            st.session_state.f_range = (pd.to_datetime(f_ini), pd.to_datetime(f_fin))

    if 'current_forecast' in st.session_state:
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Histórico', line=dict(color='#4682B4')))
        fig_f.add_trace(go.Scatter(x=st.session_state.current_forecast['ds'], y=st.session_state.current_forecast['yhat'], 
                                   name='Forecast', line=dict(color='#FF8C00', dash='dot')))
        st.plotly_chart(fig_f, use_container_width=True)
        
        c_c1, c_c2 = st.columns(2)
        if c_c1.button("⬅️ Atrás"): st.session_state.step = 2; st.rerun()
        if c_c2.button("Calcular Staffing ➡️"): st.session_state.step = 4; st.rerun()

# --- PASO 5 (o sección de resultados en Paso 4) ---
if 'current_forecast' in st.session_state:
    st.header("📊 Análisis de Capacidad y Demanda")
    
    # 1. Selector de Visión Jerárquica
    vision = st.radio(
        "Selecciona el nivel de detalle:",
        ["Mensual (Ver Semanas)", "Semanal (Ver Días)", "Diario (Ver Intervalos)"],
        horizontal=True
    )
    
    # Procesar Staffing base
    res_wfm = get_staffing_requirements(
        st.session_state.current_forecast, 
        st.session_state.aht_val, 
        sl, 
        st.session_state.shr_val
    )
    df_viz = res_wfm.copy()
    df_viz['ds'] = pd.to_datetime(df_viz['ds'])

    # 2. Lógica de Agrupación por Jerarquía
    if "Mensual" in vision:
        st.subheader("📅 Vista Mensual: Desglose por Semanas")
        # Agrupamos por semana (W)
        df_plot = df_viz.set_index('ds').resample('W').agg({
            'yhat': 'sum', 
            'agentes_netos': 'max', 
            'agentes_nominales': 'max'
        }).reset_index()
        
    elif "Semanal" in vision:
        st.subheader("📅 Vista Semanal: Desglose por Días")
        # Agrupamos por día (D) y ordenamos de Lun a Dom
        df_plot = df_viz.set_index('ds').resample('D').agg({
            'yhat': 'sum', 
            'agentes_netos': 'max', 
            'agentes_nominales': 'max'
        }).reset_index()
        
        # Ordenar días correctamente
        df_plot['dia_nombre'] = df_plot['ds'].dt.day_name()
        dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_plot['dia_nombre'] = pd.Categorical(df_plot['dia_nombre'], categories=dias_orden, ordered=True)
        df_plot = df_plot.sort_values('ds') # Mantiene orden cronológico en el gráfico

    else: # Vista Diaria por Intervalo
        st.subheader("🕒 Vista Diaria: Detalle por Intervalo (30 min)")
        # Permitir elegir qué día específico ver del forecast
        dias_disponibles = df_viz['ds'].dt.date.unique()
        dia_interes = st.selectbox("Selecciona el día a inspeccionar:", dias_disponibles)
        df_plot = df_viz[df_viz['ds'].dt.date == dia_interes]

    # 3. Renderizado de Gráficos con Colores Corporativos
    import plotly.graph_objects as go
    
    fig = go.Figure()
    # Volumen en Naranja
    fig.add_trace(go.Scatter(x=df_plot['ds'], y=df_plot['yhat'], name='Llamadas', line=dict(color='#FF8C00')))
    # Staffing en Verde
    fig.add_trace(go.Scatter(x=df_plot['ds'], y=df_plot['agentes_nominales'], name='Staffing', line=dict(color='#2E8B57', width=3)))
    
    st.plotly_chart(fig, use_container_width=True)

    # 4. Tabla de Datos y Exportación
    with st.expander("Ver tabla de resumen"):
        st.dataframe(df_plot)
        csv = df_plot.to_csv(index=False).encode('utf-8')
        st.download_button(f"📥 Descargar Vista {vision}", csv, f"WFM_{vision}.csv", "text/csv")


# PASO 4: STAFFING (ERLANG C)
elif st.session_state.step == 4:
    st.header("4️⃣ Staffing y Dimensionamiento")
    res_wfm = get_staffing_requirements(st.session_state.current_forecast, aht, sl, shrinkage)
    
    st.subheader("Plan Maestro de Agentes")
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=res_wfm['ds'], y=res_wfm['yhat'], name='Llamadas', line=dict(color='#FF8C00')))
    fig_s.add_trace(go.Scatter(x=res_wfm['ds'], y=res_wfm['agentes_nominales'], name='Staffing', line=dict(color='#2E8B57', width=3)))
    st.plotly_chart(fig_s, use_container_width=True)
    
    st.download_button("📥 Descargar Plan CSV", res_wfm.to_csv(index=False).encode('utf-8'), "plan_wfm.csv")
    if st.button("⬅️ Volver al Pronóstico"): st.session_state.step = 3; st.rerun()
