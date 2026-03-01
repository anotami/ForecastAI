import streamlit as st
import pandas as pd
import sys
import os
import math

# 1. Configuración de Rutas para evitar ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. Importación de Módulos
try:
    from modules.data_loader import load_data
    from modules.models import run_prophet
    from modules.staffing import get_staffing_requirements
    from modules.validator import get_error_heatmap
except ImportError as e:
    st.error(f"Error de módulos: {e}")
    st.stop()

# 3. Configuración de la Página
st.set_page_config(layout="wide", page_title="WFM Cascada Engine Pro")

# --- SIDEBAR: PARÁMETROS DINÁMICOS ---
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/data-report-concept-illustration_114360-883.jpg")
    st.title("⚙️ Configuración WFM")
    
    fuente = st.radio("Origen de Datos", ["Simulación Aleatoria", "Subir Archivo"])
    archivo = st.file_uploader("Cargar CSV", type=['csv']) if fuente == "Subir Archivo" else None
    
    st.markdown("---")
    st.header("Parámetros de Staffing")

    # Lógica de Ajuste Rápido para TMO
    if 'aht_val' not in st.session_state: st.session_state.aht_val = 300.0
    c1, c2 = st.columns(2)
    if c1.button("-10% TMO"): st.session_state.aht_val *= 0.9
    if c2.button("+10% TMO"): st.session_state.aht_val *= 1.1
    aht = st.number_input("TMO / AHT (Seg)", value=float(st.session_state.aht_val))
    st.session_state.aht_val = aht

    st.markdown("---")
    
    # Lógica de Ajuste Rápido para Shrinkage
    if 'shr_val' not in st.session_state: st.session_state.shr_val = 0.30
    c3, c4 = st.columns(2)
    if c3.button("-10% Shrink"): st.session_state.shr_val *= 0.9
    if c4.button("+10% Shrink"): st.session_state.shr_val = min(0.9, st.session_state.shr_val * 1.1)
    shrinkage = st.slider("Shrinkage Total (%)", 0.0, 0.9, float(st.session_state.shr_val))
    st.session_state.shr_val = shrinkage
    
    sl = st.slider("Target SL (%)", 0.5, 0.99, 0.8)

# --- CUERPO PRINCIPAL ---
st.title("🚀 Planificación WFM en Cascada")

data = load_data(fuente, archivo)

if data is not None:
    pcrc = st.selectbox("Selecciona PCRC / Skill", data['pcrc'].unique())
    df_pcrc = data[data['pcrc'] == pcrc].copy()
    df_pcrc['y'] = df_pcrc['y'].round().astype(int)

    # PASO 1: VISIÓN MENSUAL (MACRO)
    st.header("Paso 1: Pronóstico Mensual (Presupuesto)")
    with st.expander("Ver Análisis Mensual", expanded=True):
        df_monthly = df_pcrc.set_index('ds').resample('M').sum().reset_index()
        st.line_chart(df_monthly.set_index('ds')['y'])
        st.caption("Validación de volumen total por mes para presupuesto.")

    # PASO 2: VISIÓN DIARIA (DISTRIBUCIÓN SEMANAL)
    st.header("Paso 2: Pronóstico Diario (Day-of-Week)")
    with st.expander("Ver Análisis Diario"):
        df_daily_raw = df_pcrc.copy()
        df_daily_raw['dia_nombre'] = df_daily_raw['ds'].dt.day_name()
        dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_daily_raw['dia_nombre'] = pd.Categorical(df_daily_raw['dia_nombre'], categories=dias_orden, ordered=True)
        resumen_semanal = df_daily_raw.groupby('dia_nombre')['y'].mean()
        st.bar_chart(resumen_semanal)
        st.caption("Distribución: Lunes-Viernes (100%), Sábado (70%), Domingo (30%)")

    # PASO 3: VISIÓN INTERVALO (30 MIN)
    st.header("Paso 3: Pronóstico por Intervalo")
    with st.expander("Ejecutar Forecast Detallado", expanded=True):
        col_f1, col_f2 = st.columns(2)
        f_ini = col_f1.date_input("Inicio Forecast", df_pcrc['ds'].max().date())
        f_fin = col_f2.date_input("Fin Forecast", df_pcrc['ds'].max().date() + pd.Timedelta(days=7))
        
        if st.button("🚀 Generar Forecast e Intervalos"):
            dias = (f_fin - f_ini).days + 1
            with st.spinner("Entrenando modelos de doble joroba..."):
                forecast = run_prophet(df_pcrc, dias * 48)
                forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)
                st.session_state.current_forecast = forecast

    # PASO 4: STAFFING Y VISIONES DINÁMICAS
    if 'current_forecast' in st.session_state:
        st.header("Paso 4: Dimensionamiento y Visión de Datos")
        
        vision = st.radio("Cambiar visión del reporte:", ["Intervalo (30 min)", "Diario", "Semanal", "Mensual"], horizontal=True)
        
        # Procesar Staffing
        res_wfm = get_staffing_requirements(st.session_state.current_forecast, aht, sl, shrinkage)
        
        # Lógica de Resample según Visión
        df_viz = res_wfm.copy()
        if vision == "Diario":
            df_viz = df_viz.set_index('ds').resample('D').agg({'yhat':'sum', 'agentes_netos':'max', 'agentes_nominales':'max'}).reset_index()
        elif vision == "Semanal":
            df_viz = df_viz.set_index('ds').resample('W').agg({'yhat':'sum', 'agentes_netos':'max', 'agentes_nominales':'max'}).reset_index()
        elif vision == "Mensual":
            df_viz = df_viz.set_index('ds').resample('M').agg({'yhat':'sum', 'agentes_netos':'max', 'agentes_nominales':'max'}).reset_index()

        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            st.write(f"**Volumen y Staffing ({vision})**")
            st.line_chart(df_viz.set_index('ds')[['yhat']])
            st.line_chart(df_viz.set_index('ds')[['agentes_netos', 'agentes_nominales']])
        
        with col_g2:
            st.write("**Métricas Resumen**")
            st.dataframe(df_viz[['ds', 'yhat', 'agentes_nominales']].tail(10))
            
            csv = df_viz.to_csv(index=False).encode('utf-8')
            st.download_button(f"📥 Descargar Vista {vision}", csv, f"reporte_{vision}.csv", "text/csv")

        # Heatmap de apoyo
        with st.expander("Ver Mapa de Calor de Demanda"):
            st.plotly_chart(get_error_heatmap(res_wfm))

else:
    st.info("Configura la fuente de datos en el panel izquierdo para iniciar la cascada.")
