import streamlit as st
import pandas as pd
import sys
import os

# Configuración de rutas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.data_loader import load_data
    from modules.models import run_prophet
    from modules.staffing import get_staffing_requirements
    from modules.validator import get_error_heatmap
except ImportError as e:
    st.error(f"Error de módulos: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="WFM Steps Engine")

# --- LÓGICA DE NAVEGACIÓN (STEPS) ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'data' not in st.session_state:
    st.session_state.data = None

def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1
def reset_steps(): 
    st.session_state.step = 1
    st.session_state.data = None
    if 'current_forecast' in st.session_state: del st.session_state.current_forecast

# --- SIDEBAR: PARÁMETROS FIJOS ---
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/data-report-concept-illustration_114360-883.jpg")
    st.title("⚙️ Parámetros Operativos")
    
    # Análisis de Sensibilidad TMO
    if 'aht_val' not in st.session_state: st.session_state.aht_val = 300.0
    c1, c2 = st.columns(2)
    if c1.button("-10% TMO"): st.session_state.aht_val *= 0.9
    if c2.button("+10% TMO"): st.session_state.aht_val *= 1.1
    aht = st.number_input("TMO / AHT (Seg)", value=float(st.session_state.aht_val))
    
    # Análisis de Sensibilidad Shrinkage
    if 'shr_val' not in st.session_state: st.session_state.shr_val = 0.30
    c3, c4 = st.columns(2)
    if c3.button("-10% Shrink"): st.session_state.shr_val *= 0.9
    if c4.button("+10% Shrink"): st.session_state.shr_val = min(0.9, st.session_state.shr_val * 1.1)
    shrinkage = st.slider("Shrinkage Total (%)", 0.0, 0.9, float(st.session_state.shr_val))
    
    sl = st.slider("Target SL (%)", 0.5, 0.99, 0.8)
    
    if st.button("🔄 Reiniciar Proceso"): reset_steps()

# --- CUERPO PRINCIPAL ---
st.title("🚀 Planificación BPO: Flujo en Cascada")

# Indicador de Progreso
pasos = ["1. Ingesta", "2. Mensual", "3. Diario", "4. Intervalo", "5. Staffing"]
st.write(f"**Paso Actual:** {pasos[st.session_state.step - 1]}")
st.progress(st.session_state.step / len(pasos))

# --- PASO 1: INGESTA DE DATOS ---
if st.session_state.step == 1:
    st.header("1️⃣ Ingesta de Datos")
    fuente = st.radio("Selecciona origen:", ["Simulación Aleatoria", "Subir Archivo CSV"])
    archivo = st.file_uploader("Cargar histórico", type=['csv']) if fuente == "Subir Archivo CSV" else None
    
    if st.button("Confirmar y Cargar Datos"):
        with st.spinner("Cargando histórico..."):
            st.session_state.data = load_data(fuente, archivo)
            next_step()
            st.rerun()

# --- PASO 2: PRONÓSTICO MENSUAL ---
elif st.session_state.step == 2:
    st.header("2️⃣ Visión Mensual (Presupuesto)")
    df = st.session_state.data
    pcrc = st.selectbox("Selecciona PCRC", df['pcrc'].unique())
    df_pcrc = df[df['pcrc'] == pcrc].copy()
    
    df_monthly = df_pcrc.set_index('ds').resample('M').sum().reset_index()
    st.line_chart(df_monthly.set_index('ds')['y'])
    st.write(f"Volumen promedio mensual: **{int(df_monthly['y'].mean()):,} llamadas**")
    
    col_n1, col_n2 = st.columns(2)
    if col_n1.button("⬅️ Atrás"): prev_step(); st.rerun()
    if col_n2.button("Confirmar Volumen Mensual ➡️"): next_step(); st.rerun()

# --- PASO 3: DISTRIBUCIÓN DIARIA ---
elif st.session_state.step == 3:
    st.header("3️⃣ Visión Diaria (Day-of-Week)")
    df = st.session_state.data
    df['dia_nombre'] = df['ds'].dt.day_name()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['dia_nombre'] = pd.Categorical(df['dia_nombre'], categories=order, ordered=True)
    
    resumen_semanal = df.groupby('dia_nombre')['y'].mean()
    st.bar_chart(resumen_semanal)
    st.info("Distribución detectada: Lunes-Viernes (Pico), Sábado (70%), Domingo (30%)")
    
    col_n1, col_n2 = st.columns(2)
    if col_n1.button("⬅️ Atrás"): prev_step(); st.rerun()
    if col_n2.button("Confirmar Distribución Diaria ➡️"): next_step(); st.rerun()

# --- PASO 4: PRONÓSTICO POR INTERVALO ---
elif st.session_state.step == 4:
    st.header("4️⃣ Pronóstico Detallado (30 min)")
    df = st.session_state.data
    
    col_f1, col_f2 = st.columns(2)
    f_ini = col_f1.date_input("Desde", df['ds'].max().date())
    f_fin = col_f2.date_input("Hasta", df['ds'].max().date() + pd.Timedelta(days=7))
    
    if st.button("🚀 Ejecutar Prophet (Doble Joroba)"):
        dias = (f_fin - f_ini).days + 1
        with st.spinner("Calculando intervalos..."):
            forecast = run_prophet(df, dias * 48)
            forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)
            st.session_state.current_forecast = forecast
            st.success("Pronóstico generado.")
    
    if 'current_forecast' in st.session_state:
        st.line_chart(st.session_state.current_forecast.set_index('ds')['yhat'])
        col_n1, col_n2 = st.columns(2)
        if col_n1.button("⬅️ Atrás"): prev_step(); st.rerun()
        if col_n2.button("Ir al Staffing Final ➡️"): next_step(); st.rerun()

# --- PASO 5: STAFFING FINAL Y EXPORTACIÓN ---
elif st.session_state.step == 5:
    st.header("5️⃣ Dimensionamiento de Agentes")
    forecast = st.session_state.current_forecast
    
    # Procesar Erlang C
    res_wfm = get_staffing_requirements(forecast, aht, sl, shrinkage)
    
    # Selectores de Visión Dinámica
    vision = st.radio("Ver reporte por:", ["Intervalo", "Día", "Semana", "Mes"], horizontal=True)
    
    df_viz = res_wfm.copy()
    if vision == "Día":
        df_viz = df_viz.set_index('ds').resample('D').agg({'yhat':'sum', 'agentes_netos':'max', 'agentes_nominales':'max'}).reset_index()
    elif vision == "Semana":
        df_viz = df_viz.set_index('ds').resample('W').agg({'yhat':'sum', 'agentes_netos':'max', 'agentes_nominales':'max'}).reset_index()
    elif vision == "Mes":
        df_viz = df_viz.set_index('ds').resample('M').agg({'yhat':'sum', 'agentes_netos':'max', 'agentes_nominales':'max'}).reset_index()

    st.line_chart(df_viz.set_index('ds')[['agentes_netos', 'agentes_nominales']])
    
    # Métricas de impacto
    c1, c2, c3 = st.columns(3)
    c1.metric("Pico de Agentes", int(res_wfm['agentes_netos'].max()))
    c2.metric("Plantilla Nominal", int(res_wfm['agentes_nominales'].max()))
    c3.metric("Llamadas Totales", f"{int(res_wfm['yhat'].sum()):,}")

    csv = df_viz.to_csv(index=False).encode('utf-8')
    st.download_button(f"📥 Exportar Plan ({vision})", csv, f"WFM_{vision}.csv", "text/csv")
    
    if st.button("⬅️ Volver a Ajustar Fechas"): prev_step(); st.rerun()
