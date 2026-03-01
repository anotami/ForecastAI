import streamlit as st
import pandas as pd
import sys
import os

# Configuración de rutas para módulos internos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.data_loader import load_data
    from modules.models import run_prophet
    from modules.staffing import get_staffing_requirements
except ImportError as e:
    st.error(f"Error de módulos: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="WFM Cascada Engine")

# --- SIDEBAR PERSISTENTE CON BOTONES DE AJUSTE RÁPIDO ---
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/data-report-concept-illustration_114360-883.jpg")
    st.title("⚙️ Configuración WFM")
    
    fuente = st.radio("Origen de Datos", ["Simulación Aleatoria", "Subir Archivo"])
    st.markdown("---")
    
    st.header("Parámetros Staffing")

    # 1. Lógica para TMO (AHT)
    if 'aht_val' not in st.session_state:
        st.session_state.aht_val = 300.0

    col1, col2 = st.columns(2)
    if col1.button("-10% TMO"):
        st.session_state.aht_val *= 0.9
    if col2.button("+10% TMO"):
        st.session_state.aht_val *= 1.1

    aht = st.number_input("TMO / AHT (Seg)", value=float(st.session_state.aht_val), key="aht_input")
    # Sincronizar por si el usuario escribe manualmente
    st.session_state.aht_val = aht

    st.markdown("---")

    # 2. Lógica para Shrinkage
    if 'shr_val' not in st.session_state:
        st.session_state.shr_val = 0.30

    col3, col4 = st.columns(2)
    if col3.button("-10% Shrink"):
        # Reduce un 10% del valor actual (ej: de 30% a 27%)
        st.session_state.shr_val = max(0.0, st.session_state.shr_val * 0.9)
    if col4.button("+10% Shrink"):
        st.session_state.shr_val = min(1.0, st.session_state.shr_val * 1.1)

    shrinkage = st.slider("Shrinkage Total (%)", 0.0, 0.9, float(st.session_state.shr_val), step=0.01)
    # Sincronizar por si el usuario mueve el slider
    st.session_state.shr_val = shrinkage

    st.markdown("---")
    sl = st.slider("Target SL (%)", 0.5, 0.99, 0.8)

# CARGA INICIAL
data = load_data(fuente)

# TODO el proceso debe estar dentro de este bloque if para asegurar que 'data' existe
if data is not None:
    pcrc = st.selectbox("Selecciona PCRC / Skill", data['pcrc'].unique())
    df_pcrc = data[data['pcrc'] == pcrc].copy()
    df_pcrc['y'] = df_pcrc['y'].round().astype(int)

    # --- PASO 1: FORECAST MENSUAL (MACRO) ---
    st.header("Paso 1: Pronóstico Mensual (Presupuesto)")
    with st.expander("Ver Análisis Mensual", expanded=True):
        df_monthly = df_pcrc.set_index('ds').resample('M').sum().reset_index()
        st.line_chart(df_monthly.set_index('ds')['y'])
        if st.button("Validar Meses y Continuar"):
            st.success("Volumen mensual validado.")

# --- PASO 2: DIARIO (DISTRIBUCIÓN) ---
    st.header("Paso 2: Pronóstico Diario (Day-of-Week)")
    with st.expander("Ver Análisis Diario"):
        df_daily = df_pcrc.copy()
        
        # 1. Extraemos el nombre del día
        df_daily['dia_nombre'] = df_daily['ds'].dt.day_name()
        
        # 2. Definimos el orden correcto (Lunes a Domingo)
        # Nota: Si tu sistema está en español, usa los nombres en inglés aquí 
        # porque pandas.dt.day_name() devuelve nombres en inglés por defecto.
        dias_ordenados = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # 3. Convertimos a categoría con orden específico
        df_daily['dia_nombre'] = pd.Categorical(df_daily['dia_nombre'], categories=dias_ordenados, ordered=True)
        
        # 4. Agrupamos y ordenamos por la categoría
        resumen_semanal = df_daily.groupby('dia_nombre')['y'].mean()
        
        # Mostrar el gráfico
        st.bar_chart(resumen_semanal)
        
        st.caption("Validación de distribución semanal: Lunes-Viernes (Pico), Sábado (70%), Domingo (30%)")
        
        if st.button("Validar Días y Continuar"):
            st.success("Distribución diaria validada correctamente.")

# --- PASO 3: PRONÓSTICO POR INTERVALO (30 MIN) ---
    st.header("Paso 3: Pronóstico por Intervalo (30 min)")
    with st.expander("Ejecutar Forecast de Detalle", expanded=True):
        col1, col2 = st.columns(2)
        f_ini = col1.date_input("Fecha Inicio", df_pcrc['ds'].max().date(), key="f_ini")
        f_fin = col2.date_input("Fecha Fin", df_pcrc['ds'].max().date() + pd.Timedelta(days=7), key="f_fin")
        
        if st.button("Ejecutar Forecast Detallado"):
            dias_a_predecir = (f_fin - f_ini).days + 1
            periodos = dias_a_predecir * 48
            
            with st.spinner("Generando pronóstico completo..."):
                forecast = run_prophet(df_pcrc, periodos)
                forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)
                st.session_state.current_forecast = forecast
                
        if 'current_forecast' in st.session_state:
            st.subheader("📊 Pronóstico Completo del Periodo Seleccionado")
            # Mostramos todo el periodo generado, no solo las primeras 48 filas
            st.line_chart(st.session_state.current_forecast.set_index('ds')['yhat'])
            
            # Opción de Exportación Directa del Pronóstico
            csv_forecast = st.session_state.current_forecast.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exportar Pronóstico (CSV)",
                data=csv_forecast,
                file_name=f"pronostico_{pcrc}_{f_ini}_al_{f_fin}.csv",
                mime='text/csv',
            )

    # --- PASO 4: STAFFING (FINAL) ---
    st.header("Paso 4: Dimensionamiento de Personal")
    if 'current_forecast' in st.session_state:
        with st.expander("Calcular Agentes Requeridos", expanded=False):
            # Usamos los parámetros de la barra lateral (TMO, SL, Shrinkage)
            res_wfm = get_staffing_requirements(st.session_state.current_forecast, aht, sl, shrinkage)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Llamadas Totales", f"{res_wfm['yhat'].sum():,}")
            c2.metric("Agentes Requeridos (Pico)", int(res_wfm['agentes_netos'].max()))
            c3.metric("Plantilla Nominal", int(res_wfm['agentes_nominales'].max()))
            
            st.subheader("📅 Plan de Staffing por Intervalo")
            # Mostramos el staffing completo para el periodo seleccionado
            st.line_chart(res_wfm.set_index('ds')[['agentes_netos', 'agentes_nominales']])
            
            # Exportación del Plan Maestro (Staffing + Pronóstico)
            csv_wfm = res_wfm.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Plan Maestro de WFM (Excel/CSV)",
                data=csv_wfm,
                file_name=f"plan_WFM_{pcrc}_{f_ini}_al_{f_fin}.csv",
                mime='text/csv',
            )

else:
    st.info("Por favor, selecciona una fuente de datos en el panel izquierdo.")
