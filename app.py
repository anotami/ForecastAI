import streamlit as st
import pandas as pd
import sys
import os

# 1. Configuración de Rutas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. Importación Segura
try:
    from modules.data_loader import load_data
    from modules.models import run_prophet, run_sarima
    from modules.staffing import get_staffing_requirements
except ImportError as e:
    st.error(f"Error al importar módulos: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="WFM Interactive Forecast")

st.title("📞 Pronóstico de Llamadas e Intervalos")

# --- PASO 1: CARGA Y VISUALIZACIÓN INICIAL ---
if 'data' not in st.session_state:
    # Generamos o cargamos los datos (180 días por defecto)
    st.session_state.data = load_data("Simulación Aleatoria")

df = st.session_state.data
pcrcs = df['pcrc'].unique()

st.sidebar.header("Configuración Inicial")
pcrc_selected = st.sidebar.selectbox("Selecciona el PCRC", pcrcs)

# Filtrar datos del PCRC seleccionado
df_pcrc = df[df['pcrc'] == pcrc_selected].copy()
df_pcrc['y'] = df_pcrc['y'].round().astype(int) # Asegurar números enteros

st.subheader(f"Histórico de Llamadas: {pcrc_selected}")
fecha_limite_hist = df_pcrc['ds'].max()
st.write(f"Datos disponibles hasta: **{fecha_limite_hist.strftime('%d/%m/%Y %H:%M')}**")

# Gráfico del histórico
st.line_chart(df_pcrc.set_index('ds')['y'])

st.markdown("---")

# --- PASO 2: SOLICITUD DE RANGO DE PRONÓSTICO ---
st.subheader("🎯 Configuración del Pronóstico")
col1, col2 = st.columns(2)

with col1:
    fecha_inicio = st.date_input("Fecha de inicio del pronóstico", fecha_limite_hist + pd.Timedelta(days=1))
with col2:
    fecha_fin = st.date_input("Fecha final del pronóstico", fecha_limite_hist + pd.Timedelta(days=7))

# Calcular cuántos intervalos de 30 min hay en ese rango
diferencia_dias = (fecha_fin - fecha_inicio).days + 1
periodos_30min = diferencia_dias * 48

if st.button("Generar Pronóstico"):
    if fecha_fin < fecha_inicio:
        st.error("La fecha final debe ser posterior a la fecha de inicio.")
    else:
        with st.spinner(f"Calculando pronóstico para {diferencia_dias} días..."):
            # Ejecutar modelo (usamos Prophet por defecto para este ejemplo)
            forecast = run_prophet(df_pcrc, periodos_30min)
            
            # Asegurar que las llamadas proyectadas sean enteros positivos
            forecast['yhat'] = forecast['yhat'].clip(lower=0).round().astype(int)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0).round().astype(int)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0).round().astype(int)

            # Filtrar el resultado para que empiece exactamente en la fecha elegida
            forecast = forecast[forecast['ds'].dt.date >= fecha_inicio]

            st.success(f"Pronóstico generado con éxito desde {fecha_inicio} hasta {fecha_fin}")
            
            # Visualización de resultados
            tab_graf, tab_data = st.tabs(["📈 Gráfico", "📄 Datos Proyectados"])
            
            with tab_graf:
                st.line_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])
            
            with tab_data:
                st.dataframe(forecast)
                
            # Opción para continuar al Staffing
            st.session_state.forecast_final = forecast
            st.info("Ahora puedes dirigirte al módulo de Staffing en la barra lateral si deseas calcular agentes.")
