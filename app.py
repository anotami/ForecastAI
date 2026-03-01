import streamlit as st
from modules.data_loader import load_data
from modules.models import run_prophet, run_sarima
from modules.validator import calculate_metrics, get_error_heatmap

st.set_page_config(layout="wide", page_title="Forecast Master BPO")

st.title("📞 BPO Forecasting & Precision Tool")

# --- SIDEBAR: FUENTE DE DATOS ---
with st.sidebar:
    st.header("1. Configuración de Datos")
    fuente = st.radio("Fuente de datos:", ["Simulación Aleatoria", "Subir Archivo (CSV/Excel)"])
    
    uploaded_file = None
    if fuente == "Subir Archivo (CSV/Excel)":
        uploaded_file = st.file_uploader("Formato: [fecha, llamadas, pcrc]", type=['csv', 'xlsx'])
        st.info("💡 La fecha debe estar en intervalos de 30 min.")

    st.header("2. Parámetros de Pronóstico")
    modelo_choice = st.selectbox("Modelo", ["Prophet", "SARIMA"])
    horizonte = st.selectbox("Horizonte de tiempo", ["1 Día", "1 Semana", "1 Mes"])
    
    # Mapeo de periodos (intervalos de 30 min)
    mapeo_periodos = {"1 Día": 48, "1 Semana": 336, "1 Mes": 1440}
    periodos = mapeo_periodos[horizonte]

# --- PROCESAMIENTO ---
data = load_data(fuente, uploaded_file)

if data is not None:
    pcrc_lista = data['pcrc'].unique()
    pcrc_selected = st.selectbox("Selecciona PCRC para el Forecast", pcrc_lista)
    
    df_pcrc = data[data['pcrc'] == pcrc_selected].copy()
    
    if st.button("🚀 Ejecutar Forecast Masivo"):
        # Split para validación (usamos los últimos 2 días para el heatmap)
        train = df_pcrc.iloc[:-96]
        actuals = df_pcrc.iloc[-96:] 

        with st.spinner('Entrenando y proyectando...'):
            if modelo_choice == "Prophet":
                forecast = run_prophet(train, periodos)
            else:
                forecast = run_sarima(train, periodos)
            
            # --- DASHBOARD DE RESULTADOS ---
            tab1, tab2 = st.tabs(["📈 Pronóstico e Intervalos", "🎯 Análisis de Precisión"])
            
            with tab1:
                st.subheader(f"Proyección {horizonte} - {pcrc_selected}")
                # Gráfico principal con intervalos
                st.line_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])
                st.dataframe(forecast)

            with tab2:
                # Unir predicción con datos reales de test para el heatmap
                val_df = actuals.merge(forecast, on='ds', suffixes=('_real', '_pred'))
                val_df = val_df.rename(columns={'y': 'y_real'})
                
                col_m1, col_m2 = st.columns(2)
                metrics = calculate_metrics(val_df['y_real'], val_df['yhat'])
                col_m1.metric("MAPE (Precisión General)", metrics["MAPE"])
                col_m2.metric("Sesgo (Over/Under forecast)", metrics["BIAS"])
                
                # Mostrar Heatmap de error
                fig_heat = get_error_heatmap(val_df)
                st.plotly_chart(fig_heat, use_container_width=True)

else:
    st.warning("Por favor, sube un archivo o selecciona simulación para comenzar.")
