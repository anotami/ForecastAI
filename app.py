import streamlit as st
from modules.data_loader import get_data
from modules.models import run_prophet, run_sarima
from modules.validator import calculate_metrics

st.set_page_config(layout="wide", page_title="Forecast BPO")

# Carga masiva de datos (Simulación de 180 días)
if 'df' not in st.session_state:
    st.session_state.df = get_data(is_simulation=True)

df = st.session_state.df

# --- INTERFAZ ---
st.title("🚀 Engine de Pronósticos 30-min")

with st.sidebar:
    pcrc = st.selectbox("Seleccionar PCRC", df['pcrc'].unique())
    modelo = st.radio("Elegir Modelo", ["Prophet", "SARIMA"])
    dias = st.slider("Días a predecir", 1, 14, 7)

# Filtrado por PCRC
df_pcrc = df[df['pcrc'] == pcrc].copy()

if st.button("Ejecutar Pronóstico Masivo"):
    # Separamos últimos 2 días para validar
    train = df_pcrc.iloc[:-96] # 48*2 = 96 intervalos
    test = df_pcrc.iloc[-96:]
    
    with st.spinner('Entrenando modelo...'):
        if modelo == "Prophet":
            res = run_prophet(train, dias * 48)
        else:
            res = run_sarima(train, dias * 48)
            
        # 1. Mostrar Métricas
        # Comparamos la predicción contra el 'test' (lo que realmente pasó)
        # Nota: Ajustar índices para que coincidan en tiempo
        metrics = calculate_metrics(test['y'], res['yhat'].iloc[:96])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Precisión (MAPE)", metrics["MAPE"])
        c2.metric("Desviación (RMSE)", metrics["RMSE"])
        c3.metric("Sesgo (BIAS)", metrics["BIAS"])
        
        # 2. Gráfico
        st.subheader(f"Proyección de llamadas - {pcrc}")
        st.line_chart(res.set_index('ds')[['yhat']])
