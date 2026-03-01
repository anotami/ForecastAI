import streamlit as st
from modules.data_loader import load_data
from modules.models import run_prophet, run_sarima
from modules.staffing import get_staffing_requirements
from modules.validator import get_error_heatmap

# ... (Configuración inicial igual al paso anterior) ...

if data is not None:
    tab_data, tab_forecast, tab_staffing = st.tabs(["📂 Datos", "📈 Forecast", "👥 Staffing"])

    with tab_data:
        st.write("Vista previa de la carga masiva")
        st.dataframe(data.head())

    with tab_forecast:
        # Lógica de entrenamiento y gráficos de Prophet/SARIMA
        # (Lo que ya construimos antes)
        if st.button("Ejecutar Modelos"):
            st.session_state.forecast_res = run_prophet(train, periodos)
            st.line_chart(st.session_state.forecast_res.set_index('ds')['yhat'])

    with tab_staffing:
        if 'forecast_res' in st.session_state:
            st.subheader("Configuración de Operaciones (Erlang C)")
            c1, c2, c3 = st.columns(3)
            aht = c1.number_input("AHT (Segundos)", value=300)
            target_ns = c2.slider("Target SL (%)", 0.1, 1.0, 0.8)
            shrinkage = c3.slider("Shrinkage (%)", 0.0, 0.5, 0.3)
            
            # Cálculo de Agentes
            res_staffing = get_staffing_requirements(st.session_state.forecast_res, aht)
            
            # Aplicar Shrinkage (Agentes Reales = Requeridos / (1 - Shrinkage))
            res_staffing['agentes_nominales'] = (res_staffing['agentes_req'] / (1 - shrinkage)).apply(math.ceil)
            
            st.subheader("Cálculo de Agentes por Intervalo (30 min)")
            st.line_chart(res_staffing.set_index('ds')[['agentes_req', 'agentes_nominales']])
            
            # Tabla resumen
            st.dataframe(res_staffing[['ds', 'yhat', 'agentes_req', 'agentes_nominales']])
