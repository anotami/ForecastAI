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
