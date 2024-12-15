import pandas as pd
import streamlit as st
import statsmodels.api as sm
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# <======================================= session_state ========================================>

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

if "data_sources" not in st.session_state:
    st.session_state.data_sources = {}

if "sarima_model_params" not in st.session_state:
    st.session_state.sarima_model_params = None

if "sarima_results" not in st.session_state:
    st.session_state.sarima_results = None

if "sarima_model" not in st.session_state:
    st.session_state.sarima_model = None

# <======================================== SOME FUNC ==========================================>


def check_stationarity(data):
    result = sm.tsa.adfuller(data)
    return result

# <========================================= SIDEBAR ==========================================>


with st.sidebar:
    selected = option_menu(
        menu_title="Sidebar",
        options=["Home", "Plot Data Awal", "Cek Stasioner",
                 "Diferencing", "Plot ACF & PACF", "Fit Model", "Forecast"],
        default_index=0
    )

# <============================================ HOME =============================================>

if selected == "Home":
    st.title("Selamat Datang di Aplikasi Forecasting - SARIMA")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")

# <========================================= UPLOAD DATA =========================================>

if selected == "Plot Data Awal":
    st.title("Plot Data Awal")

    uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"])

    if uploaded_file is not None:
        st.session_state.uploaded_data = pd.read_csv(uploaded_file)
        st.session_state.data_sources["Data Aktual"] = st.session_state.uploaded_data

    if st.session_state.uploaded_data is not None:
        data = st.session_state.uploaded_data

        st.write("Data yang di-upload:")
        st.dataframe(data.head())

        selected_date_column = st.selectbox(
            "Pilih kolom untuk tanggal:", data.columns, index=0)
        selected_ts_column = st.selectbox(
            "Pilih kolom untuk analisis time series:", data.columns, index=1)

        data[selected_date_column] = pd.to_datetime(
            data[selected_date_column], format='%m-%Y', errors='coerce')

        data.set_index(selected_date_column, inplace=True)

        st.subheader("Visualisasi Data")
        st.line_chart(data[selected_ts_column])
        data.reset_index(inplace=True)

# <======================================= CEK STASIONER ========================================>

if selected == "Cek Stasioner":
    st.title("Cek Stasioner")

    if len(st.session_state.data_sources) == 0:
        st.warning(
            "Belum ada data yang tersedia. Upload data di 'Plot Data Awal'.")
    else:
        pilihan = list(st.session_state.data_sources.keys())

        selected_data_key = st.selectbox(
            "Pilih data yang akan diuji:", pilihan)
        selected_data = st.session_state.data_sources[selected_data_key]
        selected_column = st.selectbox(
            "Pilih kolom data:", selected_data.columns)

        if st.button("Cek"):
            data = selected_data[selected_column]
            res = check_stationarity(data)
            st.write(f'ADF Statistic: {res[0]}')
            st.write(f'p-value: {res[1]}')
            if res[1] > 0.05:
                st.warning('Data belum stasioner')
            else:
                st.success('Data stasioner')

# <======================================= DIFFERENCING ========================================>

if selected == "Diferencing":
    st.title("Differencing Data")

    if len(st.session_state.data_sources) == 0:
        st.warning(
            "Belum ada data yang tersedia. Upload data di 'Plot Data Awal'.")
    else:
        pilihan = list(st.session_state.data_sources.keys())

        selected_data_key = st.selectbox(
            "Pilih data untuk diferencing:", pilihan)
        selected_data = st.session_state.data_sources[selected_data_key]
        selected_column = st.selectbox(
            "Pilih kolom data:", selected_data.columns)

        lag = st.number_input("Pilih lag untuk differencing:",
                              min_value=1, max_value=12, value=1)
        if st.button("Lakukan Diferencing"):
            differenced_data = selected_data[selected_column].diff(
                lag).dropna()
            new_key = f"Data Diff (lag {lag})"
            st.session_state.data_sources[new_key] = pd.DataFrame(
                differenced_data, columns=[selected_column])

            st.write("Data hasil Diferencing")
            differenced_data
            st.write(f"Grafik hasil Diferencing ({new_key}):")
            st.line_chart(differenced_data)
        else:
            if st.button("Tampilkan Grafik"):
                selected_data[selected_column]
                st.write(f"Hasil Diferencing")
                st.line_chart(selected_data[selected_column])

# <======================================= DIFFERENCING ========================================>

if selected == "Plot ACF & PACF":
    st.title("Grafik ACF & PACF")
    if len(st.session_state.data_sources) == 0:
        st.warning(
            "Belum ada data yang tersedia. Upload data di 'Plot Data Awal'.")
    else:
        pilihan = list(st.session_state.data_sources.keys())

        selected_data_key = st.selectbox(
            "Pilih data untuk diferencing:", pilihan)
        selected_data = st.session_state.data_sources[selected_data_key]
        selected_column = st.selectbox(
            "Pilih kolom data:", selected_data.columns)

        with st.spinner("Membuat plot ACF dan PACF..."):

            ts_data = selected_data[selected_column].dropna()

            acf_values = acf(ts_data, nlags=20)
            pacf_values = pacf(ts_data, nlags=20)

            acf_fig = go.Figure()
            acf_fig.add_trace(
                go.Bar(x=list(range(len(acf_values))), y=acf_values, name="ACF"))
            acf_fig.update_layout(
                title="Autocorrelation Function (ACF)",
                xaxis_title="Lag",
                yaxis_title="ACF",
                height=400,
            )

            pacf_fig = go.Figure()
            pacf_fig.add_trace(
                go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name="PACF"))
            pacf_fig.update_layout(
                title="Partial Autocorrelation Function (PACF)",
                xaxis_title="Lag",
                yaxis_title="PACF",
                height=400,
            )

            st.plotly_chart(acf_fig, use_container_width=True)
            st.plotly_chart(pacf_fig, use_container_width=True)

# <==================================== FIT MODEL SARIMA =====================================>

if selected == "Fit Model":
    st.title("Fit Model SARIMA")
    if len(st.session_state.data_sources) == 0:
        st.warning(
            "Belum ada data yang tersedia. Upload data di 'Plot Data Awal'.")
    else:
        pilihan = list(st.session_state.data_sources.keys())

        selected_data_key = st.selectbox(
            "Pilih data untuk dibuat model:", pilihan)
        selected_data = st.session_state.data_sources[selected_data_key]
        selected_column = st.selectbox(
            "Pilih kolom data:", selected_data.columns)

        if st.session_state.sarima_model_params is not None:
            params = st.session_state.sarima_model_params
            pv, dv, qv, Pv, Dv, Qv, mv = params[0], params[1], params[2], params[3], params[4], params[5], params[6]
        else:
            pv = 0
            dv = 0
            qv = 0
            Pv = 0
            Dv = 0
            Qv = 0
            mv = 1

        with st.form("SARIMA Parameters"):
            st.subheader("Input Parameter Model")

            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.number_input("p (AR):", min_value=0, step=1, value=pv)
            with col2:
                d = st.number_input("d (Differencing):",
                                    min_value=0, step=1, value=dv)
            with col3:
                q = st.number_input("q (MA):", min_value=0, step=1, value=qv)

            col4, col5, col6, col7 = st.columns(4)
            with col4:
                P = st.number_input("P (Seasonal AR):",
                                    min_value=0, step=1, value=Pv)
            with col5:
                D = st.number_input(
                    "D (Seasonal differencing):", min_value=0, step=1, value=Dv)
            with col6:
                Q = st.number_input("Q (Seasonal MA):",
                                    min_value=0, step=1, value=Qv)
            with col7:
                m = st.number_input("m (Seasonal period):",
                                    min_value=1, max_value=12, step=1, value=mv)

            submit = st.form_submit_button("Fit Model")

        if submit:
            with st.spinner("Fitting model SARIMA..."):
                ts_data = selected_data[selected_column].dropna()

                try:
                    model = SARIMAX(ts_data,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, m)
                                    )
                    results = model.fit()

                    st.session_state.sarima_model_params = [
                        p, d, q, P, D, Q, m]
                    st.session_state.sarima_model = model
                    st.session_state.sarima_results = results

                    st.subheader("Ringkasan Hasil Model")
                    st.text(results.summary())

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
        else:
            if st.session_state.sarima_results is not None:
                st.subheader("Ringkasan Hasil Model")
                st.text(st.session_state.sarima_results.summary())

# <======================================= FORECASTING =======================================>

if selected == "Forecast":
    st.title("Forecasting")
    if "uploaded_data" not in st.session_state or st.session_state.uploaded_data is None:
        st.warning("Silakan upload data di tab sebelumnya!")
    else:
        data_aktual = st.session_state.uploaded_data.copy()
        selected_tcol = st.selectbox(
            "Pilih kolom tanggal:", data_aktual.columns, index=0)
        selected_dcol = st.selectbox(
            "Pilih kolom data:", data_aktual.columns,  index=1)

        forecast_steps = st.slider(
            "Pilih banyaknya forecast:", 1, 36, step=1)

        if st.button("Forecast"):
            data_aktual[selected_tcol] = pd.to_datetime(
                data_aktual[selected_tcol], format="%Y-%m-%d", errors="coerce")
            data_aktual.set_index(selected_tcol, inplace=True)

            if "sarima_results" not in st.session_state or st.session_state.sarima_results is None:
                st.warning(
                    "Model SARIMA belum di-fit. Silakan fit model di tab sebelumnya!")
            else:
                # in-sample predict
                sarima_results = st.session_state.sarima_results
                predictions = sarima_results.predict(
                    start=0, end=len(data_aktual) - 1)

                # out sample predict
                last_date = data_aktual.index[-1]
                forecast_dates = pd.date_range(start=last_date + pd.DateOffset(
                    1), periods=forecast_steps, freq="MS")
                out_sample_forecast = sarima_results.get_forecast(
                    steps=forecast_steps)
                out_sample_pred = pd.Series(
                    out_sample_forecast.predicted_mean.values, index=forecast_dates)

                combined_df = pd.DataFrame({
                    "Tanggal": data_aktual.index.tolist() + forecast_dates.tolist(),
                    "Aktual": data_aktual[selected_dcol].tolist() + [None] * len(out_sample_pred),
                    "Forecast": predictions.tolist() + out_sample_pred.tolist()
                })

                combined_df.reset_index(drop=True, inplace=True)

                st.subheader("Data Aktual VS Prediksi")
                st.dataframe(combined_df)

                st.subheader("Grafik Data Aktual VS Prediksi")
                trace_aktual = go.Scatter(
                    x=combined_df["Tanggal"],
                    y=combined_df["Aktual"],
                    mode="lines",
                    name="Data Aktual",
                    line=dict(color="cyan")
                )

                trace_prediksi = go.Scatter(
                    x=combined_df["Tanggal"],
                    y=combined_df["Forecast"],
                    mode="lines",
                    name="Forecast",
                    line=dict(color="orange", dash="dash")
                )

                layout = go.Layout(
                    title="Perbandingan Data Aktual dan Prediksi",
                    xaxis=dict(title="Tanggal"),
                    yaxis=dict(title="Wisnus"),
                    showlegend=True
                )

                data_plot = [trace_aktual, trace_prediksi]
                fig = go.Figure(data=data_plot, layout=layout)
                st.plotly_chart(fig)

                mae = mean_absolute_error(
                    data_aktual[selected_dcol], predictions)
                mape = mean_absolute_percentage_error(
                    data_aktual[selected_dcol], predictions) * 100

                st.subheader("MAE & MAPE")
                st.write(f"MAE: {mae:.2f}")

                st.write(f"MAPE: {mape:.2f}%")


# <========================================= FOOTER ==========================================>

st.sidebar.write("---")
st.sidebar.write("Made with ❤️ by Seventeen Teams")
