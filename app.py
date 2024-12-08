import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="Forecasting SARIMA", layout="wide")

# Fungsi utama
def main():
    st.title("Forecasting SARIMA untuk Data Kunjungan Wisatawan")
    
    # Upload file
    uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"])
    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:")
        st.dataframe(data.head())

        # Proses data
        data['Date'] = pd.to_datetime(data['Date'], format='%m-%Y')
        data.set_index('Date', inplace=True)
        
        # Plot data asli
        st.subheader("Visualisasi Data")
        st.line_chart(data['Kunjungan'])

        # Model SARIMA
        p, d, q = 1, 1, 1
        P, D, Q, S = 1, 1, 1, 12
        model = SARIMAX(data['Kunjungan'], 
                        order=(p, d, q), 
                        seasonal_order=(P, D, Q, S),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        results = model.fit()
        st.text(results.summary())
        
        # Forecast
        forecast_periods = st.slider("Pilih periode untuk forecasting (bulan):", 1, 24, 12)
        forecast = results.get_forecast(steps=forecast_periods)
        forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), 
                                       periods=forecast_periods, freq='MS')
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()

        # Plot hasil forecasting
        st.subheader("Hasil Forecasting")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Kunjungan'], label='Data Aktual', color='blue')
        ax.plot(forecast_index, forecast_mean, label='Forecast', color='red')
        ax.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], 
                        color='pink', alpha=0.3)
        ax.set_title("Forecasting SARIMA")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Jumlah Kunjungan")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
