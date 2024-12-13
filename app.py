import streamlit as st
import pandas as pd
import statsmodels.api as sm
from streamlit_option_menu import option_menu

# <========================================= SIDEBAR ==========================================>

with st.sidebar:
    selected = option_menu(
        menu_title="Sidebar",
        options=["Home", "Plot Data Awal", "Cek Stasioner", "Diferencing", "Plot ACF & PACF", "Fit Model", "Forecast"],
        default_index=0
    )

# <======================================= session_state ========================================>

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

# <============================================ HOME =============================================>

if selected == "Home":
    st.title("Selamat Datang di Aplikasi Forecasting - SARIMA")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit er at, sed diam nonumy eirmod tempor")

# <========================================= UPLOAD DATA =========================================>

if selected == "Plot Data Awal":
    st.title("Plot Data Awal")

    uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"])
    
    if uploaded_file is not None:
        st.session_state.uploaded_data = pd.read_csv(uploaded_file)

    if st.session_state.uploaded_data is not None:
        data = st.session_state.uploaded_data

        st.write("Data yang di-upload:")
        st.dataframe(data.head())

        selected_date_column = st.selectbox("Pilih kolom untuk tanggal:", data.columns, index=0)
        selected_ts_column = st.selectbox("Pilih kolom untuk analisis time series:", data.columns, index=1)

        data[selected_date_column] = pd.to_datetime(data[selected_date_column], format='%m-%Y', errors='coerce')

        st.subheader("Visualisasi Data")
        st.line_chart(data[selected_ts_column])

# <======================================= CEK STASIONER ========================================>

if selected == "Cek Stasioner":

    def check_stationarity(data):
        result = sm.tsa.adfuller(data)
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Stationary' if result[1] < 0.05 else 'Non-Stationary')
        return result

    st.title("Cek Stasioner")
    
    if st.session_state.uploaded_data is None:
        st.warning("Silakan upload data di tab 'Plot Data Awal' terlebih dahulu!")
    else:
        res = check_stationarity(st.session_state.uploaded_data['Wisnus'])
        st.write(f'ADF Statistic: {res[0]}')
        st.write(f'p-value: {res[1]}')
        if res[1] > 0.05:
            st.toast('Data belum stasioner')
        st.write('Stationary' if res[1] < 0.05 else 'Non-Stationary')

# Footer
st.sidebar.write("---")
st.sidebar.write("Made with ❤️ by Seventeen Teams")


        