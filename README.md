# SARIMA Forecasting Web App

## Deskripsi

Aplikasi web untuk melakukan forecasting menggunakan metode SARIMA. Aplikasi ini dapat input data dari file CSV, fitting model SARIMA, serta menampilkan prediksi dan perbandingan data aktual dengan hasil forecasting.

NB: Pastikan kolom tanggal memiliki format mm-yyyy. ex: 12-2022

---

## Cara Instalasi dan Menjalankan Aplikasi

### 1. Clone Repository

Pertama, clone repository ini menggunakan perintah berikut:

```bash
git clone https://github.com/avrzll/sarima-forecasting.git
cd sarima-forecasting
```

### 2. Buat Virtual Environment

```bash
python -m venv venv
```

### 3. Mengaktifkan Virtual Environment

#### Windows:

```bash
venv\Scripts\activate
```

#### macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependensi/Library

Setelah mengaktifkan virtual environment, install semua dependensi yang diperlukan:

```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi Web

Setelah semua dependensi terinstal, jalankan aplikasi dengan perintah berikut:

```bash
streamlit run app.py
```

### 6. Akses Aplikasi

Buka browser Anda dan akses aplikasi pada URL berikut:

```
http://localhost:8501
```

---

## Important

- Pastikan file data yang diupload memiliki kolom `Date` (mm-yyyy) dan Data Time Series.
- Gunakan slider untuk memilih langkah forecast sesuai kebutuhan.
- Perhatikan peringatan atau pesan error di aplikasi untuk memastikan kelancaran aplikasi.

---

## Kontak

Jika ada masalah atau pertanyaan, silakan hubungi:

- Instagram: @avrzll\_
- Email: avrizalrendiprayoga@gmail.com

Enjoy this app!
