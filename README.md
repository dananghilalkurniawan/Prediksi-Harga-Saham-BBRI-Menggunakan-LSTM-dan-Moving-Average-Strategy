# 📈 Prediksi Harga Saham BBRI Menggunakan LSTM dan Moving Average Strategy

Proyek ini bertujuan untuk memprediksi harga saham **BBRI (Bank Rakyat Indonesia)** dengan memanfaatkan model **LSTM (Long Short-Term Memory)** berbasis deep learning, serta memadukannya dengan strategi teknikal sederhana berupa **Moving Average (SMA)** untuk menghasilkan sinyal **beli** dan **jual**.

------------------------------------------------------------------------

## 📊 Sumber Data

Data harga historis saham BBRI diperoleh menggunakan library `yfinance` dari Yahoo Finance:

``` python
import yfinance as yf
data = yf.download('BBRI.JK', start='2015-01-01', end='2025-01-01')
```

------------------------------------------------------------------------

## 🔧 Langkah-langkah Proyek

### 1. 📥 Pengambilan dan Pemrosesan Data

-   Hanya menggunakan kolom `Close` (harga penutupan).
-   Data diindeks dengan `Date`.
-   Plot awal menunjukkan tren meningkat dengan fluktuasi signifikan.

### 2. 🔎 Normalisasi Data

-   Menggunakan `MinMaxScaler` dari `sklearn.preprocessing`.
-   Harga dinormalisasi ke range `[0, 1]` agar model LSTM dapat belajar lebih efektif.

### 3. 🧠 Transformasi Dataset untuk LSTM (Windowing)

-   Data diubah menjadi bentuk `(samples, timesteps, features)`.
-   Menggunakan jendela waktu (window size) sebanyak 60 hari untuk memprediksi hari ke-61.

Contoh bentuk input:

```         
X.shape = (2415, 60, 1)
y.shape = (2415,)
```

------------------------------------------------------------------------

## 🧱 Pembuatan dan Pelatihan Model LSTM

### Arsitektur Model:

``` python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(1)
])
```

-   Loss Function: `mean_squared_error`
-   Optimizer: `adam`
-   Epoch: 50
-   Hasil training loss terakhir: \~0.00069

------------------------------------------------------------------------

## 📉 Visualisasi Prediksi LSTM (Seluruh Data)

Model berhasil memprediksi harga saham BBRI dengan **akurasi tinggi**, mengikuti pola harga secara natural dan realistis.

![output](https://github.com/user-attachments/assets/0d994546-ee3e-445b-affe-2a6578be28a0)


------------------------------------------------------------------------

## 🔮 Prediksi Harga Saham 30 Hari ke Depan

-   Dilakukan prediksi iteratif menggunakan hasil dari window terakhir.
-   Setiap prediksi hari baru ditambahkan kembali ke input window.
-   Prediksi di-scaler inverse agar kembali ke satuan Rupiah.

``` python
future_predictions = []
window = last_60_data.reshape(1, 60, 1)
for _ in range(30):
    pred = model.predict(window)
    future_predictions.append(pred[0,0])
    window = np.append(window[:,1:,:], [[[pred[0,0]]]], axis=1)
```

------------------------------------------------------------------------

## 🧪 Evaluasi Model: Train-Test Split

### Pembagian Data:

-   **80% data** untuk pelatihan, **20% untuk pengujian** (belum pernah dilihat model).
-   Windowing tetap digunakan pada data pelatihan dan pengujian.

### Hasil Evaluasi:

-   **RMSE (Root Mean Squared Error)**: ±127.70 Rupiah
-   **MAE (Mean Absolute Error)**: ±99.00 Rupiah

### Visualisasi Prediksi vs Aktual (Data Pengujian):

![output2](https://github.com/user-attachments/assets/1e9ee523-97b1-415e-b4ea-d92130a7a543)


------------------------------------------------------------------------

## 🟩 Strategi Sinyal Beli & Jual Menggunakan Moving Average

### Metodologi:

-   Menggunakan dua Simple Moving Average (SMA):
    -   **SMA-20 (jangka pendek)**
    -   **SMA-50 (jangka menengah)**
-   Sinyal Beli: SMA-20 memotong ke atas SMA-50 (Golden Cross)
-   Sinyal Jual: SMA-20 memotong ke bawah SMA-50 (Death Cross)

### Visualisasi Sinyal:

![output3](https://github.com/user-attachments/assets/2c8c209d-df15-4838-b36d-c1de05f52971)


-   🔼 Hijau = Sinyal Beli
-   🔽 Merah = Sinyal Jual

------------------------------------------------------------------------

### Bonus Tambahan (Prediksi 30 Hari ke Depan Saham BBRI)
![output4](https://github.com/user-attachments/assets/2a770b7d-dd8c-4073-a928-c80aab82ee42)

------------------------------------------------------------------------

## ✅ Kesimpulan

-   Model **LSTM** sangat efektif dalam mempelajari pola historis harga saham BBRI.
-   Strategi kombinasi dengan **Moving Average** memperkuat proses pengambilan keputusan beli & jual.
-   Prediksi masa depan model juga menunjukkan tren realistis dan konsisten.
-   Evaluasi menunjukkan performa model **baik di data yang belum pernah dilihat**.
-   Kombinasi deep learning + analisis teknikal = pendekatan cerdas dan aplikatif untuk trading saham.

------------------------------------------------------------------------

## 📚 Teknologi yang Digunakan

-   `Python`
-   `pandas`, `numpy`, `matplotlib`, `seaborn`
-   `scikit-learn`
-   `TensorFlow / Keras`
-   `yfinance` (untuk data historis)

------------------------------------------------------------------------

## 🛠️ Potensi Pengembangan

-   Menggunakan **indikator tambahan**: RSI, MACD, Bollinger Bands
-   Implementasi **backtesting strategi trading**
-   Menyimpan dan deploy model ke API
-   Membuat dashboard interaktif untuk real-time prediksi

------------------------------------------------------------------------

## 📁 Struktur Folder Project

``` bash
.
├── data/
│   └── BBRI_data.csv
├── model/
│   └── lstm_model.h5
├── outputs/
│   ├── output.png         # Prediksi Full
│   ├── output2.png        # Prediksi di Test Split
│   └── output3.png        # Sinyal Buy/Sell (SMA)
└── notebook/
    └── lstm_bbri_analysis.ipynb
```

------------------------------------------------------------------------

## 📩 Kontak

> **Danang Hilal Kurniawan**\
> Mahasiswa Data Science\
> Email: [hilalkurniawandanang\@gmail.com](mailto:hilalkurniawandanang@gmail.com)
