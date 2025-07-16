# 🏡 House Price Predictor

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&style=flat-square) ![License](https://img.shields.io/badge/License-MIT-2ECC71?style=flat-square) ![GUI](https://img.shields.io/badge/GUI-CustomTkinter-9B59B6?style=flat-square) ![ML](https://img.shields.io/badge/Machine%20Learning-Linear%20Regression-F39C12?style=flat-square) ![Status](https://img.shields.io/badge/Status-Active-27AE60?style=flat-square)

Selamat datang di **House Price Predictor**, aplikasi prediksi harga rumah berbasis GUI yang memanfaatkan **Linear Regression** untuk memberikan estimasi harga akurat. Dibangun dengan **Python** dan **CustomTkinter**, aplikasi ini menawarkan antarmuka modern dengan pengalaman pengguna yang mulus dan intuitif.

---

## 📸 Tangkapan Layar
![Aplikasi House Price Predictor](https://assets/screenshot.png)

---

## 🌟 Fitur Unggulan
- 🔍 **Prediksi Harga Akurat**: Estimasi harga rumah berdasarkan:
  - 🏠 **Luas Bangunan** (m²)
  - 🛏️ **Jumlah Kamar**
  - 📍 **Lokasi** (Pusat Kota, Suburban, Pinggiran)
- 📊 **Visualisasi Interaktif**: Grafik scatter plot untuk memahami distribusi data
- 🎨 **Desain Modern**: Antarmuka dengan tema **dark mode** yang elegan
- 📈 **Evaluasi Model**: Menampilkan **R² Score** untuk mengukur akurasi model
- 💾 **Penyimpanan Model**: Simpan dan muat model dengan mudah menggunakan `joblib`
- ⚡ **Performa Cepat**: Proses prediksi yang efisien dan ringan

---

## 🛠️ Teknologi yang Digunakan
| Teknologi        | Deskripsi                       |
|-------------------|---------------------------------|
| 🐍 **Python 3.8+** | Bahasa pemrograman utama        |
| 🎨 **CustomTkinter** | Antarmuka pengguna modern      |
| 🤖 **scikit-learn** | Library untuk machine learning  |
| 📅 **pandas**      | Pemrosesan dan analisis data    |
| 📊 **matplotlib**  | Visualisasi data interaktif     |
| 💾 **joblib**      | Serialisasi model machine learning |

---

## 🚀 Cara Memulai
### 📦 Prasyarat
- **Python 3.8+** terinstal
- Git untuk cloning repository
- Koneksi internet untuk mengunduh dependensi

### 🛠️ Langkah Instalasi
1. **Clone Repository**:
   ```bash
   git clone https://github.com/username/house-price-predictor.git
   cd house-price-predictor
   ```

2. **Buat Virtual Environment** (disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi**:
   ```bash
   python main.py
   ```

---

## 📂 Struktur Proyek
```
house-price-predictor/
├── main.py                    # Kode utama aplikasi
├── house_data.csv             # Dataset contoh untuk pelatihan
├── model/
│   └── house_price_model.joblib  # Model Linear Regression yang disimpan
├── assets/
│   └── bg_image.jpg           # Gambar latar (opsional)
└── requirements.txt           # Daftar dependensi proyek
```

---

## 🤖 Cara Kerja Model
1. **Persiapan Data**:
   - Membaca dataset dari `house_data.csv`
   - Memisahkan fitur (luas, kamar, lokasi) dan target (harga)

2. **Pelatihan Model**:
   - Membagi data: **80% training**, **20% testing**
   - Melatih model **Linear Regression** menggunakan `scikit-learn`
   - Menghitung akurasi dengan **R² Score**

3. **Prediksi**:
   - Menerima input pengguna melalui GUI
   - Menghasilkan prediksi harga berdasarkan model
   - Menampilkan hasil dalam format yang mudah dipahami

---

## 📝 Contoh Dataset
**File**: `house_data.csv`
```csv
luas,kamar,lokasi,harga
75,3,1,800
90,4,1,950
60,2,2,500
...
```

**Penjelasan Kolom**:
- `luas`: Luas bangunan dalam meter persegi (m²)
- `kamar`: Jumlah kamar tidur
- `lokasi`: Kode lokasi (1 = Pusat Kota, 2 = Suburban, 3 = Pinggiran)
- `harga`: Harga rumah dalam juta Rupiah

---

## 📜 Lisensi
Proyek ini dilisensikan di bawah **[MIT License](LICENSE)**. Silakan gunakan, modifikasi, dan distribusikan sesuai kebutuhan Anda.

---

## 🌈 Kontribusi
Kami sangat menyambut kontribusi! Ingin membantu? Berikut caranya:
1. 🍴 **Fork** repository ini
2. 🛠️ Buat branch baru: `git checkout -b fitur-baru`
3. 💻 Lakukan perubahan dan commit: `git commit -m "Menambahkan fitur baru"`
4. 🚀 Push ke branch: `git push origin fitur-baru`
5. 📬 Buat **Pull Request** di GitHub

Ada bug atau saran? Silakan buka **[issue](https://github.com/username/house-price-predictor/issues)**.

---

## 📢 Hubungi Kami
Ikuti perkembangan proyek ini di [GitHub](https://github.com/username/house-price-predictor) atau hubungi kami melalui [email](mailto:your.email@example.com). Kami senang mendengar masukan Anda!

⭐ **Jangan lupa beri bintang di GitHub jika Anda menyukai proyek ini!**