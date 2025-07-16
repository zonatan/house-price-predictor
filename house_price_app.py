import numpy as np
import pandas as pd
import customtkinter as ctk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from joblib import dump, load
import matplotlib.pyplot as plt
from PIL import Image
import os

# Konfigurasi GUI
ctk.set_appearance_mode("dark")  # Mode: "dark", "light", "system"
ctk.set_default_color_theme("blue")  # Tema: "blue", "green", "dark-blue"

class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.load_data()
        self.train_model()
    
    def load_data(self):
        try:
            self.data = pd.read_csv('house_data.csv')
            self.X = self.data[['luas', 'kamar', 'lokasi']]
            self.y = self.data['harga']
        except FileNotFoundError:
            print("Error: File 'house_data.csv' tidak ditemukan!")
            exit()
    
    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.r2_score = r2_score(y_test, y_pred)
        
        # Simpan model jika belum ada
        if not os.path.exists('model/house_price_model.joblib'):
            os.makedirs('model', exist_ok=True)
            dump(self.model, 'model/house_price_model.joblib')
    
    def predict_price(self, luas, kamar, lokasi):
        try:
            input_data = np.array([[luas, kamar, lokasi]])
            return self.model.predict(input_data)[0]
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.predictor = HousePricePredictor()
        self.setup_ui()
    
    def setup_ui(self):
        self.title("🏠 Prediksi Harga Rumah")
        self.geometry("800x600")
        self.resizable(False, False)
        
        # Background Image (Opsional)
        try:
            bg_image = ctk.CTkImage(Image.open("assets/bg_image.jpg"), size=(800, 600))
            bg_label = ctk.CTkLabel(self, image=bg_image, text="")
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except:
            pass
        
        # Main Frame
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(pady=40, padx=40, fill="both", expand=True)
        
        # Header
        header = ctk.CTkLabel(
            main_frame,
            text="PREDIKSI HARGA RUMAH",
            font=("Arial", 24, "bold")
        )
        header.pack(pady=10)
        
        # Input Form
        input_frame = ctk.CTkFrame(main_frame)
        input_frame.pack(pady=20, padx=20, fill="x")
        
        # Input Luas
        ctk.CTkLabel(input_frame, text="Luas Bangunan (m²):").pack(pady=5)
        self.luas_entry = ctk.CTkEntry(input_frame, placeholder_text="Contoh: 75")
        self.luas_entry.pack(pady=5, padx=10, fill="x")
        
        # Input Kamar
        ctk.CTkLabel(input_frame, text="Jumlah Kamar:").pack(pady=5)
        self.kamar_entry = ctk.CTkEntry(input_frame, placeholder_text="Contoh: 3")
        self.kamar_entry.pack(pady=5, padx=10, fill="x")
        
        # Input Lokasi
        ctk.CTkLabel(input_frame, text="Lokasi (1=Pusat Kota, 2=Suburban, 3=Pinggiran):").pack(pady=5)
        self.lokasi_entry = ctk.CTkEntry(input_frame, placeholder_text="1/2/3")
        self.lokasi_entry.pack(pady=5, padx=10, fill="x")
        
        # Button Frame
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Prediksi Harga",
            command=self.predict,
            width=150,
            height=40,
            font=("Arial", 14)
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            button_frame,
            text="Lihat Grafik",
            command=self.plot_data,
            width=150,
            height=40,
            font=("Arial", 14)
        ).pack(side="left", padx=10)
        
        # Result Display
        self.result_label = ctk.CTkLabel(
            main_frame,
            text="Masukkan data rumah untuk prediksi...",
            font=("Arial", 16),
            text_color="#4CCD99"
        )
        self.result_label.pack(pady=20)
        
        # Model Accuracy
        ctk.CTkLabel(
            main_frame,
            text=f"🏆 Akurasi Model: {self.predictor.r2_score:.2f} (R² Score)",
            font=("Arial", 12),
            text_color="gray"
        ).pack(pady=5)
    
    def predict(self):
        try:
            luas = float(self.luas_entry.get())
            kamar = int(self.kamar_entry.get())
            lokasi = int(self.lokasi_entry.get())
            
            if luas <= 0 or kamar <= 0 or lokasi not in [1, 2, 3]:
                self.result_label.configure(text="❌ Input tidak valid!", text_color="#FF5733")
                return
            
            harga = self.predictor.predict_price(luas, kamar, lokasi)
            self.result_label.configure(
                text=f"💰 Perkiraan Harga: Rp {harga:,.2f} juta",
                text_color="#4CCD99"
            )
        except ValueError:
            self.result_label.configure(text="⚠️ Masukkan angka yang valid!", text_color="#FFC107")
    
    def plot_data(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.predictor.data['luas'],
            self.predictor.data['harga'],
            color='#4CCD99',
            label='Data Asli'
        )
        plt.title('Hubungan Luas Bangunan vs Harga Rumah', fontsize=14)
        plt.xlabel('Luas Bangunan (m²)', fontsize=12)
        plt.ylabel('Harga (juta Rupiah)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = App()
    app.mainloop()