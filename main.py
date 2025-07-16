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
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import plotly.express as px
import plotly.io as pio
import threading
import time

# Konfigurasi GUI
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.data_path = Path("house_data.csv")
        self.model_path = Path("model/house_price_model.joblib")
        self.history = []
        self.load_data()
        self.train_model()

    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
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
        if not self.model_path.exists():
            self.model_path.parent.mkdir(exist_ok=True)
            dump(self.model, self.model_path)

    def predict_price(self, luas, kamar, lokasi):
        try:
            input_data = np.array([[luas, kamar, lokasi]])
            prediction = self.model.predict(input_data)[0]
            # Simulasi confidence interval (sederhana)
            std_error = np.std(self.y - self.model.predict(self.X)) / np.sqrt(len(self.y))
            ci_lower = prediction - 1.96 * std_error
            ci_upper = prediction + 1.96 * std_error
            return prediction, ci_lower, ci_upper
        except Exception as e:
            print(f"Error: {str(e)}")
            return None, None, None

    def get_statistics(self):
        return {
            "rata_rata_harga": self.data['harga'].mean(),
            "min_harga": self.data['harga'].min(),
            "max_harga": self.data['harga'].max(),
            "jumlah_data": len(self.data)
        }

    def save_prediction(self, luas, kamar, lokasi, harga):
        self.history.append({"luas": luas, "kamar": kamar, "lokasi": lokasi, "harga": harga})
        # Simpan ke CSV
        history_df = pd.DataFrame(self.history)
        history_df.to_csv("prediction_history.csv", index=False)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.predictor = HousePricePredictor()
        self.setup_ui()

    def setup_ui(self):
        self.title("🏡 Prediksi Harga Rumah Modern")
        self.geometry("1000x700")
        self.resizable(True, True)

        # Background Image
        try:
            bg_image = ctk.CTkImage(Image.open("assets/bg_image.jpg"), size=(1000, 700))
            bg_label = ctk.CTkLabel(self, image=bg_image, text="")
            bg_label.place(relwidth=1, relheight=1)
        except:
            pass

        # Tab View
        self.tab_view = ctk.CTkTabview(self, fg_color="#2B2D42", segmented_button_selected_color="#4CCD99")
        self.tab_view.pack(pady=20, padx=20, fill="both", expand=True)
        self.tab_view.add("Prediksi")
        self.tab_view.add("Riwayat")
        self.tab_view.add("Statistik")

        # Tab 1: Prediksi
        self.setup_prediction_tab()

        # Tab 2: Riwayat
        self.setup_history_tab()

        # Tab 3: Statistik
        self.setup_statistics_tab()

        # Theme Toggle
        self.theme_button = ctk.CTkButton(
            self, text="🌙 Toggle Tema", command=self.toggle_theme, width=150, corner_radius=10,
            fg_color="#4CCD99", hover_color="#3A9B7A"
        )
        self.theme_button.pack(pady=10)

    def setup_prediction_tab(self):
        tab = self.tab_view.tab("Prediksi")
        main_frame = ctk.CTkFrame(tab, fg_color="transparent")
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Header
        ctk.CTkLabel(
            main_frame, text="🏠 Prediksi Harga Rumah", font=("Arial", 28, "bold"), text_color="#4CCD99"
        ).pack(pady=10)

        # Input Form
        input_frame = ctk.CTkFrame(main_frame, fg_color="#3A3B5B", corner_radius=15)
        input_frame.pack(pady=20, padx=20, fill="x")

        # Input Luas
        ctk.CTkLabel(input_frame, text="Luas Bangunan (m²):", font=("Arial", 14)).pack(pady=5)
        self.luas_entry = ctk.CTkEntry(input_frame, placeholder_text="20-500", font=("Arial", 12))
        self.luas_entry.pack(pady=5, padx=20, fill="x")
        self.create_tooltip(self.luas_entry, "Masukkan luas bangunan antara 20-500 m²")

        # Input Kamar
        ctk.CTkLabel(input_frame, text="Jumlah Kamar:", font=("Arial", 14)).pack(pady=5)
        self.kamar_entry = ctk.CTkEntry(input_frame, placeholder_text="1-10", font=("Arial", 12))
        self.kamar_entry.pack(pady=5, padx=20, fill="x")
        self.create_tooltip(self.kamar_entry, "Masukkan jumlah kamar antara 1-10")

        # Input Lokasi
        ctk.CTkLabel(input_frame, text="Lokasi:", font=("Arial", 14)).pack(pady=5)
        self.lokasi_option = ctk.CTkOptionMenu(
            input_frame, values=["1 - Pusat Kota", "2 - Suburban", "3 - Pinggiran"], font=("Arial", 12)
        )
        self.lokasi_option.pack(pady=5, padx=20, fill="x")
        self.create_tooltip(self.lokasi_option, "Pilih lokasi rumah")

        # Button Frame
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(pady=20)

        self.predict_button = ctk.CTkButton(
            button_frame, text="🔍 Prediksi Harga", command=self.predict, width=200, height=40,
            font=("Arial", 14, "bold"), fg_color="#4CCD99", hover_color="#3A9B7A"
        )
        self.predict_button.pack(side="left", padx=10)

        ctk.CTkButton(
            button_frame, text="📊 Visualisasi", command=self.plot_data, width=200, height=40,
            font=("Arial", 14, "bold"), fg_color="#3498DB", hover_color="#2A78B0"
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            button_frame, text="💾 Ekspor Riwayat", command=self.export_history, width=200, height=40,
            font=("Arial", 14, "bold"), fg_color="#E74C3C", hover_color="#C0392B"
        ).pack(side="left", padx=10)

        # Result Display
        self.result_label = ctk.CTkLabel(
            main_frame, text="Masukkan data untuk prediksi...", font=("Arial", 16, "bold"), text_color="#4CCD99"
        )
        self.result_label.pack(pady=20)

        # Model Accuracy
        ctk.CTkLabel(
            main_frame, text=f"🏆 Akurasi Model: {self.predictor.r2_score:.2f} (R² Score)",
            font=("Arial", 14), text_color="gray"
        ).pack(pady=5)

    def setup_history_tab(self):
        tab = self.tab_view.tab("Riwayat")
        history_frame = ctk.CTkFrame(tab, fg_color="transparent")
        history_frame.pack(pady=20, padx=20, fill="both", expand=True)

        ctk.CTkLabel(
            history_frame, text="📜 Riwayat Prediksi", font=("Arial", 24, "bold"), text_color="#4CCD99"
        ).pack(pady=10)

        # Table for history
        self.tree = ttk.Treeview(
            history_frame, columns=("Luas", "Kamar", "Lokasi", "Harga"), show="headings", height=10
        )
        self.tree.heading("Luas", text="Luas (m²)")
        self.tree.heading("Kamar", text="Kamar")
        self.tree.heading("Lokasi", text="Lokasi")
        self.tree.heading("Harga", text="Harga (juta)")
        self.tree.column("Luas", width=100, anchor="center")
        self.tree.column("Kamar", width=100, anchor="center")
        self.tree.column("Lokasi", width=150, anchor="center")
        self.tree.column("Harga", width=150, anchor="center")
        self.tree.pack(pady=10, fill="both", expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)

    def setup_statistics_tab(self):
        tab = self.tab_view.tab("Statistik")
        stats_frame = ctk.CTkFrame(tab, fg_color="transparent")
        stats_frame.pack(pady=20, padx=20, fill="both", expand=True)

        ctk.CTkLabel(
            stats_frame, text="📊 Statistik Dataset", font=("Arial", 24, "bold"), text_color="#4CCD99"
        ).pack(pady=10)

        stats = self.predictor.get_statistics()
        ctk.CTkLabel(
            stats_frame, text=f"📈 Rata-rata Harga: Rp {stats['rata_rata_harga']:,.2f} juta",
            font=("Arial", 16)
        ).pack(pady=5)
        ctk.CTkLabel(
            stats_frame, text=f"⬇️ Harga Terendah: Rp {stats['min_harga']:,.2f} juta",
            font=("Arial", 16)
        ).pack(pady=5)
        ctk.CTkLabel(
            stats_frame, text=f"⬆️ Harga Tertinggi: Rp {stats['max_harga']:,.2f} juta",
            font=("Arial", 16)
        ).pack(pady=5)
        ctk.CTkLabel(
            stats_frame, text=f"📑 Jumlah Data: {stats['jumlah_data']} rumah",
            font=("Arial", 16)
        ).pack(pady=5)

    def create_tooltip(self, widget, text):
        tooltip = ctk.CTkLabel(self, text=text, font=("Arial", 10), fg_color="#2B2D42", text_color="white")
        tooltip.place_forget()

        def show_tooltip(event):
            tooltip.place(x=widget.winfo_rootx() + 20, y=widget.winfo_rooty() + 30)

        def hide_tooltip(event):
            tooltip.place_forget()

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

    def toggle_theme(self):
        current_mode = ctk.get_appearance_mode()
        new_mode = "Light" if current_mode == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)
        self.theme_button.configure(text=f"{'🌞' if new_mode == 'Light' else '🌙'} Toggle Tema")

    def predict(self):
        self.predict_button.configure(state="disabled")
        self.result_label.configure(text="⏳ Sedang memproses...", text_color="#FFC107")
        self.update()

        def process_prediction():
            try:
                luas = float(self.luas_entry.get())
                kamar = int(self.kamar_entry.get())
                lokasi = int(self.lokasi_option.get().split()[0])

                if not (20 <= luas <= 500) or not (1 <= kamar <= 10) or lokasi not in [1, 2, 3]:
                    self.result_label.configure(text="❌ Input tidak valid! (Luas: 20-500, Kamar: 1-10)", text_color="#FF5733")
                    return

                harga, ci_lower, ci_upper = self.predictor.predict_price(luas, kamar, lokasi)
                if harga is not None:
                    self.result_label.configure(
                        text=f"💰 Harga: Rp {harga:,.2f} juta\n(CI: Rp {ci_lower:,.2f} - {ci_upper:,.2f} juta)",
                        text_color="#4CCD99"
                    )
                    self.predictor.save_prediction(luas, kamar, lokasi, harga)
                    self.update_history_table()
                else:
                    self.result_label.configure(text="⚠️ Gagal melakukan prediksi!", text_color="#FF5733")
            except ValueError:
                self.result_label.configure(text="⚠️ Masukkan angka yang valid!", text_color="#FFC107")
            finally:
                self.predict_button.configure(state="normal")

        threading.Thread(target=process_prediction, daemon=True).start()

    def update_history_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for entry in self.predictor.history:
            lokasi_text = {1: "Pusat Kota", 2: "Suburban", 3: "Pinggiran"}.get(entry["lokasi"], "Unknown")
            self.tree.insert("", "end", values=(
                f"{entry['luas']:.1f}", entry["kamar"], lokasi_text, f"{entry['harga']:,.2f}"
            ))

    def export_history(self):
        try:
            pd.DataFrame(self.predictor.history).to_csv("prediction_history_export.csv", index=False)
            self.result_label.configure(text="✅ Riwayat berhasil diekspor!", text_color="#4CCD99")
        except Exception as e:
            self.result_label.configure(text=f"❌ Gagal mengekspor: {str(e)}", text_color="#FF5733")

    def plot_data(self):
        try:
            import plotly.express as px
            fig = px.scatter(
                self.predictor.data, x="luas", y="harga", color="lokasi",
                title="Hubungan Luas Bangunan vs Harga Rumah",
                labels={"luas": "Luas Bangunan (m²)", "harga": "Harga (juta Rupiah)", "lokasi": "Lokasi"}
            )
            fig.update_layout(showlegend=True, template="plotly_dark")
            fig.show()
        except ImportError:
            plt.figure(figsize=(10, 6))
            plt.scatter(
                self.predictor.data['luas'], self.predictor.data['harga'],
                c=self.predictor.data['lokasi'], cmap='viridis', label='Data Asli'
            )
            plt.title('Hubungan Luas Bangunan vs Harga Rumah', fontsize=14)
            plt.xlabel('Luas Bangunan (m²)', fontsize=12)
            plt.ylabel('Harga (juta Rupiah)', fontsize=12)
            plt.colorbar(label='Lokasi')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    app = App()
    app.mainloop()