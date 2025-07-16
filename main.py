import numpy as np
import pandas as pd
import customtkinter as ctk
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from joblib import dump, load
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import plotly.express as px
import plotly.graph_objects as go
import threading
import logging
import csv
import tkintermapview
from datetime import datetime
from fpdf import FPDF

# Konfigurasi Logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Konfigurasi GUI
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class HousePricePredictor:
    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.data_path = Path("house_data.csv")
        self.model_path = Path("model")
        self.history = []
        self.language = "id"  # Default: Bahasa Indonesia
        self.load_data()
        self.train_models()

    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
            self.X = self.data[['luas', 'kamar', 'lokasi']]
            self.y = self.data['harga']
            logging.info("Dataset berhasil dimuat.")
        except FileNotFoundError:
            logging.error("File house_data.csv tidak ditemukan.")
            messagebox.showerror("Error", "File 'house_data.csv' tidak ditemukan!")
            exit()

    def train_models(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.r2_scores = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            self.r2_scores[name] = r2_score(y_test, y_pred)
            if not (self.model_path / f"{name.lower().replace(' ', '_')}_model.joblib").exists():
                self.model_path.mkdir(exist_ok=True)
                dump(model, self.model_path / f"{name.lower().replace(' ', '_')}_model.joblib")
            logging.info(f"Model {name} dilatih dengan R² Score: {self.r2_scores[name]:.2f}")

    def predict_price(self, luas, kamar, lokasi, model_name="Linear Regression"):
        try:
            input_data = np.array([[luas, kamar, lokasi]])
            model = self.models[model_name]
            prediction = model.predict(input_data)[0]
            std_error = np.std(self.y - model.predict(self.X)) / np.sqrt(len(self.y))
            ci_lower = prediction - 1.96 * std_error
            ci_upper = prediction + 1.96 * std_error
            return prediction, ci_lower, ci_upper
        except Exception as e:
            logging.error(f"Error prediksi: {str(e)}")
            return None, None, None

    def get_statistics(self):
        return {
            "rata_rata_harga": self.data['harga'].mean(),
            "min_harga": self.data['harga'].min(),
            "max_harga": self.data['harga'].max(),
            "jumlah_data": len(self.data)
        }

    def save_prediction(self, luas, kamar, lokasi, harga, model_name):
        self.history.append({
            "luas": luas,
            "kamar": kamar,
            "lokasi": lokasi,
            "harga": harga,
            "model": model_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        history_df = pd.DataFrame(self.history)
        history_df.to_csv("prediction_history.csv", index=False)
        logging.info(f"Prediksi disimpan: {luas}, {kamar}, {lokasi}, {harga}")

    def batch_predict(self, file_path):
        try:
            df = pd.read_csv(file_path)
            required_columns = ['luas', 'kamar', 'lokasi']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("File CSV harus memiliki kolom: luas, kamar, lokasi")
            predictions = []
            for _, row in df.iterrows():
                harga, _, _ = self.predict_price(row['luas'], row['kamar'], row['lokasi'])
                predictions.append({"luas": row['luas'], "kamar": row['kamar'], "lokasi": row['lokasi'], "harga": harga})
            pd.DataFrame(predictions).to_csv("batch_predictions.csv", index=False)
            logging.info("Batch prediction berhasil.")
            return True
        except Exception as e:
            logging.error(f"Error batch prediction: {str(e)}")
            return False

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.predictor = HousePricePredictor()
        self.language = "id"
        self.texts = {
            "id": {
                "title": "🏡 Prediksi Harga Rumah Premium",
                "predict_tab": "Prediksi",
                "history_tab": "Riwayat",
                "stats_tab": "Statistik",
                "dashboard_tab": "Dashboard",
                "predict_button": "🔍 Prediksi Harga",
                "visualize_button": "📊 Visualisasi",
                "export_button": "💾 Ekspor Riwayat",
                "batch_button": "📑 Prediksi Batch",
                "map_button": "🗺️ Peta Lokasi",
                "report_button": "📄 Buat Laporan PDF",
                "welcome": "Selamat datang! Masukkan data rumah untuk memulai.",
                "invalid_input": "❌ Input tidak valid! (Luas: 20-500, Kamar: 1-10)",
                "invalid_number": "⚠️ Masukkan angka yang valid!",
                "processing": "⏳ Sedang memproses...",
                "success_export": "✅ Riwayat berhasil diekspor!",
                "error_export": "❌ Gagal mengekspor: {}",
                "success_batch": "✅ Batch prediction berhasil!",
                "error_batch": "❌ Gagal batch prediction: {}",
                "tooltip_luas": "Masukkan luas bangunan antara 20-500 m²",
                "tooltip_kamar": "Masukkan jumlah kamar antara 1-10",
                "tooltip_lokasi": "Pilih lokasi rumah",
                "tooltip_model": "Pilih model machine learning"
            },
            "en": {
                "title": "🏡 Premium House Price Predictor",
                "predict_tab": "Prediction",
                "history_tab": "History",
                "stats_tab": "Statistics",
                "dashboard_tab": "Dashboard",
                "predict_button": "🔍 Predict Price",
                "visualize_button": "📊 Visualize",
                "export_button": "💾 Export History",
                "batch_button": "📑 Batch Prediction",
                "map_button": "🗺️ Location Map",
                "report_button": "📄 Generate PDF Report",
                "welcome": "Welcome! Enter house data to start.",
                "invalid_input": "❌ Invalid input! (Area: 20-500, Rooms: 1-10)",
                "invalid_number": "⚠️ Enter valid numbers!",
                "processing": "⏳ Processing...",
                "success_export": "✅ History exported successfully!",
                "error_export": "❌ Failed to export: {}",
                "success_batch": "✅ Batch prediction successful!",
                "error_batch": "❌ Batch prediction failed: {}",
                "tooltip_luas": "Enter building area between 20-500 m²",
                "tooltip_kamar": "Enter number of rooms between 1-10",
                "tooltip_lokasi": "Select house location",
                "tooltip_model": "Select machine learning model"
            }
        }
        self.setup_ui()

    def setup_ui(self):
        self.title(self.texts[self.language]["title"])
        self.geometry("1200x800")
        self.resizable(True, True)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        ctk.CTkLabel(self.sidebar, text="🏠 Menu", font=("Arial", 20, "bold")).pack(pady=20)
        ctk.CTkButton(
            self.sidebar, text="🌐 Toggle Bahasa", command=self.toggle_language, fg_color="#4CCD99", hover_color="#3A9B7A"
        ).pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(
            self.sidebar, text="🌙 Toggle Tema", command=self.toggle_theme, fg_color="#4CCD99", hover_color="#3A9B7A"
        ).pack(pady=10, padx=20, fill="x")

        # Main Content
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(side="right", fill="both", expand=True)

        # Tab View
        self.tab_view = ctk.CTkTabview(self.main_frame, fg_color="#2B2D42", segmented_button_selected_color="#4CCD99")
        self.tab_view.pack(pady=20, padx=20, fill="both", expand=True)
        self.tab_view.add(self.texts[self.language]["predict_tab"])
        self.tab_view.add(self.texts[self.language]["history_tab"])
        self.tab_view.add(self.texts[self.language]["stats_tab"])
        self.tab_view.add(self.texts[self.language]["dashboard_tab"])

        # Setup Tabs
        self.setup_prediction_tab()
        self.setup_history_tab()
        self.setup_statistics_tab()
        self.setup_dashboard_tab()

        # Welcome Message
        self.welcome_label = ctk.CTkLabel(
            self.main_frame, text=self.texts[self.language]["welcome"], font=("Arial", 16), text_color="#4CCD99"
        )
        self.welcome_label.pack(pady=10)

    def setup_prediction_tab(self):
        tab = self.tab_view.tab(self.texts[self.language]["predict_tab"])
        input_frame = ctk.CTkFrame(tab, fg_color="#3A3B5B", corner_radius=15)
        input_frame.pack(pady=20, padx=20, fill="x")

        ctk.CTkLabel(input_frame, text="Luas Bangunan (m²):", font=("Arial", 14)).pack(pady=5)
        self.luas_entry = ctk.CTkEntry(input_frame, placeholder_text="20-500", font=("Arial", 12))
        self.luas_entry.pack(pady=5, padx=20, fill="x")
        self.create_tooltip(self.luas_entry, self.texts[self.language]["tooltip_luas"])

        ctk.CTkLabel(input_frame, text="Jumlah Kamar:", font=("Arial", 14)).pack(pady=5)
        self.kamar_entry = ctk.CTkEntry(input_frame, placeholder_text="1-10", font=("Arial", 12))
        self.kamar_entry.pack(pady=5, padx=20, fill="x")
        self.create_tooltip(self.kamar_entry, self.texts[self.language]["tooltip_kamar"])

        ctk.CTkLabel(input_frame, text="Lokasi:", font=("Arial", 14)).pack(pady=5)
        self.lokasi_option = ctk.CTkOptionMenu(
            input_frame, values=["1 - Pusat Kota", "2 - Suburban", "3 - Pinggiran"], font=("Arial", 12)
        )
        self.lokasi_option.pack(pady=5, padx=20, fill="x")
        self.create_tooltip(self.lokasi_option, self.texts[self.language]["tooltip_lokasi"])

        ctk.CTkLabel(input_frame, text="Model:", font=("Arial", 14)).pack(pady=5)
        self.model_option = ctk.CTkOptionMenu(
            input_frame, values=["Linear Regression", "Random Forest"], font=("Arial", 12)
        )
        self.model_option.pack(pady=5, padx=20, fill="x")
        self.create_tooltip(self.model_option, self.texts[self.language]["tooltip_model"])

        button_frame = ctk.CTkFrame(tab, fg_color="transparent")
        button_frame.pack(pady=20)
        self.predict_button = ctk.CTkButton(
            button_frame, text=self.texts[self.language]["predict_button"], command=self.predict, width=200, height=40,
            font=("Arial", 14, "bold"), fg_color="#4CCD99", hover_color="#3A9B7A"
        )
        self.predict_button.pack(side="left", padx=10)
        ctk.CTkButton(
            button_frame, text=self.texts[self.language]["visualize_button"], command=self.plot_data, width=200, height=40,
            font=("Arial", 14, "bold"), fg_color="#3498DB", hover_color="#2A78B0"
        ).pack(side="left", padx=10)
        ctk.CTkButton(
            button_frame, text=self.texts[self.language]["batch_button"], command=self.batch_predict, width=200, height=40,
            font=("Arial", 14, "bold"), fg_color="#E74C3C", hover_color="#C0392B"
        ).pack(side="left", padx=10)
        ctk.CTkButton(
            button_frame, text=self.texts[self.language]["map_button"], command=self.show_map, width=200, height=40,
            font=("Arial", 14, "bold"), fg_color="#F1C40F", hover_color="#D4AC0D"
        ).pack(side="left", padx=10)
        ctk.CTkButton(
            button_frame, text=self.texts[self.language]["report_button"], command=self.generate_pdf_report, width=200, height=40,
            font=("Arial", 14, "bold"), fg_color="#9B59B6", hover_color="#7D3C98"
        ).pack(side="left", padx=10)

        self.result_label = ctk.CTkLabel(
            tab, text="", font=("Arial", 16, "bold"), text_color="#4CCD99"
        )
        self.result_label.pack(pady=20)

    def setup_history_tab(self):
        tab = self.tab_view.tab(self.texts[self.language]["history_tab"])
        history_frame = ctk.CTkFrame(tab, fg_color="transparent")
        history_frame.pack(pady=20, padx=20, fill="both", expand=True)

        ctk.CTkLabel(
            history_frame, text="📜 Riwayat Prediksi", font=("Arial", 24, "bold"), text_color="#4CCD99"
        ).pack(pady=10)

        self.tree = ttk.Treeview(
            history_frame, columns=("Timestamp", "Luas", "Kamar", "Lokasi", "Harga", "Model"), show="headings", height=15
        )
        self.tree.heading("Timestamp", text="Waktu")
        self.tree.heading("Luas", text="Luas (m²)")
        self.tree.heading("Kamar", text="Kamar")
        self.tree.heading("Lokasi", text="Lokasi")
        self.tree.heading("Harga", text="Harga (juta)")
        self.tree.heading("Model", text="Model")
        self.tree.column("Timestamp", width=150, anchor="center")
        self.tree.column("Luas", width=100, anchor="center")
        self.tree.column("Kamar", width=100, anchor="center")
        self.tree.column("Lokasi", width=150, anchor="center")
        self.tree.column("Harga", width=150, anchor="center")
        self.tree.column("Model", width=150, anchor="center")
        self.tree.pack(pady=10, fill="both", expand=True)

        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)

        ctk.CTkButton(
            history_frame, text=self.texts[self.language]["export_button"], command=self.export_history, width=200, height=40,
            font=("Arial", 14, "bold"), fg_color="#E74C3C", hover_color="#C0392B"
        ).pack(pady=10)

    def setup_statistics_tab(self):
        tab = self.tab_view.tab(self.texts[self.language]["stats_tab"])
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

        ctk.CTkLabel(
            stats_frame, text="📊 Perbandingan Model", font=("Arial", 20, "bold"), text_color="#4CCD99"
        ).pack(pady=10)
        for name, score in self.predictor.r2_scores.items():
            ctk.CTkLabel(
                stats_frame, text=f"{name}: R² Score = {score:.2f}", font=("Arial", 16)
            ).pack(pady=5)

    def setup_dashboard_tab(self):
        tab = self.tab_view.tab(self.texts[self.language]["dashboard_tab"])
        dashboard_frame = ctk.CTkFrame(tab, fg_color="transparent")
        dashboard_frame.pack(pady=20, padx=20, fill="both", expand=True)

        ctk.CTkLabel(
            dashboard_frame, text="📈 Dashboard", font=("Arial", 24, "bold"), text_color="#4CCD99"
        ).pack(pady=10)

        stats = self.predictor.get_statistics()
        card_frame = ctk.CTkFrame(dashboard_frame, fg_color="#3A3B5B", corner_radius=15)
        card_frame.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(
            card_frame, text=f"🏠 Total Rumah: {stats['jumlah_data']}", font=("Arial", 16)
        ).pack(pady=5)
        ctk.CTkLabel(
            card_frame, text=f"💰 Rata-rata Harga: Rp {stats['rata_rata_harga']:,.2f} juta", font=("Arial", 16)
        ).pack(pady=5)

        ctk.CTkButton(
            dashboard_frame, text="📊 Analisis Sensitivitas", command=self.sensitivity_analysis, width=200, height=40,
            font=("Arial", 14, "bold"), fg_color="#3498DB", hover_color="#2A78B0"
        ).pack(pady=10)

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

    def toggle_language(self):
        self.language = "en" if self.language == "id" else "id"
        self.title(self.texts[self.language]["title"])
        self.tab_view.set(self.texts[self.language]["predict_tab"])
        self.tab_view.tab(self.texts[self.language]["predict_tab"]).configure(text=self.texts[self.language]["predict_tab"])
        self.tab_view.tab(self.texts[self.language]["history_tab"]).configure(text=self.texts[self.language]["history_tab"])
        self.tab_view.tab(self.texts[self.language]["stats_tab"]).configure(text=self.texts[self.language]["stats_tab"])
        self.tab_view.tab(self.texts[self.language]["dashboard_tab"]).configure(text=self.texts[self.language]["dashboard_tab"])
        self.predict_button.configure(text=self.texts[self.language]["predict_button"])
        self.welcome_label.configure(text=self.texts[self.language]["welcome"])
        self.setup_prediction_tab()
        self.setup_history_tab()
        self.setup_statistics_tab()
        self.setup_dashboard_tab()

    def predict(self):
        self.predict_button.configure(state="disabled")
        self.result_label.configure(text=self.texts[self.language]["processing"], text_color="#FFC107")
        self.update()

        def process_prediction():
            try:
                luas = float(self.luas_entry.get())
                kamar = int(self.kamar_entry.get())
                lokasi = int(self.lokasi_option.get().split()[0])
                model_name = self.model_option.get()

                if not (20 <= luas <= 500) or not (1 <= kamar <= 10) or lokasi not in [1, 2, 3]:
                    self.result_label.configure(text=self.texts[self.language]["invalid_input"], text_color="#FF5733")
                    messagebox.showerror("Error", self.texts[self.language]["invalid_input"])
                    return

                harga, ci_lower, ci_upper = self.predictor.predict_price(luas, kamar, lokasi, model_name)
                if harga is not None:
                    stats = self.predictor.get_statistics()
                    recommendation = "Wajar" if abs(harga - stats['rata_rata_harga']) < stats['rata_rata_harga'] * 0.2 else "Tinggi/Rendah"
                    self.result_label.configure(
                        text=f"💰 Harga: Rp {harga:,.2f} juta\n(CI: Rp {ci_lower:,.2f} - {ci_upper:,.2f} juta)\n📊 Rekomendasi: {recommendation}",
                        text_color="#4CCD99"
                    )
                    self.predictor.save_prediction(luas, kamar, lokasi, harga, model_name)
                    self.update_history_table()
                    messagebox.showinfo("Sukses", f"Prediksi berhasil! Harga: Rp {harga:,.2f} juta")
                else:
                    self.result_label.configure(text="⚠️ Gagal melakukan prediksi!", text_color="#FF5733")
                    messagebox.showerror("Error", "Gagal melakukan prediksi!")
            except ValueError:
                self.result_label.configure(text=self.texts[self.language]["invalid_number"], text_color="#FFC107")
                messagebox.showerror("Error", self.texts[self.language]["invalid_number"])
            finally:
                self.predict_button.configure(state="normal")

        threading.Thread(target=process_prediction, daemon=True).start()

    def update_history_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for entry in self.predictor.history:
            lokasi_text = {1: "Pusat Kota", 2: "Suburban", 3: "Pinggiran"}.get(entry["lokasi"], "Unknown")
            self.tree.insert("", "end", values=(
                entry["timestamp"], f"{entry['luas']:.1f}", entry["kamar"], lokasi_text, f"{entry['harga']:,.2f}", entry["model"]
            ))

    def export_history(self):
        try:
            pd.DataFrame(self.predictor.history).to_csv("prediction_history_export.csv", index=False)
            self.result_label.configure(text=self.texts[self.language]["success_export"], text_color="#4CCD99")
            messagebox.showinfo("Sukses", self.texts[self.language]["success_export"])
        except Exception as e:
            self.result_label.configure(text=self.texts[self.language]["error_export"].format(str(e)), text_color="#FF5733")
            messagebox.showerror("Error", self.texts[self.language]["error_export"].format(str(e)))

    def batch_predict(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            if self.predictor.batch_predict(file_path):
                self.result_label.configure(text=self.texts[self.language]["success_batch"], text_color="#4CCD99")
                messagebox.showinfo("Sukses", self.texts[self.language]["success_batch"])
            else:
                self.result_label.configure(text=self.texts[self.language]["error_batch"].format("File tidak valid"), text_color="#FF5733")
                messagebox.showerror("Error", self.texts[self.language]["error_batch"].format("File tidak valid"))

    def show_map(self):
        try:
            map_window = ctk.CTkToplevel(self)
            map_window.title("Peta Lokasi")
            map_window.geometry("800x600")
            map_widget = tkintermapview.TkinterMapView(map_window, width=800, height=600)
            map_widget.pack(fill="both", expand=True)
            map_widget.set_position(-6.2088, 106.8456)  # Jakarta (contoh)
            map_widget.set_zoom(10)
            for entry in self.predictor.history:
                lokasi = {1: (-6.2088, 106.8456), 2: (-6.3000, 106.8000), 3: (-6.3500, 106.7500)}.get(entry["lokasi"], (-6.2088, 106.8456))
                map_widget.set_marker(lokasi[0], lokasi[1], text=f"Harga: Rp {entry['harga']:,.2f} juta")
        except ImportError:
            messagebox.showerror("Error", "Modul tkintermapview tidak ditemukan. Install dengan: pip install tkintermapview")

    def plot_data(self):
        try:
            import plotly.express as px
            fig = px.scatter_3d(
                self.predictor.data, x="luas", y="kamar", z="harga", color="lokasi",
                title="Hubungan Luas, Kamar, dan Harga Rumah",
                labels={"luas": "Luas (m²)", "kamar": "Kamar", "harga": "Harga (juta Rupiah)", "lokasi": "Lokasi"}
            )
            fig.update_layout(template="plotly_dark")
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

    def sensitivity_analysis(self):
        try:
            fig = go.Figure()
            luas_range = np.linspace(20, 500, 10)
            for luas in luas_range:
                prices = [self.predictor.predict_price(luas, 3, lokasi)[0] for lokasi in [1, 2, 3]]
                fig.add_trace(go.Scatter(x=[luas] * 3, y=prices, mode='markers+lines', name=f'Luas {luas:.0f} m²'))
            fig.update_layout(
                title="Analisis Sensitivitas: Luas vs Harga",
                xaxis_title="Luas Bangunan (m²)",
                yaxis_title="Harga (juta Rupiah)",
                template="plotly_dark"
            )
            fig.show()
        except ImportError:
            messagebox.showerror("Error", "Modul plotly tidak ditemukan. Install dengan: pip install plotly")

    def generate_pdf_report(self):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Laporan Prediksi Harga Rumah", ln=True, align="C")
            pdf.ln(10)
            pdf.cell(200, 10, txt="Riwayat Prediksi:", ln=True)
            for entry in self.predictor.history:
                pdf.cell(200, 10, txt=f"Waktu: {entry['timestamp']}, Luas: {entry['luas']} m², Kamar: {entry['kamar']}, "
                                      f"Lokasi: {entry['lokasi']}, Harga: Rp {entry['harga']:,.2f} juta, Model: {entry['model']}", ln=True)
            pdf.output("prediction_report.pdf")
            messagebox.showinfo("Sukses", "Laporan PDF berhasil dibuat: prediction_report.pdf")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membuat laporan PDF: {str(e)}")

if __name__ == "__main__":
    app = App()
    app.mainloop()