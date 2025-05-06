import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
from tensorflow.keras.models import load_model
from utils.dpf_logic import hitung_dpf
from utils.bmi import hitung_bmi


# --- Load CNN Model ---
model_path = os.path.join("model", "cnn_model.h5") 
model = load_model(model_path)

# --- Load Scalers ---
with open(os.path.join("model", "scaler_standard.pkl"), "rb") as f:
    scaler_standard = pickle.load(f)

with open(os.path.join("model", "scaler_minmax.pkl"), "rb") as f:
    scaler_minmax = pickle.load(f)

st.title("Prediksi Diabetes dengan CNN")

default_values = {
    "Glucose": 99,
    "BloodPressure": 74,
    "SkinThickness": 27.24,
    "Insulin": 102.50
}

st.markdown("Data Diri")
age = st.number_input("Usia", min_value=0, max_value=120, step=1)
jenis_kelamin = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"]
                         , horizontal=True)

if jenis_kelamin != "Perempuan":
    pregnancies = 0
    st.number_input("Jumlah Kehamilan", value=pregnancies, disabled=True)
    st.info("Jumlah kehamilan hanya relevan untuk perempuan, " \
    "jadi input ini dinonaktifkan dan diset ke 0.")
else:
    pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, 
                                  max_value=20, step=1)

checkup = st.checkbox("Apakah sebelumnya pernah checkup kesehatan?")

if not checkup:
    st.number_input(f"**Glukosa**", value = default_values['Glucose'],  disabled=True)
    st.number_input(f"**Tekanan Darah**", default_values['BloodPressure'] ,disabled=True)
    st.number_input(f"**Tebal Kulit**", default_values['SkinThickness'] ,disabled=True)
    st.number_input(f"**Insulin**", default_values['Insulin'] ,disabled=True)
else:
    glucose = st.number_input("Glukosa", min_value=0, max_value=200, step=1)
    blood_pressure = st.number_input("Tekanan Darah", min_value=0, max_value=150, step=1)
    skin_thickness = st.number_input("Tebal Kulit", min_value=0, max_value=100, step=1)
    insulin = st.number_input("Insulin", min_value=0, max_value=1000, step=1)

berat = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=200.0, step=0.1)
tinggi = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=250.0, step=0.1)

try:
    bmi = hitung_bmi(berat, tinggi)
except ValueError as e:
    st.error(str(e))
    bmi = 0.0

#DPF
riwayat_orangtua = st.radio("Riwayat Diabetes pada Orang Tua", ["Kedua", "Salah satu", "Tidak ada"])
riwayat_kakek = st.checkbox("Riwayat Diabetes pada Kakek/Nenek")

# Prediksi
if st.button("Prediksi"):
    # Hitung DPF berdasarkan logika domain
    dpf = hitung_dpf(riwayat_orangtua, jenis_kelamin, riwayat_kakek)

    # Skala input sesuai dengan jenis scalert
    fitur_standard = np.array([[pregnancies, insulin, dpf, age]])
    fitur_minmax = np.array([[blood_pressure, skin_thickness, glucose, bmi]])

    scaled_standard = scaler_standard.transform(fitur_standard)
    scaled_minmax = scaler_minmax.transform(fitur_minmax)

    # Gabungkan fitur sesuai urutan input model
    input_data = np.concatenate([
        scaled_standard[:, [0]],  # Pregnancies
        scaled_minmax[:, [2]],    # Glucose
        scaled_minmax[:, [0]],    # BloodPressure
        scaled_minmax[:, [1]],    # SkinThickness
        scaled_standard[:, [1]],  # Insulin
        scaled_minmax[:, [3]],    # BMI
        scaled_standard[:, [2]],  # DPF
        scaled_standard[:, [3]]   # Age
    ], axis=1)

    # Ubah bentuk input untuk CNN
    input_reshaped = input_data.reshape((1, 8, 1))

    # Prediksi
    prediction = model.predict(input_reshaped)
    label = (prediction[0][0] > 0.5).astype(int)
    hasil = "POSITIF Diabetes" if label == 1 else "NEGATIF Diabetes"

    st.success(f"Hasil Prediksi: {hasil} (Probabilitas: {prediction[0][0]:.2f})")
