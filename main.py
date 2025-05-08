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

if "page" not in st.session_state:
    st.session_state.page = 1

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

# ---------------- Page 1 ----------------
if st.session_state.page == 1:
    st.title("🩺 Prediksi Risiko Diabetes")
    st.write("""
    Selamat datang di aplikasi prediksi risiko diabetes! Aplikasi ini menggunakan model deep learning **Convolutional Neural Network (CNN)** untuk membantu memperkirakan risiko diabetes berdasarkan data pribadi dan kesehatan Anda.

    **Ketentuan Penggunaan:**
    - Data yang Anda masukkan hanya digunakan untuk keperluan simulasi dan **tidak akan disimpan maupun dibagikan** kepada pihak lain.
    - Mohon isi data dengan sebenar-benarnya agar hasil prediksi lebih akurat.
    - Hasil prediksi ini **bukan diagnosis medis**, melainkan estimasi berbasis data yang bertujuan untuk mendukung **deteksi dini dan pencegahan** diabetes.

    Dataset yang digunakan adalah [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

    Apakah Anda sudah membaca dan memaham ketentuan di atas?
    """)


    ketentuan = st.checkbox(
    "Ya, saya sudah membaca dengan seksama dan setuju memberikan data pribadi dan data kesehatan saya", 
    value=st.session_state.get("ketentuan", False)
)
    st.session_state["ketentuan"] = ketentuan
    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])
    with col3:
        st.button(
        "Mulai", 
        on_click=next_page,
        disabled=not ketentuan 
    )

# ---------------- Page 2: Data Diri ----------------
elif st.session_state.page == 2:
    st.header("Data Diri")

    show_warning = False

    st.session_state.age = st.number_input("Usia", min_value=5, max_value=120, step=1, value=st.session_state.get("age"))
    st.session_state.jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], index=["Laki-laki", "Perempuan"].
                                                  index(st.session_state.get("jenis_kelamin", "Perempuan")))

    if st.session_state.jenis_kelamin != "Perempuan" or (
    st.session_state.age is not None and st.session_state.age <= 10):
        st.session_state.pregnancies = 0
        st.number_input("Berapa kali Anda pernah hamil?", value=0, disabled=True)

        if st.session_state.jenis_kelamin != "Perempuan":
            st.info("Jumlah kehamilan tidak relevan untuk laki-laki.")
        else:
            st.info("Jumlah kehamilan hanya relevan untuk perempuan berusia di atas 10 tahun.")
    else:
        st.session_state.pregnancies = st.number_input(
        "Berapa kali Anda pernah hamil?",
        min_value=0,
        max_value=20,
        step=1,
        value=st.session_state.get("pregnancies", 0)
    )



    error_flag = False
    if st.session_state.age is None or st.session_state.age == 0:
        st.warning("⚠️ Usia tidak boleh 0.")
        error_flag = True

    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])
    with col1:
        st.button("Kembali", on_click=prev_page)
    with col4:
        if st.button("Lanjut"):
            if not error_flag:
                next_page()
            else:
                st.error("Mohon lengkapi data yang wajib diisi terlebih dahulu.")

# ---------------- Page 3: Data Kesehatan ----------------
elif st.session_state.page == 3:
    st.header("Data Kesehatan")

    st.subheader("🩺 Riwayat Check-up Kesehatan")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Ya"):
            st.session_state.checkup = True

    with col2:
        if st.button("❌ Tidak"):
            st.session_state.checkup = False

    # Tampilkan status pilihan (opsional)
    if "checkup" in st.session_state:
        st.write(f"Anda memilih: {'Ya' if st.session_state.checkup else 'Tidak'}")
    
    if not st.session_state.checkup:
        st.info("Nilai default digunakan karena Anda belum pernah melakukan check-up. " \
        "Nilai ini diambil berdasarkan nilai modus dari data kesehatan individu yang tidak menderita diabetes.")
        st.session_state.glucose = 99
        st.session_state.blood_pressure = 70
        st.session_state.skin_thickness = 27.74
        st.session_state.insulin = 102.05

        st.number_input("Kadar Glukosa", value=st.session_state.glucose, disabled=True)
        st.number_input("Tekanan Darah Diastolik (mm/Hg)", value=st.session_state.blood_pressure, disabled=True)
        st.number_input("Ketebalan Lipatan Kulit Trisep (mm)", value=st.session_state.skin_thickness, disabled=True)
        st.number_input("Kadar Insulin (muU/ml)", value=st.session_state.insulin, disabled=True)

    else:
        st.session_state.glucose = st.number_input("Kadar Glukosa", 30, 200, step=1, value=st.session_state.get("glucose", "Input here"))
        st.session_state.blood_pressure = st.number_input("Tekanan Darah Diastolik (mm/Hg)", 40, 150, step=1, value=st.session_state.get("blood_pressure", "Input here"))
        st.session_state.skin_thickness = st.number_input("Ketebalan Lipatan Kulit Trisep (mm)", 3.0, 100.0, step=1.0, value=st.session_state.get("skin_thickness", "Input here"))
        st.session_state.insulin = st.number_input("Kadar Insulin (muU/ml)", 2.0, 1000.0, step=1.0, value=st.session_state.get("insulin", "Input here"))

    # Validasi input
    error_flag = False

    # Validasi jika ada field yang kosong
    if st.session_state.glucose == 0:
        st.warning("⚠️ Mohon isi kadar glukosa.")
        error_flag = True
    if st.session_state.blood_pressure == 0:
        st.warning("⚠️ Mohon isi tekanan darah diastolik.")
        error_flag = True
    if st.session_state.skin_thickness == 0:
        st.warning("⚠️ Mohon isi ketebalan lipatan kulit.")
        error_flag = True
    if st.session_state.insulin == 0:
        st.warning("⚠️ Mohon isi kadar insulin.")
        error_flag = True

    col1, col2 = st.columns(2)
    with col1:
        st.button("Kembali", on_click=prev_page)
    with col2:
        st.button("Lanjut", on_click=next_page,
                  disabled=error_flag)

# ---------------- Page 4: BMI dan Riwayat ----------------
elif st.session_state.page == 4:
    st.header("Data Tambahan")

    st.session_state.berat = st.number_input("Berat Badan (kg)", 1.0, 200.0, step=0.1, value=st.session_state.get("berat", 60.0))
    st.session_state.tinggi = st.number_input("Tinggi Badan (cm)", 30, 250, step=1, value=st.session_state.get("tinggi", 160))

    try:
        st.session_state.bmi = hitung_bmi(st.session_state.berat, st.session_state.tinggi)
        st.success(f"BMI Anda: {st.session_state.bmi}")
    except ValueError as e:
        st.error(str(e))
        st.session_state.bmi = 0.0

    st.session_state.riwayat_orangtua = st.radio("Riwayat Diabetes pada Orang Tua", ["Kedua", "Salah satu Ayah/Ibu", "Tidak ada"], 
                                                  index=["Kedua", "Salah satu Ayah/Ibu", "Tidak ada"].
                                                  index(st.session_state.get("riwayat_orangtua", "Tidak ada")),
                                                  horizontal=True)
    st.session_state.riwayat_kakek = st.checkbox("Riwayat Diabetes pada Kakek/Nenek", value=st.session_state.get("riwayat_kakek", False))

    col1, col2 = st.columns(2)
    with col1:
        st.button("Kembali", on_click=prev_page)
    with col2:
        st.button("Prediksi", on_click=next_page)

# ---------------- Page 5: Hasil Prediksi ----------------
elif st.session_state.page == 5:
    st.header("Hasil Prediksi Risiko Diabetes")
    dpf = hitung_dpf(st.session_state.riwayat_orangtua, st.session_state.jenis_kelamin, st.session_state.riwayat_kakek)

    # Buat input array sesuai skala
    fitur_standard = pd.DataFrame([[
        st.session_state.pregnancies,
        st.session_state.insulin,
        dpf,
        st.session_state.age
    ]], columns=["Pregnancies", "Insulin", "DPF", "Age"])

    fitur_minmax = pd.DataFrame([[
        st.session_state.blood_pressure,
        st.session_state.skin_thickness,
        st.session_state.glucose,
        st.session_state.bmi
    ]], columns=["BloodPressure", "SkinThickness", "Glucose", "BMI"])

    scaled_standard = scaler_standard.transform(fitur_standard)
    scaled_minmax = scaler_minmax.transform(fitur_minmax)

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

    input_reshaped = input_data.reshape((1, 8, 1))

    # Prediksi
    prediction = model.predict(input_reshaped)
    probabilitas = prediction[0][0]
    label = (probabilitas > 0.5).astype(int)
    hasil = "POSITIF Diabetes" if label == 1 else "NEGATIF Diabetes"

    # Menentukan kategori risiko berdasarkan probabilitas
    if probabilitas < 0.3:
        risiko = "🟢 Risiko Rendah"
    elif 0.3 <= probabilitas < 0.6:
        risiko = "🟡 Risiko Sedang"
    elif 0.6 <= probabilitas < 0.85:
        risiko = "🟠 Risiko Tinggi"
    else:
        risiko = "🔴 Risiko Sangat Tinggi"

    # Tampilkan hasil
    st.success(f"Hasil Prediksi: **{hasil}**")
    st.markdown(f"**Probabilitas Positif Diabetes:** `{probabilitas:.2%}`")
    st.markdown(f"**Kategori Risiko:** {risiko}")

    # Tombol ulangi
    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])
    with col3:
        st.button("🔁 Ulangi", on_click=lambda: st.session_state.update({"page": 1}))


