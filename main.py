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

def tampilkan_footer():
    st.markdown("""
    <style>
    .footnote-container {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 90%; /* Atur agar tidak selebar layar */
        max-width: 700px; /* Sesuaikan dengan lebar konten Streamlit default */
        background-color: white;
        padding: 10px 20px;
        font-size: 12px;
        color: #555;
        text-align: left;
        border-top: 1px solid #eee;
        z-index: 100;
    }
    .footnote-container a {
        color: #555;
    }
    </style>

    <div class="footnote-container">
        <em>Supervised by Mrs. Retno Aulia Vinarti, S.Kom., M.Kom., Ph.D.</em><br>
        <em>Developed by Nisa Salvia Najmi, a student of the Information Systems Department, Institut Teknologi Sepuluh Nopember, Class of 2021.</em><br>
        <a href="mailto:nisasalvia96@gmail.com">nisasalvia96@gmail.com</a><br><br>
    </div>
    """, unsafe_allow_html=True)

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
    
    tampilkan_footer()

# ---------------- Page 2: Data Diri ----------------
elif st.session_state.page == 2:
    st.header("Data Diri")

    error_flag = False

    # Usia dan Jenis Kelamin
    st.session_state.age = st.number_input(
        "Usia", min_value=5, max_value=120, step=1, value=st.session_state.get("age")
    )
    st.session_state.jenis_kelamin = st.selectbox(
        "Jenis Kelamin", ["Laki-laki", "Perempuan"],
        index=["Laki-laki", "Perempuan"].index(st.session_state.get("jenis_kelamin", "Perempuan"))
    )

    # Berat dan Tinggi Badan
    st.session_state.berat = st.number_input(
        "Berat Badan (kg)", 1.0, 200.0, step=0.1, value=st.session_state.get("berat", 60.0)
    )
    st.session_state.tinggi = st.number_input(
        "Tinggi Badan (cm)", 30, 250, step=1, value=st.session_state.get("tinggi", 160)
    )

    try:
        st.session_state.bmi = hitung_bmi(st.session_state.berat, st.session_state.tinggi)
        st.success(f"BMI Anda: {st.session_state.bmi}")
    except ValueError as e:
        st.error(str(e))
        st.session_state.bmi = 0.0

    # Berapa kali hamil - dipindah ke bawah setelah berat/tinggi
    if st.session_state.jenis_kelamin != "Perempuan" or (st.session_state.age and st.session_state.age <= 10):
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

        # Pertanyaan utama
    st.markdown("### Riwayat Diabetes Keluarga")

    riwayat_diabetes = st.multiselect(
        "Apakah memiliki riwayat diabetes di keluarga?",
        options=["Orang Tua", "Kakek/Nenek"],
        default=[
            opt for opt in ["Orang Tua", "Kakek/Nenek"]
            if st.session_state.get("riwayat_orangtua") or st.session_state.get("riwayat_kakek")
        ]
    )

    # Tampilkan radio jika "Orang Tua" dipilih
    if "Orang Tua" in riwayat_diabetes:
        default_radio = st.session_state.get("riwayat_orangtua")
        if default_radio not in ["Keduanya Diabetes", "Salah satu Orang Tua"]:
            default_radio = "Keduanya Diabetes"

        st.session_state.riwayat_orangtua = st.radio(
            "Riwayat diabetes pada orang tua:",
            ["Keduanya Diabetes", "Salah satu Orang Tua"],
            index=["Keduanya Diabetes", "Salah satu Orang Tua"].index(default_radio)
        )
    else:
        st.session_state.riwayat_orangtua = None


    # Checkbox untuk Kakek/Nenek
    st.session_state.riwayat_kakek = "Kakek/Nenek" in riwayat_diabetes


    # Navigasi
    col_kiri, col_tengah, col_kanan = st.columns([2, 10, 1])

    with col_kiri:
        st.button("⬅", on_click=prev_page)

    with col_kanan:
        if st.button("⮕"):
            st.session_state["lanjut_dipencet"] = True
            if not error_flag:
                next_page()

    if st.session_state.get("lanjut_dipencet") and error_flag:
        st.error("Mohon lengkapi data yang wajib diisi terlebih dahulu.")

    tampilkan_footer()
# ---------------- Page 3: Data Kesehatan ----------------
elif st.session_state.page == 3:
    st.header("Data Kesehatan")

    checkup_option = st.radio(
        "Apakah sebelumnya pernah check-up kesehatan?",
        options=["Ya", "Tidak"],
        index=None if "checkup" not in st.session_state else (0 if st.session_state.checkup else 1),
        horizontal=True
    )

    # Simpan pilihan ke session_state
    if checkup_option:
        st.session_state.checkup = (checkup_option == "Ya")

    # Tampilkan form input jika Ya
    if st.session_state.get("checkup") is True:
        st.session_state.glucose = st.number_input("Kadar Glukosa (mg/dL)", min_value=30, max_value=200, value=None, placeholder="Masukkan kadar glukosa")
        st.session_state.blood_pressure = st.number_input("Tekanan Darah Diastolik (mm/Hg)", min_value=40, max_value=150, value=None, placeholder="Masukkan tekanan darah")
        st.session_state.skin_thickness = st.number_input("Ketebalan Lipatan Kulit Trisep (mm)", min_value=3.0, max_value=100.0, value=None, placeholder="Masukkan ketebalan kulit", format="%.1f")
        st.session_state.insulin = st.number_input("Kadar Insulin (muU/ml)", min_value=2.0, max_value=1000.0, value=None, placeholder="Masukkan kadar insulin", format="%.1f")
    elif st.session_state.get("checkup") is False:
        st.info(
            "Nilai default digunakan karena belum pernah checkup. "
            "Nilai ini berdasarkan modus dari data non-diabetes."
        )
        st.number_input("Kadar Glukosa (mg/dL)", value=99, disabled=True)
        st.number_input("Tekanan Darah Diastolik (mm/Hg)", value=70, disabled=True)
        st.number_input("Ketebalan Lipatan Kulit Trisep (mm)", value=27.74, disabled=True)
        st.number_input("Kadar Insulin (muU/ml)", value=102.05, disabled=True)

    # Navigasi tombol (panah)
    col_kiri, col_tengah, col_kanan = st.columns([2, 10, 1])

    with col_kiri:
        st.button("⬅", on_click=prev_page)

    with col_kanan:
        lanjut_ditekan = st.button("⮕", key="lanjut_checkup")

    # Tampilkan error jika tombol ditekan dan belum ada pilihan
    if lanjut_ditekan:
        st.session_state["lanjut_dipencet"] = True

        if "checkup" not in st.session_state:
            st.error("Mohon pilih apakah Anda pernah check-up terlebih dahulu.")
        elif st.session_state.checkup and (
            st.session_state.glucose is None or
            st.session_state.blood_pressure is None or
            st.session_state.skin_thickness is None or
            st.session_state.insulin is None
        ):
            st.error("Mohon lengkapi semua data hasil check-up.")
        else:
            next_page()

    tampilkan_footer()

# ---------------- Page 4: BMI dan Riwayat ----------------
elif st.session_state.page == 4:
    st.header("Data Tambahan")

    st.session_state.berat = st.number_input(
        "Berat Badan (kg)", 1.0, 200.0, step=0.1, value=st.session_state.get("berat", 60.0)
    )
    st.session_state.tinggi = st.number_input(
        "Tinggi Badan (cm)", 30, 250, step=1, value=st.session_state.get("tinggi", 160)
    )

    try:
        st.session_state.bmi = hitung_bmi(st.session_state.berat, st.session_state.tinggi)
        st.success(f"BMI Anda: {st.session_state.bmi}")
    except ValueError as e:
        st.error(str(e))
        st.session_state.bmi = 0.0

    st.session_state.riwayat_orangtua = st.radio(
        "Riwayat Diabetes pada Orang Tua",
        ["Kedua", "Salah satu Ayah/Ibu", "Tidak ada"],
        index=["Kedua", "Salah satu Ayah/Ibu", "Tidak ada"].index(
            st.session_state.get("riwayat_orangtua", "Tidak ada")
        ),
        horizontal=True
    )
    st.session_state.riwayat_kakek = st.checkbox(
        "Riwayat Diabetes pada Kakek/Nenek", value=st.session_state.get("riwayat_kakek", False)
    )

    col_kiri, col_tengah, col_kanan = st.columns([2, 10, 1])

    with col_kiri:
        st.button("⬅", on_click=prev_page)

    with col_kanan:
        if st.button("Prediksi"):
            st.session_state["lanjut_dipencet"] = True

            valid = True
            if st.session_state.berat == 0:
                st.warning("⚠️ Mohon isi berat badan Anda.")
                valid = False
            if st.session_state.tinggi == 0:
                st.warning("⚠️ Mohon isi tinggi badan Anda.")
                valid = False

            if valid:
                next_page()

    if st.session_state.get("lanjut_dipencet") and (
        st.session_state.berat == 0 or st.session_state.tinggi == 0
    ):
        st.error("Mohon lengkapi data yang wajib diisi terlebih dahulu.")

    tampilkan_footer()

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

    prediction = model.predict(input_reshaped)
    probabilitas = prediction[0][0]
    label = (probabilitas > 0.5).astype(int)

    # Tentukan hasil dan risiko
    if probabilitas < 0.3:
        risiko = "🟢 Risiko Rendah"
        pesan = "Probabilitas Anda untuk mengidap diabetes tergolong rendah. Tetaplah menjaga pola hidup sehat dan lakukan pemeriksaan secara berkala."
    elif 0.3 <= probabilitas < 0.6:
        risiko = "🟡 Risiko Sedang"
        pesan = "Anda memiliki kemungkinan sedang untuk mengidap diabetes. Disarankan untuk mulai memperhatikan pola makan, aktivitas fisik, dan berkonsultasi dengan tenaga medis."
    elif 0.6 <= probabilitas < 0.85:
        risiko = "🟠 Risiko Tinggi"
        pesan = "Kemungkinan Anda mengidap diabetes cukup tinggi. Sebaiknya segera lakukan pemeriksaan medis dan ubah gaya hidup ke arah yang lebih sehat."
    else:
        risiko = "🔴 Risiko Sangat Tinggi"
        pesan = "Kemungkinan Anda sangat tinggi untuk mengidap diabetes. Segera konsultasikan dengan dokter untuk penanganan lebih lanjut."

    # Tampilkan hasil tanpa box
    if label == 1:
        warna_bg = "#f8d7da"
        warna_teks = "#721c24"
        hasil_teks = "⚠️ Anda kemungkinan <strong>mengidap diabetes.</strong>"
    else:
        warna_bg = "#d4edda"
        warna_teks = "#155724"
        hasil_teks = "✅ Anda kemungkinan <strong>tidak mengidap diabetes.</strong>"

    # Tampilkan hasil dalam box berwarna
    st.markdown(f"""
    <div style='
        background-color: {warna_bg};
        color: {warna_teks};
        border: 1px solid {warna_teks};
        border-radius: 8px;
        padding: 15px 20px;
        margin-top: 20px;
        font-size: 18px;
    '>
        {hasil_teks}
    </div>
    """, unsafe_allow_html=True)

    # Penjelasan tambahan di bawah box
    st.markdown(f"""
    <div style='text-align: justify; font-size: 16px; margin-top: 20px;'>
        Berdasarkan hasil prediksi, Anda memiliki probabilitas sebesar <b>{probabilitas:.2%}</b> untuk mengidap diabetes, 
        yang tergolong dalam kategori <b>{risiko}</b>.<br><br>
        {pesan}
    </div>
    """, unsafe_allow_html=True)

    # Spasi sebelum tombol ulangi
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Tombol ulangi
    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])
    with col3:
        st.button("Ulangi", on_click=lambda: st.session_state.update({"page": 1}))

    tampilkan_footer()

