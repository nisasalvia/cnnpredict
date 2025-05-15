# Fungsi hitung DPF
def hitung_dpf(riwayat_orangtua, jenis_kelamin, riwayat_kakek):
    base_dpf = 0.08

    if riwayat_orangtua == 'Kedua':
        base_dpf += 1.2 if jenis_kelamin == 'Laki-laki' else 1.0
    elif riwayat_orangtua == 'Salah satu Ayah/Ibu':
        base_dpf += 0.8 if jenis_kelamin == 'Laki-laki' else 0.6
    elif riwayat_orangtua == 'Tidak ada':
        base_dpf += 0.4 if riwayat_kakek else 0.1

    return min(base_dpf, 1.32)