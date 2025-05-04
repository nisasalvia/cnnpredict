
def hitung_dpf(riwayat_orangtua, jenis_kelamin_anak, riwayat_kakek_nenek):
    """
    Hitung nilai DPF berdasarkan riwayat keluarga dan jenis kelamin.
    Range nilai DPF 0.08 - 1.32
    """
    base_dpf = 0.08

    if riwayat_orangtua == 'Kedua':
        base_dpf += 1.2 if jenis_kelamin_anak == 'Laki-laki' else 1.0
    elif riwayat_orangtua == 'Salah satu':
        base_dpf += 0.8 if jenis_kelamin_anak == 'Laki-laki' else 0.6
    elif riwayat_orangtua == 'Tidak ada':
        base_dpf += 0.4 if riwayat_kakek_nenek else 0.1

    return min(base_dpf, 1.32)
