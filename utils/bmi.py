def hitung_bmi(berat_kg: float, tinggi_cm: float) -> float:
    """
    Menghitung Body Mass Index (BMI).

    Parameters:
        berat_kg (float): Berat badan dalam kilogram.
        tinggi_cm (float): Tinggi badan dalam sentimeter.

    Returns:
        float: Nilai BMI (dibulatkan 1 desimal).
    """
    if tinggi_cm <= 0:
        raise ValueError("Tinggi badan harus lebih dari 0 cm.")
    
    tinggi_m = tinggi_cm / 100
    bmi = berat_kg / (tinggi_m ** 2)
    return round(bmi, 1)
