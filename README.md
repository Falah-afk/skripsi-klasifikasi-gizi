# Sistem Klasifikasi Tingkat Gizi Siswa dan Rekomendasi

## Deskripsi

Project ini merupakan implementasi dari penelitian skripsi yang berjudul **"Model Klasifikasi Tingkat Gizi Siswa Sekolah dan Rekomendasi Menggunakan Machine Learning"**.

Sistem dikembangkan untuk melakukan klasifikasi tingkat gizi siswa berdasarkan data antropometri dan pola hidup. Berdasarkan hasil klasifikasi, sistem memberikan informasi kebutuhan energi dan zat gizi serta rekomendasi menu makanan dan aktivitas olahraga.

## Tujuan

Sistem ini bertujuan untuk membantu proses klasifikasi status gizi siswa serta memberikan rekomendasi pola hidup yang disesuaikan dengan kondisi pengguna berdasarkan hasil analisis data.

## Metode

Metode yang digunakan dalam sistem meliputi:

- **Random Forest** untuk melakukan klasifikasi tingkat gizi.
- **SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous)** untuk menangani ketidakseimbangan data pada proses pelatihan model.
- **K-Nearest Neighbors (KNN)** untuk menghasilkan rekomendasi menu makanan berdasarkan kemiripan kebutuhan zat gizi.
- **WHO LMS** untuk proses pelabelan status gizi berdasarkan BMI-for-age.
- **TEE (Total Energy Expenditure)** untuk menghitung kebutuhan energi harian.
- Perhitungan kebutuhan makronutrien berupa karbohidrat, protein, dan lemak.
- Perhitungan kebutuhan energi untuk rekomendasi aktivitas olahraga.

## Kategori Status Gizi

Sistem melakukan klasifikasi ke dalam beberapa kategori status gizi:

- Gizi Buruk
- Gizi Kurang
- Gizi Baik
- Gizi Lebih
- Obesitas

## Fitur Sistem

Sistem menyediakan beberapa fitur utama:

1. Input data pengguna.
2. Klasifikasi status gizi.
3. Menampilkan probabilitas hasil klasifikasi.
4. Menghitung kebutuhan energi basal (BMR).
5. Menghitung total kebutuhan energi harian (TEE).
6. Menentukan target kalori harian.
7. Menghitung kebutuhan makronutrien.
8. Menghitung kebutuhan cairan harian.
9. Memberikan rekomendasi menu makanan.
10. Memberikan rekomendasi snack.
11. Memberikan rekomendasi olahraga.

## Alur Sistem

Alur utama sistem adalah sebagai berikut:

```text
Input Data Pengguna
        ↓
Preprocessing Data
        ↓
Klasifikasi Status Gizi
        ↓
Perhitungan Kebutuhan Energi dan Zat Gizi
        ↓
Rekomendasi Menu Makanan
        ↓
Rekomendasi Olahraga
        ↓
Menampilkan Hasil

Dataset

Dataset yang digunakan dalam sistem terdiri dari:

Dataset data siswa dan label status gizi.
Dataset makanan.
Dataset snack.

Struktur dataset:
data/
├── Data_Label_Gizi.csv
├── makanan_berat_bersih.csv
└── snack_bersih.csv
Dataset yang digunakan dalam repository tidak mencantumkan informasi pribadi yang dapat mengidentifikasi pengguna.

Struktur Project
skripsi-klasifikasi-gizi/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    ├── Data_Label_Gizi.csv
    ├── makanan_berat_bersih.csv
    └── snack_bersih.csv

Teknologi

Project ini dikembangkan menggunakan:

Python
Pandas
NumPy
Scikit-learn
Imbalanced-learn
OpenPyXL
Joblib
Matplotlib
Seaborn

Hasil
Sistem menghasilkan informasi berupa:
Status gizi pengguna.
Probabilitas klasifikasi.
Kebutuhan energi basal (BMR).
Total kebutuhan energi harian (TEE).
Target kalori.
Kebutuhan karbohidrat, protein, dan lemak.
Kebutuhan cairan.
Rekomendasi menu makanan.
Rekomendasi snack.
Rekomendasi olahraga.

