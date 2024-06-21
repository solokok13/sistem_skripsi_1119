import pandas as pd

# Membaca data dari file Excel
file_path = 'Data_FINAL.xlsx'
df = pd.read_excel(file_path)

# Menampilkan beberapa baris pertama dari data
print(df.head())

# Menampilkan tipe data dari setiap kolom
print(df.dtypes)

# Memeriksa distribusi kelas dalam kolom target 'HASIL_BELAJAR'
target = 'HASIL_BELAJAR'
class_distribution = df[target].value_counts()
print("Distribusi kelas dalam kolom target:")
print(class_distribution)

# Menghitung persentase distribusi kelas
class_percentage = df[target].value_counts(normalize=True) * 100
print("Persentase distribusi kelas:")
print(class_percentage)
