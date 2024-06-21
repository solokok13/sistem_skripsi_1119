import pandas as pd
from sklearn.model_selection import train_test_split

# Membaca data dari file Excel
file_path = 'Data_FINAL.xlsx'
df = pd.read_excel(file_path)

# Misalkan kolom target adalah 'HASIL_BELAJAR'
target = 'HASIL_BELAJAR'
features = [col for col in df.columns if col != target]

# Mengidentifikasi fitur kategorikal dan melakukan encoding
df_encoded = pd.get_dummies(df[features])

# Mengubah kolom target menjadi numerik
y = df[target].apply(lambda x: 1 if x == 'Memuaskan' else 0)

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=42)

# Mereset indeks untuk set pelatihan dan pengujian
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Menampilkan ukuran dari masing-masing set
print("Ukuran X_train:", X_train.shape)
print("Ukuran X_test:", X_test.shape)
print("Ukuran y_train:", y_train.shape)
print("Ukuran y_test:", y_test.shape)

# Menampilkan beberapa contoh dari masing-masing set
print("\nContoh data X_train:\n", X_train.head())
print("\nContoh data X_test:\n", X_test.head())
print("\nContoh label y_train:\n", y_train.head())
print("\nContoh label y_test:\n", y_test.head())
