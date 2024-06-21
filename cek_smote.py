import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Membaca data dari file Excel
file_path = 'Data_FINAL.xlsx'
df = pd.read_excel(file_path)

# Pra-pemrosesan data
df.dropna(inplace=True)  # Menghapus baris dengan nilai yang hilang (opsional)

# Misalkan kolom target adalah 'PlayTennis' dan fitur lainnya adalah kategorikal
target = 'HASIL_BELAJAR'
features = [col for col in df.columns if col != target]

# Encoding fitur kategorikal menjadi numerik
df_encoded = pd.get_dummies(df[features])
y = df[target].apply(lambda x: 1 if x == 'Memuaskan' else 0)

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=42)

# Menghitung distribusi kelas sebelum SMOTE
class_counts_before = y_train.value_counts()
print("Distribusi kelas sebelum SMOTE:")
print(class_counts_before)

# Visualisasi distribusi kelas sebelum SMOTE
plt.figure(figsize=(10, 6))
class_counts_before.plot(kind='bar')
plt.xlabel('Kelas')
plt.ylabel('Jumlah Sampel')
plt.title('Distribusi Kelas Sebelum SMOTE')
plt.tight_layout()
plt.show()

# Mengaplikasikan SMOTE pada data latih
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Menghitung distribusi kelas sesudah SMOTE
class_counts_after = pd.Series(y_train_resampled).value_counts()
print("Distribusi kelas sesudah SMOTE:")
print(class_counts_after)

# Visualisasi distribusi kelas sesudah SMOTE
plt.figure(figsize=(10, 6))
class_counts_after.plot(kind='bar')
plt.xlabel('Kelas')
plt.ylabel('Jumlah Sampel')
plt.title('Distribusi Kelas Sesudah SMOTE')
plt.tight_layout()
plt.show()
