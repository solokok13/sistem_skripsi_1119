import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import graphviz
import pydotplus

# Membaca data dari file Excel
file_path = 'Data_FINAL.xlsx'  # Pastikan file path ini sesuai
df = pd.read_excel(file_path)

# Pra-pemrosesan data
df.dropna(inplace=True)  # Menghapus baris dengan nilai yang hilang

# Misalkan kolom target adalah 'HASIL_BELAJAR' dan fitur lainnya adalah kategorikal
target = 'HASIL_BELAJAR'
features = [col for col in df.columns if col != target]

# Encoding fitur kategorikal menjadi numerik
df_encoded = pd.get_dummies(df[features])
y = df[target].apply(lambda x: 1 if x == 'Memuaskan' else 0)

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=42)

# Menggunakan SMOTE untuk mengatasi ketidakseimbangan data (jika perlu)
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# Melatih model dengan data training
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Prediksi pada data test
y_pred = model.predict(X_test)

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))

# Menyimpan model ke file
joblib_file = "model_c4_5.pkl"
joblib.dump(model, joblib_file)
print(f'Model disimpan ke {joblib_file}')

# Visualisasi pohon keputusan menggunakan graphviz dan pydotplus
dot_data = export_graphviz(model, out_file=None, 
                           feature_names=df_encoded.columns,  
                           class_names=['Tidak Memuaskan', 'Memuaskan'],  
                           filled=True, rounded=True,  
                           special_characters=True)  

# Mengonversi dot_data menjadi grafik
graph = pydotplus.graph_from_dot_data(dot_data)  

# Menyimpan grafik ke file
graph.write_png("decision_tree.png")
print("Visualisasi pohon keputusan disimpan sebagai decision_tree.png")
