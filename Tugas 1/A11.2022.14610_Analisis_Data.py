import pandas as pd
import seaborn as sns
import warnings
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Re-import the necessary libraries to fix the NameError and reload the data
import pandas as pd

# Load the dataset again
df = pd.read_excel(file_path, sheet_name='2022')

# Previewing the first few rows to inspect the structure of the data
df.head()

# Langkah 1: Hapus kolom 'Unnamed: 12' yang tidak diperlukan
df_cleaned = df.drop(columns=['Unnamed: 12'])

# Langkah 2: Konversi kolom 'USIA' menjadi numerik dengan mengekstrak digitnya
df_cleaned['USIA'] = df_cleaned['USIA'].str.extract(r'(\d+)').astype(float)

# Langkah 3: Standarisasi kolom biner menjadi format 1/0 menggunakan map()
binary_cols = ['RIW HIPERTENSI', 'RIW PE', 'OBESITAS', 'RIW DM', 
               'RIW HIPERTENSI/PE DALAM KELUARGA', 'SOSEK RENDAH', 'PE/Non PE']

# Definisikan fungsi mapping untuk kolom biner
binary_mapping = lambda x: 1 if x == 'Ya' or x == 'PE' else 0

# Terapkan mapping pada kolom biner menggunakan .map()
df_cleaned[binary_cols] = df_cleaned[binary_cols].apply(lambda col: col.map(binary_mapping))

# Tampilkan data yang sudah dibersihkan
df_cleaned.head()


# Hanya memilih kolom numerik untuk analisis korelasi
numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64'])

# Korelasi antar variabel numerik
correlation_matrix = numeric_cols.corr()

# Menampilkan matriks korelasi
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn')
plt.title('Heatmap Korelasi Antar Variabel')
plt.show()


# Tetapkan variabel lingkungan untuk menghindari peringatan LOKY_MAX_CPU_COUNT
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
warnings.filterwarnings("ignore", message=".*LOKY_MAX_CPU_COUNT*")

# Hapus kolom 'Unnamed: 12'
df_cleaned = df.drop(columns=['Unnamed: 12'])

# Ubah kolom 'USIA' menjadi numerik dengan mengekstrak digitnya
df_cleaned['USIA'] = df_cleaned['USIA'].str.extract(r'(\d+)').astype(float)

# Tangani nilai yang hilang dengan mengisinya dengan mean
imputer = SimpleImputer(strategy='mean')
X = df_cleaned[['USIA', 'PARITAS']]
X_imputed = imputer.fit_transform(X)

# Melakukan pengelompokan menggunakan K-Means pada data yang diimputasi
kmeans = KMeans(n_clusters=3, random_state=42)
df_cleaned['Cluster'] = kmeans.fit_predict(X_imputed)

# Visualisasi hasil pengelompokan dengan palet warna yang diperbarui
plt.figure(figsize=(8, 6))
plt.scatter(X_imputed[:, 0], X_imputed[:, 1], c=df_cleaned['Cluster'], cmap='plasma')  # Changed to 'plasma' color map
plt.xlabel('USIA')
plt.ylabel('PARITAS')
plt.title('Clustering berdasarkan USIA dan PARITAS')
plt.colorbar(label='Cluster')
plt.show()


# Misalkan df_cleaned sudah ada, dan kolom-kolom tersebut valid
# df_cleaned = pd.read_csv("dataKasus-1.csv") # Ganti dengan loading dataset Anda

# Definisikan fitur dan label (target)
X = df_cleaned[['USIA', 'PARITAS', 'RIW HIPERTENSI', 'OBESITAS']]
y = df_cleaned['PE/Non PE']

# Label encoding untuk kolom kategorikal
label_encoder = LabelEncoder()

# Menggunakan .loc untuk menghindari SettingWithCopyWarning
X.loc[:, 'RIW HIPERTENSI'] = label_encoder.fit_transform(X['RIW HIPERTENSI'])
X.loc[:, 'OBESITAS'] = label_encoder.fit_transform(X['OBESITAS'])

# Membagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model Decision Tree
model = DecisionTreeClassifier()

# Melatih model
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Menampilkan laporan klasifikasi
print(classification_report(y_test, y_pred, zero_division=0))

# Visualisasi pentingnya fitur dalam model
feature_importances = pd.Series(model.feature_importances_, index=X.columns)

# Membuat grafik batang
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='barh', color='Green', title='Laporan', edgecolor='black')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# Misalkan df_cleaned sudah ada
# df_cleaned = pd.read_csv("dataKasus-1.csv") # Ganti dengan loading dataset Anda

# Definisikan fitur dan label (target)
X = df_cleaned[['USIA', 'PARITAS', 'RIW HIPERTENSI', 'OBESITAS']]
y = df_cleaned['PE/Non PE']

# Label encoding untuk kolom kategorikal
label_encoder = LabelEncoder()

# Menggunakan .loc untuk menghindari SettingWithCopyWarning
X.loc[:, 'RIW HIPERTENSI'] = label_encoder.fit_transform(X['RIW HIPERTENSI'])
X.loc[:, 'OBESITAS'] = label_encoder.fit_transform(X['OBESITAS'])

# Membagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model Decision Tree
model = DecisionTreeClassifier()

# Melatih model
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Menampilkan laporan klasifikasi
print(classification_report(y_test, y_pred, zero_division=0))

# Mendapatkan pentingnya fitur dalam model
feature_importances = pd.Series(model.feature_importances_, index=X.columns)

# Visualisasi pentingnya fitur dalam model menggunakan grafik garis
plt.figure(figsize=(10, 6))
plt.plot(feature_importances.index, feature_importances.values, marker='o', color='red', linestyle='-', linewidth=2)
plt.title('Laporan')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.show()