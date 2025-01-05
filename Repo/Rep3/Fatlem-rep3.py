#!/usr/bin/env python
# coding: utf-8

# ## A11.2022.1424_Faathir El Tasleem

# In[20]:


# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report


# ### Load Dataset:
# 
# - Menggunakan pd.read_excel() untuk memuat dataset dari file Excel.
# - Menggunakan data.info() untuk menampilkan informasi struktur dataset seperti tipe data dan nilai kosong.
# - Menampilkan beberapa data awal dengan data.head() untuk melihat isi dataset.

# In[21]:


# Load Dataset from Excel
data = pd.read_excel('C:/Users/Fatlem/Desktop/Data-Mining/Fatlem-Rep3/dataKasus-1.xlsx')  # Memuat dataset dari file Excel
print("Data Loaded Successfully!")
print("Info Data:", data.info())
print("Beberapa data awal:\n", data.head())


# ### Data Preprocessing:
# 
# - Menghapus kolom yang tidak relevan (NO, NAMA, Unnamed: 12).
# - Mengatasi nilai kosong pada kolom USIA dan JARAK KELAHIRAN dengan menggantinya menggunakan modus (nilai yang paling sering muncul).
# - Membersihkan nilai non-numeric dalam kolom USIA dan mengonversinya ke tipe integer.

# In[25]:


# Drop irrelevant columns
data_cleaned = data.drop(columns=['NO', 'NAMA', 'Unnamed: 12'])


# In[23]:


# Handle missing values by filling with the most frequent value (mode)
data_cleaned['USIA'] = data_cleaned['USIA'].fillna(data_cleaned['USIA'].mode()[0])
data_cleaned['JARAK KELAHIRAN'] = data_cleaned['JARAK KELAHIRAN'].fillna(data_cleaned['JARAK KELAHIRAN'].mode()[0])


# In[29]:


# Isi nilai NaN dengan modus terlebih dahulu untuk menghindari masalah konversi
data_cleaned['USIA'] = data_cleaned['USIA'].fillna(data_cleaned['USIA'].mode()[0])

# Remove non-numeric characters (like " TH") in the 'USIA' column
data_cleaned['USIA'] = data_cleaned['USIA'].str.extract(r'(\d+)')

# Pastikan tidak ada nilai NaN setelah proses ekstraksi angka
data_cleaned['USIA'] = data_cleaned['USIA'].fillna(data_cleaned['USIA'].mode()[0])

# Konversi ke tipe integer setelah memastikan tidak ada NaN
data_cleaned['USIA'] = data_cleaned['USIA'].astype(int)


# ### Encode Categorical Data:
# 
# - Mengonversi data kategori menjadi format numerik menggunakan LabelEncoder, yang diperlukan untuk algoritma pembelajaran mesin.

# In[30]:


# Encode categorical data
label_encoders = {}
for column in data_cleaned.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data_cleaned[column] = le.fit_transform(data_cleaned[column])
    label_encoders[column] = le


# ### Split Data:
# 
# - Memisahkan dataset menjadi fitur (X) dan target (y), di mana PE/Non PE adalah variabel target.

# In[31]:


# Splitting data into Features (X) and Target (y)
target_column_name = 'PE/Non PE'
X = data_cleaned.drop(target_column_name, axis=1)
y = data_cleaned[target_column_name]


# ### Exploratory Data Analysis (EDA):
# 
# - Menampilkan statistik deskriptif dari data yang telah dibersihkan dengan data_cleaned.describe().
# - Membuat heatmap korelasi untuk memahami hubungan antar fitur.
# - Menampilkan distribusi variabel target untuk melihat keseimbangan kelas.

# In[32]:


# Exploratory Data Analysis (EDA)
print("Descriptive Statistics:\n", data_cleaned.describe())
plt.figure(figsize=(12, 8))
sns.heatmap(data_cleaned.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# In[33]:


# Exploratory Data Analysis (EDA)
print("Descriptive Statistics:\n", data_cleaned.describe())
plt.figure(figsize=(12, 8))
sns.heatmap(data_cleaned.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# In[34]:


# Visualisasi distribusi variabel target
plt.figure(figsize=(8, 6))
sns.countplot(data_cleaned[target_column_name])
plt.title("Distribusi Kelas Target")
plt.show()


# ### Feature Selection:
# 
# - Menggunakan metode Recursive Feature Elimination (RFE) dengan model DecisionTreeClassifier untuk memilih fitur yang paling penting.

# In[35]:


# Feature Selection (15 AH) using Recursive Feature Elimination (RFE)
print("\n---Feature Selection---")
model = DecisionTreeClassifier(random_state=42)
selector = RFE(model, n_features_to_select=15, step=1)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.support_]
print("Selected Features:", selected_features)


# In[36]:


# Train-test split
X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_selected, X_test_selected, _, _ = train_test_split(X_selected, y, test_size=0.2, random_state=42)


# In[37]:


# Standardize data
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test_full = scaler.transform(X_test_full)
X_train_selected = scaler.fit_transform(X_train_selected)
X_test_selected = scaler.transform(X_test_selected)


# ### Modeling and Evaluation:
# 
# - Menerapkan beberapa model seperti Naive Bayes, K-Nearest Neighbors, dan Decision Tree.
# - Melakukan evaluasi menggunakan metrik seperti Confusion Matrix dan Classification Report.

# In[38]:


# Modeling and Evaluation with Cross-Validation
print("\n---Modeling and Evaluation---")
models = {
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Initialize dictionaries to store scores
scores_full = {}
scores_selected = {}

for name, model in models.items():
    # Original Data
    print(f"\nModel: {name} on Original Data")
    model.fit(X_train_full, y_train)
    y_pred_full = model.predict(X_test_full)
    cm_full = confusion_matrix(y_test, y_pred_full)
    print("Confusion Matrix (Original):\n", cm_full)
    print("Classification Report (Original):\n", classification_report(y_test, y_pred_full))
    cv_scores_full = cross_val_score(model, X_train_full, y_train, cv=5)
    scores_full[name] = (cv_scores_full.mean(), cv_scores_full.std())
    
    # Selected Data
    print(f"\nModel: {name} on Selected Features")
    model.fit(X_train_selected, y_train)
    y_pred_selected = model.predict(X_test_selected)
    cm_selected = confusion_matrix(y_test, y_pred_selected)
    print("Confusion Matrix (Selected):\n", cm_selected)
    print("Classification Report (Selected):\n", classification_report(y_test, y_pred_selected))
    cv_scores_selected = cross_val_score(model, X_train_selected, y_train, cv=5)
    scores_selected[name] = (cv_scores_selected.mean(), cv_scores_selected.std())


# ### Comparison Analysis:
# 
# - Membandingkan performa model pada data asli dan data dengan fitur yang telah dipilih berdasarkan skor validasi silang (Cross-Validation).

# In[40]:


# Comparison Analysis
for name in models.keys():
    print(f"{name}:")
    print(f"Original Data - Mean CV Score: {scores_full[name][0]:.4f} ± {scores_full[name][1]:.4f}")
    print(f"Selected Data - Mean CV Score: {scores_selected[name][0]:.4f} ± {scores_selected[name][1]:.4f}")
    print()


# ### Visualisasi Hasil:
# 
# - Membuat grafik batang untuk membandingkan performa model pada data asli dan data dengan fitur terpilih.

# In[41]:


# Visualisasi Hasil
labels = list(models.keys())
original_scores = [scores_full[model][0] for model in labels]
selected_scores = [scores_selected[model][0] for model in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, original_scores, width, label='Original Data')
rects2 = ax.bar(x + width/2, selected_scores, width, label='Selected Features')

ax.set_ylabel('Mean CV Score')
ax.set_title('Comparison of Model Performance on Original vs Selected Features')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()

