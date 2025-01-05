#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# ##### Tahap 1: Memuat Data
# Pertama, kita perlu memuat data dari file Excel yang telah Anda sediakan. Kita akan menggunakan pustaka pandas untuk ini.

# In[3]:


# Langkah 1: Memuat data dari file Excel
kepadatan_data = pd.read_excel('Kepadatan_Penduduk_menurut_Provinsi.xlsx')
jumlah_data = pd.read_excel('Jumlah_Penduduk.xlsx')
covid_data = pd.read_excel('Statistik_Harian_per_Provinsi_COVID19_Indonesia_Rev.xlsx')


# - Memuat Data: Membaca data dari tiga file Excel ke dalam tiga DataFrame terpisah (kepadatan_data, jumlah_data, dan covid_data). Setiap DataFrame berisi data terkait kepadatan penduduk, jumlah penduduk, dan statistik COVID-19.

# ##### Tahap 2: Memeriksa dan Menyesuaikan Nama Kolom
# Sebelum menggabungkan data, kita perlu memeriksa nama kolom di setiap DataFrame untuk memastikan bahwa kita dapat menggabungkannya dengan benar.

# In[4]:


# Langkah 2: Mengganti nama kolom untuk memastikan konsistensi
kepadatan_data.rename(columns={'Provinsi': 'Province'}, inplace=True)
jumlah_data.rename(columns={'Provinsi': 'Province'}, inplace=True)
covid_data.rename(columns={'Provinsi': 'Province'}, inplace=True)


# Jika nama kolom tidak konsisten (misalnya, ada spasi atau perbedaan huruf besar/kecil), kita perlu menyesuaikannya. Misalnya, jika kolom provinsi di semua DataFrame tidak konsisten, kita bisa menggantinya.

# ##### Langkah 3: Menggabungkan Dataset Berdasarkan Kolom 'Province'

# In[5]:


# Langkah 3: Menggabungkan dataset berdasarkan kolom 'Province'
merged_data = pd.merge(kepadatan_data, jumlah_data, on='Province')
merged_data = pd.merge(merged_data, covid_data, on='Province')


# - Menggabungkan DataFrame: Menggabungkan ketiga DataFrame menjadi satu (merged_data) berdasarkan kolom Province. Ini menggabungkan data kepadatan penduduk, jumlah penduduk, dan statistik COVID-19.

# ##### Langkah 4: Memeriksa Nama Kolom pada DataFrame yang Digabungkan

# In[19]:


# Langkah 4: Memeriksa nama kolom pada DataFrame yang digabungkan
print("Merged Data Columns:")
for column in merged_data.columns:
    print(f"- {column}")


# - Memeriksa Nama Kolom: Mencetak nama kolom dari DataFrame yang digabungkan untuk memverifikasi bahwa penggabungan berhasil dan untuk melihat kolom apa saja yang tersedia.

# ##### Langkah 5: Menghapus Spasi di Awal dan Akhir Nama Kolom

# In[7]:


# Langkah 5: Menghapus spasi di awal dan akhir nama kolom
merged_data.columns = merged_data.columns.str.strip()


# - Membersihkan Nama Kolom: Menghapus spasi di awal dan akhir dari nama kolom dalam DataFrame yang digabungkan untuk menghindari masalah saat merujuk nama kolom di kemudian hari.

# ##### Langkah 6: Memilih Fitur yang Relevan untuk Clustering

# In[24]:


# Langkah 6: Memilih fitur yang relevan untuk clustering
try:
    features = merged_data[['Kepadatan Penduduk', 'Jumlah Penduduk', 'Kasus COVID', 
                            'PDB', 'Tingkat Urbanisasi', 'Indeks Kesehatan', 
                            'Tingkat Pendidikan', 'Tingkat Ketenagakerjaan']]
    
    # Menampilkan nama kolom yang dipilih dengan rapi
    print("Selected Features Columns:")
    for column in features.columns:
        print(f"- {column}")  # Menampilkan setiap kolom di baris baru
except KeyError as e:
    print(f"KeyError: {e}. Silakan periksa nama kolom di merged_data.")
    features = None  # Set features menjadi None untuk menghindari kesalahan lebih lanjut


# - Memilih Fitur: Mencoba untuk membuat DataFrame baru (features) yang hanya berisi kolom-kolom relevan yang diperlukan untuk clustering. Jika ada kolom yang tidak ada dalam merged_data, maka akan muncul KeyError, dan pesan kesalahan akan dicetak. Variabel features diatur menjadi None untuk mencegah kesalahan lebih lanjut jika pemilihan gagal.

# ##### Langkah 7: Menormalkan Data

# In[12]:


# Langkah 7: Menormalkan data
if features is not None:
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(features)
    print("Data telah dinormalisasi.")  # Baris debugging
else:
    print("Fitur tidak berhasil dipilih, normalisasi tidak dilakukan.")
    normalized_data = None  # Set normalized_data menjadi None untuk menghindari kesalahan lebih lanjut


# - Tujuan: Menstandarkan data agar semua fitur memiliki skala yang sama (rata-rata 0 dan deviasi standar 1).
# - Proses:
#     - Memeriksa apakah fitur yang relevan telah dipilih.
#     - Menggunakan StandardScaler untuk menormalkan data.
#     - Jika normalisasi berhasil, data disimpan dalam variabel normalized_data.

# ##### Langkah 8: Melakukan Clustering dengan K-Means

# In[13]:


# Langkah 8: Melakukan clustering dengan K-Means
if normalized_data is not None:
    cluster_numbers = [3, 4, 5, 6, 7]
    cluster_labels = {}

    for n_clusters in cluster_numbers:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(normalized_data)
        cluster_labels[n_clusters] = labels

        # Visualisasi hasil clustering
        plt.figure(figsize=(8, 6))
        plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=labels, cmap='viridis')
        plt.title(f'K-Means Clustering dengan {n_clusters} Cluster')
        plt.xlabel('Fitur 1')
        plt.ylabel('Fitur 2')
        plt.show()
else:
    print("Normalisasi data gagal, clustering tidak dilakukan.")


# - Tujuan: Mengelompokkan data ke dalam beberapa cluster menggunakan algoritma K-Means.
# - Proses:
#     - Memeriksa apakah data yang dinormalisasi ada.
#     - Menguji berbagai jumlah cluster (3, 4, 5, 6, dan 7).
#     - Untuk setiap jumlah cluster, K-Means menghitung centroid dan mengelompokkan data.
#     - Hasil clustering divisualisasikan dengan plot sebar untuk menunjukkan distribusi data dalam cluster yang berbeda.

# ##### Langkah 9: Menetapkan Zona Berdasarkan Label Cluster

# In[14]:


# Langkah 9: Menetapkan zona berdasarkan label cluster
zone_mapping = {
    0: 'Hijau',
    1: 'Kuning',
    2: 'Merah',
    3: 'Hitam',
}


# - Pemetaan Zona: Mendefinisikan pemetaan label cluster ke nama zona. Setiap cluster diberikan nama deskriptif (misalnya, "Hijau" untuk zona hijau, "Kuning" untuk zona kuning, dll.).

# ##### Langkah 10: Menambahkan Informasi Zona ke DataFrame yang Digabungkan

# In[15]:


# Langkah 10: Menambahkan informasi zona ke data yang digabungkan
for n_clusters, labels in cluster_labels.items():
    merged_data[f'Zone_{n_clusters}'] = [zone_mapping[label] for label in labels]


# - Menambahkan Informasi Zona: Menambahkan kolom baru ke DataFrame merged_data untuk setiap jumlah cluster, yang menunjukkan zona yang ditetapkan untuk setiap provinsi berdasarkan label cluster-nya.

# ##### Langkah 11: Menyimpan DataFrame Akhir ke File Excel

# In[25]:


# Langkah 11: Menyimpan DataFrame akhir ke file Excel
merged_data.to_excel('Final_Clustered_Data.xlsx', index=False)

print("Clustering selesai dan hasil disimpan ke 'Final_Clustered_Data.xlsx'.")


# - Menyimpan Hasil: Menyimpan DataFrame merged_data, yang sekarang mencakup hasil clustering dan informasi zona, ke file Excel baru bernama Final_Clustered_Data.xlsx. Argumen index=False mencegah pandas menulis indeks baris ke file.

# ##### Memfilter Data untuk Provinsi Tertentu
# Jika Anda ingin memfilter data hanya untuk provinsi tertentu seperti Jawa Timur dan DKI Jakarta,

# In[17]:


# Memfilter data untuk provinsi tertentu (Jawa Timur dan DKI Jakarta)
filtered_data = merged_data[merged_data['Province'].isin(['Jawa Timur', 'DKI Jakarta'])]


# - Kemudian, gunakan filtered_data sebagai pengganti merged_data di langkah-langkah berikutnya.
