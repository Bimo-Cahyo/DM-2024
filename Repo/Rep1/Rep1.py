#!/usr/bin/env python
# coding: utf-8

# ## A11.2022.14610_Bimo Cahyo WIdyanto

# # Deskripsi Dataset
# > Dataset yang Anda analisis adalah data kesehatan yang berisi informasi tentang partisipan dalam suatu studi atau penelitian. Dataset ini diambil dari lembar kerja Excel yang bernama '2022' dan mencakup berbagai variabel yang berkaitan dengan kesehatan dan kondisi sosial ekonomi partisipan.

# In[1]:


import pandas as pd


# In[29]:


# Load the Excel file
file_path = r'C:/Users/Fatlem/Desktop/Data-Mining/Fatlem-Rep1/dataKasus-1.xlsx'
xls = pd.ExcelFile(file_path)


# In[30]:


# Load the data from the '2022' sheet
data_2022 = pd.read_excel(xls, sheet_name='2022')


# In[31]:


# Drop the unnamed column that seems to be irrelevant
data_cleaned = data_2022.drop(columns=['Unnamed: 12'])


# In[20]:


data_cleaned['USIA'] = pd.to_numeric(data_cleaned['USIA'].str.extract(r'(\d+)')[0], errors='coerce')


# In[21]:


data_cleaned.dropna(subset=['USIA'], inplace=True)


# In[22]:


data_cleaned['USIA'] = data_cleaned['USIA'].astype(int)


# In[23]:


print(data_cleaned.head())


# In[24]:


pe_counts = data_cleaned['PE/Non PE'].value_counts()
print("\nJumlah kasus PE dan Non-PE:")
print(pe_counts)


# In[25]:


average_age = data_cleaned['USIA'].mean()
print("\nRata-rata usia partisipan:", average_age)


# In[26]:


conditions = ['RIW HIPERTENSI', 'OBESITAS', 'RIW DM']
for condition in conditions:
    condition_counts = data_cleaned[condition].value_counts()
    print(f"\nJumlah partisipan berdasarkan {condition}:")
    print(condition_counts)


# In[27]:


sosek_counts = data_cleaned['SOSEK RENDAH'].value_counts()
print("\nStatus sosial ekonomi partisipan:")
print(sosek_counts)


# # Hasil Analisis
# 1. - Jumlah Kasus PE dan Non-PE:
# Dataset menunjukkan jumlah partisipan dengan kondisi PE dan Non PE, yang memberikan gambaran tentang prevalensi penyakit dalam populasi yang diteliti.
# 
# 2. - Rata-rata Usia Partisipan:
# Rata-rata usia partisipan memberikan wawasan tentang demografi peserta. Hal ini penting untuk memahami karakteristik populasi dalam studi ini.
# 
# 3. - Distribusi Kondisi Kesehatan:
# Analisis terhadap kondisi kesehatan seperti hipertensi, obesitas, dan diabetes memberikan gambaran tentang faktor risiko yang mungkin ada dalam populasi. Ini dapat membantu dalam merumuskan intervensi kesehatan yang lebih baik.
# 
# 4. - Status Sosial Ekonomi:
# Mengetahui status sosial ekonomi partisipan bisa menjadi indikator penting dalam penelitian kesehatan. Ada kemungkinan bahwa status sosial ekonomi berkorelasi dengan kesehatan dan akses terhadap perawatan kesehatan.

# # Kesimpulan
# > Dataset ini menawarkan informasi yang berharga untuk penelitian kesehatan, terutama dalam memahami kondisi kesehatan masyarakat. Dengan menganalisis usia, prevalensi penyakit, dan status sosial ekonomi, peneliti dapat mengidentifikasi pola, faktor risiko, dan area di mana intervensi kesehatan mungkin diperlukan. Dataset ini dapat menjadi dasar untuk analisis lebih lanjut dan pengembangan strategi kesehatan masyarakat yang lebih efektif.
