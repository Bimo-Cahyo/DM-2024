#!/usr/bin/env python
# coding: utf-8

# - NAMA : Bimo Cahyo Widyanto
# - NIM : A11.2022.14610

# ## Crop Yield Prediction Datase

# In[7]:


from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


# In[8]:


# Load the dataset to explore its content
file_path = 'yield.csv'
dataset = pd.read_csv(file_path)


# ### Sample Data (5 Baris Pertama)

# In[9]:


# Display the first few rows and basic information about the dataset
dataset_info = dataset.info()
sample_data = dataset.head(5)

dataset_info, sample_data


# - Selanjutnya, kita akan melakukan eksperimen pada subset kecil (5 data) untuk analisis sederhana menggunakan regresi linear. Saya akan memilih data untuk satu negara (Afghanistan) dan satu jenis tanaman (Maize) dari tahun 1961-1965 untuk prediksi hasil panen di tahun berikutnya.

# In[10]:


# Filter dataset for a small subset: Afghanistan, Maize, 1961-1965
subset = dataset[(dataset['Area'] == 'Afghanistan') & 
                 (dataset['Item'] == 'Maize') & 
                 (dataset['Year'].between(1961, 1965))]


# - Data yang digunakan untuk eksperimen regresi linear adalah hasil panen tanaman jagung (Maize) di Afghanistan dari tahun 1961 hingga 1965.

# In[11]:


# Prepare data for regression: Year as independent variable, Value as dependent variable
subset_for_regression = subset[['Year', 'Value']]

subset_for_regression


# - Regresi linear akan dilakukan dengan variabel independen Tahun dan variabel dependen Hasil Panen (Value). Tujuannya adalah untuk memprediksi hasil panen di tahun-tahun berikutnya, misalnya tahun 1966.

# In[24]:


# Extract Year and Value as numpy arrays for regression
X = subset_for_regression['Year'].values.reshape(-1, 1)
y = subset_for_regression['Value'].values

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the yield for 1966
year_to_predict = np.array([[1966]])
predicted_value = model.predict(year_to_predict)


# #### 1. Persamaan Regresi:
#     - Y=105.7×Year−193305.7
#         - Slope (kemiringan): 105.7 (peningkatan rata-rata hasil panen per tahun).
#         - Intercept (titik potong): -193305.7 (nilai teoretis ketika tahun = 0).
# #### 2. Prediksi Tahun 1966:
#    > Menggunakan model ini, hasil panen diprediksi sebesar 14.500,5 hg/ha untuk tahun 1966.
# 
# 

# In[25]:


# Coefficients and prediction result
slope = model.coef_[0]
intercept = model.intercept_
predicted_value[0], slope, intercept


# ### Penjelasan:
# - Model ini menunjukkan bahwa hasil panen di Afghanistan untuk jagung meningkat secara konsisten sekitar 105.7 hg/ha setiap tahun dari 1961 hingga 1965.
# - Prediksi ini mungkin tidak akurat untuk jangka panjang karena asumsi tren linear, sehingga validasi tambahan pada data lebih luas diperlukan.
