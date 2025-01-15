#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize


# In[17]:


# Load the datasets with the local path
training_data = pd.read_csv('twitter_training.csv')
validation_data = pd.read_csv('twitter_validation.csv')


# In[18]:


# Display the first few rows and structure of each dataset
training_info = training_data.info(), training_data.head()
validation_info = validation_data.info(), validation_data.head()

training_info, validation_info


# In[19]:


# Ganti nama kolom untuk kedua dataset agar lebih mudah dipahami
# Asumsikan kolom ketiga adalah teks tweet dan kolom kedua adalah label sentimen

# Ganti nama kolom untuk data pelatihan
training_data.columns = ['ID', 'Source', 'Sentiment', 'Tweet']

# Ganti nama kolom untuk data validasi
validation_data.columns = ['ID', 'Source', 'Sentiment', 'Tweet']


# In[20]:


# Periksa struktur kumpulan data yang telah dibersihkan
training_data_cleaned = training_data.head()
validation_data_cleaned = validation_data.head()

training_data_cleaned, validation_data_cleaned


# In[21]:


# Secara manual menentukan daftar kata-kata berhenti umum dalam bahasa Inggris untuk menghindari masalah pengunduhan
custom_stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
    'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', 
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}


# In[22]:


# Download tokenizer data if not already downloaded
nltk.download('punkt')

# Function to clean text
def clean_text_local(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+|[^\w\s]', '', text)
    tokens = word_tokenize(text)  # Using word_tokenize from nltk
    tokens = [word for word in tokens if word not in custom_stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Apply the cleaning function to training and validation data
training_data['Cleaned_Tweet'] = training_data['Tweet'].apply(clean_text_local)
validation_data['Cleaned_Tweet'] = validation_data['Tweet'].apply(clean_text_local)

# Display a sample of the cleaned data
cleaned_training_sample_updated = training_data[['Tweet', 'Cleaned_Tweet']].head()
print(cleaned_training_sample_updated)

