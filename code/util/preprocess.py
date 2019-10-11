from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np
import os

path = '../'
data_list = os.listdir(path)


data = pd.read_csv(path + 'text-label.csv', encoding='utf8', index_col=False)

data_text = data['text']
data_label = data['label'].drop_duplicates()
print("Labels ", data_label[0])
print('Text sample: ', data_text[0])

# Applying count vectorizer
text = data[data['label'] == data_label[0]]['text']

count = CountVectorizer()
count_vec = count.fit_transform(text.values.astype('U'))
print("Counting shape: ", count_vec.shape)

# Applying TF_IDF transform

tf_idf = TfidfTransformer()
tf_idf_vec = tf_idf.fit_transform(count_vec)
print("TF-IDF shape: ", tf_idf_vec.shape)

