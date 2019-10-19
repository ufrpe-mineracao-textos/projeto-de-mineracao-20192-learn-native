from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from util import util
import pandas as pd
import numpy as np
import os
import re
import math
import string
import nltk




def count_words():
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


class autoStemm:

    path_dir = ''
    sufix_freq = {}
    letter_freq = {}

def freq_counter(self, text):

    
    raw_text = ' '.join(text).lower()

    tokenizer = RegexpTokenizer(r'\w+', flags=re.UNICODE)
    tokens = tokenizer.tokenize(raw_text)

    for token in tokens:

        for l in token:

            try:
                freq = self.letter_freq.get(l.lower())
                freq += 1
                self.letter_freq[l.lower()] = freq
            
            except TypeError:
                
                freq = 1
                self.letter_freq[l.lower()] = freq

        for size in range(1, 6):

            if len(token) > size+3:
                sufix = token[-size:]
                
                try:
                    freq = self.sufix_freq.get(sufix)
                    freq += 1
                    self.sufix_freq[sufix] = freq
                except TypeError:
                    freq = 1
                    self.sufix_freq[sufix] = freq

    def auto_stemm(self):

        signature = {}
