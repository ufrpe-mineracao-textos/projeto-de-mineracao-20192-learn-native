import os
import re
import math
import string
import nltk
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from util.preprocess import prepData, autoStemm

# --- Tirando referências -----


path = r'../datasets/'
data_list = os.listdir(path)
prep = prepData(path)

#prep.label_data('text-label.csv')
dataset = prep.get_datasets()
data = dataset[data_list[0]]
scripture = data['Scripture']

stem = autoStemm(scripture)

suf_freq = stem.freq_counter()


stem_text = stem.stem_words()

df = pd.DataFrame({
    'sufix':list(suf_freq.keys()),
    'count':list(suf_freq.values())
})



df.to_csv('sufix.csv', index=False)
