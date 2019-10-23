import os
import re
import math
import string
from itertools import combinations

import nltk
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from util.preprocess import prepData, AutoStemm

# --- Tirando referências -----

path = r'../datasets/'
data_list = os.listdir(path)
prep = prepData(path)

# prep.label_data('text-label.csv')
dataset = prep.get_datasets()
data = dataset[data_list[0]]
scripture = data['Scripture']

stem = AutoStemm(scripture)

suf_freq = stem.freq_counter()

stemmed_data = stem.stem_words()

stems = []
words = []
sufix = []
s_freq = []
# (stem, word, sufix, freq)
for el in stemmed_data.values():
    stems.append(el[0])
    words.append(el[1])
    sufix.append(el[2])
    s_freq.append(el[3])

df = pd.DataFrame({
    'stems': stems,
    'words': words,
    'sufix': sufix,
    'freq': s_freq
})

df.to_csv('stem.csv', index=False)
