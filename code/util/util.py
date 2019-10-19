
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import matplotlib.pyplot as plt
import random
import re
import numpy as np
from numpy.random.mtrand import shuffle
from pandas import Series
import os
from nltk import RegexpTokenizer

__name__ = "preprocess.py"

def get_noise(new=[]):
    livros = []
    penta = ['Gênesis', 'Êxodo', 'Levítico', 'Números', 'Deuteronômio']

    history = ['Josué', 'Juízes', 'Rute', '1 e 2 Samuel', '1 e 2 Reis', '1 e 2 Crônicas', 'Esdras', 'Neemias',
               'Tobias',
               'Judite', 'Ester', '1 e 2 Macabeus']

    poete = ['Jó', 'Salmo', 'Provérbios', 'Eclesiastes', 'Cântico dos Cânticos', 'Sabedoria', 'Eclesiástico']

    profe = ['Isaías', 'Jeremias', 'Lamentações', 'Baruc', 'Ezequiel', 'Daniel', 'Oséias', 'Joel', 'Amós', 'Abdias',
             'Jonas', 'Miquéias', 'Naum', 'Habacuque', 'Sofonias', 'Ageu', 'Zacarias', 'Malaquias']

    evan = ['Mateus', 'Marcos', 'Lucas', 'João']

    cartas = ['Atos', 'Romanos', '1 e 2 Coríntios', 'Gálatas', 'Efésios', 'Filipenses', 'Colossenses',
              '1 e 2 Tessalonicenses', '1 e 2 Timóteo', 'Tito', 'Filemon', 'Hebreus', 'Tiago', '1 e 2 Pedro',
              '1 a 3 João', 'Judas', 'Apocalipse']

    livros.extend(penta)
    livros.extend(history)
    livros.extend(poete)
    livros.extend(profe)
    livros.extend(evan)
    livros.extend(cartas)
    noise = []

    for livro in livros:
        livro += ' [0-9]*[.]*[0-9]*[-]*[0-9]*[;]*'
        livro = livro.replace('1 e 2', '[0-9]')
        livro = livro.replace('1 a 3', '[0-9]')
        noise.append(livro)

    noise.append(r'[0-9][0-9]*\s[-]\s[0-9][0-9]*')
    noise.append(r'[-][0-9][0-9]*')
    noise.append(r'[0-9][0-9]*[-][0-9][0-9]*')
    noise.append(r'<sup>[(][0-9]*[-][0-9]*[)]<[/]sup>')
    noise.append(r'<.*?>\w+\s[0-9]*[.][0-9]*[-]*[0-9]*<.*?>[,;]*')
    noise.append(r'<.*?>[0-9]*[.][0-9]*[-]*[0-9]*<.*?>[,;]*')
    noise.append(r'\w+\s[0-9][0-9]*[.][0-9][0-9]*')
    noise.append(r'<.*?>\w*[0-9]*[.]*[0-9]*[-]*[0-9]*[,;]*<.*?>')
    noise.append(r'<.*?>[,;]*')
    noise.append(r'[;]\s[)]')
    noise.append(r'[()]')
    noise.append(r'Series[(,\s)],\s[()]')
    noise.append(r'[(]\s[0-9]*\s[-]\s[0-9]*\s[)]')
    noise.append('Veja verso [0-9]*')
    noise.append(r'Series\(\[\], \)')
    noise.append(r'\\\\')
    noise.append(r'\[*\]')
    noise.append(r'\{*\}')
    noise.extend(new)
    return noise


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
