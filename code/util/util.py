import math
import re
import sys
import time
from collections import Counter

import pandas as pd
from nltk import RegexpTokenizer
import matplotlib.pyplot as plt
from spacy.symbols import punct


def draw_plot(params):
    x = params['x']
    y = params['y']

    plt.plot(x, y)
    plt.title(params['title'])
    plt.xticks(rotation=45)
    plt.xlabel(params['x_label'])
    plt.ylabel(params['y_label'])
    plt.figure(dpi=300)
    plt.show()


import pdb
from tensorflow.keras.preprocessing.text import Tokenizer


def get_tokens(document, stop_list=None):
    if stop_list is None:
        stop_list = ['.', '-', '!', '?', '', ',', 'Series([],']

    if document is None:
        raise ValueError("Document is None")

    tokenizer = RegexpTokenizer(r'\w+', flags=re.UNICODE)
    tokens = tokenizer.tokenize(' '.join(document).lower())

    tokens = list(filter(lambda x: type(x) == str, tokens))

    tokens = [token for token in tokens if token not in stop_list]

    return tokens


def stem_document(document, stems):
    document = pd.Series(document)

    for stem in stems:
        document.replace(to_replace=stem + r'\w+', value=stem, regex=True)

    return document.get(0)


def prep_text_to_stem(text):
    """
    Remove partes indesejadas como números e palavras na stop_list. Além disso adicionar # ao final da
    palavra a fim de facilitar no stems de uma única letra
    :param text:
    :return:
    """
    text = list(filter(lambda x: type(x) == str, text))

    tokenizer = RegexpTokenizer(r'\w+', flags=re.UNICODE)
    tokens = tokenizer.tokenize(' '.join(text).lower())

    new_tokens = []

    stop_list = Counter(tokens).most_common(300)
    stop_list = [tup[0] for tup in stop_list]
    stop_list.append('series([],')

    for token in tokens:
        if token not in stop_list:
            token = ''.join([letter for letter in token if not letter.isdigit()])
            for pun in punct:
                token.replace(pun, '')
            new_token = token + '#'
            new_tokens.append(new_token)

    return ' '.join(new_tokens)


def format_result(result):
    x, y = [], []
    for res in result:
        y.append(res[0])
        x.append('{0:.2f}%'.format(res[1]))
    return x, y


def letter_count(token, count_dic):
    for l in token[:-1]:

        try:
            freq = count_dic.get(l.lower())
            freq += 1
            count_dic[l.lower()] = freq
        except TypeError:
            freq = 1
            count_dic[l.lower()] = freq
    return count_dic


def suffix_count(token, count_dic):
    for size in range(1, 8):

        if len(token) > size + 2:
            suffix = token[-size:-1]

            try:
                if suffix:
                    freq = count_dic.get(suffix)
                    freq += 1
                    count_dic[suffix] = freq
            except TypeError:
                freq = 1
                count_dic[suffix] = freq

    return count_dic


def calculate_freq(count_dic):
    freq_dic = {}
    total_count = sum(count_dic.values())

    for key, count in zip(count_dic.keys(), count_dic.values()):
        freq_dic[key] = count / total_count

    return freq_dic


def word_count(text, words):
    word_freq = {}
    for word in words:
        counting = len(list(re.findall(word, text)))
        word_freq[word] = counting

    return sorted(word_freq.items(), key=lambda tup: tup[1], reverse=True)


def animated_loading():
    chars = "/—\|"
    for char in chars:
        sys.stdout.write(char)
        time.sleep(.1)
        sys.stdout.flush()


def to_dic(list_tup):
    suffixes = []
    coherence = []

    for tup in list_tup:
        suffixes.append(tup[0])
        coherence.append(tup[1])

    suffix_dic = {
        'suffix': suffixes,
        'coherence': coherence
    }

    return suffix_dic


def coherence(suffix_data, letter_freq):
    suffixes = suffix_data[0]
    s_freq = suffix_data[1]
    ls_freq = 1
    for l in suffixes:
        l_freq = letter_freq.get(l.lower())
        ls_freq *= l_freq
    coh = s_freq * math.log((s_freq / ls_freq), 10)

    return coh


def save_stemms(stemmed_data, path):
    stems = []
    words = []
    sufix = []
    s_freq = []
    # (stem, word, sufix, freq)
    for list_tup in stemmed_data['signature'].values():

        for el in list_tup:
            stems.append(el[0])
            words.append(el[1])
            sufix.append(el[2])
            s_freq.append(el[3])

    df = pd.DataFrame({
        'stems.csv': stems,
        'words': words,
        'suffix': sufix,
        'freq': s_freq
    })

    df.to_csv(path, index=False)


def get_noise(new=None):
    if new is None:
        new = []
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
    # Remember create file pattern to remove undesired text
    noise.extend(new)
    return noise
