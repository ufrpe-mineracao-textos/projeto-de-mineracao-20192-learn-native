import math
import sys
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import os, re


def count_words(text, threshold=100):
    """
    Retorna as palavras mais frequentes limitadas pelo threshold que
    é 100 por default.
    """
    word_freq = {}.fromkeys(set(text.split(' ')))
    text = text.split(' ')
    for word in text:
        try:
            word_freq[word] += 1
        except TypeError:
            word_freq[word] = 1

    word_freq.pop('')

    return sorted(word_freq.items(), key=lambda kv: kv[1], reverse=True)[:threshold]


def stem_text(text, stems):

    text = ' '.join(list(filter(lambda x: type(x) == str, text))).lower()
    
    for stem in stems:
        words = re.findall(stem + r'\w+', text)
        for word in words:
            text = text.replace(word, stem)
    return text


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


