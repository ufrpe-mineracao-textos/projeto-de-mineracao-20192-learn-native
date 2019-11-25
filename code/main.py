import math
import os, re
import threading
from sys import intern

import pandas as pd
import matplotlib.pyplot as plt

from util.preprocess import PrepData, AutoStem, stem_words
from language_clf import LangClf
from util.util import count_words, stem_text, draw_plot, text_prep
import pprint
from nltk import RegexpTokenizer

# --- Tirando referências -----

path = r'../Resources/datasets/'
stems_path = r'../Resources/stems/'
data_list = os.listdir(path)
prep = PrepData(path)
dataset = prep.get_datasets()


def classify():
    """
    Makes the classification of all languages
    :return: two vectors with the text label, predicted, similarity of all predictions
    """
    trains = []
    testes = {}
    stems_list = os.listdir(stems_path)
    stems_dic = {}
    threshold = 44
    data_sizes = []
    for name in stems_list:
        data = pd.read_csv(path + name, encoding='utf-8')
        train_data = data[data['Book'] < threshold]['Scripture']
        test_data = data[data['Book'] >= threshold]['Scripture']

        train_words = text_prep(train_data)
        test_words = text_prep(test_data)

        total = len(train_words) + len(test_words)
        print('Train size: ', 100 * (len(train_words) / total))

        label = name.replace('.csv', '')
        data_sizes.append((label, len(train_words), len(test_words)))
        trains.append(train_data)
        testes[label] = test_data
        stems_dic[label] = pd.read_csv(stems_path + name, encoding='utf-8')

    prep = PrepData(path)
    lang_dic = prep.get_datasets()

    clf = LangClf(stems_dic, lang_dic)
    clf.fit(trains, stems_list)

    results = []
    hits = 0
    for key, txt in zip(testes.keys(), testes.values()):
        print('Predicting {}...'.format(key))
        predicted, similarity = clf.predict(txt)
        print('Match: {} with similarity: {}%'.format(predicted, similarity))
        if intern(key) is intern(predicted):
            hits += 1

        results.append((key, predicted, similarity))

    return sorted(results, key=lambda tup: tup[2]), data_sizes


def stems_analysis():
    stems = []
    for label in os.listdir(stems_path):
        size = len(pd.read_csv(stems_path + label))
        stems.append((label.replace('.csv', ''), size))

    stems = sorted(stems, key=lambda tup: tup[1])

    labels = []
    sizes = []

    for tup in stems:
        labels.append(tup[0])
        sizes.append(tup[1])

    draw_plot({'x': sizes, 'y': labels})


def format_result(result):
    x, y = [], []
    for res in result:
        y.append(res[0])
        x.append('{}%'.format(res[2]))

    return x, y


def main():
    """for name in data_list:
        text = pd.read_csv(path + name, encoding='utf8')['Scripture']
        stem_words(text, name)"""
    result, data_sizes = classify()
    data_sizes = sorted(data_sizes, key=lambda tup: tup[1])

    x, y = format_result(result)
    draw_plot({'x': x, 'y': y})
    stems_analysis()


if __name__ == "__main__":
    main()
