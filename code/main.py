import math
import os, re
import threading
from sys import intern

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util.preprocess import PrepData, AutoStem, stem_words
from language_clf import LangClf
from util.util import count_words, stem_text, draw_plot, text_prep
import pprint
from nltk import RegexpTokenizer

# --- Tirando referências -----

path = r'../Resources/bibles/'
stems_path = r'../Resources/stems/'
data_list = os.listdir(path)
prep = PrepData(path)
dataset = prep.get_datasets()
PATH = r'../out/table'


def init_table():
    file = open(PATH, 'w')
    header = r"\begin{center}" + "\n" + r"\begin{tabular}{ ||c c c c|| }" + '\n'
    header += r'\hline' + '\n'
    header += r'Mean Number of Words & Mean Word Match & Match Standard Deviation & Hits\\ [0.5ex]' + '\n'
    header += r'\hline\hline' + '\n'
    file.write(header)


def add_to_table(values):
    file = open(PATH, 'a')
    row = ''
    for el in values:
        row += '{0:.2f} & '.format(el)
    row = row[:-1] + '\\' + '\\'
    file.write(row)
    file.write('\n')
    file.close()


def close_table():
    file = open(PATH, 'a')
    bottom = r'\hline' + '\n' + r'\end{tabular}' + r'\end{center}'
    file.write(bottom)
    file.close()


def load_data(threshold=44):
    trains = []
    testes = {}
    stems_list = os.listdir(stems_path)
    stems_dic = {}

    for name in stems_list:
        data = pd.read_csv(path + name, encoding='utf-8')
        train_data = data[data['Book'] < threshold]['Scripture']
        test_data = data[data['Book'] >= threshold]['Scripture']

        label = name.replace('.csv', '')

        trains.append(train_data)
        testes[label] = test_data
        stems_dic[label] = pd.read_csv(stems_path + name, encoding='utf-8')

    return stems_dic, trains, stems_list, testes


def get_mean_std(train_data):
    train_sizes = []
    for data in train_data:
        train_words = text_prep(data)
        train_sizes.append(len(train_words))

    return np.mean(train_sizes), np.std(train_sizes)


def classify(threshold=44):
    """
    Makes the classification of all languages
    :return: two vectors with the text label, predicted, similarity of all predictions
    """

    prep = PrepData(path)
    lang_dic = prep.get_datasets()
    stems_dic, trains, stems_list, testes = load_data(threshold)

    clf = LangClf(stems_dic, lang_dic)
    clf.fit(trains, stems_list)

    results = {}
    hits = 0
    for key, txt in zip(testes.keys(), testes.values()):
        print()
        print('Predicting {}...'.format(key))
        predicted, similarity = clf.predict(txt)
        print('Prediction: {} Match: {}%'.format(predicted, similarity))
        if intern(key) is intern(predicted):
            hits += 1
        print('-' * 40)
        results[key] = (predicted, similarity)

    mean_size, std_size = get_mean_std(trains)

    mean_res = np.mean([tup[1] for tup in results.values()])
    std_res = np.std([tup[1] for tup in results.values()])

    print("Result: Mean: {} Standard Deviation {}".format(mean_res, std_res))
    clf_data = {
        'means': (mean_size, mean_res),
        'stds': (std_size, std_res)
    }

    add_to_table([mean_size, mean_res, std_res, hits])
    return sorted(results.values(), key=lambda tup: tup[1]), clf_data


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

    draw_plot({'x': sizes, 'y': labels, "title": "Stem Analysis",
               'x_label': "Quantity of Stems",
               'y_label': "Language"
               })


def format_result(result):
    x, y = [], []
    for res in result:
        y.append(res[0])
        x.append('{0:.2f}%'.format(res[1]))
    return x, y


def main():
    """for name in data_list:
        text = pd.read_csv(path + name, encoding='utf8')['Scripture']
        stem_words(text, name)"""
    means = []
    stds = []
    init_table()
    for threshold in range(41, 45):

        print('Threshold: ', threshold-1)
        print('-' * 20)
        result, clf_data = classify(threshold)
        means.append(clf_data['means'])
        stds.append(clf_data['stds'])
        x, y = format_result(result)
        draw_plot({'x': x, 'y': y, 'title': "Similarity Plot",
                   'x_label': "Match Percentage",
                   'y_label': "Language"
                   })
        print()
        print('-' * 20)
    means = sorted(means, key=lambda tup: tup[1])
    stds = sorted(stds, key=lambda tup: tup[1])
    x, y = format_result(means)

    train_param = {'x': x,
                   'y': y,
                   'title': "Relation Between the Size and Performance",
                   'x_label': "Match Percentage",
                   'y_label': "Mean of Number of Words"}
    draw_plot(train_param)

    x, y = format_result(stds)
    draw_plot({'x': x,
               'y': y,
               'title': "Test Size",
               'x_label': "Match Percentage",
               'y_label': "Standard Deviation of Number of  Words"
               })

    stems_analysis()


if __name__ == "__main__":
    main()
