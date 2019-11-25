import math
import os, re
import threading
import pandas as pd
import matplotlib.pyplot as plt

from util.preprocess import PrepData, AutoStem
from language_clf import LangClf
from util.util import count_words, stem_text, draw_plot
import pprint
from nltk import RegexpTokenizer

# --- Tirando referências -----

path = r'../Resources/datasets/'
stems_path = r'../Resources/stems/'
data_list = os.listdir(path)
prep = PrepData(path)
dataset = prep.get_datasets()


def classify():
    trains = []
    testes = {}
    stems_list = os.listdir(stems_path)
    stems_dic = {}

    for name in stems_list:
        data = pd.read_csv(path + name, encoding='utf-8')
        train_data = data[data['Book'] < 44]['Scripture']
        test_data = data[data['Book'] >= 44]['Scripture']
        trains.append(train_data)
        testes[name.replace('.csv', '')] = test_data
        stems_dic[name.replace('.csv', '')] = pd.read_csv(stems_path + name, encoding='utf-8')

    prep = PrepData(path)
    lang_dic = prep.get_datasets()

    clf = LangClf(stems_dic, lang_dic)
    clf.fit(trains, stems_list)

    for key, txt in zip(testes.keys(), testes.values()):
        print('Predicting {}...'.format(key))
        clf.predict(txt)


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


def main():
    """for name in data_list:
        stem_words(name)"""
    # classify()
    stems_analysis()


if __name__ == "__main__":
    main()
