import math
import os, re
import threading
from sys import intern

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from util.preprocess import TextPreprocess, AutoStem, stem_words
from language_clf import LangClf
from util.util import count_words, stem_text, draw_plot, text_prep
import pprint
from nltk import RegexpTokenizer

# --- Tirando referências -----

path = r'Resources/bibles/'
stems_path = r'Resources/stems/stems.csv'

PATH = r'../out/table'


def load_data(threshold=4):
    """
     It loads the data according to a given threshold. The threshold will tell us the number of books to be loaded.
     Here the train, test, and stems are loaded.
    :param threshold: Tells the number of books to be loaded
    :return: the training, test and stem dictionary
    """
    print("Loading data...")
    trains = {}
    testes = {}

    threshold += 40
    stems_dic = pd.read_csv(stems_path, encoding='utf-8').dropna()

    for name in os.listdir(path):
        data = pd.read_csv(path + name.replace(' ', ''), encoding='utf-8')
        train_data = data[data['Book'] < threshold]['Scripture'].to_list()
        test_data = data[data['Book'] >= threshold]['Scripture'].to_list()

        label = name.replace('.csv', '')

        trains[label] = list(train_data)
        testes[label] = list(test_data)

    return stems_dic, trains, testes


def classify(threshold=4):
    """
    Makes the classification of all languages
    :return: two vectors with the text label, predicted, similarity of all predictions
    """

    prep = TextPreprocess(path)
    lang_dic = prep.get_datasets()  # Retrieve all data sets
    stems_dic, trains, testes = load_data(threshold)  # Loads the data according to the established threshold in
    # terms of number of books

    clf = LangClf(stems_dic)
    clf.fit(trains, testes)  # Fits the Classifier with the training and test set

    #clf.load_clf(pd.read_csv('top_ranked_words.csv', encoding='utf8'))
    results = clf.test()
    print(results)
    print("Mean similarity: ", clf.get_mean_similarity())
    print("Standard Deviation similarity: ", clf.get_std_similarity())
    print("Accuracy: ", clf.get_accuracy())
    print("Mean train size: ", clf.get_train_mean_size())
    
    return sorted(results.values(), key=lambda tup: tup[1])


def format_result(result):
    x, y = [], []
    for res in result:
        y.append(res[0])
        x.append('{0:.2f}%'.format(res[1]))
    return x, y


def get_results():
    """for name in data_list:
            text = pd.read_csv(path + name, encoding='utf8')['Scripture']
            stem_words(text, name)"""
    means = []
    stds = []

    # Test
    for threshold in range(1, 5):
        print('Threshold: ', threshold)
        print('-' * 20)
        result = classify(threshold)
        print()
        print('-' * 20)

    means = sorted(means, key=lambda tup: tup[1])
    stds = sorted(stds, key=lambda tup: tup[1])
    x, y = format_result(means)

def main():
    classify(4)


if __name__ == "__main__":
    main()
