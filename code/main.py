import math
import os, re
import threading
from sys import intern

import numpy as np
import pandas as pd
import sys
from util.preprocess import TextPreprocess, AutoStem, stem_words
from language_clf import LangClf
from util.util import get_tokens
import pprint
from nltk import RegexpTokenizer

# --- Tirando referências -----

path = r'../Resources/bibles/'
stems_path = r'../Resources/stems/stems.csv'

PATH = r'../out/table'


def load_data(threshold=4):
    """
     It loads the data according to a given threshold. The threshold will tell us the number of books to be loaded.
     Here the train, test, and stems are loaded.
    :param threshold: Tells the number of books to be loaded
    :return: the training, test and stem dictionary
    """

    trains = {}
    testes = {}

    threshold += 40
    stems_dic = pd.read_csv(stems_path, encoding='utf-8').dropna()

    for name in os.listdir(path):
        label = name.replace('.csv', '')
        data = pd.read_csv(path + name.replace(' ', ''), encoding='utf-8')
        trains[label] = data[data['Book'] < threshold]['Scripture']
        testes[label] = data[data['Book'] >= threshold]['Scripture']

    return stems_dic, trains, testes


def classify(threshold=4):
    """
    Makes the classification of all languages according to a threshold
    :return: two vectors with the text label, predicted, similarity of all predictions
    """

    print("Loading data...")
    stems_dic, trains, testes = load_data(threshold)  # Loads the data according to the established threshold in
    # terms of number of books
    print("Load done")

    clf = LangClf(stems_dic)
    print("Fitting the Classifier...")
    clf.fit(trains, testes)  # Fits the Classifier with the training and test set

    # clf.load_clf(pd.read_csv('top_ranked_words.csv', encoding='utf8'))
    print("Testing...")
    results = clf.test()
    print(results)

    print("Mean similarity: ", clf.get_mean_similarity())
    print("Standard Deviation similarity: ", clf.get_std_similarity())
    print("Accuracy: ", clf.get_accuracy())
    print("Mean train size: ", clf.get_train_mean_size())

    clf.get_test_plot()
    return sorted(results.values(), key=lambda tup: tup[1])


def main():
    for i in range(1, 5):
        classify(i)


if __name__ == "__main__":
    main()
