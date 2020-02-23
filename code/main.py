import math
import os, re
import threading
from sys import intern

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util.preprocess import TextPreprocess, AutoStem, stem_words
from language_clf import LangClf
from util.util import count_words, stem_text, draw_plot, text_prep
import pprint
from nltk import RegexpTokenizer

# --- Tirando referências -----

path = r'../Resources/bibles/'
stems_path = r'../Resources/stems/'

PATH = r'../out/table'


def load_data(threshold=4):
    """
     It loads the data according to a given threshold. The threshold will tell us the number of books to be loaded.
     Here the train, test, and stems are loaded.
    :param threshold: Tells the number of books to be loaded
    :return: the training, test and stem dictionary
    """
    trains = []
    testes = {}

    stems_dic = {}
    threshold += 40
    for name in os.listdir(stems_path):
        data = pd.read_csv(path + name.replace(' ', ''), encoding='utf-8')
        train_data = data[data['Book'] < threshold]['Scripture']
        test_data = data[data['Book'] >= threshold]['Scripture']

        label = name.replace('.csv', '')

        trains.append(train_data)
        testes[label] = test_data
        stems_dic[label] = pd.read_csv(stems_path + name, encoding='utf-8')

    return stems_dic, trains, testes


def get_mean_std(train_data):
    train_sizes = []
    for data in train_data:
        train_words = text_prep(data)
        train_sizes.append(len(train_words))

    return np.mean(train_sizes), np.std(train_sizes)


def classify(threshold=4):
    """
    Makes the classification of all languages
    :return: two vectors with the text label, predicted, similarity of all predictions
    """

    prep = TextPreprocess(path)
    lang_dic = prep.get_datasets()  # Retrieve all data sets
    stems_dic, trains, testes = load_data(threshold)  # Loads the data according to the established threshold

    clf = LangClf(stems_dic, lang_dic)
    clf.fit(trains)  # Fits the Classifier with the training set the stems for each language

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
        results[key] = (predicted, similarity)  # Stores the prediction and similarity between test and training set.

    mean_size, std_size = get_mean_std(trains)

    mean_res = np.mean([tup[1] for tup in results.values()])
    std_res = np.std([tup[1] for tup in results.values()])

    print("Result: Mean: {} Standard Deviation {}".format(mean_res, std_res))
    clf_data = {
        'means': (mean_size, mean_res),
        'stds': (std_size, std_res)
    }

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


def get_results():
    """for name in data_list:
            text = pd.read_csv(path + name, encoding='utf8')['Scripture']
            stem_words(text, name)"""
    means = []
    stds = []

    # Test
    for threshold in range(1, 5):
        print('Threshold: ', threshold - 1)
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


def main():
    data_list = os.listdir(path)
    prep = TextPreprocess(path)
    datasets = prep.get_datasets()
    stems = {}

    for key, dataset in zip(datasets.keys(), datasets.values()):
        print("\nStemming: ", key)
        auto_stem = AutoStem(dataset['Scripture'])
        auto_stem.stem_words()
        selected = auto_stem.select_stem()
        stems[key.replace('.csv', '')] = selected

    print(stems)
    df = pd.DataFrame.from_dict(stems, orient='index')
    df = df.T

    df.to_csv(stems_path + 'stems.csv', index=False)


if __name__ == "__main__":
    main()
