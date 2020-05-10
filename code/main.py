import itertools
import os
import re
import sre_constants
import time
import matplotlib.colors as mcolors
import pandas as pd
import somoclu
import random
from language_clf import LangClf

# --- Tirando referências -----

path = r'../Resources/bibles/'
stems_path = r'../Resources/stems/stems.csv'


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
        label = name.split()[0]
        data = pd.read_csv(path + name, encoding='utf-8').dropna()
        trains[label] = data[data['Book'] < threshold]['Scripture']
        testes[label] = data[data['Book'] >= threshold]['Scripture']

    return stems_dic, trains, testes


def classify(threshold=4):
    """
    Makes the classification of all languages according to a threshold
    :return: two vectors with the text label, predicted, similarity of all predictions
    """
    start = time.time()  # initial time

    stems_dic, trains, testes = load_data(threshold)  # Loads the data according to the established threshold in
    # terms of number of books

    clf = LangClf(stems_dic)

    clf.fit(trains, testes)  # Fits the Classifier with the training and test set

    # clf.load_clf(pd.read_csv('top_ranked_words.csv', encoding='utf8'))

    results = clf.test()

    time_taken = (time.time() - start)
    mean = clf.get_mean_similarity()
    std = clf.get_std_similarity()
    accuracy = clf.get_accuracy()
    train_mean_size = clf.get_train_mean_size()
    clf.run_som()

    print("Threshold: ", threshold)
    print("Mean similarity: {:.5f}".format(mean))
    print("Standard Deviation similarity: {:.5f}".format(std))
    print("Accuracy: ", accuracy)
    print("Mean train size: {:.5f} words".format(train_mean_size))
    print("Final time: {:.5f}secs".format(time_taken))
    print(results)
    print()
    print('-' * 40)
    print()
    # clf.get_test_plot()
    return {
        "mean": mean,
        "std": std,
        "accuracy": accuracy,
        "train_mean_size": train_mean_size,
        "time_taken": time_taken
    }


def run_experiment():
    results_tup = []
    for i in range(1, 2):
        result = classify(i)
        results_tup.append((i, result["mean"], result["std"],
                            result["accuracy"], result["train_mean_size"],
                            result['time_taken']))


def main():
    run_experiment()


if __name__ == "__main__":
    main()
