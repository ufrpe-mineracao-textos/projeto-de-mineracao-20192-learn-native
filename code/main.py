import os
import time

import pandas as pd
from sklearn.metrics import accuracy_score

from language_clf import LangClf
import pdb

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

    X_train, y_train = [], []
    X_test, y_test = [], []

    threshold += 40
    stems_dic = pd.read_csv(stems_path, encoding='utf-8').dropna()

    for name in os.listdir(path):
        label = name.split()[0]
        data = pd.read_csv(path + name, encoding='utf-8').dropna()
        X_train.append(data[data['Book'] < threshold]['Scripture'].to_numpy())
        y_train.append(label)
        X_test.append(data[data['Book'] >= threshold]['Scripture'].to_numpy())

    y_test = y_train
    return stems_dic, X_train, y_train, X_test, y_test


def classify(threshold=4):
    """
    Makes the classification of all languages according to a threshold
    :return: two vectors with the text label, predicted, similarity of all predictions
    """
    start = time.time()  # initial time

    stems_dic, X_train, y_train, X_test, y_test = load_data(
        threshold)  # Loads the data according to the established threshold in
    # terms of number of books

    clf = LangClf(stems_dic)

    clf.fit(X_train, y_train)  # Fits the Classifier with the training and test set

    y_pred = clf.test(X_test)

    time_taken = (time.time() - start)

    accuracy = accuracy_score(y_test, y_pred)

    print("Threshold: ", threshold)
    print("Accuracy: ", accuracy)
    print("Final time: {:.5f}min".format(time_taken / 60))
    clf.run_som()

    print()
    print('-' * 40)
    print()
    # clf.get_test_plot()
    return {"Threshold": threshold,
            "Accuracy": accuracy,
            "Time": time_taken / 60}


def run_experiment():
    for i in range(1, 2):
        classify(i)


def main():
    run_experiment()


if __name__ == "__main__":
    main()
