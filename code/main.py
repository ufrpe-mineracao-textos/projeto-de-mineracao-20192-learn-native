import os

import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from language_clf import LangClf
import pdb

# --- Tirando referências -----

path = r'../Resources/bibles/'
stems_path = r'../Resources/stems/stems.csv'
new_dir = r'../Resources/texts/'


def load_data(training_split=.8, number_documents=100):
    """
     It loads the data according to a given threshold. The threshold will tell us the number of books to be loaded.
     Here the train, test, and stems are loaded.
    :param number_documents: The total number of documents to be retrieved
    :param training_split: How you want to split them
    :return: the training set, testing set and stem dictionary
    """

    x_text, y_text = [], []

    stems_dic = pd.read_csv(stems_path, encoding='utf-8').dropna()
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    for filename in os.listdir(new_dir):
        file = open(new_dir + filename, encoding='utf-8')
        text = [line.strip('\n') for line in file.readlines()]

        text = text[:number_documents]
        train_size = int(len(text) * training_split)
        X_train.append(text[:train_size])
        y_train.append(filename.split('.')[0])
        X_test.append(text[train_size:])
        y_test.append(filename.split('.')[0])

    return stems_dic, np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def classify(threshold=4):
    """
    Makes the classification of all languages according to a threshold
    :return: two vectors with the text label, predicted, similarity of all predictions
    """
    start = time.time()  # initial time

    stems_dic, X_train, y_train, X_test, y_test = load_data(.1)  # Loads the data according to the established
    print('Train shape: ', X_train.shape)
    print('X_test shape: ', X_test.shape)
    # threshold in
    # terms of number of books
    clf = LangClf(stems_dic)
    print('Fitting the clf...')

    clf.fit(X_train, y_train)  # Fits the Classifier with the training and test set
    print('Testing...')
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
