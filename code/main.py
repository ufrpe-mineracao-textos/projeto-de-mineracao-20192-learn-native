import os
import time

import pandas as pd
import matplotlib.pyplot as plt
from language_clf import LangClf, Imag_path

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


# plt.title("Mean Match rate Evolution")
# plt.xlabel("Number of books")
# plt.ylabel("Mean Match rate")
# plt.savefig(Imag_path + 'mean.pdf', dpi=600)
def main():
    results_tup = []
    for i in range(1, 5):
        result = classify(i)
        results_tup.append((i, result["mean"], result["std"],
                            result["accuracy"], result["train_mean_size"],
                            result['time_taken']))
    y_s = [tup[1] for tup in results_tup]
    x_s = [str(tup[0]) for tup in results_tup]


if __name__ == "__main__":
    main()
