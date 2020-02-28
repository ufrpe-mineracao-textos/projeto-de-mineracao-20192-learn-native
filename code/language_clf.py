import os
from collections import Counter
from sys import intern
import threading
import pandas as pd
from util.preprocess import TextPreprocess, AutoStem
from util.util import count_words, stem_text
from multiprocessing.pool import ThreadPool, Pool
import numpy as np
import sys


def check_if_inside(set_1, set_2):
    # Calculate the similarity
    similarity = 0
    for w_t in set_2:
        if w_t in set_1:
            similarity += 1

    return similarity / len(set_1)


class LangClf:
    stem_dic = None
    test_texts = None
    test_count = None
    train_text = None
    lang_count_dic = {}
    top_ranked_words = {}
    test_results = {}
    hits = 0

    def __init__(self, stem_dic=None, words_threshold=100):

        """
        :param stem_dic: The dictionary of stems
        :param words_threshold: The threshold for the most frequent words to be selected in the training process
        """

        if stem_dic is None:
            stem_dic = {}

        self.hits = 0
        self.train_text = {}
        self.stem_dic = stem_dic
        self.top_ranked_words = {}
        self.test_results = {}
        self.train_labels = []
        self.test_labels = []
        self.test_texts = {}
        self.words_threshold = words_threshold

    def fit(self, train_texts, test_texts):
        """
        Receives the training set and count the words
        :param train_texts: The texts dictionary that will feed our classifier
        key:Language value: text
        :return:
        """
        self.test_texts = test_texts
        self.train_texts = train_texts
        for train_text, label in zip(self.train_texts.values(), self.train_texts.keys()):
            text = list(filter(lambda x: type(x) == str, train_text))
            stemmed_txt = stem_text(text, self.stem_dic[label])
            train_count = count_words(stemmed_txt, self.words_threshold)
            self.train_text[label] = stemmed_txt
            self.top_ranked_words[label] = [word[0] for word in train_count]

    def get_lang_count(self):
        return self.lang_count_dic

    def load_clf(self, train_data):
        self.top_ranked_words = train_data

    def check_similarity(self, params):

        train_lang = params[0]
        document = params[1]
        stemmed_text = stem_text(document, self.stem_dic[train_lang])
        words_count = count_words(stemmed_text, len(self.top_ranked_words[train_lang]))
        test_words = [w[0] for w in words_count]
        # Obtain the words count in the training set
        match = check_if_inside(self.top_ranked_words[train_lang], test_words)

        return train_lang, match

    def predict(self, document):

        """
          Predicts the language of a given language
        :param document:
        :return: return the predicted language with its similarity.
        """
        training_list = []
        params_list = []
        # Test procedure

        for train_lang in self.train_text.keys():
            params_list.append((train_lang, document))

        p = Pool(5)
        similarity_list = p.map(self.check_similarity, params_list)

        most_similar = sorted(similarity_list, key=lambda tup: tup[1], reverse=True)[0]

        return most_similar

    def test(self):

        test_list = []

        results_array = []
        for original, txt in zip(self.test_texts.keys(), self.test_texts.values()):
            print("Predicting :", original)
            predicted = self.predict(txt)
            print("Predicted: ", predicted)
            if sys.intern(predicted[0]) is sys.intern(original):
                self.hits += 1
            self.test_results[original] = predicted

        return self.test_results

    def get_test_results(self):
        return self.test_results

    def get_test_mean_size(self):
        """
        :return:The mean size of the test set in terms of number of words
        """
        sizes = []
        for text in self.test_texts.values():
            sizes.append(len(text.split()))

        return np.mean(sizes)

    def get_train_mean_size(self):
        """
        :return: The mean size of the training set in terms of number of words
        """
        sizes = []
        for text in self.train_text.values():
            sizes.append(len(text.split()))

        return np.mean(sizes)

    def get_mean_similarity(self):
        """
        :return: The mean of similarity between of the test and training set of top words
        """
        similarities = [tup[1] for tup in self.test_results.values()]
        return np.mean(similarities)

    def get_std_similarity(self):
        """
        :return: The Standard Deviation of similarity between of the test and training set of top words
        """
        similarities = [tup[1] for tup in self.test_results.values()]
        return np.std(similarities)

    def save_clf(self):
        df = pd.DataFrame(self.top_ranked_words)
        df.to_csv('top_ranked_words.csv', index=False)

    def get_accuracy(self):
        return self.hits / len(self.test_texts.keys())

    def get_scatter(self):
        pass
