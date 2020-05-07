import sys
from collections import Counter
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util.util import get_tokens
from util.util import stem_document

Imag_path = './images/'


class LangClf:
    stem_dic = None
    test_documents = None
    number_train_words = None
    train_documents = None
    lang_count_dic = {}
    train_recurrent_words = {}
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
        self.train_documents = {}
        self.test_documents = {}
        self.stem_dic = stem_dic
        self.train_recurrent_words = {}
        self.test_results = {}
        self.words_threshold = words_threshold
        self.number_train_words = []

    def extract_features(self, params):

        """
         Creates the profile features of training document obtaining the
         100 most frequent words.
        :param params: A tuple where the first item is the document label and second the training corpus
        :return: a tuple of (label, list of 100 most frequent words)
        """
        train_document = params[1]
        label = params[0]
        tokens = get_tokens(train_document)

        train_document = stem_document(' '.join(tokens), self.stem_dic[label])

        tokens = train_document.split(' ')
        words_count = Counter(tokens)
        self.number_train_words.append((label, sum(words_count.values())))
        return label, words_count.most_common(self.words_threshold)

    def fit(self, train_documents, test_documents):
        """
        Receives the training set and count the words
        :param test_documents:
        :param train_documents: The texts dictionary that will feed our classifier
        key:Language value: text
        :return: nothing
        """
        self.test_documents = test_documents
        self.train_documents = train_documents
        p = Pool(5)

        self.train_recurrent_words = dict(p.map(self.extract_features, list(self.train_documents.items())))

    def get_lang_count(self):
        return self.lang_count_dic

    def load_clf(self, train_data):
        self.train_recurrent_words = train_data

    def check_similarity(self, params):
        """
            Checks the similarity between the training dataset train_lang and the test_words data_set
        :param params: (train_lang, list of test words)
        :return:
        """
        train_lang = params[0]
        test_words = params[1]

        # Obtain the words count in the training set
        train_words = [tup[0] for tup in self.train_recurrent_words[train_lang]]

        # Calculate the similarity

        similarity = 0
        for w_t in train_words:
            if w_t in test_words:
                similarity += 1

        match = similarity / len(train_words)

        return train_lang, match

    def predict(self, document):

        """
          Predicts the language of a given language
        :param document:
        :return: return the predicted language with its similarity.
        """

        similarity_list = []

        # Test procedure

        for train_lang in self.train_recurrent_words.keys():
            test_recurrent_words = self.extract_features((train_lang, document))
            test_words = [tup[0] for tup in test_recurrent_words[1]]

            similarity = self.check_similarity((train_lang, test_words))
            similarity_list.append(similarity)

        most_similar = sorted(similarity_list, key=lambda tup: tup[1], reverse=True)[0]

        return most_similar

    def _run_test(self, params):

        original = params[0]
        text = params[1]
        predicted = self.predict(text)

        self.test_results[original] = predicted

        return original, predicted

    def test(self):

        test_list = []

        p = Pool(3)
        for original, txt in zip(self.test_documents.keys(), self.test_documents.values()):
            test_list.append((original, txt))

        self.test_results = dict(p.map(self._run_test, test_list))
        return self.test_results

    def get_test_results(self):
        return self.test_results

    def get_test_mean_size(self):
        """
        :return:The mean size of the test set in terms of number of words
        """
        sizes = []
        for text in self.test_documents.values():
            sizes.append(len(text.split()))

        return np.mean(sizes)

    def get_train_mean_size(self):
        """
        :return: The mean size of the training set in terms of number of words
        """
        sizes = []
        for text in self.train_documents:
            sizes.append(len(text))
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
        df = pd.DataFrame(self.train_recurrent_words)
        df.to_csv('top_ranked_words.csv', index=False)

    def get_accuracy(self):
        for predicted, original in zip(self.test_results.values(), self.test_results.keys()):
            if sys.intern(predicted[0]) is sys.intern(original):
                self.hits += 1

        return self.hits / len(self.test_documents.keys())

    def get_test_plot(self, title='Test Plot'):

        x = [tup[1] for tup in self.test_results.values()]
        x.sort(reverse=True)
        plt.barh(list(self.test_results.keys()), x)
        plt.title(title)
        plt.xticks(rotation=45)
        filename = title.lower() + str(self.get_mean_similarity()) + '.pdf'
        plt.savefig(Imag_path+filename, dpi=600)

    def save_results(self):
        file = open('test_results.txt', 'a')
        file.write("Threshold: " + str(self.words_threshold) + '\n')
        file.write('-' * 40)
        file.write("\nMean similarity: " + str(self.get_mean_similarity()))
        file.write("\nStandard Deviation similarity: " + str(self.get_std_similarity()))
        file.write("\nAccuracy: " + str(self.get_accuracy()))
        file.write("\nMean train size: " + str(self.get_train_mean_size()))
        file.write('-' * 40 + '\n')
        file.close()
