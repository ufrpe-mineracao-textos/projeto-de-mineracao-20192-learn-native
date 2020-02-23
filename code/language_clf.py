import os
import pandas as pd
from util.preprocess import TextPreprocess, AutoStem
from util.util import count_words, stem_text


class LangClf:
    stem_dic = None
    lang_dic = None
    test_count = None
    train_count = None
    lang_count_dic = {}
    text_label = ''
    words_threshold = 100

    def __init__(self, stem_dic=None, lang_dic=None, threshold=100):

        """
        :param stem_dic:
        :param lang_dic:
        :param threshold: The threshold for the most frequent words to be selected
        """

        if lang_dic is None:
            lang_dic = {}
        if stem_dic is None:
            stem_dic = {}
        self.stem_dic = stem_dic
        self.lang_dic = lang_dic
        self.lang_count_dic = {}
        self.text_label = ''
        self.words_threshold = threshold

    def fit(self, train_texts):
        """
        Receives the training set and count the words
        :param train_texts: The text that will feed our classifier

        :return:
        """
        for train_text, label in zip(train_texts, self.lang_dic.keys()):
            train_text = list(filter(lambda x: type(x) == str, train_text))
            stemmed_txt = stem_text(train_text, self.stem_dic[label])
            train_count = count_words(stemmed_txt, self.words_threshold)
            self.lang_count_dic[label] = train_count

    def predict(self, text):

        """
          Predicts the language of a given language
        :param text: The text that must be identified
        :return: return the predicted language with its similarity.
        """

        # Stems the new Text
        sim_list = []
        auto_stem = AutoStem(text)
        auto_stem.stem_words()
        selected = auto_stem.select_stem()

        # Preprocess the test text and get the words count
        stemmed_text = stem_text(text, selected)
        words_count = count_words(stemmed_text)
        test_words = [w[0] for w in words_count]

        for key in self.lang_count_dic.keys():

            # Obtain the words count in the training set
            words_tup = self.lang_count_dic[key]
            train_words = [w[0] for w in words_tup]

            # Calculate the similarity
            similarity = 0
            for w_t in test_words:
                if w_t in train_words:
                    similarity += 1

            sim_list.append((key.replace('.csv', ''), similarity / self.words_threshold))

        match = sorted(sim_list, key=lambda kv: kv[1], reverse=True)[0]

        return match[0].replace('.csv', ''), match[1]
