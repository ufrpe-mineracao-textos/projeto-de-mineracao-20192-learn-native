import os
import pandas as pd
from util.preprocess import PrepData, AutoStem
from util.util import count_words, stem_text


class LangClf:
    stem_dic = None
    lang_dic = None
    test_count = None
    train_count = None
    lang_count_dic = {}
    text_label = ''

    def __init__(self, stem_dic=None, lang_dic=None):
        if lang_dic is None:
            lang_dic = {}
        if stem_dic is None:
            stem_dic = {}
        self.stem_dic = stem_dic
        self.lang_dic = lang_dic
        self.lang_count_dic = {}
        self.text_label = ''

    def fit(self, train_texts, labels):

        for train_text, label in zip(train_texts, labels):
                   
            train_text = list(filter(lambda x: type(x) == str, train_text))
            stemmed_txt = stem_text(train_text, self.stem_dic[label.replace('.csv', '')])
            train_count = count_words(stemmed_txt)
            self.lang_count_dic[label.replace('.csv', '')] = train_count

    def predict(self, text, threshold=100):

        """
          Predicts the language of a given language
        :param text: The text that must be identified
        :param threshold: The threshold for the most frequent words to be selected
        :return: return the predicted language with its similarity.
        """
        sim_list = []
        for key in self.lang_count_dic.keys():

            # Preprocess the test text and get the words count
            stemmed_text = stem_text(text, self.stem_dic[key.replace('.csv', '')])
            words_count = count_words(stemmed_text)
            test_words = [w[0] for w in words_count]

            # Obtain the words count in the training set
            words_tup = self.lang_count_dic[key]
            train_words = [w[0] for w in words_tup]

            # Calculate the similarity
            similarity = 0
            for w_t in test_words:
                if w_t in train_words:
                    similarity += 1
                    
            sim_list.append((key.replace('.csv', ''), similarity / threshold))

        match = sorted(sim_list, key=lambda kv: kv[1], reverse=True)[0]

        return match[0].replace('.csv', ''), match[1]
