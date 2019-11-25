import math
import os, re
import threading
import pandas as pd
from util.preprocess import PrepData, AutoStem
from language_clf import LangClf
from util.util import count_words, stem_text
import pprint
from nltk import RegexpTokenizer

# --- Tirando referências -----

path = r'../Resources/datasets/'
stems_path = r'../Resources/stems/'
data_list = os.listdir(path)
prep = PrepData(path)
dataset = prep.get_datasets()


def stem_words(name):
    data = pd.read_csv(path + name)
    label = name.replace('.csv', '')
    print('Processing: ', label)
    scripture = data['Scripture']

    stem = AutoStem(scripture)
    stem.freq_counter()

    stem.stem_words()
    data = {label: list(filter(lambda x: type(x) == str, stem.select_stem()))}

    df = pd.DataFrame(data)
    df.to_csv(stems_path + name, index=False)
    print()


def classify():
    trains = []
    testes = {}
    stems_list = os.listdir(stems_path)
    stems_dic = {}

    for name in stems_list:
        data = pd.read_csv(path + name, encoding='utf-8')
        train_data = data[data['Book'] < 44]['Scripture']
        test_data = data[data['Book'] >= 44]['Scripture']
        trains.append(train_data)
        testes[name.replace('.csv', '')] = test_data
        try:
            stems_dic[name.replace('.csv', '')] = pd.read_csv(stems_path + name, encoding='utf-8')
        except pd.errors.EmptyDataError:
            pass

    prep = PrepData(path)
    lang_dic = prep.get_datasets()

    clf = LangClf(stems_dic, lang_dic)
    clf.fit(trains, stems_list)

    for key, txt in zip(testes.keys(), testes.values()):
        print('Predicting {}...'.format(key))
        clf.predict(txt)


def main():
    for name in data_list:
        stem_words(name)
    classify()


if __name__ == "__main__":
    main()
