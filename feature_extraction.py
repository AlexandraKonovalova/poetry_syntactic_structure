from readfiles import read_corpus, read_lines, read_verse, \
    read_meta, split_meta
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from math import log
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

import re

tqdm.pandas()

cols = ['NOUN_MOD', 'MOD_NOUN', 'MOD_ADV', 'ADV_MOD', 'VERB_ADV', 'ADV_VERB',
       'NOUN_GEN', 'GEN_NOUN', 'NOUN_GEN_MOD', 'NOUN_MOD_GEN', 'MOD_NOUN_GEN',
       'GEN_NOUN_MOD', 'GEN_MOD_NOUN', 'MOD_GEN_NOUN', 'MOD_NOUN_MOD']

# Constants

POS = {'NOUN': 'NOUN',
       'PRON': 'NOUN',
       'ADJ': 'MOD',
       'DET': 'MOD',
       'ANUM': 'MOD',
       'ADV': 'ADV',
       'VERB': 'VERB',
       'ADP': 'ADP'}

PARAMETERS = [('NOUN', 'MOD'),
              ('MOD', 'ADV'),
              ('VERB', 'ADV')]

DECADES = ['2_decades', '3_decades', '4_decades', '5_decades',
           '6_decades', '7_decades', '8_decades']
TWENTIES_INVERSE = [2001, 1981, 1961, 1941, 1921, 1901, 1881, 1861, 1841,
                     1821, 1801]

METERS = ["логаэд","гетерометрия", "тонический", "Пе", "Ан","Аф", "Дк", "Д", "Х", 'Я']
FEETS = ['вольная', 'регулярная']
RHYMES = ["охватная", "монорим", "скользящая", "спорадическая",
          "затянутая", "четверная", "тройная", "вольная",
          "нечетная", "четная", 'сложная', "регулярная",
          "цепная", "парная", "перекрестная"]
# Сбор данных


def normalize(corpus):
    """
    :param corpus: pandas object in conll-u format
    :return: corpus grouped by head-dependencies
    """
    corpus = corpus[corpus.UPOS.isin(pos.keys())]
    corpus['IS_GEN'] = corpus.FEATS.str.contains('Case=Gen')
    corpus['POS'] = corpus.UPOS.progress_apply(lambda x: pos[x])
    heads = corpus.groupby(['IDline', 'HEAD']).agg({'ID': list,
                             'FORM': list, 'POS': list,
                             'DEPREL': list, 'IS_GEN': list}).reset_index()
    result = pd.merge(heads, corpus[['IDline', 'ID', 'POS',
                                       'DEPREL', 'IS_GEN', 'FORM']],
                      left_on=['IDline', 'HEAD'],
                      right_on=['IDline', 'ID'],
                      suffixes=['_dep', '_head'])
    return result


def add_2grams(corpus, n):
    """
    :param corpus: pandas object
    :return: corpus with extracted bigrams features
    """

    for par in PARAMETERS:
        name = par[0] + '_' + par[1]
        name2 = par[1] + '_' + par[0]
        corpus[name] = np.zeros(n)
        corpus[name2] = np.zeros(n)

    for id_line, line in tqdm(corpus.iterrows()):
        # n_h is number of heads
        for n_el, el in enumerate(line.ID_dep):
            for par in PARAMETERS:
                if line.POS_head == par[0] and line.POS_dep[n_el] == par[1]:
                    # if head is first
                    if line.HEAD < el:
                        name = par[0] + '_' + par[1]
                        corpus[name][id_line] = 1
                    # if head is second
                    else:
                        name = par[1] + '_' + par[0]
                        corpus[name][id_line] = 1

    return corpus


def add_gen(corpus, n):
    """
    :param corpus: pandas object
    :return: corpus with extracted gen+noun\noun+gen
    """

    corpus['NOUN_GEN'] = np.zeros(n)
    corpus['GEN_NOUN'] = np.zeros(n)

    for id_line, line in tqdm(corpus.iterrows()):
        # n_h is number of heads
        for n_el, el in enumerate(line.ID_dep):
            if (line.POS_head == 'NOUN') \
            and (line.POS_dep[n_el] == 'NOUN') \
            and (line.IS_GEN_dep[n_el]):
                # if head is first
                if line.HEAD < el:
                    name = 'NOUN_GEN'
                    corpus[name][id_line] = 1
                # if head is second
                else:
                    name = 'GEN_NOUN'
                    corpus[name][id_line] = 1
    return corpus


def add_3grams_gen_1(corpus, n):
    """
    :param corpus: pandas object
    :return: corpus with extracted feature NOUN + GEN + MOD
    """
    corpus['NOUN_GEN_MOD'] = np.zeros(n)
    corpus['NOUN_MOD_GEN'] = np.zeros(n)

    # one head with dependent adverbial modifier and noun-genitive modifier
    for id_line, line in corpus[(corpus.NOUN_GEN == 1)
                                & (corpus.NOUN_MOD == 1)].iterrows():
        # n_h - number of heads
        is_gen = 0  # flag that genitive noun is found
        is_mod = 0  # flag that modifier is found
        if len(line.ID_dep) > 1:
            for n_el, el in enumerate(line.ID_dep):
                # genitive noun is first
                if line.POS_dep[n_el] == 'NOUN' and line.IS_GEN_dep[n_el]:
                    is_gen = 1
                if is_gen > is_mod:
                    corpus['NOUN_GEN_MOD'][id_line] = 1
                # modifier is first
                if line.POS_dep[n_el] == 'MOD':
                    is_mod = 1
                if is_mod > is_gen:
                    corpus['NOUN_MOD_GEN'][id_line] = 1
    return corpus


def add_3grams_gen_2(corpus, n):
    """
    :param corpus: pandas object
    :return: corpus with extracted feature MOD + NOUN + GEN
    """
    corpus['MOD_NOUN_GEN'] = corpus.NOUN_GEN * corpus.MOD_NOUN
    corpus['GEN_NOUN_MOD'] = corpus.GEN_NOUN * corpus.NOUN_MOD

    corpus['GEN_MOD_NOUN'] = np.zeros(n)
    corpus['MOD_GEN_NOUN'] = np.zeros(n)

    for id_line, line in tqdm(corpus[(corpus.GEN_NOUN == 1)
                                     & (corpus.MOD_NOUN == 1)].iterrows()):
        is_gen = 0
        is_mod = 0
        if len(line.ID_dep) > 1:
            for n_el, el in enumerate(line.ID_dep):
                # genitive noun is first
                if line.POS_dep[n_el] == 'NOUN' and line.IS_GEN_dep[n_el]:
                    is_gen = 1
                if is_gen > is_mod:
                    corpus['GEN_MOD_NOUN'][id_line] = 1
                # modifier is first
                if line.POS_dep[n_el] == 'MOD':
                    is_mod = 1
                if is_mod > is_gen:
                    corpus['MOD_GEN_NOUN'][id_line] = 1

    return corpus


def add_3grams_other(corpus):
    """
    :param corpus: pandas object
    :return: pandas object with MOD + NOUN + MOD feature
    """
    corpus['MOD_NOUN_MOD'] = corpus.MOD_NOUN * corpus.NOUN_MOD
    return corpus


def add_meta(corpus, lines, verses, meta):
    """
    :param corpus: corpus
    :param lines: word - line - doc
    :param verses: parameters of verse
    :param meta: corpus with meta info
    :return: corpus with merged features
    """
    lines = lines[['IDline', 'IDdoc']]
    lines = lines.drop_duplicates(subset=['IDline', 'IDdoc'])
    corpus = corpus.merge(lines, left_on='IDline', right_on='IDline')
    corpus = corpus.merge(verses[['IDline', 'LineText',
                                  'NoStrophe', 'Meter',
                                  'RhymedPattern']],
                          left_on='IDline',
                          right_on='IDline')
    corpus = corpus.merge(meta[['IDdoc', 'Author',
                                'Gender', 'YearBegin',
                                'YearEnd', 'Decade', 'Rhyme']],
                          left_on='IDdoc',
                          right_on='IDdoc')
    l = pd.Series(lines.groupby('IDline')['IDword'].count(), name='CountWords')
    corpus = corpus.merge(l, left_on='IDline', right_on='IDline')
    return corpus


def concat_corpora(corpus_19, corpus_20, meta):
    """
    :param corpus_19: pd-object with extracted features
    :param corpus_20: pd-object with extracted features
    :param meta: meta info
    :return: merged corpus for two centuries
    """
    df = pd.concat([corpus_19, corpus_20], ignore_index=True)
    df = df.merge(meta[['IDdoc', 'Meter', 'Feet', 'Strophe']],
                  left_on='IDdoc',
                  right_on='IDdoc')
    df = df[(df.Decade > 1800) & (df.Decade < 2001)]
    return df


def get_decade(decs, yb):
    """
    :param decs: value in column 'Decade'
    :param yb: value in column 'YearBegin'
    :return: list of decades for each year
    """
    decades_list = []
    for idx, dec in enumerate(decs):
        if dec in decades_dict:
            decades_list.append(yb[idx])
        else:
            decades_list.append(dec)

    decades_list = [(int(d)//10)*10 + 1 for d in decades_list]
    return decades_list


def get_twenty(dec):
    """
    :param dec: value in column 'Decade_num'
    :return: value for twenty year period
    """
    for i in TWENTIES_INVERSE:
        if dec >= i:
            return i


def idf(col):
    return log(N_doc/np.count_nonzero(col.to_numpy()))


def get_meter(string):
    for r in METERS:
        if re.search(r, string):
            return r
        else:
            continue


def get_feet(string):
    for r in FEETS:
        if re.search(r, str(string)):
            return r
    return str(string)[:2].strip('+,(# ')


def get_rhyme(string):
    for r in RHYMES:
        if re.search(r, string):
            return r
        return string


def half(label):
    if label < 2:
        return 0
    elif label < 5:
        return 1
    elif label < 7:
        return 2
    else:
        return 3


def binary(label):
    if label > 5:
        return 1
    else:
        return 0


def get_doc(df, line_sum):
    doc_groups = df.groupby('IDdoc')
    doc_sum = doc_groups[df.columns[12:27]].sum()
    doc_sum['CountWords'] = line_sum.groupby('IDdoc')['CountWords'].sum()
    for col in cols:
        name = col + '_norm'
        doc_sum[name] = doc_sum[col]/doc_sum.CountWords

    idfs = np.array([idf(doc_sum[col]) for col in cols])
    cols_norm = [i + '_norm' for i in cols]
    for i, col in enumerate(cols):
        name = col + '_norm'
        doc_sum[name] = doc_sum[name] * idfs[i]
    X = doc_sum.loc[:, cols_norm]
    scaler = MinMaxScaler()
    X['n_words'] = scaler.fit_transform(doc_sum.CountWords.to_numpy().reshape(-1, 1))
    X['Feet'] = doc_groups['Feet'].first()
    X['Meter'] = doc_groups['Meter'].first()
    X['Rhyme'] = doc_groups['Rhyme'].first()
    le = preprocessing.LabelEncoder()
    X['label'] = le.fit_transform(doc_groups['Decade_num'].first())
    X.dropna(inplace=True)
    X['Meter_new'] = X.Meter.progress_apply(get_meter)
    X['Feet_new'] = X.Feet.apply(get_feet)
    X['Rhyme_new'] = X.Rhyme.progress_apply(get_rhyme)
    X = pd.get_dummies(X, columns=['Feet_new', 'Meter_new', 'Rhyme_new'])
    X.drop(columns=['Feet', 'Meter', 'Rhyme'], inplace=True)
    X['half_class'] = np.zeros(X.shape[0])
    X['half_class'] = X.label.progress_apply(half)
    X['binary_class'] = np.zeros(X.shape[0])
    X['binary_class'] = X.label.progress_apply(binary)
    features = [col for col in X.iloc[:, :-2].columns if not col == 'label']
    features_all = [col + '_norm' for col in cols]
    # Только синтаксические признаки
    X_synt = X.loc[:, features_all]
    X_synt['label'] = X.label
    X_synt['binary_class'] = X.binary_class
    X_synt['half_class'] = X.half_class
    return X, X_synt, features, features_all


def main():
    c19 = read_corpus(input('Path to corpus of 19 century: '))
    c20 = read_corpus(input('Path to corpus of 20 century: '))
    lines = read_lines(input('Path to lines corpus: '))
    verses = read_verse(input('Path to verses corpus: '))
    meta = read_meta(input('Path to meta corpus: '))
    c19 = split_meta(c19, lines)
    c20 = split_meta(c20, lines)
    c20 = c20.drop_duplicates()


if __name__ == '__main__':
    main()
