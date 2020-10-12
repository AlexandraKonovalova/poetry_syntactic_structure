import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score
from readfiles import *
from feature_extraction import *
from plot_graphics import *
tqdm.pandas()


TWENTIES = [1801, 1821, 1841, 1861, 1881, 1901, 1921, 1941, 1961, 1981]


def train_predict(X, target, name):
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(columns=['label', 'half_class', 'binary_class']),
        X[target],
        random_state=17,
        test_size=0.25,
        stratify=target)
    clf = RandomForestClassifier()

    kfold = KFold(n_splits=5)
    grid = GridSearchCV(clf, cv=kfold,
                        param_grid=[{'n_estimators': [100, 200, 250, 300],
                                     'max_depth': [10, 20, 30]}])
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    clf = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                 max_depth=best_params['max_depth'])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    feats = {i: k for i, k in zip(clf.feature_importances_, X_train.columns)}

    fig, ax = plt.subplots(figsize=(20, 20))

    plot_confusion_matrix(clf, X_test, y_test,
                          display_labels=X_train.columns,
                          cmap=plt.cm.Greys,
                          normalize='true', ax=ax)
    plt.savefig('confusion' + 'name' + '.png',
                bbox_inches='tight')

    conf = confusion_matrix(y_test, y_pred)
    pd.DataFrame(conf, columns=X_train.columns,
                 index=X_train.columns)

    scores = pd.DataFrame(columns=[name],
                          index=X_train.columns)

    acc = []
    for i in range(0, X.label.nunique()):
        acc.append(conf[i][i] / conf[i].sum(axis=0))
    scores[name] = acc
    return accuracy, feats, scores


def main():
    c19 = read_corpus(input('Path to corpus of 19 century: '))
    c20 = read_corpus(input('Path to corpus of 20 century: '))
    lines = read_lines(input('Path to lines corpus: '))
    verses = read_verse(input('Path to verses corpus: '))
    meta = read_meta(input('Path to meta corpus: '))
    result19 = normalize(split_meta(c19, lines))
    result20 = normalize(split_meta(c20, lines).drop_duplicates())
    result19 = add_3grams_other\
        (add_3grams_gen_2(add_3grams_gen_1(add_gen(add_2grams(result19)))))
    result20 = add_3grams_other\
        (add_3grams_gen_2(add_3grams_gen_1(add_gen(add_2grams(result20)))))
    result19 = add_meta(result19, lines, verses, meta)
    result20 = add_meta(result20, lines, verses, meta)
    df = concat_corpora(result19, result20, meta)
    df['Decade_num'] = df.Decade_num.progress_apply(get_decade)
    df['Decade_num'] = df.Decade_num.progress_apply(get_twenty)
    line_sum, n_ng_per_year, n_words_per_year, indices = get_lines(df)
    X, X_synt, features, features_all = get_doc(df, line_sum)
    plot_all(n_ng_per_year)
    plot_size(n_words_per_year)
    plot_bigrams()
    plot_trigrams()
    plot_mnm()
    accuracy, feats, scores = train_predict(X, 'label', 'decades')
    accuracy_synt, feats_synt, scores_synt = train_predict(X_synt,
                                                                    'label',
                                                                    'decades')
    accuracy_half, feats_half, scores_half = train_predict(X, 'half_class',
                                                           'half')
    accuracy_half_synt, feats_half_synt, \
    scores_half_synt = train_predict(X_synt, 'half_class', 'half')
    accuracy_bin, feats_bin, scores_bin = train_predict(X,
                                                        'binary_class', 'bin')
    accuracy_bin_synt, feats_bin_synt, \
    scores_bin_synt = train_predict(X_synt,'binary_class', 'bin')
    plt.figure(figsize=(15, 5))
    plt.bar([i for i in feats_bin.values()], [i for i in feats_bin.keys()],
            color='maroon')
    plt.xticks(rotation=90)
    plt.title('Важность признаков для классификации по векам')
    plt.xlabel('Признак')
    plt.ylabel('Важность признака, доля')
    plt.ylim((0, 0.15))
    ax = plt.gca()
    ax.set_facecolor('#F5F5F5')
    plt.savefig('/content/drive/My Drive/Все признаки по векам.png',
                bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar([i for i in feats_bin_synt.values()],
            [i for i in feats_bin_synt.keys()],
            color='#E23A3A')
    plt.xticks(rotation=90)
    plt.title('Важность синтаксических признаков для классификации по векам')
    plt.xlabel('Признак')
    plt.ylabel('Важность признака, доля')
    plt.ylim((0, 0.15))
    ax = plt.gca()
    ax.set_facecolor('#F5F5F5')
    plt.savefig('/content/drive/My Drive/Синтаксические признаки по векам.png',
                bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.bar([i for i in feats_half.values()], [i for i in feats_half.keys()],
            color='#000980')
    plt.xticks(rotation=90)
    plt.title('Важность признаков для классификации по половине века')
    plt.xlabel('Признак')
    plt.ylabel('Важность признака, доля')
    plt.ylim((0, 0.15))
    ax = plt.gca()
    ax.set_facecolor('#F5F5F5')
    plt.savefig('/content/drive/My Drive/Все признаки по полувекам.png',
                bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar([i for i in feats_half_synt.values()],
            [i for i in feats_half_synt.keys()],
            color='#4751E1')
    plt.xticks(rotation=90)
    plt.title(
        'Важность синтаксических признаков для классификации по половине века')
    plt.xlabel('Признак')
    plt.ylabel('Важность признака, доля')
    plt.ylim((0, 0.15))
    ax = plt.gca()
    ax.set_facecolor('#F5F5F5')
    plt.savefig(
        '/content/drive/My Drive/Синтаксические признаки по полувекам.png',
        bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.bar([i for i in feats.values()], [i for i in feats.keys()],
            color='#026405')
    plt.xticks(rotation=90)
    plt.title('Важность признаков для классификации по двадцатилетиям')
    plt.xlabel('Признак')
    plt.ylabel('Важность признака, доля')
    plt.ylim((0, 0.15))
    ax = plt.gca()
    ax.set_facecolor('#F5F5F5')
    plt.savefig('/content/drive/My Drive/Все признаки по двадцатилетиям.png',
                bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar([i for i in feats_synt.values()], [i for i in feats_synt.keys()],
            color='#40BC44')
    plt.xticks(rotation=90)
    plt.title(
        'Важность синтаксических признаков для классификации по двадцатилетиям')
    plt.xlabel('Признак')
    plt.ylabel('Важность признака, доля')
    plt.ylim((0, 0.15))
    ax = plt.gca()
    ax.set_facecolor('#F5F5F5')
    plt.savefig(
        '/content/drive/My Drive/Синтаксические признаки по двадцатилетиям.png',
        bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(20, 5))

    plt.bar(range(0, 168, 3), [i for i in feats.keys()],
            color='maroon', label='По веку')
    plt.bar([i + 1 for i in range(0, 168, 3)], [i for i in feats_bin.keys()],
            color='#000980', label='По двадцатилетиям')
    plt.bar([i + 2 for i in range(0, 168, 3)], [i for i in feats_half.keys()],
            color='#026405', label='По половине века')
    plt.xticks(range(0, 168, 3), [i for i in feats.values()], rotation=90)
    plt.title('Важность всех признаков для классификации')
    plt.xlabel('Признак')
    plt.ylabel('Важность признака, доля')
    plt.ylim((0, 0.15))
    plt.legend()
    ax = plt.gca()
    ax.set_facecolor('#F5F5F5')
    plt.savefig('/content/drive/My Drive/Важность всех признаков.png',
                bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(20, 5))

    plt.bar(range(0, 48, 3), [i for i in feats_synt.keys()],
            color='#E23A3A', label='По веку')
    plt.bar([i + 1 for i in range(0, 48, 3)],
            [i for i in feats_bin_synt.keys()],
            color='#4751E1', label='По двадцатилетиям')
    plt.bar([i + 2 for i in range(0, 48, 3)],
            [i for i in feats_half_synt.keys()],
            color='#40BC44', label='По половине века')
    plt.xticks(range(0, 48, 3), [i for i in feats.values()], rotation=90)
    plt.title('Важность синтаксических признаков для классификации')
    plt.xlabel('Признак')
    plt.ylabel('Важность признака, доля')
    plt.ylim((0, 0.15))
    ax = plt.gca()
    ax.set_facecolor('#F5F5F5')
    plt.savefig(
        '/content/drive/My Drive/Важность синтаксических признаков.png',
        bbox_inches='tight')
    plt.legend(loc=(0.7, 0.7))
    plt.show()
    print(pd.concat([scores.T, scores_synt.T], ignore_index=False))

    print(pd.concat([scores_half.T, scores_half_synt.T], ignore_index=False))

    print(pd.concat([scores_bin, scores_bin_synt], ignore_index=False))


if __name__ == '__main__':
    main()
