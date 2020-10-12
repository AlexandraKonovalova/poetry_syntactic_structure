import feature_extraction
from feature_extraction import normalize, add_2grams, add_gen, \
    add_3grams_gen_1, add_3grams_gen_2, add_3grams_other, add_meta, \
    concat_corpora, get_decade, add_meta
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

tqdm.pandas()
TWENTIES = [1801, 1821, 1841, 1861, 1881, 1901, 1921, 1941, 1961, 1981]
BIGRAMS = [['NOUN_MOD', 'MOD_NOUN'], ['MOD_ADV', 'ADV_MOD'],
           ['VERB_ADV', 'ADV_VERB'], ['GEN_NOUN', 'NOUN_GEN']]
TRIGRAMS = [['NOUN_GEN_MOD','NOUN_MOD_GEN'],
            ['MOD_NOUN_GEN','GEN_NOUN_MOD'],
            ['GEN_MOD_NOUN','MOD_GEN_NOUN']]


def get_lines(df):
    line_groups = df.groupby('IDline')
    line_sum = line_groups[df.columns[12:27]].sum()
    line_sum['CountWords'] = line_groups['CountWords'].first()
    line_sum['Decade'] = line_groups['Decade_num'].first()
    line_sum['IDdoc'] = line_groups['IDdoc'].first()
    n_words_per_year = line_sum.groupby('Decade')['CountWords'].sum()
    n_ng_per_year = df.groupby('Decade_num')[df.columns[12:27]].sum().div\
        (n_words_per_year.to_numpy(), axis=0)
    indices = n_ng_per_year.index
    return line_sum, n_ng_per_year, n_words_per_year, indices


def plot_all(n_ng_per_year):
    n_ng_per_year.plot(figsize=(15, 5),
                       title='Распределение количества n-граммов во времени',
                       xticks=TWENTIES,
                       xlabel='Двадцатилетия',
                       ylabel='Доля от общего числа слов').legend(loc='center left',
                                                                  bbox_to_anchor=(1.0, 0.5))
    plt.grid(alpha=0.5)
    ax = plt.gca()
    ax.set_facecolor('#F5F5F5')
    plt.savefig('total_in_time.png')


def plot_size(n_words_per_year):
    n_words_per_year.plot.bar(figsize=(15, 5),
                              title='Размер корпуса по двадцатилетиям',
                              color='maroon',
                              xlabel='Двадцатилетия',
                              ylabel='Размер корпуса, токены')
    plt.yticks((500000, 1000000, 1500000, 2000000, 2500000),
               ('500 000', '1 000 000', '1 500 000', '2 000 000', '2500000'))
    ax = plt.gca()
    ax.set_facecolor('#F5F5F5')
    plt.savefig('corpus_size.png')
    plt.show()


def plot_bigrams():
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 10))
    orders = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for idx, pair in enumerate(BIGRAMS):
        i = orders[idx]
        ax[i[0]][i[1]].plot(indices, n_ng_per_year[pair[0]],
                            label=BIGRAMS[idx][0], c='maroon')
        ax[i[0]][i[1]].plot(indices, n_ng_per_year[pair[1]],
                            label=BIGRAMS[idx][1])
        ax[i[0]][i[1]].legend()
        ax[i[0]][i[1]].set_xlim((1801, 1991))
        ax[i[0]][i[1]].set_ylim((0, 0.07))
        title = 'Распределение ' + ' '.join(pair)
        ax[i[0]][i[1]].set_title(title)
        ax[i[0]][i[1]].set_xlabel('Двадцатилетия')
        ax[i[0]][i[1]].set_ylabel('Доля от общего числа слов')
        ax[i[0]][i[1]].set_xticks(indices)
        ax[i[0]][i[1]].grid(alpha=0.5)
        ax[i[0]][i[1]].set_facecolor('#F5F5F5')
        plt.savefig(title + '.png')

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 10))
    orders = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for idx, pair in enumerate(BIGRAMS):
        i = orders[idx]
        ax[i[0]][i[1]].plot(indices, n_ng_per_year[pair[0]],
                            label=BIGRAMS[idx][0], c='maroon')
        ax[i[0]][i[1]].plot(indices, n_ng_per_year[pair[1]],
                            label=BIGRAMS[idx][1])
        ax[i[0]][i[1]].legend()
        title = 'Распределение ' + ' '.join(pair) + ' (Приближение)'
        ax[i[0]][i[1]].set_title(title)
        ax[i[0]][i[1]].set_xlabel('Двадцатилетия')
        ax[i[0]][i[1]].set_ylabel('Доля от общего числа слов')
        ax[i[0]][i[1]].set_xticks(indices)
        ax[i[0]][i[1]].grid(alpha=0.5)
        ax[i[0]][i[1]].set_facecolor('#F5F5F5')
        plt.savefig(title + '.png')

def plot_trigrams():
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 5))

    for idx, pair in enumerate(TRIGRAMS):
        ax[idx].plot(indices, n_ng_per_year[pair[0]], label=TRIGRAMS[idx][0],
                     c='maroon')
        ax[idx].plot(indices, n_ng_per_year[pair[1]], label=TRIGRAMS[idx][1])

        ax[idx].legend()
        ax[idx].set_xlim((1801, 1991))
        ax[idx].set_ylim((0, 0.005))
        ax[idx].set_xticks(indices)
        title = 'Распределение ' + ' '.join(pair)
        ax[idx].set_title(title)
        ax[idx].set_xlabel('Двадцатилетия')
        ax[idx].set_ylabel('Доля от общего числа слов')
        ax[idx].set_xticks(indices)
        ax[idx].grid(alpha=0.5)
        ax[idx].set_facecolor('#F5F5F5')
        plt.savefig(title + '.png')

    for idx, pair in enumerate(TRIGRAMS):
        ax[idx].plot(indices, n_ng_per_year[pair[0]], label=TRIGRAMS[idx][0],
                     c='maroon')
        ax[idx].plot(indices, n_ng_per_year[pair[1]], label=TRIGRAMS[idx][1])
        ax[idx].legend()
        ax[idx].set_xticks(indices)
        title = 'Распределение ' + ' '.join(pair) + ' (приближение)'
        ax[idx].set_title(title)
        ax[idx].set_xlabel('Двадцатилетия')
        ax[idx].set_ylabel('Доля от общего числа слов')
        ax[idx].set_xticks(indices)
        ax[idx].grid(alpha=0.5)
        ax[idx].set_facecolor('#F5F5F5')
        plt.savefig(title + '.png')


def plot_mnm():
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 5))
    ax[0].plot(indices, n_ng_per_year['MOD_NOUN_MOD'], c='maroon')
    title = 'Распределение ' + 'MOD_NOUN_MOD'
    ax[0].set_xlim((1801, 1991))
    ax[0].set_ylim((0, 0.003))
    ax[0].set_title(title)
    ax[0].set_xlabel('Двадцатилетия')
    ax[0].set_ylabel('Доля от общего числа слов')
    ax[0].set_xticks(indices)
    ax[0].grid(alpha=0.5)
    ax[0].set_facecolor('#F5F5F5')
    plt.savefig(title + '.png')

    ax[1].plot(indices, n_ng_per_year['MOD_NOUN_MOD'], c='maroon')
    title = 'Распределение MOD_NOUN_MOD (приближение)'
    ax[1].set_title(title)
    ax[1].set_xlabel('Двадцатилетия')
    ax[1].set_ylabel('Доля от общего числа слов')
    ax[1].set_xticks(indices)
    ax[1].grid(alpha=0.5)
    ax[1].set_facecolor('#F5F5F5')
    plt.savefig(title + '.png')
    plt.show()


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
    plot_all(n_ng_per_year)
    plot_size(n_words_per_year)
    plot_bigrams()
    plot_trigrams()
    plot_mnm()


if __name__ == '__main__':
    main()
