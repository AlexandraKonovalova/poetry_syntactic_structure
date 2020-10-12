# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt



def read_corpus(path):
    corpus = pd.read_csv(path,
                         delimiter='\t',
                         names=['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS',
                                'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC'])
    return corpus


def read_verse(path):
    verse_headers = ['IDline', 'NoInVerse', 'NoStrophe', 'NoInStrophe',
                 'LineCat', 'NoIndex', 'Meter', 'LineText',
                 'BeginRhymeWithAcc', 'BeginRhyme', 'IDdoc',
                 'Rhymed', 'RhymedNwords', 'RhymedVowels', 
                 'RhymedNsyl', 'RhymedPattern', 'RhymedUPoS',
                 'RhymedWith', 'RhymedWithIDline', 'RhymedWithNwords',
                 'RhymedWithUPoSLast', 'RhymedStat']

    verses = pd.read_csv(path,
                         delimiter='\t',
                         names=verse_headers)
    return verses


def read_lines(path):
    lines = pd.read_csv(path,
                        delimiter='\t')
    return lines


def read_meta(path):
    meta = pd.read_csv('path',
                       delimiter='\t')
    return meta


def split_meta(corpus, lines, ):
    
    '''
    Split MISC column in corpus to separate columns
    Add IDline column from lines dataset
    corpus: DataFrame in .conllu format
    '''

    corpus['IDword'] = pd.to_numeric(corpus['MISC'].apply(lambda x: x.split('|')[0]))
    corpus['IDdoc'] = pd.to_numeric(corpus['MISC'].apply(lambda x: x.split('|')[1]))
    corpus['DocPath'] = corpus['MISC'].apply(lambda x: x.split('|')[2])
    
    corpus = corpus.merge(lines, left_on='IDword', right_on='IDword')
    corpus.drop(columns=['IDdoc_y'], inplace=True)
    corpus.rename(columns={'IDdoc_x': 'IDdoc'}, inplace=True)
    
    return corpus


if __name__ == '__main__':
    main()
