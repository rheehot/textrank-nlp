# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer
)
from sklearn.preprocessing import normalize
from typing import List, Tuple
from konlpy.tag import Mecab


import numpy as np


class DocTokenizer(object):
    def __init__(self, stop_words: List[str]=None, min_df: int=2) -> None:
        super(DocTokenizer, self).__init__()
        self.stop_words = stop_words
        self.min_df = min_df
        self.tokenizer = Mecab() # Mecab 은 기본 Tokenizer 임.
        self.bow = CountVectorizer(tokenizer=self.tokenizer.nouns, min_df=min_df, stop_words=stop_words)

    def get_word_corr_matrix(self, doc: List[str]) -> Tuple[np.ndarray, dict]:
        """

        :param sent:
        :return:
        """
        word_matrix = normalize(self.bow.fit_transform(doc).toarray().astype(float), axis=0)
        vocab = self.bow.vocabulary_
        return np.dot(word_matrix.T, word_matrix), {vocab[word]: word for word in vocab}
