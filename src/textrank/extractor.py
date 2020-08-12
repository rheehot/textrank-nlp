# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

from typing import List, Tuple


class Extractor(object):
    def __init__(self, rank: List[Tuple[str, float]]) -> None:
        super(Extractor, self).__init__()
        self.rank = rank

    def extract_top_k(self, top_k: int) -> List[Tuple[str, float]]:
        if top_k >= len(self.rank) - 1:
            return self.rank

        return self.rank[:top_k]