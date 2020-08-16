# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

from typing import List, Dict
from itertools import combinations

import torch
import torch.nn.functional as F
import torch as nn
import numpy as np


def normalize_graph(M: torch.Tensor) -> torch.Tensor:
    """
    M = D**(-1/2) M D**(-1/2)
    """
    deg = M.sum(axis=1).squeeze()
    deg = deg ** (-1/2)
    D = torch.diag(deg)
    return D.mm(M.mm(D))


def build_graph(tokens: List[int]) -> Dict[str, torch.Tensor]:
    """
    build the graph representation.
    rules:
        - bi-directional representation (https://arxiv.org/pdf/1905.07689.pdf)
            - backward A_{ij} = \sum_{p_i \in P(w_i)} \sum_{p_j \in P(w_j)} ReLU(\frac{1}{p_i-p_j})
            - forward A_{ij} = \sum_{p_i \in P(w_i)} \sum_{p_j \in P(w_j)} ReLU(\frac{1}{p_j-p_i})
            - P(w_i) is the set of the position offset p_i of word w_i in thedocument.
    """

    n = len(tokens)
    g_f, g_b = torch.zeros([n, n]), torch.zeros([n, n]) # square matrix

    for i, _ in enumerate(tokens):
        for j, __ in enumerate(tokens[i:]):
            if i != j:
                g_b[i][j] = max(0, (1 / (i - j)))
                g_f[i][j] = max(0, (1 / (j - i)))

    return {
        'backward': g_b + torch.eye(n, n),
        'forward': g_f + torch.eye(n, n)
    }


if __name__ == '__main__':
    from pprint import pprint

    tokens = [(i) for i in range(0, 10)]
    graph = build_graph(tokens)
    n_g_b, n_g_f = normalize_graph(graph['backward']), normalize_graph(graph['forward'])
    pprint(n_g_b)
    pprint(n_g_f)
    pprint(n_g_b + n_g_f)
