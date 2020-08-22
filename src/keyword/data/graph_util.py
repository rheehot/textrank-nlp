# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

from typing import  Dict

import torch
import numpy as np


def normalize_graph(M: torch.Tensor) -> torch.Tensor:
    """
    M = D**(-1/2) M D**(-1/2)
    In order to stabilize the iterative message propagation progcess in gcn encoder, graph should be normalized.
    The purpose of this re-normalization trick is to constrain the eigenvalues of the normalized adjacency matrices
    close to 1.
    """
    deg = M.sum(axis=1).squeeze()
    deg = deg ** (-1/2)
    D = torch.diag(deg)
    return D.mm(M.mm(D))


def build_graph(actual_src_len: int=None, max_src_len: int=512) -> Dict[str, torch.Tensor]:
    """
    build the graph representation.
    rules:
        - bi-directional representation (https://arxiv.org/pdf/1905.07689.pdf)
            - backward A_{ij} = \sum_{p_i \in P(w_i)} \sum_{p_j \in P(w_j)} ReLU(\frac{1}{p_i-p_j})
            - forward A_{ij} = \sum_{p_i \in P(w_i)} \sum_{p_j \in P(w_j)} ReLU(\frac{1}{p_j-p_i})
            - P(w_i) is the set of the position offset p_i of word w_i in the document.

    :tokens -> position offset of given sequence.
    :max_src_len -> max length of given sequence.
    """

    n = max_src_len
    tokens = [i for i in range(1, actual_src_len + 1)]

    assert len(tokens) == actual_src_len

    B, F = torch.zeros([n, n]), torch.zeros([n, n]) # square matrix

    for i, offset_i in enumerate(tokens):
        for j, offset_j in enumerate(tokens):
            # if each index is same, it will occur ZeroDivisionError
            if i == j: continue

            B[i][j] = np.maximum(0, (1 / (offset_i - offset_j)))
            F[i][j] = np.maximum(0, (1 / (offset_j - offset_i)))

    # self-loops
    i_m = torch.eye(n, n)

    B += i_m
    F += i_m

    return {
        'backward': B,
        'forward': F
    }


if __name__ == '__main__':
    from pprint import pprint

    graph = build_graph(6, 16)
    B = graph['backward']
    F = graph['forward']
    pprint(normalize_graph(B))
    pprint(normalize_graph(F))


