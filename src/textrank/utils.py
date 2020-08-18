# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def get_page_rank(graph: np.ndarray, vocab: dict, d:float=0.85) -> list:
    graph = nx.from_numpy_array(graph)
    page_rank = nx.pagerank(graph, alpha=d)
    print(page_rank)
    word_rank = {vocab[v]: page_rank[v] for v in vocab}
    return sorted(word_rank.items(), key=lambda x: x[1], reverse=True)


def get_ranks(directed_graph_weights, d=0.85):
    A = directed_graph_weights
    matrix_size = A.shape[0]
    for id in range(matrix_size):
        A[id, id] = 0
        col_sum = np.sum(A[:, id])
        if col_sum != 0:
            A[:, id] /= col_sum
        A[:, id] *= -d
        A[id, id] = 1

    B = (1 - d) * np.ones((matrix_size, 1))

    ranks = np.linalg.solve(A, B)
    return {idx: r[0] for idx, r in enumerate(ranks)}

def visualize_graph(graph: np.ndarray, output_file: str='pagerank_graph.png') -> None:
    graph = nx.from_numpy_array(graph)
    pos = nx.spring_layout(graph, iterations=200)
    nx.draw(graph, pos=pos, with_labels=True)
    plt.savefig(output_file)
    plt.show()
