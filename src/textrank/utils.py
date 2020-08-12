# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def get_page_rank(graph: np.ndarray, vocab: dict, d:float=0.85) -> list:
    graph = nx.from_numpy_array(graph)
    page_rank = nx.pagerank(graph, alpha=d)
    word_rank = {vocab[v]: page_rank[v] for v in vocab}
    return sorted(word_rank.items(), key=lambda x: x[1], reverse=True)


def visualize_graph(graph: np.ndarray, output_file: str='pagerank_graph.png') -> None:
    graph = nx.from_numpy_array(graph)
    pos = nx.spring_layout(graph, iterations=200)
    nx.draw(graph, pos=pos, with_labels=True)
    plt.savefig(output_file)
    plt.show()
