# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

from typing import Tuple

import torch
from torchtext.vocab import GloVe
from torchtext.data import Field, TabularDataset, Iterator, RawField
from tqdm import tqdm
from src.keyword.data.graph_util import build_graph, normalize_graph


def get_dataset(file_path: str, min_freq: int=0, vectors: Tuple[str, int] = None) -> Tuple[TabularDataset, Field, Field]:
    SRC = Field(
        tokenize=lambda x: x.split(' '),
        lower=True,
        batch_first=True)

    TRG = Field(
        tokenize=lambda x: x.split(' '),
        init_token='<sos>',
        eos_token='<eos>',
        lower=True,
        batch_first=True)

    dataset = TabularDataset(path=file_path,
                             format='json',
                             fields={
                                 'doc_words': ('text', SRC),
                                 'keyphrases': ('label', TRG)}
                             )

    if vectors is not None:
        SRC.build_vocab(dataset, vectors=GloVe(name=vectors[0], dim=vectors[1]), min_freq=min_freq)
        TRG.build_vocab(dataset, vectors=GloVe(name=vectors[0], dim=vectors[1]), min_freq=min_freq)
    else:
        SRC.build_vocab(dataset, min_freq=min_freq)
        TRG.build_vocab(dataset, min_freq=min_freq)

    return dataset, SRC, TRG


def batch_graph(grhs: torch.Tensor):
    """ batch a list of graphs
    @param grhs: tensor
    """
    b = len(grhs)  # batch size
    graph_dims = [len(g) for g in grhs]
    s = max(graph_dims)  # max seq length

    G = torch.zeros([b, s, s])
    for i, g in enumerate(grhs):
        s_ = graph_dims[i]
        G[i, :s_, :s_] = g
    return G


def build_graph_dataset(dataset: TabularDataset):
    GRH = RawField(postprocessing=batch_graph)

    for d in tqdm(dataset):
        token_len = len(d.text)
        G = build_graph(token_len, token_len)
        A_f = G['forward']
        A_b = G['backward']
        d.A_f = normalize_graph(A_f)
        d.A_b = normalize_graph(A_b)

    dataset.fields['A_f'] = GRH
    dataset.fields['A_b'] = GRH

    return dataset

if __name__ == '__main__':
    dataset = build_graph_dataset(get_dataset('../rsc/preprocessed/kp20k.valid_100_lines.json')[0])
    BATCH_SIZE = 8

    train_iterator = Iterator(
        dataset,
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True
    )

    for i, batch in enumerate(train_iterator):
        print(batch.A_f.shape)
        for j in range(BATCH_SIZE):
            print(batch.text[j].shape)
        for j in range(BATCH_SIZE):
            print(batch.A_f[j].shape)
        break
