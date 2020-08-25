# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import argparse
import logging
import torch
import dill
import sys
import os

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))

logger = logging.getLogger()

from tqdm import tqdm
from typing import Tuple
from torchtext.vocab import GloVe
from torchtext.data import Field, TabularDataset, Iterator, RawField
from src.keyword.data.graph_util import build_graph, normalize_graph


def get_dataset(file_path: str,
                train_file: str,
                valid_file: str,
                test_file: str,
                min_freq: int = 0,
                vectors: Tuple[str, int] = None) -> \
        Tuple[TabularDataset, TabularDataset, TabularDataset, Field, Field]:
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

    train_dataset, valid_dataset, test_dataset = TabularDataset.splits(
        path=file_path,
        train=train_file,
        validation=valid_file,
        test=test_file,
        format='json',
        fields={
            'doc_words': ('text', SRC),
            'keyphrases': ('label', TRG)
        }
    )

    if vectors is not None:
        SRC.build_vocab(train_dataset, vectors=GloVe(name=vectors[0], dim=vectors[1]), min_freq=min_freq)
        TRG.build_vocab(train_dataset, vectors=GloVe(name=vectors[0], dim=vectors[1]), min_freq=min_freq)
    else:
        SRC.build_vocab(train_dataset, min_freq=min_freq)
        TRG.build_vocab(train_dataset, min_freq=min_freq)

    return train_dataset, valid_dataset, test_dataset, SRC, TRG


def batch_graph(grhs: torch.Tensor) -> torch.Tensor:
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


def build_graph_dataset(dataset: TabularDataset) -> TabularDataset:
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


def build_iterator(dataset: TabularDataset,
                   batch_size: int,
                   sort_key=lambda x: len(x.text)) -> Iterator:
    return Iterator(
        dataset=dataset,
        batch_size=batch_size,
        sort_key=sort_key,
        sort_within_batch=True,
    )

def save_dataset(dataset: TabularDataset, output_path: str) -> None:
    with open(output_path, 'wb') as f:
        dill.dump(list(dataset), f)
        f.close()

def save_fields(field: Field, output_path: str) -> None:
    with open(output_path, 'wb') as f:
        dill.dump(field, f)
        f.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset_path', type=str, required=True,
                        help="source_dataset_path")
    parser.add_argument('--train_file', type=str, required=True,
                        help="train_file")
    parser.add_argument('--valid_file', type=str, required=True,
                        help="valid_file")
    parser.add_argument('--test_file', type=str, required=True,
                        help="test_file")
    parser.add_argument('--output_train_path', type=str, required=True,
                        help="output_train_path")
    parser.add_argument('--output_valid_path', type=str, required=True,
                        help="output_valid_path")
    parser.add_argument('--output_test_path', type=str, required=True,
                        help="output_test_path")
    parser.add_argument('--src_field_path', type=str, required=True,
                        help="src_field_path")
    parser.add_argument('--trg_field_path', type=str, required=True,
                        help="trg_field_path")
    parser.add_argument('--min_freq', type=int, default=2,
                        help="GloVe Dim")
    parser.add_argument('--glove_dim', type=int, default=100,
                        help="GloVe Dim")
    parser.add_argument('--glove_words', type=str, default='6B',
                        help="GloVe Words")

    args = parser.parse_args()

    print('building torchtext dataset...')
    logger.info('building torchtext dataset...')

    train_dataset, valid_dataset, test_dataset, SRC, TRG = get_dataset(
        args.source_dataset_path,
        args.train_file,
        args.valid_file,
        args.test_file,
        args.min_freq,
        (args.glove_words, args.glove_dim)
    )

    # print('building graph dataset...')
    # logger.info('building graph dataset...')
    # train_dataset, valid_dataset, test_dataset = build_graph_dataset(train_dataset), \
    #                                              build_graph_dataset(valid_dataset), \
    #                                              build_graph_dataset(test_dataset)

    print('saving dataset...')
    logger.info('saving dataset...')
    save_dataset(train_dataset, args.output_train_path)
    save_dataset(valid_dataset, args.valid_train_path)
    save_dataset(test_dataset, args.output_test_path)

    print('saving fields...')
    logger.info('saving fields...')
    save_fields(SRC, args.src_field_path)
    save_fields(SRC, args.trg_field_path)

    """
    How to load dataset?
    
    1. First, Load the dataset, and custom fields.
    with open('./SRC.pkl', 'rb') as f:
        src_data = dill.load(f)
        f.close()

    with open('./TRG.pkl', 'rb') as f:
        trg_data = dill.load(f)
        f.close()
    with open('./train_dataset.pkl', 'rb') as f:
        loaded_dataset = dill.load(f)
        f.close()
        
    2. wrap the loaded_dataset with fields. 
    GRH = RawField(postprocessing=None)
    data_fields = [('text', src_data), ('label', trg_data), ('A_f', GRH), ('A_b', GRH)]
    load = Dataset(loaded_dataset, data_fields)
    
    3. build iterator.
    it = Iterator(
        dataset=load,
        batch_size=4,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
    )
    
    4. Done!
    for batch in it:
        print(batch)
    """


if __name__ == '__main__':
    main()