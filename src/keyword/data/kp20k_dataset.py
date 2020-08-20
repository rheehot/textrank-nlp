# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import json
import torch
import argparse


from typing import Dict
from torch.utils.data import Dataset
from tqdm import tqdm
from src.keyword.data.graph_util import build_graph, normalize_graph
from src.keyword.data.token import get_bert_tokenizer


class KP20KDataset(Dataset):
    def __init__(self,
                 doc_words,
                 keywords,
                 max_src_len: int = 256) -> None:
        super(KP20KDataset, self).__init__()
        self.doc_words = doc_words
        self.keywords = keywords
        self.max_src_len = max_src_len

    def __getitem__(self, index: int) -> Dict[str, str]:
        # 102 -> [SEP]
        G = build_graph((self.doc_words[index] == 102).nonzero(as_tuple=False).item(), self.max_src_len)
        A_b = normalize_graph(G['backward'])
        A_f = normalize_graph(G['forward'])

        return {
            'doc_words': self.doc_words[index],
            'keyphrases': self.keywords[index],
            'A_b': A_b,
            'A_f': A_f
        }

    def __len__(self) -> int:
        return len(self.doc_words)


def save_dataset(input_file: str, output_file: str, max_src_len: int, max_trg_len: int) -> None:
    assert max_src_len < 512

    doc_words = []
    keyphrases = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f)):
            json_object = json.loads(line)
            doc_words.append(json_object['doc_words'])
            keyphrases.append(json_object['keyphrases'])
        f.close()

    tokenizer, num_tokens = get_bert_tokenizer()
    print('tokenizer loaded', tokenizer, sep=', ')

    assert len(doc_words) == len(keyphrases)
    doc_words = tokenizer(doc_words, max_length=max_src_len, padding=True, truncation=True, return_tensors='pt')['input_ids']
    keyphrases = tokenizer(keyphrases, max_length=max_trg_len, padding=True, truncation=True, return_tensors='pt')['input_ids']

    dataset = KP20KDataset(doc_words, keyphrases, max_src_len)
    torch.save(dataset, output_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset', type=str, required=True,
                        help="The path to the source dataset (preprocessed json).")
    parser.add_argument('--output_path', type=str, required=True,
                        help="The output path")
    parser.add_argument('--max_src_seq_len', type=int, default=256,
                        help="Maximum document sequence length")
    parser.add_argument('--max_trg_seq_len', type=int, default=16,
                        help="Maximum keyphrases sequence length to keep.")

    args = parser.parse_args()

    save_dataset(args.source_dataset, args.output_path, args.max_src_seq_len, args.max_trg_seq_len)


if __name__ == '__main__':
    if __name__ == '__main__':
