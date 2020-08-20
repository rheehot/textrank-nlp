# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import json
import torch


from typing import List, Dict
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BatchEncoding
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


if __name__ == '__main__':
    # save_dataset('../rsc/preprocessed/kp20k.valid.json', './valid_dataset_test.pt', 256, 16)
    dataset = torch.load('./valid_dataset_test.pt')
    dataloader = DataLoader(dataset, 16)

    for batch in dataloader:
        print(batch)
        break