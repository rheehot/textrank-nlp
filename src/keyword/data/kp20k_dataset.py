# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import json
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
        print(self.doc_words[index].shape)
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


if __name__ == '__main__':
    doc_words = []
    keyphrases = []

    with open('../rsc/preprocessed/kp20k.valid.json', 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f)):
            json_object = json.loads(line)
            doc_words.append(json_object['doc_words'])
            keyphrases.append(json_object['keyphrases'])
        f.close()

    tokenizer, num_tokens = get_bert_tokenizer()
    # print(doc_words)
    print(tokenizer)

    import logging

    logging.basicConfig(level=logging.INFO)

    print(len(doc_words))
    print(len(keyphrases))

    doc_words = tokenizer(doc_words, max_length=256, padding=True, truncation=True, return_tensors='pt')['input_ids']
    keyphrases = tokenizer(keyphrases, max_length=16, padding=True, return_tensors='pt')['input_ids']

    test_dataloader = DataLoader(KP20KDataset(doc_words, keyphrases, 256), batch_size=4)

    for batch in test_dataloader:
        print(batch['doc_words'])
        print(batch['doc_words'].shape)
        print(batch['A_b'])
        print(batch['A_b'].shape)
        break
