# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from collections import defaultdict, Counter
from typing import List


class Vocab(object):
    def __init__(self, special_tokens=None) -> None:
        super(Vocab, self).__init__()
        self.vocab2idx = defaultdict()
        self.idx2vocab = defaultdict()

        if special_tokens is None:
            special_tokens = ['<unk>', '<pad>']

        self.initialize(special_tokens=special_tokens)

    def initialize(self, special_tokens=None) -> None:
        if special_tokens is None:
            special_tokens = ['<unk>', '<pad>']

        for idx, token in enumerate(special_tokens):
            self.vocab2idx[token] = idx
            self.idx2vocab[idx] = token

    def add_tokens(self, tokens: List[str], min_c: int=1) -> None:
        cursor = len(self.vocab2idx)
        for idx, token in enumerate({x: count for x, count in Counter(tokens).items() if count >= min_c}):
            idx += cursor
            self.vocab2idx[token] = idx
            self.idx2vocab[idx] = token

    def vocab_2_txt(self, output_path: str=None) -> None:
        vocab_str = '\n'.join([token for token in self.vocab2idx])

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(vocab_str)
            f.close()



if __name__ == '__main__':
    vocab = Vocab()
    vocab.add_tokens(['hello', 'world', 'world', 'token'])
