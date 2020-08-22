# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

from typing import Dict
from transformers import BertModel

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
References
    - Convolutional Sequence to Sequence Learning
    - Deep Residual Learning for Image Recognition
    - Graph Convolution Network
"""

class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(GCNLayer, self).__init__()
        self.W_f = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=True)
        self.W_b = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=True)
        self.W = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=True)

    def forward(self,
                x: torch.Tensor,
                a_f: torch.Tensor,
                a_b: torch.Tensor):
        h_f = torch.bmm(a_f, x).matmul(self.W_f)
        h_b = torch.bmm(a_b, x).matmul(self.W_f)
        h_w = torch.matmul(x, self.W)

        # $H_{l+1} = H_l + f_l(H_l) \otimes \sigmoid(g_l(H_l))$
        # In the paper, g_l is function defined in a similar way as f_l.
        f_h = h_f + h_b + h_w
        return x + (f_h * F.sigmoid(f_h))


class GCNEncoder(nn.Module):
    """
    GCNEncoder

    TODO => 설명 쓰기.
    """
    def __init__(self,
                 bert: BertModel,
                 output_dim: int,
                 num_gcn_layers: int):
        super(GCNEncoder, self).__init__()
        self.bert = bert
        self.num_gcn_layers = num_gcn_layers
        self.gcn_layers = [GCNLayer(self.bert.config.hidden_size, output_dim)] + \
                          [GCNLayer(output_dim, output_dim) for _ in range(num_gcn_layers - 1)]
        self.linear = nn.Linear(2 * output_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, batch: Dict):
        x = self.bert(batch['doc_words'])[0]
        hidden = []
        for layer in self.gcn_layers:
            x = layer(x, batch['A_f'], batch['A_b'])
            print(x[:, 0, :].shape, x.shape)
            # [1 x hidden_dim], 1 means each hidden states of corresponding word.
            hidden.append(x[:, 0, :])

        mean = x.mean(dim=1)
        maxm = x.max(dim=1)[0]
        x = torch.cat((mean, maxm), dim=1)
        out = self.linear(self.dropout(x))
        hidden = torch.stack(hidden)

        """
        TODO: Decoder-Level
        
        Example:
        
        gru = nn.GRU(input_size=128, hidden_size=128, num_layers=2)
        gru(seq.shape (seq_len x bs x hidden_dim), hidden (num_layers x bs x hidden_dim))
        num_layers is following num_gcn_layers
        """
        return out, hidden


if __name__ == '__main__':
    from transformers import BertModel, BertTokenizerFast
    from src.keyword.data.graph_util import build_graph

    model_name = 'bert-base-uncased'
    bert = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")['input_ids']
    graphs = build_graph(8, 8)
    A_b = graphs['backward']
    A_f = graphs['forward']

    batch = {
        'doc_words': input_ids,
        'A_f': A_f.unsqueeze(dim=0),
        'A_b': A_b.unsqueeze(dim=0)
    }

    graph_encoder = GCNEncoder(bert, 768, 3)
    x, hidden = graph_encoder(batch)

