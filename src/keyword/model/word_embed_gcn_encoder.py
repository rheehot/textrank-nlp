# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return x + (f_h * torch.sigmoid(f_h))


class WordEmbedGCNEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 gcn_hid_dim: int,
                 num_gcn_layers: int,
                 pad_idx: int,
                 d_rate: int = 0.2,
                 pretrained_embed: Any = None):
        super(WordEmbedGCNEncoder, self).__init__()

        assert num_gcn_layers > 0
        assert d_rate > 0.0

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(p=d_rate)
        gcn_layers = [GCNLayer(embed_dim, gcn_hid_dim)] + \
                     [GCNLayer(gcn_hid_dim, gcn_hid_dim) for _ in range(num_gcn_layers - 1)]
        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.bn = nn.BatchNorm1d(gcn_hid_dim)

        if pretrained_embed is not None:
            # pretrained_embed = torchtext.field.vocab.vectors
            self.embed.from_pretrained(pretrained_embed)

    def forward(self,
                docs: torch.Tensor,
                a_f: torch.Tensor,
                a_b: torch.Tensor) -> torch.Tensor:

        x = self.dropout(self.embed(docs))

        for layer in self.gcn_layers:
            x = layer(x, a_f, a_b)

        x = F.dropout(x, p=0.5)
        x = torch.einsum('ijk->ikj', x)
        x = self.bn(x)

        return x
