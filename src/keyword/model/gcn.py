# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import BertModel

"""
References
    - Convolutional Sequence to Sequence Learning
    - Deep Residual Learning for Image Recognition
    - Graph Convolution Network
"""

class BertGCNEmbedding(nn.Module):
    def __init__(self, bert: BertModel, input_dim: int, output_dim: int):
        super(BertGCNEmbedding, self).__init__()
        self.bert = bert

        """
        W_f, W_b: Trainable weights for each adjacency matrix of backwards and forward.
        W: Trainable weights for output of embedding.
        """
        self.W_f = nn.Parameter(torch.randn(input_dim, output_dim))
        self.W_b = nn.Parameter(torch.randn(input_dim, output_dim))
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, batch: Dict) -> torch.Tensor:
        """
        batch['input_ids']: (bs x max_seq_len)
        batch['A_f']: (bs x max_seq_len x max_seq_len)
        batch['A_b']: (bs x max_seq_len x max_seq_len)
        """

        # [bs x seq_len x hidden_dim]
        last_hidden_states = self.bert(batch['input_ids'])[0]

        assert last_hidden_states.shape[1] == batch['A_f'].shape[1]
        assert last_hidden_states.shape[1] == batch['A_b'].shape[1]
        assert batch['A_f'].shape == batch['A_b'].shape

        # [bs x seq_len x hidden_dim]
        forward_hidden_states = torch.bmm(batch['A_f'], last_hidden_states)
        backward_hidden_states = torch.bmm(batch['A_b'], last_hidden_states)

        # [bs x seq_len x output_dim]
        hs_w_f = forward_hidden_states.matmul(self.W_f)
        hs_w_b = backward_hidden_states.matmul(self.W_b)
        hs_w = last_hidden_states.matmul(self.W)

        # [bs x seq_len x output_dim]
        return F.relu(hs_w + hs_w_b + hs_w_f)
