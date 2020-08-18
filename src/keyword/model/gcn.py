# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import torch
import torch.nn as nn


"""
References
    - Convolutional Sequence to Sequence Learning
    - Deep Residual Learning for Image Recognition
    - Graph Convolution Network
"""

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout = 0.2):
        """
        each layer has the following form of computation
        H = f(A * H * W)
        H = A_f * H * W_f + A_b * H * W_b + H * W
        H: (b, seq len, ninp)
        A: (b, seq len, seq len)
        W: (ninp, nout)
        """
        super(GCNLayer, self).__init__()
        self.W_f = nn.Parameter(torch.randn(input_dim, output_dim))
        self.W_b = nn.Parameter(torch.randn(input_dim, output_dim))
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))

        self.b = nn.Parameter(torch.randn(output_dim))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A_f, A_b):
        """
        H = relu(A * x * W)
        x: (b, seq len, ninp) -> embedded representation
        A: (b, seq len, seq len) -> adjacency matrix
        W: (ninp, nout)
        """
        x = self.dropout(x)
        try:
            x_f = torch.bmm(A_f, x)
            x_b = torch.bmm(A_b, x)
        except:
            import pdb; pdb.set_trace()

        x_f = x_f.matmul(self.W_f)
        x_b = x_b.matmul(self.W_b)
        h = x.matmul(self.W)

        return x_f + x_b + h
        # try:
        #     x = torch.bmm(A, x)  # x: (b, seq len, ninp)
        # except:
        #     import pdb; pdb.set_trace()
        #
        # x = x.matmul(self.W) + self.b
        # x = self.relu(x)
        # return x