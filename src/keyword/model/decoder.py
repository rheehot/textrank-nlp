# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import torch.nn as nn


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self):
        pass


class GRUDecoder(nn.Module):
    def __init__(self,
                 input_size:int,
                 hidden_size:int,
                 num_layers: int):
        super(GRUDecoder, self).__init__()
        """
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers
        """
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self):
        pass