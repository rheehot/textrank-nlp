# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import numpy as np


g = [[0, 0, 0.5, 0],
     [0, 0, 0.5, 1],
     [1, 0.5, 0, 0],
     [0, 0.5, 0, 0]]

g = np.array(g)

pr = np.array([1, 1, 1, 1])
d = 0.85

for iter in range(10):
    pr = 0.15 + 0.85 * np.dot(g, pr)
    print(iter, pr, sep=', ')