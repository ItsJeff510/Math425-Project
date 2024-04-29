#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:04:44 2024

@author: henryboateng
"""

from efficient_cancer_data import read_training_data
A, b = read_training_data("train.data")

import numpy as np
from sympy import Matrix

#print(A.row)
#print(len(A))

A = np.array(A).astype(float)
b = np.array(b).astype(float)

print("Before: ") 
print(A[0][1])
for i in range(len(A)):
    for j in range(len(A[i])):
        if j == 2:
            A[i][j] /= 2

print("After: ") 
print(A[0][1])
    

Q, R = np.linalg.qr(A)

x_ls = np.linalg.solve(R, np.dot(Q.T, b))

print("Solution to least squares problem:", x_ls)

