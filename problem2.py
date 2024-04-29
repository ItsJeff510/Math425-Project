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

# Making the Array a 2-Dimensional Matrix
A = np.array(A).astype(float) 
b = np.array(b).astype(float)

# Applying the weights to dataset #2, 8, 26, 27, 30
for i in range(len(A)):
    for j in range(len(A[i])):
        if j == 1:
            A[i][1] /= 2
        
        elif j == 7:
            A[i][7] /= 2

        elif j == 25:
            A[i][25] /= 2
        
        elif j == 26:
            A[i][26] /= 2

        elif j == 29:
            A[i][29] /= 2

# Getting the Q and R Matrix
Q, R = np.linalg.qr(A)


# Applying Least Squares
least_squares = np.linalg.solve(R, np.dot(Q.T, b))

# Displaying the results
print("Solution to least squares problem:\n", least_squares)

