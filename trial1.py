#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:04:44 2024

@author: henryboateng
"""

from efficient_cancer_data import read_training_data
A, b = read_training_data("train.data")

print(len(A))
print(len(b))