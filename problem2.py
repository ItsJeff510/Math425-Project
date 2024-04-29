#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:04:44 2024

@author: henryboateng
"""

# Importing neccesary libraries 
from efficient_cancer_data import read_training_data
import numpy as np
from sympy import Matrix

# Classifier function to determine if results are malignant or benign
def predict_cancer(data, least_squares):
    prediction = np.dot(data, least_squares)
    return np.where(prediction >= 0, 1, -1)

# Importing data into variables
A, b = read_training_data("train.data")
val_A, val_b = read_training_data('validate.data')

# Making the Array a 2-Dimensional Matrix
A = np.array(A).astype(float) 
b = np.array(b).astype(float)
val_A = np.array(val_A).astype(float)
val_b = np.array(val_b).astype(float)

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
for i in range(len(val_A)):
    for j in range(len(val_A[i])):
        if j == 1:
            val_A[i][1] /= 2
        
        elif j == 7:
            val_A[i][7] /= 2

        elif j == 25:
            val_A[i][25] /= 2
        
        elif j == 26:
            val_A[i][26] /= 2

        elif j == 29:
            val_A[i][29] /= 2

# Getting the Q and R Matrix
Q, R = np.linalg.qr(A)

# Applying Least Squares
least_squares = np.linalg.solve(R, np.dot(Q.T, b))

# Displaying the results
print("The solution to the weighted least squares problem is:\n", least_squares)

# Predicting Malignancy of validate and train data
training_results = predict_cancer(A, least_squares)
validate_results = predict_cancer(val_A, least_squares)

# Defining variables to keep track of positive and negative results for train and validate data
apos = 0
aneg = 0
pos = 0
neg = 0

# Counting the Negative and Positive predictions in the train and validate dataset
for i in training_results:
    if i == 1:
        apos += 1
    else:
        aneg += 1
for i in validate_results:
    if i == 1:
        pos += 1
    else:
        neg += 1

# Displaying the percentage of negatives in the train and validate dataset
print("\nTrain data negatives: ", aneg / (aneg + apos))
print("Validate data negatives: ", neg / (neg + pos))
print("\nThe validate dataset has a lower success rate than the train data.\n")