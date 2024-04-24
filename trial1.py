#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:04:44 2024

@author: henryboateng
"""

from efficient_cancer_data import read_training_data
A, b = read_training_data("train.data")

import numpy as np
import statsmodels.api as sm

# Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 2)  # Predictor variables
y = X.dot(np.array([1, 2])) + np.random.normal(scale=0.5, size=100)  # Response variable
weights = np.random.rand(100)  # Weights for each observation

# Add a constant to the predictor variables for the intercept term
X = sm.add_constant(X)

# Perform weighted least squares regression
model = sm.WLS(y, X, weights=1.0/weights)  # Weighted least squares model
results = model.fit()

# Print the summary of the regression results
print(results.summary())
