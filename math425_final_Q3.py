#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
# read data for training set and store in A_train as pandas DataFrame
A_train = pd.read_csv("handwriting_training_set.txt", header=None)
# convert the DataFrame to a NumPy array
A_train = A_train.to_numpy()


# In[2]:


# read data for training set labels and store in b_train as pandas DataFrame
b_train = pd.read_csv("handwriting_training_set_labels.txt", header=None)
# convert the DataFrame to a NumPy array
b_train = b_train.to_numpy()


# In[3]:


# read data for test set and store in A_test as pandas DataFrame
A_test = pd.read_csv("handwriting_test_set.txt", header=None)
# convert the DataFrame to a NumPy array
A_test = A_test.to_numpy()


# In[4]:


# read data for test set labels and store in b_test as pandas DataFrame
b_test = pd.read_csv("handwriting_test_set_labels.txt", header=None)
# convert the DataFrame to a NumPy array
b_test = b_test.to_numpy()


# In[5]:


# Use the training set to compute the SVD of each class/digit matrix
A_train_0 = A_train[0:400,:]
U_0, S_0, VT_0 = np.linalg.svd(A_train_0)
A_train_1 = A_train[400:800,:]
U_1, S_1, VT_1 = np.linalg.svd(A_train_1)
A_train_2 = A_train[800:1200,:]
U_2, S_2, VT_2 = np.linalg.svd(A_train_2)
A_train_3 = A_train[1200:1600,:]
U_3, S_3, VT_3 = np.linalg.svd(A_train_3)
A_train_4 = A_train[1600:2000,:]
U_4, S_4, VT_4 = np.linalg.svd(A_train_4)
A_train_5 = A_train[2000:2400,:]
U_5, S_5, VT_5 = np.linalg.svd(A_train_5)
A_train_6 = A_train[2400:2800,:]
U_6, S_6, VT_6 = np.linalg.svd(A_train_6)
A_train_7 = A_train[2800:3200,:]
U_7, S_7, VT_7 = np.linalg.svd(A_train_7)
A_train_8 = A_train[3200:3600,:]
U_8, S_8, VT_8 = np.linalg.svd(A_train_8)
A_train_9 = A_train[3600:4000,:]
U_9, S_9, VT_9 = np.linalg.svd(A_train_9)


# In[6]:


# Create a matrix to store all VT matrices
VT_matrices_5 = [VT_0[0:5,:], VT_1[0:5,:], VT_2[0:5,:], VT_3[0:5,:], VT_4[0:5,:], VT_5[0:5,:], VT_6[0:5,:], VT_7[0:5,:], VT_8[0:5,:], VT_9[0:5,:]]
# Create an empty list to store all predictions
b_pred_5 = []

for j in range(1000): # Create a loop for each test data
    d = A_test[j, :].T # Transpose row vector to column vector    
    distances = [] # Initialize distances and reset for each d    
    for VT in VT_matrices_5: # Loop over each VT matrix
        proj_d = VT.T @ VT @ d # Compute projection of d onto the subspace spanned by V
        distance = np.linalg.norm(d - proj_d) # Compute the norm of the difference
        distances.append(distance) # Add distance to distances list
    min_index = distances.index(min(distances))  # Find the index of the minimum distance
    b_pred_5.append(min_index)  # Append the index of the closest subspace

b_test[b_test == 10] = 0 # reset 10 to 0 in b_test to match index in b_pred
b_pred_5 = np.array(b_pred_5).reshape(-1, 1) # reshape the list b_pred to 1000x1 matrix
correct_predictions_5 = np.sum(b_pred_5 == b_test) # sum up correct predictions
print("Number of correctly identified digits for 5 sigular values:", correct_predictions_5)


# In[7]:


# Create a matrix to store all VT matrices
VT_matrices_10 = [VT_0[0:10,:], VT_1[0:10,:], VT_2[0:10,:], VT_3[0:10,:], VT_4[0:10,:], VT_5[0:10,:], VT_6[0:10,:], VT_7[0:10,:], VT_8[0:10,:], VT_9[0:10,:]]
# Create an empty list to store all predictions
b_pred_10 = []

for j in range(1000): # Create a loop for each test data
    d = A_test[j, :].T # Transpose row vector to column vector    
    distances = [] # Initialize distances and reset for each d    
    for VT in VT_matrices_10: # Loop over each VT matrix
        proj_d = VT.T @ VT @ d # Compute projection of d onto the subspace spanned by V
        distance = np.linalg.norm(d - proj_d) # Compute the norm of the difference
        distances.append(distance) # Add distance to distances list
    min_index = distances.index(min(distances))  # Find the index of the minimum distance
    b_pred_10.append(min_index)  # Append the index of the closest subspace

b_test[b_test == 10] = 0 # reset 10 to 0 in b_test to match index in b_pred
b_pred_10 = np.array(b_pred_10).reshape(-1, 1) # reshape the list b_pred to 1000x1 matrix
correct_predictions_10 = np.sum(b_pred_10 == b_test) # sum up correct predictions
print("Number of correctly identified digits for 10 sigular values:", correct_predictions_10)


# In[8]:


# Create a matrix to store all VT matrices
VT_matrices_15 = [VT_0[0:15,:], VT_1[0:15,:], VT_2[0:15,:], VT_3[0:15,:], VT_4[0:15,:], VT_5[0:15,:], VT_6[0:15,:], VT_7[0:15,:], VT_8[0:15,:], VT_9[0:15,:]]
# Create an empty list to store all predictions
b_pred_15 = []

for j in range(1000): # Create a loop for each test data
    d = A_test[j, :].T # Transpose row vector to column vector    
    distances = [] # Initialize distances and reset for each d    
    for VT in VT_matrices_15: # Loop over each VT matrix
        proj_d = VT.T @ VT @ d # Compute projection of d onto the subspace spanned by V
        distance = np.linalg.norm(d - proj_d) # Compute the norm of the difference
        distances.append(distance) # Add distance to distances list
    min_index = distances.index(min(distances))  # Find the index of the minimum distance
    b_pred_15.append(min_index)  # Append the index of the closest subspace

b_test[b_test == 10] = 0 # reset 10 to 0 in b_test to match index in b_pred
b_pred_15 = np.array(b_pred_15).reshape(-1, 1) # reshape the list b_pred to 1000x1 matrix
correct_predictions_15 = np.sum(b_pred_15 == b_test) # sum up correct predictions
print("Number of correctly identified digits for 15 sigular values:", correct_predictions_15)


# In[9]:


# Create a matrix to store all VT matrices
VT_matrices_20 = [VT_0[0:20,:], VT_1[0:20,:], VT_2[0:20,:], VT_3[0:20,:], VT_4[0:20,:], VT_5[0:20,:], VT_6[0:20,:], VT_7[0:20,:], VT_8[0:20,:], VT_9[0:20,:]]
# Create an empty list to store all predictions
b_pred_20 = []

for j in range(1000): # Create a loop for each test data
    d = A_test[j, :].T # Transpose row vector to column vector    
    distances = [] # Initialize distances and reset for each d    
    for VT in VT_matrices_20: # Loop over each VT matrix
        proj_d = VT.T @ VT @ d # Compute projection of d onto the subspace spanned by V
        distance = np.linalg.norm(d - proj_d) # Compute the norm of the difference
        distances.append(distance) # Add distance to distances list
    min_index = distances.index(min(distances))  # Find the index of the minimum distance
    b_pred_20.append(min_index)  # Append the index of the closest subspace

b_test[b_test == 10] = 0 # reset 10 to 0 in b_test to match index in b_pred
b_pred_20 = np.array(b_pred_20).reshape(-1, 1) # reshape the list b_pred to 1000x1 matrix
correct_predictions_20 = np.sum(b_pred_20 == b_test) # sum up correct predictions
print("Number of correctly identified digits for 20 sigular values:", correct_predictions_20)


# In[10]:


# i. Give a table or graph of the percentage of correctly classified digits as a function of the number of basis vectors
import matplotlib.pyplot as plt

basis_vectors = [5,10,15,20]
percentage = [correct_predictions_5/1000, correct_predictions_10/1000, correct_predictions_15/1000, correct_predictions_20/1000]
plt.bar(basis_vectors, percentage)
plt.title('Bar Plot of Correct Percentages')  # Title for the plot
plt.xlabel('number of basis vectors')  # Label for the x-axis
plt.ylabel('correct percentage')  # Label for the y-axis
plt.show()  # Display the plot


# In[13]:


# ii. Check if all digits are equally easy or difficult to classify.
# Also look at some of the difficult ones, and see that in may cases they are very badly written.

# Dictionary to store accuracy for each digit
digit_accuracy_20 = {}

# use for loop to check each number
for i in range(0,10):
    # Indices where the true label is equal to the current digit
    idx = (b_test == i)    
    # Subset predictions and true values where the true value is digit
    pred_digit = b_pred_20[idx]
    true_digit = b_test[idx]    
    # Calculate accuracy for this digit
    correct = np.sum(pred_digit == true_digit)
    total = len(true_digit)
    accuracy = correct / total    
    # Store accuracy in dictionary with digit as key
    digit_accuracy_20[i] = accuracy

# print accuracy for each digit
for i, accuracy in digit_accuracy_20.items():
    print(f"Accuracy for digit {i}: {accuracy:.2f}")


# $2$ and $8$ have accuracy rate of $0.89$ with all other digits above $0.94$.

# In[16]:


# iii. check the singular values of the different classes. Is there evidence to support using different number of basis for different digits

# Dictionary to store accuracy for each digit
digit_accuracy_5 = {}
# use for loop to check each number
for i in range(0,10):
    # Indices where the true label is equal to the current digit
    idx = (b_test == i)    
    # Subset predictions and true values where the true value is digit
    pred_digit = b_pred_5[idx]
    true_digit = b_test[idx]    
    # Calculate accuracy for this digit
    correct = np.sum(pred_digit == true_digit)
    total = len(true_digit)
    accuracy = correct / total    
    # Store accuracy in dictionary with digit as key
    digit_accuracy_5[i] = accuracy

# Dictionary to store accuracy for each digit
digit_accuracy_10 = {}
# use for loop to check each number
for i in range(0,10):
    # Indices where the true label is equal to the current digit
    idx = (b_test == i)    
    # Subset predictions and true values where the true value is digit
    pred_digit = b_pred_10[idx]
    true_digit = b_test[idx]    
    # Calculate accuracy for this digit
    correct = np.sum(pred_digit == true_digit)
    total = len(true_digit)
    accuracy = correct / total    
    # Store accuracy in dictionary with digit as key
    digit_accuracy_10[i] = accuracy

# Dictionary to store accuracy for each digit
digit_accuracy_15 = {}
# use for loop to check each number
for i in range(0,10):
    # Indices where the true label is equal to the current digit
    idx = (b_test == i)    
    # Subset predictions and true values where the true value is digit
    pred_digit = b_pred_15[idx]
    true_digit = b_test[idx]    
    # Calculate accuracy for this digit
    correct = np.sum(pred_digit == true_digit)
    total = len(true_digit)
    accuracy = correct / total    
    # Store accuracy in dictionary with digit as key
    digit_accuracy_15[i] = accuracy

# list for all digits
digits = list(digit_accuracy_20.keys())
# list for accuracy for 5, 10, 15 ,20 singular values
accuracy_5 = list(digit_accuracy_5.values())
accuracy_10 = list(digit_accuracy_10.values())
accuracy_15 = list(digit_accuracy_15.values())
accuracy_20 = list(digit_accuracy_20.values())

# Create a side-by-side plot
x = np.arange(len(digits))  # label locations
width = 0.20  # width of the bars

fig, ax = plt.subplots(figsize=(12, 8))
rects_5 = ax.bar(x - 1.5*width, accuracy_5, width, label='Accuracy at SV=5')
rects_10 = ax.bar(x - 0.5*width, accuracy_10, width, label='Accuracy at SV=10')
rects_15 = ax.bar(x + 0.5*width, accuracy_15, width, label='Accuracy at SV=15')
rects_20 = ax.bar(x + 1.5*width, accuracy_20, width, label='Accuracy at SV=20')

# Add text for labels, title, and custom x-axis tick labels.
ax.set_xlabel('Digits')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by digit and singular value setting')
ax.set_xticks(x)
ax.set_xticklabels(digits)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# As we can see from graph above, in most cases, when we increase the number of sigular values, the accuracy increases and total prediction accuracy increases.
# In[18]:


# Create a matrix to store all VT matrices
VT_matrices_1 = [VT_0[0:1,:], VT_1[0:1,:], VT_2[0:1,:], VT_3[0:1,:], VT_4[0:1,:], VT_5[0:1,:], VT_6[0:1,:], VT_7[0:1,:], VT_8[0:1,:], VT_9[0:1,:]]
# Create an empty list to store all predictions
b_pred_1 = []

for j in range(1000): # Create a loop for each test data
    d = A_test[j, :].T # Transpose row vector to column vector    
    distances = [] # Initialize distances and reset for each d    
    for VT in VT_matrices_1: # Loop over each VT matrix
        proj_d = VT.T @ VT @ d # Compute projection of d onto the subspace spanned by V
        distance = np.linalg.norm(d - proj_d) # Compute the norm of the difference
        distances.append(distance) # Add distance to distances list
    min_index = distances.index(min(distances))  # Find the index of the minimum distance
    b_pred_1.append(min_index)  # Append the index of the closest subspace

b_test[b_test == 10] = 0 # reset 10 to 0 in b_test to match index in b_pred
b_pred_1 = np.array(b_pred_1).reshape(-1, 1) # reshape the list b_pred to 1000x1 matrix
correct_predictions_1 = np.sum(b_pred_1 == b_test) # sum up correct predictions
print("Number of correctly identified digits for 1 sigular values:", correct_predictions_1)


# In[20]:


# Dictionary to store accuracy for each digit
digit_accuracy_1 = {}
# use for loop to check each number
for i in range(0,10):
    # Indices where the true label is equal to the current digit
    idx = (b_test == i)    
    # Subset predictions and true values where the true value is digit
    pred_digit = b_pred_1[idx]
    true_digit = b_test[idx]    
    # Calculate accuracy for this digit
    correct = np.sum(pred_digit == true_digit)
    total = len(true_digit)
    accuracy = correct / total    
    # Store accuracy in dictionary with digit as key
    digit_accuracy_1[i] = accuracy

accuracy_1 = list(digit_accuracy_1.values())

# Create a side-by-side plot
x = np.arange(len(digits))  # label locations
width = 0.20  # width of the bars

fig, ax = plt.subplots(figsize=(12, 8))
rects_1 = ax.bar(x - 0.5*width, accuracy_1, width, label='Accuracy at SV=1')
rects_20 = ax.bar(x + 0.5*width, accuracy_20, width, label='Accuracy at SV=20')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Digits')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by digit and singular value setting')
ax.set_xticks(x)
ax.set_xticklabels(digits)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

#Using one sigular value to predict gives us an overall accuracy rate of 0.797, which is significantly lower than using 20 sigular values. For the prediction of each digit, the accuracy is lower as well.