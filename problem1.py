from efficient_cancer_data import read_training_data

A, b = read_training_data("train.data")

# Problem 1a) Use the QR algorithm to find the least-squares linear model for the data.

# QR Factorization is just rewriting A = QR,
# where Q is an orthonormal basis for Col(A)
# and R is a triangular invertible matrix with positive diagonal entries.

# Find Q and R
Q, R = A.QRdecomposition()

# The least-squares solution with QR factorization is denoted by x_hat = R^-1 * Q_t * b
R_inv = R.inv()
Q_t = Q.T
least_squares = R_inv * Q_t * b # Could also have used A.QRSolve(b)

# print(least_squares)

# Problem 1b) Use the linear model from (a) to the data set validate.data
# and predict the malignancy of the tissues. You will have to define a
# classifier function C(y) = 1 if prediction is non-negative, -1 otherwise

validate_data, actual_malignancy = read_training_data("validate.data")
predicted_malignancy = validate_data * least_squares
# print(len(predicted_malignancy))

# Replaces elements in vector y with +-1 in place
def classifier(y):
    for i in range(len(y)):
        if y[i] >= 0:
            y[i] = 1
        else:
            y[i] = -1
    
    return y

predicted_malignancy = classifier(predicted_malignancy)
print(predicted_malignancy)

# Problem 1c) What is the percentage of samples that are incorrectly classified?
# Is it greater or smaller than the success rate on the training data?

def percentage_error(x,y):
    if len(x) == len(y):
        counter = 0
        for i in range(len(y)):
            if x[i] != y[i]:
                counter += 1
        return counter/len(x)
    else:
        return -1

print("Success rate of linear model applied to validate data:")
print(1 - percentage_error(predicted_malignancy,actual_malignancy))
print("Success rate of linear model applied to training data:")
predicted_malignancy_train_data = A * least_squares
predicted_malignancy_train_data = classifier(predicted_malignancy_train_data)
print(1 - percentage_error(predicted_malignancy_train_data, b))
print("The linear model has a greater success rate on validate data than the training data")


# Setting up the least squares solution (A_t * A * x_hat = A_t * b)
# A_t = A.transpose()
# lhs = A_t*A
# rhs = A_t*b
# concat = (lhs.row_join(rhs))
# concat = concat.rref()
#print(concat)

# coefficients_list = [] #will contain the coefficients for our linear model

# output the equation for the linear model
# print("Linear model: y = ", end="")
# for i in range(len(concat[1])):
#     print((concat[0])[i,len(concat[1])], end="")
#     coefficients_list.append((concat[0])[i,len(concat[1])])
#     print("x", end="")
#     print(i, end="")
#     if i != (len(concat[1]) - 1):
#         print(" + ", end="")

# print("\n\nList of coefficients:\n")
# print(coefficients_list)
# print()


# for i in range(A.shape[0]):
#     for j in range(A.shape[1]):
#         print(str(i) + " " + str(j))




