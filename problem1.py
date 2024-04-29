from efficient_cancer_data import read_training_data

A, b = read_training_data("train.data")
#print(A)
A_t = A.transpose()
lhs = A_t*A
rhs = A_t*b
concat = (lhs.row_join(rhs))
concat = concat.rref()
#print(concat)

coefficients_list = [] #will contain the coefficients for our linear model

# output the equation for the linear model
print("Linear model: y = ", end="")
for i in range(len(concat[1])):
    print((concat[0])[i,len(concat[1])], end="")
    coefficients_list.append((concat[0])[i,len(concat[1])])
    print("x", end="")
    print(i, end="")
    if i != (len(concat[1]) - 1):
        print(" + ", end="")

print("\n\nList of coefficients:\n")
print(coefficients_list)





#x = A.QRsolve(b)

#print(x)

# for i in range(A.shape[0]):
#     for j in range(A.shape[1]):
#         print(str(i) + " " + str(j))




