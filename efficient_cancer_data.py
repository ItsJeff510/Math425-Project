# Copyright 2013 Philip N. Klein
from vec import Vec
from vecutil import vec2list
from sympy import Matrix

def read_training_data(fname, D=None):
    """Given a file in appropriate format, and given a set D of features,
    returns the pair (A, b) consisting of
    a P-by-D matrix A and a P-vector b,
    where P is a set of patient identification integers (IDs).

    For each patient ID p,
      - row p of A is the D-vector describing patient p's tissue sample,
      - entry p of b is +1 if patient p's tissue is malignant, and -1 if it is benign.

    The set D of features must be a subset of the features in the data (see text).
    """
    file = open(fname)
    params = ["radius", "texture", "perimeter","area","smoothness","compactness","concavity","concave points","symmetry","fractal dimension"]
    stats = ["(mean)", "(stderr)", "(worst)"]
    feature_labels = set([y+x for x in stats for y in params])
    feature_map = {params[i]+stats[j]:(i*3)+j for i in range(len(params)) for j in range(len(stats))} #changed some variables here
    # ^^^  FIXED: mapping of feature to value isn't right??, it seems it is ordered by 10 means, 10 stderror, then 10 worst
    #             as opposed to the assignment which suggests the order 3 radius, 3 texture, 3 ..., etc.
    if D is None: D = feature_labels

    # print(feature_map)

    feature_vectors = {}
    A = []
    b = []
    for line in file:
        row = line.split(",")
        patient_ID = int(row[0])
        b.append(-1) if row[1] == 'B' else b.append(1)
        feature_vectors[patient_ID] = Vec(D, {f:float(row[feature_map[f]+2]) for f in D})

        #change some stuff so that data is returned in a consistent order
        #print(feature_map) # contains the right order for the keys
        #print((feature_vectors[patient_ID].D))

        # build a list in the correct order, by iterating through feature map keys
        # using the keys as an index into the values in feature_vectors[patient_ID]
        ordered_list = []
        for key in feature_map.keys():
            ordered_list.append((feature_vectors[patient_ID].f).get(key))
            #print(key), print((feature_vectors[patient_ID].f).get(key))
        #print(ordered_list)
        A.append(ordered_list) # appends a consistently ordered list
        # A.append(vec2list(feature_vectors[patient_ID])) # this one stinks, vec2list randomizes the order of the elements
    return Matrix(A), Matrix(b)
        