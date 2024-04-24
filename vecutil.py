# Copyright 2013 Philip N. Klein
from vec import Vec

def list2vec(L):
    """Given a list L of field elements, return a Vec with domain {0...len(L)-1}
    whose entry i is L[i]

    >>> list2vec([10, 20, 30])
    Vec({0, 1, 2},{0: 10, 1: 20, 2: 30})
    """
    return Vec(set(range(len(L))), {k:L[k] for k in range(len(L))})

def zero_vec(D):
    """Returns a zero vector with the given domain
    """
    return Vec(D, {})

def vec2list(d):
    """
    Given a Vec, return a list of the dictionary values in it's field 
    >>> vec2list(Vec({0, 1, 2},{0: 10, 1: 20, 2: 30}))
    [10, 20, 30]


    """
    return list((d.f).values())