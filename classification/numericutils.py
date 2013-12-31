'''
Created on Feb 5, 2013

@author: dicle
'''

import math
import numpy as np
from nltk import FreqDist
import numpy as np

# round the float values of the input dictionary
def roundvalues(dct):
    for k,v in dct.items():
        dct[k] = round(v, 4)
    return dct



def sum_normalise(vec):
    return vec / float(sum(vec))

# if frequency is not important, use this one
def length_normalise(vec):
    return vec / np.sqrt(np.dot(vec, vec))

def euclidean_distance(u, v):
    diff = u - v
    return math.sqrt(np.dot(diff, diff))

def cosine_similarity(u, v):
    m = np.dot(u, v)
    n = np.linalg.norm(u) * np.linalg.norm(v)
    if n == 0.0:
        return 0.0
    return m / n

def row_normalize(matrix):
    normalisedmatrix = matrix.copy()
    for i,row in enumerate(matrix):
        vlength = np.sqrt(np.dot(row, row))
        if vlength > 0.0:
            row = row / vlength
        normalisedmatrix[i] = row
    return normalisedmatrix
        

# returns the second maximum value and its index of the input np array
def get2ndmax(x):
    n = x.argmax()
    t = np.append(x[ : n ], x[n+1 : ])
    n2 = t.argmax()
    max2 = t.max()
    if n2 >= n:
            n2 = n2+1
    return max2, n2



# row-wise element reduction (eliminate some rows) improce to include columnwise operation
def compress_term_matrix(matrix, words):
    initials = [item[0] for item in words]
    
    fdist = FreqDist(initials)
    
    letterindices = []
    for letter in sorted(fdist.keys()):
        letterindices.append((letter, fdist[letter]))
    
    indexmatrix = []
    start = 0
    for letter, occ in letterindices:
        newocc = occ / 5
        
        print letter,"  ",occ
        print " range: ", start,"  ", start+occ,"  ",newocc
        indexes = np.random.random_integers(start, start+occ, newocc)
        indexmatrix.append((letter, indexes.tolist()))
        start = start+ occ
    
    allindices = []
    for _,v in indexmatrix:
        allindices.extend(v)
    smatrix = matrix[allindices, :]
    return indexmatrix, smatrix                
                             
    

if __name__ == "__main__":
    x = [[0.5, 0.7, 1.2],[1.2, 0.8, 2.3], [0,0,0]]
    x = np.array(x)
    nx = row_normalize(x)
    for vec in nx:
        print np.sqrt(np.dot(vec, vec))
    
    
    print cosine_similarity(x[0], x[1])
    print cosine_similarity(nx[0], nx[1])
    
    print "row normalized"
    print nx
    
    xzeromean = (x-np.mean(x,axis=0))
    print "0-mean"
    print xzeromean
    
    print "x sigma"
    print np.cov(xzeromean)
    
    
    
