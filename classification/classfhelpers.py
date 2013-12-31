'''
Created on May 21, 2013

@author: dicle
'''

import numpy as np
import codecs
from mpl_toolkits.mplot3d import Axes3D

from sentimentfinding import IOtools



def classlabelindicing(labels):
    classlabels_encode = {}
    classlabels_decode = {}
    for i,label in enumerate(labels):
        classlabels_encode[label] = i
        classlabels_decode[i] = label
    return classlabels_encode, classlabels_decode



def get_matrix(matrixpath, N=0, outfeatures=[]):
    lines = IOtools.readtextlines(matrixpath)
    
    
    classinfo = lines[0]
    classes = classinfo.split(",")
    classes = map(lambda x : x.strip(), classes)
    classencoding, _ = classlabelindicing(classes)

    '''
    header = lines[1]
    items = header.split("\t")
    # HANDLE CHOOSING FEATURES - featureindices, how to pass them as indices or names.. ??? 
    features = items[1:-1]
    '''
    
    numofdatapoints = N
    if N == 0:
        numofdatapoints = len(lines)-2
    
    
    X = []
    Y = []
    for i in range(2,numofdatapoints):
        items = lines[i].split()
        classlabelindex = classencoding[items[-1]]    # class encoding
        values = [float(val) for val in items[1:-1]]
        X.append(values)
        Y.append(classlabelindex)

    X = np.array(X)
    Y = np.array(Y)    
    
    return X, Y


def get_matrix_metadata(matrixpath):
    f = codecs.open(matrixpath,"r", encoding='utf8')
    classinfo = f.readline()
    classlabels = classinfo.split(",")
    classlabels = map(lambda x : x.strip(), classlabels)
    
    header = f.readline()
    items = header.split("\t") 
    featurenames = items[1:-1]

    count = 0
    while f.readline():
        count += 1
    numofpoints = count
    f.close()
    
    return classlabels, featurenames, numofpoints


if __name__ == "__main__":
    
    p = "/home/dicle/Dicle/Tez/output/CLASSTEST/t60.m"
    print get_matrix_metadata(p)
    

