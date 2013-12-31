'''
Created on Jan 29, 2013

@author: dicle
'''

import random
import numpy
from numpy import mean,cov,double,cumsum,dot,linalg,array,rank
from pylab import plot,subplot,axis,stem,show,figure




# returns a double matrix of type numpy.array containing random numbers in the range rnd_floor and rnd_ceil
def random_matrix(rows, cols, rnd_floor, rnd_ceil):
    M = []
    for i in range(rows):
        row = []
        row = random.sample(range(rnd_floor, rnd_ceil),cols)
        M.append(row)
    return numpy.asarray(M, dtype=double)





A = array([[-1, 1, 2, 2],
           [-2, 3, 1, 0],
           [ 4, 0, 3,-1]],dtype=double)

B = random_matrix(4, 6, 2, 10)


C = numpy.dot(A, B)
print C


print B


D = numpy.zeros((5,5))
D[0,0] = 1
D[1,1] = 2
D[2,2] = 3
D[3,3] = 6
d = numpy.diag(D)

l = [i for i in d]
t = sum(d[:2])
print type(t)
print type(d) 
print len(B)

print A[:,:2]


M = (A-mean(A.T,axis=1)).T
print M
print cov(M)


    
    
    
    
    