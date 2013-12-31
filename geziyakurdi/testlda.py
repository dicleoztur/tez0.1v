'''
Created on Jul 22, 2013

@author: dicle
'''

import numpy as np
from scipy import linalg
import pandas as pd




def apply_svd(tfidfmatrix):
    u, sigma, vt = np.linalg.svd(tfidfmatrix)
    n,m = tfidfmatrix.shape
    sigma = linalg.diagsvd(sigma, n, m)
    return u, sigma, vt



# calls svd, finds singular values and compares the ratio of first k singular values iteratively to the all singular values 
# until 99% of variance in the data is preserved. The point it stops is the optimal value of k, i.e. npc
def find_optimal_npc(sigma, preserve = 0.99):
    #print "sigma dim: ",sigma.shape
    singular_values = sigma.diagonal()
    
    n,m = sigma.shape
    if n < m:
        singular_values = np.append(singular_values, np.zeros(m-n))
    
    total = np.sum(singular_values)
    
    print "begin NPC"
    #print numpy.shape(S)
    k = 1
    while(True):
        sum1 = np.sum(singular_values[:k]) 
        #print "singular values: ", len(singular_values)," ",type(singular_values)
        #print "Sums:",type(sum1)," ",type(sum2)
        ratio = sum1 / total
        print k," ",ratio
        
        if ratio >= preserve:
            return k-1
            
        k = k+1
    return k


def reconstruct(u, sigma, vt, k):
    reconstructedmatrix = np.dot(np.dot(u[:,:k], sigma[:k,:k]), vt[:k, :k])
    return reconstructedmatrix




if __name__ == "__main__":
    print np.__version__

   
        

