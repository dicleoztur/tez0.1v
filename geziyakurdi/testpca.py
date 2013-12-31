'''
Created on Jul 19, 2013

@author: dicle
'''

import os
import numpy as np
import pandas as pd
import pylab as plt

from sentimentfinding import IOtools

rootpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/pcatests/"

def read_matrix(path):
    data = pd.read_csv(path, index_col=0)
    #matrix = data.values
    return data



# performs column-wise 0-mean
def get_zeromean_matrix(matrix, axis=0):
    mean_vect = matrix.mean(axis=0)
    mean0_matrix = np.subtract(matrix, mean_vect)
    return mean0_matrix   


def _pca(matrix, nosort=True, kDim=0):
  
    mean0matrix= get_zeromean_matrix(matrix)
    print "org: ",matrix.shape
    print "mean0: ",mean0matrix.shape
    
    Sigma = np.cov(mean0matrix)
    
    print "sigma: ",Sigma.shape
    
    eigvalues, eigvectors = np.linalg.eig(Sigma)
    
    print "eigvl: ",eigvalues.shape
    print "eigvc: ",eigvectors.shape
    
    # caution
    del Sigma
    
    
    if not nosort:
        indices = np.argsort(eigvalues)
        indices = indices[::-1]
        eigvectors = eigvectors[:,indices]
        
        print "indices: ",np.max(indices)
        
        
        eigvalues = eigvalues[indices]
 
    
 
    if kDim > 0:
        eigvectors = eigvectors[:,:kDim]
    
    projecteddata = np.dot(eigvectors.T, mean0matrix)
    projecteddata = np.real(projecteddata)
    return projecteddata   # return eigvals !
    

def apply_pca(inpath, outpath):
    data = read_matrix(inpath)
    matrix = data.values
        
    projectedmatrix = _pca(matrix)
    projecteddata = pd.DataFrame(projectedmatrix, index=data.index.values.tolist(), columns=data.columns.values.tolist())
    return projecteddata




def analyse_principal_components(projectedmatrix):
    return


def visualize_vectors(matrix, indexrange):
    for i in indexrange:
        v = matrix[:, i]
        print len(v)
        plt.plot(range(len(v)), v)
    plt.legend(indexrange)
    plt.xticks(range(0, len(matrix[:,0]), 15), range(0, len(matrix[:,0]), 15), rotation=90)
    plt.show()

def visualize_matrix(Sigma, path, xlabels=[]):
    #plt.pcolor(Sigma)
   
    plt.pcolor(Sigma)
    v = np.linspace(np.min(Sigma), np.max(Sigma), 20, endpoint=True)
    plt.colorbar(ticks=v)
    m,n = Sigma.shape
    
    if not xlabels:
        plt.xticks(range(0, m, 50))
        plt.yticks(range(0, n , 50))
    else:
        r = np.arange(len(xlabels))+0.5
        r = r.tolist()
        plt.xticks(r, xlabels,rotation=90)
        plt.yticks(r, xlabels)
    plt.savefig(path,dpi=200)
    plt.show()

# indis mi dondurelim?
# num of zeros with indices
def covariance_analysis(Sigma):
    covs = Sigma.diagonal()
    covs = np.sort(covs)
    covs = covs[::-1]
    return covs  



# since the matrix is compressed, we map selected indices to the original ones to see the real words in effect
def view_words(mapped, original, wordlist):
    for i in mapped:
        print i," ",original[i],"   ",wordlist[original[i]]
        



