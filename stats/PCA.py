'''
Created on Jan 21, 2013

@author: dicle
# http://glowingpython.blogspot.com/2011/07/principal-component-analysis-with-numpy.html

'''

import numpy
from numpy import mean,cov,double,cumsum,dot,linalg,array,rank
from pylab import plot,subplot,axis,stem,show,figure



def principlecompanalysis(A):
    """ performs principal components analysis 
        (PCA) on the n-by-p data matrix A
        Rows of A correspond to observations, columns to variables. 
    
    Returns :  
     coeff :
       is a p-by-p matrix, each column containing coefficients 
       for one principal component.
     score : 
       the principal component scores; that is, the representation 
       of A in the principal component space. Rows of SCORE 
       correspond to observations, columns to components.
    
     latent : 
       a vector containing the eigenvalues 
       of the covariance matrix of A.
    """
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
    [latent,coeff] = linalg.eig(cov(M))
    score = dot(coeff.T,M) # projection of the data in the new space
    return coeff,score,latent

def singularvaldecomp(A):
    return linalg.svd(A)


def result_svd(U, S, Vh):
    
    print "Singular Value Decomposition:"
    print "Coordinates of each word on the concept space"
    print U
    
    print "Singular values"
    print S
    
    print "Coordinates of each feature on the concept space"
    print Vh
    
    

def resultt(A):
    coeff, score, latent = principlecompanalysis(A.T)
    
    figure()
    subplot(121)
    # every eigenvector describe the direction
    # of a principal component.
    m = mean(A,axis=1)
    plot([0, -coeff[0,0]*2]+m[0], [0, -coeff[0,1]*2]+m[1],'--k')
    plot([0, coeff[1,0]*2]+m[0], [0, coeff[1,1]*2]+m[1],'--k')
    plot(A[0,:],A[1,:],'ob') # the data
    axis('equal')
    subplot(122)
    # new data
    plot(score[0,:],score[1,:],'*g')
    axis('equal')
    show()


def result_pca(coeff, score, latent):
    print "Principal Component Analysis:" 
    print "Coefficients of the principal component"
    print coeff
    
    print "\nRepresentation of the matrix in the principal component space"
    print score.T
    
    print "\nEigenvalues of the covariance of the given matrix"
    print latent
    print "# of eigenvalues: ",len(latent)
    print 'The rank of A is'

    

#reconstructs the original data matrix L according to the first k eigenvalues of U.
def reconstruction_from_pca(U, L, k):
    print "begin reconstruction:"
    print "shapeA: ",numpy.shape(L)
    print "shapeU",numpy.shape(U)
    Ureduce = U[:,:k]   # nxk 
    print "shapeUreduce",numpy.shape(Ureduce)
    UreduceTr = numpy.transpose(Ureduce)   # kxn
    Z = numpy.dot(L, Ureduce)   # mxk = mxn * nxk
    Lapprox = numpy.dot(Z, UreduceTr)  # mxn = mxk * kxn
    return Z, Lapprox    





# calls svd, finds singular values and compares the ratio of first k singular values iteratively to the all singular values 
# until 99% of variance in the data is preserved. The point it stops is the optimal value of k, i.e. npc
def find_optimal_npc(L, preserve = 0.99):
    U, S, Vh = singularvaldecomp(L)
    singular_values = numpy.diag(S)
    sum2 = numpy.sum(singular_values)
    
    print "begin NPC"
    #print numpy.shape(S)
    k = 1
    while(True):
        sum1 = numpy.sum(singular_values[:k]) 
        #print "singular values: ", len(singular_values)," ",type(singular_values)
        #print "Sums:",type(sum1)," ",type(sum2)
        ratio = sum1 / sum2
        print k," ",ratio
        
        if ratio >= preserve:
            return k-1
            
        k = k+1
        


# @author: dicle
# convert the given pairmap(wordpair : numofcooccurrences) to mxn matrix where m is the number of rows 
# and n is number of columns and len(pairmap) is n*m
def convert_pairmap2matrix(pairmap, m, n):
    matrix = []
    start = 0
    for i in range(m):
        row = pairmap[start:start+n]
        rowvalues = []
        for item in row:
            (wordpair, numofcooccurrences) = item
            rowvalues.append(numofcooccurrences)
        matrix.append(rowvalues)
        start = start+n
        
    return matrix
            


#main

if __name__ == "__main__":
    
    '''
    L = [[2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
    [1, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 1, 0, 0, 1, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 1, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], 
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    '''
    
    
    # adv by adj co-occurrence matrix (sentencewise)
    L = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    
    
    
    
     
    
    
    L_ = numpy.asarray(L, dtype=double)
    L_ = numpy.transpose(L_)   # for L(mxn), if m < n, we transpose it. just for now.
    Lzeromean = (L_-mean(L_,axis=0)).T
    
    
    Xcov = numpy.cov(Lzeromean)   # covariance matrix of L
    
    print Xcov
    
    print len(Xcov)
    
    coeff, score, latent = principlecompanalysis(Xcov)
    result_pca(coeff, score, latent)
    
    U, S, Vh = singularvaldecomp(Xcov)
    result_svd(U, S, Vh)
    
    
    #reconstruction
    # approximate data points that lie on the projection surface
    k = 4
    Z, Lapprox = reconstruction_from_pca(U, L_, k)
    print Z
    
    print "L approximated"
    print Lapprox
    
    
    
    
    
    '''
    Ureduce = U[:,:k]
    UreduceTr = numpy.transpose(Ureduce)
    Z = numpy.dot(UreduceTr, L)
    print Z
    '''
    
    
    print find_optimal_npc(L)
    
    
    
    
    #numpy.save("/home/dicle/workspacepy/sentimentv2/out/arraytest.npy", Xcov)
    
    #numpy.where(x == np.max(x))
    
    
    
    
    
    
         
    #main
    
    
    
    '''
    print A[2,:]   # 3rd row
    print A[:,2]   # 3rd col
    print "for words"
    print U[:,:3]
    print "for features"
    print Vh[:3,:]
    
    print numpy.argmax(A, axis=1)   # axis=1 for col, 0 for row
    '''
    
    
    ''' 
    A = array([[-1, 1, 2, 2],
               [-2, 3, 1, 0],
               [ 4, 0, 3,-1]],dtype=double)
    
    coeff, score, latent = principlecompanalysis(A)
    
    print "Coefficients of the principal component"
    print coeff
    
    print "\nRepresentation of the matrix in the principal component space"
    print score.T
    
    print "\nEigenvalues of the covariance of the given matrix"
    print latent
    
    print 'The rank of A is'
    print rank(A)  # indeed, the rank of A is 2
    
    result_pca(A)
    '''
    
    
    
    
    #toooo old
    '''
    A = array([ [2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9],
                [2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1] ])
    
    result_pca(A)
    
    ''' 
    
    '''
    perc = cumsum(latent)/sum(latent)
    figure()
    # the following plot shows that first two components 
    # account for 100% of the variance.
    stem(range(len(perc)),perc,'--b')
    axis([-0.3,4.3,0,1.3])
    show()
    print 'the principal component scores'
    print score.T # only the first two columns are nonzero
    print 'The rank of A is'
    print rank(A)  # indeed, the rank of A is 2
    '''
    

