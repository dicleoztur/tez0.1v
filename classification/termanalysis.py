'''
Created on May 22, 2013

@author: dicle
'''

import numpy as np
import codecs
import os
import math
from scipy import linalg

from sentimentfinding import IOtools
import languagetools
from languagetools import SAKsParser




# her sey csv den okunmali

def get_termdoc_matrix(matrixpath):
    lines = IOtools.readtextlines(matrixpath)
    doclist = []
    
    header = lines[0]
    terms = header.split()
    terms = map(lambda x : x.strip(), terms)
    
    matrix = []
    for i in range(1,len(lines)):
        items = lines[i].split()
        doclist.append(items[0])
        values = [float(val) for val in items[1:-1]]
        matrix.append(values)
    
    return np.array(matrix), terms, doclist
        
        
    '''
    f = codecs.open(matrixpath, "r", encoding='utf8')
    
    line = f.readline()
    terms = line.split()
    terms = map(lambda x : x.strip(), terms)
    
    matrix = []
    while line:
        line = f.readline()
        items = line.split()
        values = items[1:-1]
        values = [float(val) for val in values]
        print type(values)," ",len(values)
        matrix.append(values)
        #print line
    f.close()
    
    matrix = np.array(matrix)
    return matrix
    '''

def get_doc_NOUN_matrix(matrix, terms):
    nounindices = []
    nouns = []
    
    for i,term in enumerate(terms):
        _, postag = SAKsParser.find_word_POStag(term)
        if postag == "Noun":
            nounindices.append(i)
            nouns.append(term)
    
    #docNounmatrix = np.array((matrix.shape[0], len(nounindices)))
    docNounmatrix = matrix[:, nounindices]
    
    '''
    for i in nounindices:
        values = map(lambda x : [x], matrix[:,i].tolist())
        print len(values)," ",values
        docNounmatrix = np.append(docNounmatrix, values, axis=1)
        '''
    
    return docNounmatrix, nouns


def find_tfidf(doctermmatrix):        
    """ 
    MODIFY FROM http://blog.josephwilk.net/projects/latent-semantic-analysis-in-python.html
    tfidfmatrix is a numpy array
    Apply TermFrequency(tf)*inverseDocumentFrequency(idf) for each matrix element. 
        This evaluates how important a word is to a document in a corpus
           
        With a document-term matrix: matrix[x][y]
        tf[x][y] = frequency of term y in document x / frequency of all terms in document x
        idf[x][y] = log( abs(total number of documents in corpus) / abs(number of documents with term y)  )
        Note: This is not the only way to calculate tf*idf
    """


    numofdocs = doctermmatrix.shape[0]
    rows,cols = doctermmatrix.shape
    tfidfmatrix = np.array(doctermmatrix, copy=True)
   
    for row in xrange(0, rows): #For each document
       
        wordTotal= reduce(lambda x, y: x+y, tfidfmatrix[row] )

        for col in xrange(0,cols): #For each term
        
            #For consistency ensure all self.matrix values are floats
            tfidfmatrix[row][col] = float(tfidfmatrix[row][col])

            if tfidfmatrix[row][col]!=0:
                termDocumentOccurences = sum(tfidfmatrix[:,col])
                termFrequency = tfidfmatrix[row][col] / float(wordTotal)
                inverseDocumentFrequency = math.log(abs(numofdocs / float(termDocumentOccurences)))
                tfidfmatrix[row][col]=termFrequency*inverseDocumentFrequency

    return tfidfmatrix



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
        

def lsa_transform(matrix,dimensions):
    """ Calculate SVD of objects matrix: U . SIGMA . VT = MATRIX 
        Reduce the dimension of sigma by specified factor producing sigma'. 
        Then dot product the matrices:  U . SIGMA' . VT = MATRIX'
    """
    rows,cols= matrix.shape

    if dimensions <= rows: #Its a valid reduction

        #Sigma comes out as a list rather than a matrix
        u,sigma,vt = linalg.svd(matrix)

        #Dimension reduction, build SIGMA'
        for index in xrange(rows-dimensions, rows):
            sigma[index]=0

        #print linalg.diagsvd(sigma,len(self.matrix), len(vt))        

        #Reconstruct MATRIX'
        reconstructedMatrix= np.dot(np.dot(u,linalg.diagsvd(sigma,len(matrix),len(vt))),vt)

        #return transform
        return reconstructedMatrix

    else:
        print "dimension reduction cannot be greater than %s" % rows
        


def get_N_terms(reducedmatrix, N):
    topicterms = []
    
    rows, cols = reducedmatrix.shape
    
    for i in range(rows):
        termindexpairs = []
        termindexpairs = [(j, value) for j,value in enumerate(reducedmatrix[i])]
        termindexpairs.sort(key=lambda tup : tup[1], reverse=True)
        #indices = [indis for indis,_ in termindexpairs]
        
        #topicterms.append(indices[:N])
        topicterms.append(termindexpairs[:N])
    return topicterms

def report_topic_terms(topictermmatrix, doclist, termlist):
    rows = len(doclist)
    cols = len(topictermmatrix[0])
   
    for i in range(rows):
        print "topic terms in ",doclist[i]
        for j in range(cols):
            indis,val = topictermmatrix[i][j]
            #print indis," ",val
            print "   ",termlist[indis]," : ",val
        print
    
            

if __name__ == "__main__":
    matrixpath = "/home/dicle/Dicle/Tez/output/CLASSTEST/docterm60.m"
    
    docTermmatrix, terms, doclist = get_termdoc_matrix(matrixpath)
    print docTermmatrix.shape
    
    
    '''
    for term in terms:        
        w, postag = SAKsParser.find_word_POStag(term)
        print w," ",postag
    '''
    #print matrix[rows-1,cols-10 : cols]
    
    
    nounmatrix, nouns = get_doc_NOUN_matrix(docTermmatrix, terms)
    print nounmatrix.shape
    
    outpath = "/home/dicle/Dicle/Tez/output/topicdetect/"
    IOtools.todisc_matrix(nounmatrix, outpath+os.sep+"nounmatrix60docs.m")
    
    nountfidfmatrix = find_tfidf(nounmatrix)
    IOtools.todisc_matrix(nountfidfmatrix, outpath+os.sep+"nounTFIDFmatrix60docs.m")
    
    lsa_tfidfmatrix = lsa_transform(nountfidfmatrix)
    lsa_occrmatrix = lsa_transform(nounmatrix)
    
    IOtools.todisc_matrix(lsa_tfidfmatrix, outpath+os.sep+"lsa_nounTFIDFmatrix60docs.m")
    IOtools.todisc_matrix(lsa_occrmatrix, outpath+os.sep+"lsa_doctermmatrix60docs.m")
    
    
    # get topic terms
    
    N = 10
    
    '''
    termindices = get_N_terms(lsa_occrmatrix, N)
    report_topic_terms(termindices, doclist, nouns)
    
    '''
    termindices = get_N_terms(lsa_tfidfmatrix, N)
    report_topic_terms(termindices, doclist, nouns)
    
    
    '''
    print nouns[:10]
    print lsa_tfidfmatrix[:5,:10]
    print nountfidfmatrix[:5,:10]
    '''
    # lsa perform both doctermocc and tfidf matrix
    
    
    