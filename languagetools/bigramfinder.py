'''
Created on Sep 27, 2012

@author: dicle
'''

import nltk
from nltk.collocations import *

import numpy
from pylab import *
from nltk.cluster import KMeansClusterer, euclidean_distance

import math


def bigram_bywordlist(wordlist):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    
    finder = BigramCollocationFinder.from_words(wordlist)
    x = finder.nbest(bigram_measures.pmi, 10) 
    #scored = finder.score_ngrams( bigram_measures.likelihood_ratio  )
    scored = finder.score_ngrams(bigram_measures.raw_freq )
    return scored


# inserts missing possibilities of bigrams in the bigramtuples where bigramtuples contains a tag pair's frequency of occurrence
# which may not include a possible pair not seen in the data
def ensure_allTagPairs(bigramtuples, tags):   
    tagpairs = []
    tagpairs = [(tag1, tag2) for tag1 in tags for tag2 in tags]   # all possibilities of tag pairs
        
    # find missing pairs
    currenttags = []
    for k,v in bigramtuples:
        currenttags.append(k)
    
    missingtags = []
    for tag in tagpairs:
        if tag not in currenttags:
            missingtags.append(tag)
    for mtag in missingtags:
        bigramtuples = bigramtuples + [(mtag, 0.0)]
    
    bigramdict = {}
    for k,v in bigramtuples:
        bigramdict[k] = v
    
    return bigramdict
 

def bigram_bytags(wordtagtuples, taglist):

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    
    finder = BigramCollocationFinder.from_words(t for w, t in wordtagtuples)
    x = finder.nbest(bigram_measures.pmi, 10) 
    #scored = finder.score_ngrams( bigram_measures.likelihood_ratio  )
    scored = finder.score_ngrams(bigram_measures.raw_freq )
    return ensure_allTagPairs(scored, taglist)
     





def sumnormalise_matrix(matrix):
    smatrix = []
    for row in matrix:
        srow = sum_normalise(row)
        smatrix.append(srow)
        print "type srow ",type(srow)
        print "type row ",type(row)
    print "type smatrix ",type(smatrix)
    print "type matrix ",type(matrix)
    return smatrix
        
        

def sum_normalise(vec):
    return vec / float(sum(vec))

# if frequency is not important, use this one
def length_normalise(vec):
    return vec / numpy.sqrt(numpy.dot(vec, vec))

def euclidean_distance(u, v):
    diff = u - v
    return math.sqrt(numpy.dot(diff, diff))

def cosine_similarity(u, v):
    m = numpy.dot(u, v)
    n = numpy.linalg.norm(u) * numpy.linalg.norm(v)
    if n == 0.0:
        return 0.0
    return m / n

# if notarray: cmatrix = {instanceID : values} else cmatrix is a list of arrays that contain values
def clusterer(cmatrix, numofclusters, notarray = False):
    matrix = []
    if notarray:
        matrix = [ array(row) for row in cmatrix.values() ]
    else:
        matrix = cmatrix
    clusterer = nltk.KMeansClusterer(numofclusters, euclidean_distance, normalise=True)
    clusters = clusterer.cluster(matrix, assign_clusters=True, trace=False)
        
    print "cluster assignments: ",clusters
    print " means: ", clusterer.means()
    txt_cluster = {}    # {txtid : cluster found to be member of}
    for i, cluster_index in enumerate(clusters):      # bu membership i yazdirmak lazim
        #print clusterer.cluster_name[cluster_index]
        print "Text ",cmatrix.keys()[i]," : ",cluster_index
        txt_cluster[cmatrix.keys()[i]] = cluster_index
    
    cl_asgnmntlist = {}
    for i in range(numofclusters):
        l = []
        for (j,clindex) in enumerate(clusters):
            if i == clindex:
                l.append(cmatrix.keys()[j])
            #print str(i)," ",str(j)," ",clindex," ",cmatrix.keys()[j]," ",l
        cl_asgnmntlist[i] = l
    
    #print cmatrix.keys(),"  ",cl_asgnmntlist
    
    for cl_index, members in cl_asgnmntlist.iteritems():
        print str(cl_index)," members:"
        print members
        
    return txt_cluster




'''
tags = ['neut','pos','neg']

sent1 = (('yayin','pos'),('habere','pos'),('gore','neut'),('habere','pos'),('gore','neut'),('rusyanin','neut'),('en','neut'),('sessiz','neg'),('saldiri','neg'),('gucu','pos'),('endise','neg'))
sent2 = (('yayin','pos'),('habere','pos'),('gore','neut'),('habere','pos'),('gore','pos'),('rusyanin','neut'),('en','neg'),('sessiz','neg'),('saldiri','neg'),('gucu','neg'),('endise','pos'))
sent3 = (('yayma','neg'),('habere','pos'),('gore','neg'),('habere','pos'),('gore','neut'),('rusyanin','neut'),('en','pos'),('sessiz','pos'),('saldiri','neg'),('gucu','pos'),('endise','neut'))

score1 = bigram_bytags(sent1, tags)
score2 = bigram_bytags(sent2, tags)
score3 = bigram_bytags(sent3, tags)

print score1
print score2

scorematrix = []
scorematrix.append(score1.values())
scorematrix.append(score2.values())
scorematrix.append(score3.values())

print len(score1)," ",len(score2)," ",len(score3)
print score1

for row in scorematrix:
    print row


clustermatrix = [ array(row) for row in scorematrix ]
print clustermatrix

# normalise

smatrix = sumnormalise_matrix(clustermatrix)
print "smatrix"
print smatrix

numofclusters = 2
clusterer(scorematrix, numofclusters,True)   
a1 = smatrix[0]
a2 = smatrix[1]
print type(a1)," ",a1[0]
print type(a2)," ",a2[0]
print euclidean_distance(smatrix[0], smatrix[1])

'''


'''
# y is a multidim list
array( [[i for i in row] for row in y])

'''


'''
wordpolpairs = (('yayin','pos'),('habere','pos'),('gore','neut'),('habere','pos'),('gore','neut'),('rusyanin','neut'),('en','neut'),('sessiz','neg'),('saldiri','neg'),('gucu','pos'),('endise','neg'))

scored = bigram_bytags(wordpolpairs)
scored2 = bigram_bywordlist([w for w,t in wordpolpairs])
print scored
print scored2
'''

'''
for k,v in prefix_keys.iteritems():
    print k," ",v
'''
