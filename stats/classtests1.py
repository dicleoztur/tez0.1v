'''
Created on Apr 23, 2013

@author: dicle
'''


# nltk Ch6 p227 doc. classf.

import nltk
from nltk.corpus import movie_reviews
import random
from datetime import datetime


def movie_features(docwordlist, wordfeatures=None):
    docwords = set(docwordlist)
    features = {}
    for word in wordfeatures:
        features['contains(%s)' % word] = (word in docwords)
    return features


def movie_features2(docwordlist):
    wordfreq = nltk.FreqDist(docwordlist)
    features = {}
    for word,occ in wordfreq.iteritems():
        features['%s' % word] = occ
    return features

def getdata(percentsplit=80, numofdocs=100):
    docs = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
    random.shuffle(docs)
    
    if numofdocs == 0:
        numofdocs = len(docs)
    print len(docs)
    numoftrainingexamples = numofdocs * percentsplit / 100
    traindocs, testdocs = docs[:numoftrainingexamples], docs[numoftrainingexamples:]
    return traindocs, testdocs

def getfeatureset(doclist, wordfeatures, choice=1):
    if choice == 1:
        return [(movie_features(d, wordfeatures), c) for (d,c) in doclist]
    else:
        return [(movie_features2(d), c) for (d,c) in doclist]

def classify_movies(featurechoice=1, numofdocs=0):
    traindocs, testdocs = getdata(numofdocs=0)
  
    allwords = nltk.FreqDist(w.lower() for w in movie_reviews.words())
    wordfeatures = allwords.keys()[:2000]    # the first most frequent 2000 words
    
    trainset = getfeatureset(traindocs, wordfeatures, featurechoice)
    testset = getfeatureset(testdocs, wordfeatures, featurechoice)
    classifier = nltk.NaiveBayesClassifier.train(trainset)
    return classifier, testset

def classificationreport(classifier, testdata):
    print "Acc: ",nltk.classify.accuracy(classifier, testdata) 
    print classifier.show_most_informative_features(10)

if __name__ == "__main__":
    
    
    numofdocs = 0
    
    for choice in [1,2]:
        start = datetime.now()
        movieclsfr, movietests = classify_movies(choice, numofdocs)
        classificationreport(movieclsfr, movietests)
        end = datetime.now()
        d = end - start
        print "duration: ",str(d)
    #print nltk.classify.accuracy(movieclsfr, movietests) 
    #print movieclsfr.show_most_informative_features(10)
    
    
    
    
    


    

