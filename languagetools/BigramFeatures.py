# -*- coding: utf-8 -*-

'''
Created on Sep 13, 2012

@author: dicle
'''

import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import pylab as plt
import numpy as np

import nltk
from nltk.collocations import *
from txtprocessor import texter





# MAIN

'''
Notlar:
- tag'lerin (mesela polarity) bigramlarını bulabiliriz.


'''


def viewbigramscores(bigramscorepairs):
    for wordpair, score in bigramscorepairs:
        w1, w2 = wordpair
        print w1,", ",w2," : ",score

bigram_measures = nltk.collocations.BigramAssocMeasures()

path = "/home/dicle/Dicle/Tez/dataset/readingtest/402966.txt"
rawtext = texter.readtxtfile(path)
words = nltk.wordpunct_tokenize(rawtext)

finder = BigramCollocationFinder.from_words(words)

bigramscores = {}

raw_nbestbigrams = finder.score_ngrams(bigram_measures.raw_freq)
bigramscores["raw_freq"] = raw_nbestbigrams
chi_nbestbigrams = finder.score_ngrams(bigram_measures.chi_sq)
bigramscores["chi_sq"] = chi_nbestbigrams
lratio_nbestbigrams = finder.score_ngrams(bigram_measures.likelihood_ratio)
bigramscores["like_ratio"] = lratio_nbestbigrams
pmi_nbestbigrams = finder.score_ngrams(bigram_measures.pmi)
bigramscores["pmi"] = pmi_nbestbigrams



# nbestbigrams = finder.nbest(bigram_measures.pmi, 10)



xitems = [wordpair for wordpair, score in raw_nbestbigrams]


pairscoresdict = {}

dct = {}
dct["a"] = [1,2]
dct["b"] = [8,9]
dct["a"].insert(7,9)
print dct

for bigramitem in xitems:
    pairscoresdict[bigramitem] = []
    pairscoresdict[bigramitem] = [0 for i in range(len(bigramscores.keys()))]

for i,metric in enumerate(bigramscores.keys()):
    for bigramitem, score in bigramscores[metric]:
        # pairscoresdict[bigramitem] = []
        pairscoresdict[bigramitem][i] = score
        print metric," ",i," ",score," : ",bigramitem," ",pairscoresdict[bigramitem]

print len(xitems)," vs ",len(pairscoresdict)   
colors = "bgrcmykw"


for k,v in pairscoresdict.iteritems():
    w1, w2 = k
    print w1,",",w2,"  : ",v



plt.figure()
xitems = pairscoresdict.keys()
'''
for i,metric in enumerate(bigramscores.keys()):
    yitems = [0 for x in range(len(pairscoresdict.keys()))]
    for j,pair in enumerate(pairscoresdict.keys()):
        yitems[j] = round(pairscoresdict[pair][i],3)
        #print "j: ",j," : ",pairscoresdict[pair][i]
     
    xitems = map(lambda x : repr(x), xitems)
    print xitems[0]," ",yitems[0],type(yitems[0]),len(xitems)," ",len(yitems)
    plt.plot(xitems, yitems,  linestyle="-", label=metric)    #color=colors[i],
    plt.xticks(np.arange(len(xitems),dtype="int"), xitems, rotation=90)
    plt.xlim(range(len(xitems)))
'''
s = [item[0] for item in pairscoresdict.values()]
print len(xitems)," ",len(s)

plt.xticks(np.arange(len(xitems),dtype="int"), xitems, rotation=90)
#plt.xlim(range(len(xitems)))
plt.plot(xitems, s,  linestyle="-", label=metric)    #color=colors[i],

  
plt.legend(loc="upper-left")
plt.show()

#for item, scores in pairscoresdict:
    #plot(xitems, )

# add subplots



for (k,v) in chi_nbestbigrams:
    print k,", ",v


'''
http://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/

def evaluate_classifier(featx):
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')
 
    negfeats = [(featx(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(featx(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
 
    negcutoff = len(negfeats)*3/4
    poscutoff = len(posfeats)*3/4
 
    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
 
    classifier = NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
 
    for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
 
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
    print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
    print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
    print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
    print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
    classifier.show_most_informative_features()
    
 
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
 
evaluate_classifier(bigram_word_feats)


'''