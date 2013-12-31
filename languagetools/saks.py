# -*- coding: utf-8 -*-

'''
Created on Nov 27, 2012

@author: dicle
'''

#Description: An example python script to use the stochastic morphological parser for Turkish.

import sys
import re
import nltk
import TurkishMorphology

from txtprocessor import texter





def parse_corpus(filepath):
    TurkishMorphology.load_lexicon('turkish.fst');
    n = 0
    e = 0
    
    f = open(filepath, 'r')
    for line in f:
        print '<S> <S>+BSTag'
        line = line.rstrip()
        words = re.split('\s+', line)
        for w in words:
            parses = TurkishMorphology.parse(w)
            if not parses:
                print w, w+"[Unknown]"
                continue
            print w,
            for p in parses: #There may be more than one possible morphological analyses for a word
                (parse, neglogprob) = p #An estimated negative log probability for a morphological analysis is also returned
                print parse, ':', neglogprob,
            print
        print '</S> </S>+ESTag'
    f.close()


# returns (phrase,parsing) pairs where 'parsing' is the most probable parse of 'phrase' 
def parse_wordlist(words):
    TurkishMorphology.load_lexicon('turkish.fst');
 
    wordparselist = []  # contains (word/phrase, best parse) list
    for word in words:
        
        wordparsepair = parse_word(word) 
        
        wordparselist.append(wordparsepair)        
        
    return wordparselist


# returns a tuple of (word, parsing)
def parse_word(word):
    TurkishMorphology.load_lexicon('turkish.fst');
 

    ws = re.split('\s+', word)
    
    selectedparse = []
    
    for w in ws:
        parses = TurkishMorphology.parse(w)
        if not parses:
            return (word, " ")
        
        parsechoices = []
        for p in parses: #There may be more than one possible morphological analyses for a word
            (parse, neglogprob) = p #An estimated negative log probability for a morphological analysis is also returned
            #print parse, ':', neglogprob,
            parsechoices.append((parse, neglogprob))
        
        parsechoices.sort(key=lambda tup: tup[1])
        '''for key,value in parsechoices:
            print key, ":",value'''
            
        # find the least, append in parselist
        bestparse = parsechoices[0][0]   # take the most probable (least negative log prob valued) parse as the selection
        selectedparse.append(bestparse)        
    
    return (word, selectedparse)


def findroot(wordparsepair):
    (literal, parse) = wordparsepair
    root = ""
    rootPOS = ""
    for p in parse:
        if p is " ":
            root = literal
        else:
            morphemes = re.split(r'\][+-]\[', p)   #bizim parsingde sorun var. POSlar iyi gorunmuyor            
            rootal = re.split(r'\[', morphemes[0])
            root = root + rootal[0] + " "
            rootPOS = rootPOS + rootal[1] + " "
    
    root = root.decode('utf-8').lower().encode('utf-8')        
    return (literal, root.strip().lower(), rootPOS.strip())



def lemmatizeword(word):
    wordparsepair = parse_word(word)
    wordrootPOStuple = findroot(wordparsepair)
    return wordrootPOStuple

# morphemes is a list of tuples, [(word/phrase, parse)]
def findroots(parses):    
    wordroots = []
    for i in range(len(parses)):    
        wordrootPOStuple = findroot(parses[i])        
        wordroots.append(wordrootPOStuple)   # pos root strip()
    
    return wordroots


def findrootsinlexicon(wordlist):
    parses = parse_wordlist(wordlist)
    wordroots =  findroots(parses)      
    return wordroots
 


# parse the parser output - replace verbal words by their roots 
def findverbalroots(wordparsepairs):
    for word,parse in wordparsepairs:
        morphemes = re.split()
    return ""
            

'''
word = "iran'dan"
print findroot(parse_word(word))
'''


'''


filepath = "/home/dicle/Desktop/Tez/polaritybase/positive.txt"
#parse_corpus(filepath)

positivewords = texter.readtextlines(filepath)
parses = parse_wordlist(positivewords)
wordroots =  findroots(parses)

for (literal, root, POStag) in wordroots:
    print literal," - ",root," : ",POStag 


positivewords1 = [root for (word,root,POS) in wordroots if POS.lower().find("ver") > -1]
positivewords2 = [word for (word,root,POS) in wordroots if POS.lower().find("ver") == -1]

l = 'karar'
parse2 = lemmatizeword(l)
print parse2

print positivewords1


'''

'''
sent = "internet üzerinden yayın yapan pentagona yakın washington free beaconın haberine üzerinden yayın yapan göre meksika körfezi ve amerikanın stratejik sularında bir ay kadar dolaşan rus denizaltıdan bölgeden ayrıldıktan sonra haberdar olundu"
s = sent.split()
bigrams = nltk.bigrams(s)

for item in bigrams:
    print item
    

fd = nltk.FreqDist(bigrams)

print fd," ",type(fd)

fd.plot()
'''