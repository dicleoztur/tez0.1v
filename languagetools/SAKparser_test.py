# -*- coding: utf-8 -*-

'''
Created on Sep 6, 2012

@author: dicle
'''

#Description: An example python script to use the stochastic morphological parser for Turkish.

import sys
import re
import TurkishMorphology

from txtprocessor import texter

'''
if len(sys.argv) < 2:
    print 'usage:', sys.argv[0], 'corpus[ex:test.txt]'
    exit(1)
'''


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
        print "<S>"
        ws = re.split('\s+', word)
        
        selectedparse = []
        
        for w in ws:
            parses = TurkishMorphology.parse(w)
            if not parses:
                print w, w+"[Unknown]"
                continue
            print w,
            
            parsechoices = []
            for p in parses: #There may be more than one possible morphological analyses for a word
                (parse, neglogprob) = p #An estimated negative log probability for a morphological analysis is also returned
                #print parse, ':', neglogprob,
                parsechoices.append((parse, neglogprob))
            parsechoices.sort(key=lambda tup: tup[1])
            for key,value in parsechoices:
                print key, ":",value
                
            # find the least, append in parselist
            bestparse = parsechoices[0][0]   # take the most probable (least negative log prob valued) parse as the selection
            selectedparse.append(bestparse)
        wordparselist.append((word, selectedparse))        
        print "</S"
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
        for key,value in parsechoices:
            print key, ":",value
            
        # find the least, append in parselist
        bestparse = parsechoices[0][0]   # take the most probable (least negative log prob valued) parse as the selection
        selectedparse.append(bestparse)        
    
    return (word, selectedparse)


def findroot(wordparsepair):
    (literal, parse) = wordparsepair
    root = ""
    rootPOS = ""
    for p in parse:
        morphemes = re.split(r'\][+-]\[', p)   #bizim parsingde sorun var. POSlar iyi gorunmuyor
        rootal = re.split(r'\[', morphemes[0])
        root = root + rootal[0] + " "
        rootPOS = rootPOS + rootal[1] + " "
    return (literal, root, rootPOS)



def lemmatizeword(word):
    wordparsepair = parse_word(word)
    wordrootPOStuple = findroot(wordparsepair)
    return wordrootPOStuple

# morphemes is a list of tuples, [(word/phrase, parse)]
def findroots(parses):    
    wordroots = []
    for i in range(len(parses)):
        
        (literal, parse) = parses[i]
        root = ""
        rootPOS = ""
        for p in parse:
            morphemes = re.split(r'\][+-]\[', p)   #bizim parsingde sorun var. POSlar iyi gorunmuyor
            rootal = re.split(r'\[', morphemes[0])
            root = root + rootal[0] + " "
            rootPOS = rootPOS + rootal[1] + " "
        
        
        '''
        print literal," ** ",parse, " -- ",type(parse)  #expected string (parse is a list since literal may contain >1 words)
        morphemes = re.split(r'\][+-]\[', parse)  
        rootal = re.split(r'\[', morphemes[0])
        print morphemes[0], " ", rootal
        root = rootal[0]
        rootPOS = rootal[1]
        
        rootinfo = (root, rootPOS)
        print "Root: ",rootinfo
        '''
        wordroots.append((literal, root, rootPOS))   # pos root strip()
    return wordroots


def lemmatize(wordlist):
    parses = parse_wordlist(positivewords)
    wordroots =  findroots(parses)      
    return wordroots
 


# parse the parser output - replace verbal words by their roots 
def findverbalroots(wordparsepairs):
    for word,parse in wordparsepairs:
        morphemes = re.split()
    return ""
            


filepath = "/home/dicle/Desktop/Tez/polaritybase/positive.txt"
#parse_corpus(filepath)
'''
positivewords = texter.readtextlines(filepath)
parses = parse_wordlist(positivewords)
wordroots =  findroots(parses)

for (literal, root, POStag) in wordroots:
    print literal," - ",root," : ",POStag 
'''

l = 'karar'
parse2 = lemmatizeword(l)
print parse2

'''
words = ['deger bicmek', 'yarar','gelmek','gitmek']
parses = parse_wordlist(words)

print "PARSES"
for w,p in parses:
    print w," - ", p
    
'''