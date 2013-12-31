# -*- coding: utf-8 -*-

'''
Created on Sep 6, 2012

@author: dicle
'''

#Description: An example python script to use the stochastic morphological parser for Turkish.

import sys
import re
import nltk
import TurkishMorphology
import os

from txtprocessor import texter
from sentimentfinding import IOtools


poslabels = {'Noun' : '(\[Nom|Abl|Loc|Gen|Ins|Acc)', 'ADJ' : 'Adj|NoHats', 'ADV' : 'Adv', 'PostPos' : 'PCNom', 'Conj' : 'Conj', 'Interjection' : 'Interj'}
                


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
        '''
        for key,value in parsechoices:
            print key, ":",value
           ''' 
        # find the least, append in parselist
        bestparse = parsechoices[0][0]   # take the most probable (least negative log prob valued) parse as the selection
        selectedparse.append(bestparse)        
        #print parsechoices
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
    
    #print "root1 ",root
    root = root.decode('utf-8').lower().encode('utf-8')
    #print "root2 ",root        
    return (literal, root.strip().lower(), rootPOS.strip())



def lemmatizeword(word):
    wordparsepair = parse_word(word)
    (literal, root, rootPOS) = findroot(wordparsepair)
    (literal, literalPOS) = find_wordparsepair_POStag(wordparsepair)
    
    p = r"\].*$"
    rootPOS = re.sub(p, "", rootPOS)
    literalPOS = re.sub(p, "", literalPOS)
    return (literal, literalPOS, root, rootPOS)


def lemmatize_lexicon(wordlist):
    lemmata = []
    for word in wordlist:
        lemmaquadruple = lemmatizeword(word)
        lemmata.append(lemmaquadruple)
    return lemmata

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



def find_word_POStag(word):
    wordparsepair = parse_word(word)
    return find_wordparsepair_POStag(wordparsepair)

def find_wordparsepair_POStag(wordparsepair):
    (literal, parse) = wordparsepair   
    #root = ""
    literalPOS = ""
    
    for p in parse:
        if p is " ":
            root = literal
        else:
            morphemes = re.split(r'\][+-]', p)   #bizim parsingde sorun var. POSlar iyi gorunmuyor            
            
            if morphemes:
                lastmorpheme = morphemes[-1]  # to find if a word is adj or adv, the label of the last morphem is inspected because it carries that info
                #print lastmorpheme
                
                forms = re.split(r'\[', lastmorpheme)
                morph_surface = forms[0]
                morph_lexical = forms[1]
                #print morph_surface,"  ",morph_lexical 
                morph_lexical = "["+morph_lexical
                
                #pronoun missing
                #poslabels = {'Noun' : '(\[Nom|Abl|Loc|Gen|Ins|Acc)', 'ADJ' : 'Adj|NoHats', 'ADV' : 'Adv', 'PostPos' : 'PCNom', 'Conj' : 'Conj', 'Interjection' : 'Interj'}
                posindex = -1
                for k,v in poslabels.items():
                    #posindex = morph_lexical.find(v)
                    #print k," ",v," ::: ",posindex
                    m = re.search(v, morph_lexical)
                    
                    '''
                    if posindex > -1:
                        literalPOS = k
                        break
                    '''
                    if m:
                        literalPOS = k
                        break
                    
                if literalPOS == '':
                    literalPOS = 'Verb'    
               
                #print "literal POS: ",literalPOS,"  ",morph_lexical   
    
    return (literal, literalPOS)
    #return (literal, root.strip().lower(), rootPOS.strip())  
    



###   improve    
def getcooccurrence_insent(word, wordsofsentence):
    #words = sent.getwords()
    count = wordsofsentence.count(word)
    return (word, count)

    

if __name__ == "__main__":
    
    s = "atağı"
    print parse_word(s)
    print lemmatizeword(s)
    
    
    
    
    '''
    fname = "tr_strongsubjective.txt"
    trpath = "/home/dicle/Dicle/Tez/languageresources/tr_subjectivitylexicon/edit/"
    words = texter.readtextlines(trpath+os.sep+fname)
    words = list(set(words))
    lemmata = lemmatize_lexicon(words)
    print lemmata
    #lemmata = [(x.decode('utf-8'), y.decode('utf-8'), z.decode('utf-8'), t.decode('utf-8')) for (x,y,z,t) in lemmata]
    lemmata = [" ".join(item) for item in lemmata]
    txt = "\n".join(lemmata)
    IOtools.todisc_txt(txt, trpath+os.sep+"lemmatized_"+fname)
    #IOtools.todisc_list(trpath+os.sep+"lemmatized_"+fname, lemmata)
    '''

'''
word = "iran'dan"
print findroot(parse_word(word))
'''

'''
word = "hey"
wordparsepair = parse_word(word)
print "parse_word ",wordparsepair
print

print find_wordparsepair_POStag(wordparsepair)




#get a txtfile and split it to its sentences
txt = texter.readtxtfile("/home/dicle/Dicle/Tez/dataset/readingtest/402970.txt")
sents = texter.splitToSentences(txt)


# find wordlist of the txt. get morphparses and pos tags of words
words = texter.eliminatepunctuation(re.split(r'\s+', txt))
parses = parse_wordlist(words)

wordposdict = {}
for w,p in zip(words, parses):
    (wrd, tag) = find_wordparsepair_POStag(p)
    
    if not wordposdict.has_key(tag):
        wordposdict[tag] = []
    else:
        wordposdict[tag].append(wrd)
    print w," : ",tag
    
IOtools.printdictionary(wordposdict)

for k,v in wordposdict.items():
    s = ""
    for item in v:
        s = s + item + ", "
    s = s + "\n"
    print k," : ",s    

'''
   
    

#print lemmatizeword("veren")



'''

filepath = "/home/dicle/Desktop/Tez/polaritybase/positive.txt"
parse_corpus(filepath)


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
sent = u"internet üzerinden yayın yapan pentagona yakın washington free beaconın haberine üzerinden yayın yapan göre meksika körfezi ve amerikanın stratejik sularında bir ay kadar dolaşan rus denizaltıdan bölgeden ayrıldıktan sonra haberdar olundu"
s = sent.split()
bigrams = nltk.bigrams(s)

for item in bigrams:
    k, v = item
    t = (k.encode('utf-8'),)
    print item," -> ",k," ",t[0]
    

fd = nltk.FreqDist(bigrams)

print fd," ",type(fd)

fd.plot()
'''