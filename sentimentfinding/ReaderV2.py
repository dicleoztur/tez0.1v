'''
Created on Mar 9, 2013

@author: dicle
'''
import os
from datetime import datetime
from nltk import FreqDist

import IOtools
from languagetools import SAKsParser
import keywordhandler

from Word import Word
from Sentence import Sentence
from Text import Text
from Corpus import Corpus
from txtprocessor import texter
from txtprocessor import listutils


def buildcorpus(corpus, rootpath, filelimit = 0):
    
    #rootpath = corpus.rootpath
    fileids = os.listdir(rootpath)
    
    hugewordlist = []   
    hugewordlist.extend(corpus.words)   # will contain distinct Word instances

    numoffiles = 0
    
    corpus.set_corpusname(str(max(filelimit, len(fileids)))+"texts")
    
    for fileid in fileids:
    
        
        allwords = FreqDist()    # will contain all words in this text
        
        doc_id = fileid.split(".")[0]
        # corpus.inserttext(doc_id)    ##### !   text in kendisini gondermeli
        newtext = Text(doc_id)
        
        path = rootpath + os.sep + fileid
        #lines = readtextlines(path)
    
        #rawtext = texter.readtxtfile(path)
        rawtext = texter.readnewstext(path)
        lines = texter.splitToSentences(rawtext)
        
        sntindex = 0
        # each line is a sentence
        for line in lines:
            words = []   # words in this sentence
            words = line.split()
            words = texter.eliminatepunctuation(words)
            words = [word for word in words if not word.isspace()]
            
            
            
            for word in words:
                allwords.inc(word)
                
                
                newword = Word(word)
                newword.insertsentenceid(doc_id+"_"+str(sntindex))
                
                if allwords[word] <= 1:    # if this was not added to the hugelist before, add it
                    hugewordlist.append(newword)
                
                    
            sentence = Sentence(sntindex)
            sntindex = sntindex + 1
            
            # sentence'a Word mu wordindex mi atalim?
            for word in words:
                index = hugewordlist.index(Word(word))
                hugewordlist[index].insertsentenceid(doc_id+"_"+str(sntindex-1))
                sentence.insertword(index)
                
            newtext.insertsentence(sentence)
            
        if (not rawtext.isspace()) or (len(allwords) != 0):   
            corpus.inserttext(newtext)    
            
            print str(numoffiles)," : finished handling the words-snts-txts ",doc_id 
    
                
            numofwords = reduce(lambda x,y : x+y, allwords.values())
            
            for word in hugewordlist:
                cnt =  allwords[word.literal]
                #freq = cnt / float(numofwords)
                word.assigntermfreq(cnt, numofwords, doc_id)
                #hugewordlist[index].toscreen()
        
        numoffiles = numoffiles + 1
        if filelimit == numoffiles:
            break       

        
    # end for - docs
    

    numofdocs = len(fileids)
    print "computing tf*idf"
    for word in hugewordlist:
        word.computeinvdocfreq(numofdocs)
        word.computeTFIDF()
        #word.toscreen()
        
    corpus.assignwords(hugewordlist)
    print "corpus length ",str(len(corpus.words))," words"
    print "huges length ",str(len(hugewordlist))," words"
    print "exiting buildcorpus()"
    
    print "pickle-dumping words"
    corpus.pickledumpwords()
    #return hugewordlist
  


def assignPOStags(corpus):
    for word in corpus.words:
        (literal, literalPOS, root, rootPOS) = SAKsParser.lemmatizeword(word.literal)
        word.root = root.lower()
        rootPOStag = rootPOS.upper()[:4]
        if rootPOStag.startswith("AD"):
            rootPOStag = rootPOStag[:3]
        word.rootPOStag = rootPOStag 
        word.setPOStag(literalPOS)
        
  
  

def shell(filelimit = 0):       
    #rootpath = "/home/dicle/Dicle/Tez/dataset/readingtest30/"
    corpuspath = "/home/dicle/Dicle/Tez/dataset/readingtest300/"
    rootpath = corpuspath
    folders = IOtools.getfoldernames_of_dir(corpuspath)
    foldername = ""
    corpus = Corpus(rootpath)
    singlefolder = False
    if len(folders) == 0:
        singlefolder = True
    
    
    if singlefolder:                                                    
        rootpath = corpuspath 
        #corpus = Corpus(rootpath, foldername)
        starttime = datetime.now()
        buildcorpus(corpus, rootpath, filelimit)
        endtime_buildcorpus = datetime.now()
        print "build corpus took: ",str(endtime_buildcorpus - starttime)
        print "corpus length ",str(len(corpus.words))," words"
    
    else:     
        for foldername in folders:
            
            print "Folder: ",foldername
            rootpath = corpuspath + os.sep + foldername + os.sep
            
            #corpus = Corpus(rootpath, foldername)
            
            starttime = datetime.now()
            
            buildcorpus(corpus, rootpath)
            endtime_buildcorpus = datetime.now()
            print "build corpus took: ",str(endtime_buildcorpus - starttime)
            print "corpus length ",str(len(corpus.words))," words"
            
    print "pickle-getting words"
    corpus.picklegetwords()    
    print "assigning pos tags" 
    assignPOStags(corpus)
    endtime_postags = datetime.now()
    print "postag assignment took: ",str(endtime_postags - endtime_buildcorpus)
    
    
    '''
    get_magnitudewords_doc_matrix(corpus)
    
    adjectives = get_words_ofPOStag(corpus, "ADJ")
    print "numof adjectives, ",len(adjectives),"  ",adjectives[:-10]
    get_docterm_matrix(corpus, adjectives, "adjective-doc-matrix.txt", record = True)
    '''
    endtime = datetime.now()
    passtime = endtime - starttime
    print "Elapsed time: ",passtime," on folder ",foldername
    
    print "pickle-dumping words"
    endtimep = datetime.now() 
    corpus.pickledumpwords()   
    print "Corpus length: ",len(corpus.words)  
    print "Elapsed time for pickle: ",str(endtimep - endtime)
    
    # PICKLE words
    print "pickle-getting words"
    corpus.picklegetwords()

    print "corpus first 20 words:"
    for word in corpus.words[:20]:
        word.toscreen()
        
    print "pickle-dumping words"
    corpus.pickledumpwords()
    
    
    #polarity matrix (by word and polarity type)
    # polarity bigram matrix
    # big matrix, many texts





#if __name__ == "__main__":

filelimit = 50    
shell(filelimit)
      

