'''
Created on Aug 27, 2012

@author: dicle
'''

import codecs
import re
import os
import math
import pickle
import nltk
import numpy as np
from datetime import datetime


import IOtools
from languagetools import SAKsParser
from languagetools import bigramfinder
from Word import Word
from Sentence import Sentence
from Text import Text
from Corpus import Corpus
from txtprocessor import texter
from txtprocessor import listutils
from stats import PCA
import keywordhandler

stopwordspath = ""

writewordspath = "/home/dicle/Dicle/Tez/dataset/output/words/"
writesentencespath = "/home/dicle/Dicle/Tez/dataset/output/sentences/"

polaritybase = "/home/dicle/Dicle/Tez/polaritybase/"

positivevalue = 1
negativevalue = -1
    
polaritytags = {"positive" : 1, "negative": -1, "neutral" : 0}
    

def picklewrite(path, obj):
    f = codecs.open(path,"a", encoding='utf8')
    pickle.dumps(obj, f)
    
    
    
    
def writelist(path, lst):
    f = codecs.open(path,"w", encoding='utf8')
    
    i = 0
    for item in lst:
        f.write(item.tostring_record()+"\n")
        #print str(i)," - ",item.__str__()
        i = i+1
    f.close()
    
    



def readtextlines(path):
    f = codecs.open(path,"r", encoding='utf8')
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    f.close()
    return lines

def printlist(lst):
    for w in lst:
        print w

'''
def eliminatepunctuation(wordlist):
    cleanwords = []    
    for w in wordlist:
        cleanword = ''.join(re.findall(r'\w+',w, flags=re.UNICODE))
        cleanwords.append(cleanword.lower())      # cleanwords.append(cleanword)
    return cleanwords
'''

def removestopwords(wordlist):
    sws = readtextlines(stopwordspath)
    #print "stopwords "+str(len(sws))
    removed = [w for w in wordlist if w not in sws]
    return removed


def assignTFIDF(corpus):
    
    rootpath = corpus.rootpath
    fileids = os.listdir(rootpath)
    
    hugewordlist = []   # will contain distinct Word instances
    
    for fileid in fileids:
    
        allwords = []    # will contain all words in this text
        alldistinctwords = []
        
        doc_id = fileid.split(".")[0]
        corpus.inserttext(doc_id)
        
        path = rootpath + os.sep + fileid
        lines = readtextlines(path)
    
        # each line is a sentence?
        for line in lines:
            words = []
            words = line.split()
            words = texter.eliminatepunctuation(words)
            
            allwords = allwords + words
            
            for word in words:
                newword = Word(word)
                if not hugewordlist.count(newword):
                    hugewordlist.append(newword)
                if not alldistinctwords.count(newword):
                    alldistinctwords.append(newword)
        
        '''
        print doc_id,": all distinct words:"
        printlist(alldistinctwords)
        '''
        numofwords = len(allwords)
        for word in alldistinctwords:
            cnt =  allwords.count(word.literal)
            freq = cnt / float(numofwords)
            
            index = hugewordlist.index(word)
            hugewordlist[index].assigntermfreq(freq, doc_id)
            #hugewordlist[index].toscreen()

    numofdocs = len(fileids)
    
    for word in hugewordlist:
        word.computeinvdocfreq(numofdocs)
        word.computeTFIDF()
        #word.toscreen()
    
    writepath = writewordspath+os.sep+"samplewords.txt"
    writelist(writepath, hugewordlist)
    corpus.assignwords(hugewordlist)
    
    return hugewordlist
   


''' read news items,
    for each text:
     get sentences
     get words
     calculate tf idf weights
'''



def buildcorpus(corpus, rootpath, filelimit = 0):
    
    #rootpath = corpus.rootpath
    fileids = os.listdir(rootpath)
    
    hugewordlist = []   
    hugewordlist.extend(corpus.words)   # will contain distinct Word instances

    numoffiles = 0
    
    corpus.set_corpusname(str(max(filelimit, len(fileids)))+"texts")
    
    for fileid in fileids:
    
        
        allwords = nltk.FreqDist()    # will contain all words in this text
        
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
  


'''   closed on 8 March
def buildcorpus(corpus):
    
    rootpath = corpus.rootpath
    fileids = os.listdir(rootpath)
    
    hugewordlist = []   # will contain distinct Word instances
    hugedoclist = []
    
    
    for fileid in fileids:
    
        
        allwords = []    # will contain all words in this text
        alldistinctwords = []
        sentencelist = []    # list of sentences in this text  (Sentence instances)
        
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
            words = []
            words = line.split()
            words = texter.eliminatepunctuation(words)
            words = [word for word in words if not word.isspace()]
            allwords = allwords + words
            
            for word in words:
                newword = Word(word)
                newword.insertsentenceid(doc_id+"_"+str(sntindex))
                if not hugewordlist.count(newword):
                    hugewordlist.append(newword)
                if not alldistinctwords.count(newword):
                    alldistinctwords.append(newword)
                    
            sentence = Sentence(sntindex)
            sntindex = sntindex + 1
            
            # sentence'a Word mu wordindex mi atalim?
            for word in words:
                index = hugewordlist.index(Word(word))
                hugewordlist[index].insertsentenceid(doc_id+"_"+str(sntindex-1))
                sentence.insertword(index)
                
            newtext.insertsentence(sentence)
            
            
        corpus.inserttext(newtext)    
        
        print "finished handling the words-snts-txts" 

            
        numofwords = len(allwords)
        for word in alldistinctwords:
            cnt =  allwords.count(word.literal)
            #freq = cnt / float(numofwords)
            
            index = hugewordlist.index(word)
            hugewordlist[index].assigntermfreq(cnt, numofwords, doc_id)
            #hugewordlist[index].toscreen()

        
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
    return hugewordlist
'''   



def assignpolarity_onwords(corpus):
    positivevalue = 1
    negativevalue = -1
    
    pos_path = polaritybase + os.sep + "positive.txt"
    neg_path = polaritybase + os.sep + "negative.txt" 
    
    positivewords = IOtools.readtextlines(pos_path)
    positivewords = sorted(positivewords)
    negativewords = IOtools.readtextlines(neg_path)
    negativewords = sorted(negativewords)
    
    for word in corpus.words:
        if word.literal in positivewords:
            word.setpolarity(positivevalue)
        elif word.literal in negativewords:
            word.setpolarity(negativevalue)


def assignpolarity_onroots(corpus):
    #positivevalue = 1
    #negativevalue = -1
    
    pos_path = polaritybase + os.sep + "positive.txt"
    neg_path = polaritybase + os.sep + "negative.txt" 
    
    positivewords = IOtools.readtextlines(pos_path)
    
    ling_triple = SAKsParser.findrootsinlexicon(positivewords)
    positivewords1 = [root for (word,root,POS) in ling_triple if POS.lower().find("ver") > -1]
    positivewords2 = [word for (word,root,POS) in ling_triple if POS.lower().find("ver") == -1]
    
    positivewords = positivewords1 + positivewords2
    positivewords = sorted(positivewords)
    
    
    
    negativewords = IOtools.readtextlines(neg_path)
    ling_triple = SAKsParser.findrootsinlexicon(negativewords)
    negativewords1 = [root for (word,root,POS) in ling_triple if POS.lower().find("ver") > -1]
    negativewords2 = [word for (word,root,POS) in ling_triple if POS.lower().find("ver") == -1]
    
    negativewords = negativewords1 + negativewords2
    negativewords = sorted(negativewords)  
      
    for word in corpus.words:
        if word.root in positivewords:
            word.setpolarity(positivevalue)
        elif word.root in negativewords:
            word.setpolarity(negativevalue)




def assignPOStags(corpus):
    for word in corpus.words:
        (literal, literalPOS, root, rootPOS) = SAKsParser.lemmatizeword(word.literal)
        word.root = root.lower()
        rootPOStag = rootPOS.upper()[:4]
        if rootPOStag.startswith("AD"):
            rootPOStag = rootPOStag[:3]
        word.rootPOStag = rootPOStag 
        word.setPOStag(literalPOS)
        
                   

def assignTFIDF3(rootpath):
    
    fileids = os.listdir(rootpath)
    
    hugewordlist = []   # will contain distinct Word instances
    
    for fileid in fileids:
    
        allwords = []    # will contain all words in this text
        alldistinctwords = []
        
        doc_id = fileid.split(".")[0]
        path = rootpath + os.sep + fileid
        lines = readtextlines(path)
    
        
        for line in lines:
            words = []
            words = line.split()
            words = texter.eliminatepunctuation(words)
            
            allwords = allwords + words
            
            for word in words:
                newword = Word(word)
                if not hugewordlist.count(newword):
                    hugewordlist.append(newword)
                if not alldistinctwords.count(newword):
                    alldistinctwords.append(newword)
        
        '''
        print doc_id,": all distinct words:"
        printlist(alldistinctwords)
        '''
        numofwords = len(allwords)
        for word in alldistinctwords:
            cnt =  allwords.count(word.literal)
            freq = cnt / float(numofwords)
            
            index = hugewordlist.index(word)
            hugewordlist[index].assigntermfreq(freq, doc_id)
            #hugewordlist[index].toscreen()

    numofdocs = len(fileids)
    
    for word in hugewordlist:
        word.computeinvdocfreq(numofdocs)
        word.computeTFIDF()
        #word.toscreen()
    
    writepath = writewordspath+os.sep+"samplewords.txt"
    writelist(writepath, hugewordlist)

    return hugewordlist
  


#precaution against memory shortages..should take file path as parameter
def recordwords():
    write2 = writewordspath+os.sep+"objects"+os.sep+"samples.txt"
    fwrite = open(write2, "w")
    pickle.dump(corpus.words, fwrite, -1)
    fwrite.close()
    
    fread = open(write2, "r")
    words = pickle.load(fread)
    
    return words


def findbigrams(corpus):
   
    hugewordlist = corpus.words
    
    for text in corpus.texts:
        #print "in text ",text.txtid
        wordpoltuples = []     # keep (word,poltag) pairs for each text
        polaritybigramscore = {}
        for snt in text.sentences:
            
            for wordindex in snt.words:
                polaritylabel = "neut"  # in case there are errors on polarityvalue(polval), let default label be neutral
                literalword = hugewordlist[wordindex].literal
                
                polval = hugewordlist[wordindex].polarity
                polaritylabel = corpus.polaritytags.get(polval)
                              
                wordpoltuples.append((literalword, polaritylabel))
                #find bigram! 
        
        polaritybigramscore =  bigramfinder.bigram_bytags(wordpoltuples, corpus.polaritytags.values())       # should be assigned to text     
        text.assignpolaritybigramvector(polaritybigramscore)



def extractfeatures_bigrampolarity(corpus):
    scorematrix = {}
    for text in corpus.texts:
        scorematrix[text.txtid] = text.polaritybigramVector.values()
    return scorematrix
    
def cluster_bigrampolarity(corpus):
    txtassignments = bigramfinder.clusterer(extractfeatures_bigrampolarity(corpus), numofclusters = 4, notarray = True)
    
    for text in corpus.texts:
        text.set_clustermembership(txtassignments.get(text.txtid))
  



def search2countries_insentences(corpus):
    countryrelationSentences = []    # will store the id's of the sentences that contain 
 
    country1 = texter.readtextlines(IOtools.countrypath+os.sep+IOtools.countries[0]+".txt")
    country1.sort()
    country1 = [w.lower() for w in country1]
    
    country2 = texter.readtextlines(IOtools.countrypath+os.sep+IOtools.countries[1]+".txt")
    country2.sort()
    country2 = [w.lower() for w in country2] #the related words/names of the predetermined two countries 
    
    '''
    print "Country1"
    IOtools.printlist(country1)
    print "Country2"
    IOtools.printlist(country2)
    '''
    for text in corpus.texts:
        i = 0 
        for snt in text.sentences:
            #print "start sentence ",snt.sntid,"  i: ",str(i)
            #list of keywords related to country_i

            wordsofsent = []
            for wordindex in snt.words:
                #wordsofsent.append(corpus.words[wordindex].literal)
                wordsofsent.append(corpus.words[wordindex].root)
                
            contains_country1 = False
            contains_country2 = False
                        
            contains_country1 = texter.keywords_search(wordsofsent, country1)
            
            if contains_country1:
                contains_country2 = texter.keywords_search(wordsofsent, country2)
                

            if contains_country1 and contains_country2:
                countryrelationSentences.append((snt.sntid, text.txtid))
                snt.contains2countries = True
                
    return countryrelationSentences

    
'''
def search2countries_insentences(corpus):
    countryrelationSentences = []    # will store the id's of the sentences that contain 
 
    country1 = texter.readtextlines(IOtools.countrypath+os.sep+IOtools.countries[0]+".txt")
    country1.sort()
    country1 = [w.lower() for w in country1]
    
    country2 = texter.readtextlines(IOtools.countrypath+os.sep+IOtools.countries[1]+".txt")
    country2.sort()
    country2 = [w.lower() for w in country2] #the related words/names of the predetermined two countries 
    
    print "Country1"
    IOtools.printlist(country1)
    print "Country2"
    IOtools.printlist(country2)
    
    for text in corpus.texts:
        i = 0 
        for snt in text.sentences:
            #print "start sentence ",snt.sntid,"  i: ",str(i)
            #list of keywords related to country_i

            wordsofsent = []
            for wordindex in snt.words:
                #wordsofsent.append(corpus.words[wordindex].literal)
                wordsofsent.append(corpus.words[wordindex].root)
                
            contains_country1 = False
            contains_country2 = False
            
            print "sntid- ",snt.sntid," : ",wordsofsent
            word1 = ""
            word2 = ""
            
            for word in wordsofsent:
                if word in country1:
                    #print "word1: ",word," text ",text.txtid," snt",snt.sntid
                    contains_country1 = True
                    word1 = word
                    break
            if contains_country1:
                for word in wordsofsent:
                    if word in country2:
                        #print "word2: ",word," text ",text.txtid," snt",snt.sntid
                        contains_country2 = True
                        word2 = word
                        break

            if contains_country1 and contains_country2:
                countryrelationSentences.append((snt.sntid, text.txtid))
                snt.contains2countries = True
                print "ADDED: ",snt.sntid," ",text.txtid," by words",word1," ~~ ",word2
            
            i = i+1
    return countryrelationSentences
'''

def printAllSentences(corpus):
    for txt in corpus.texts:
        print "Text: ",txt," ",txt.txtid
        i = 0
        for snt in txt.sentences:
            snttxt = ""
            print " snt ",snt.sntid," i: ",str(i)
            for wordindex in snt.words:
                snttxt = snttxt + corpus.words[wordindex].literal + " "
            print snttxt
            i = i+1    


''' for each sentence:
     1- find # of pos, neg, neut words 
     2- assign verb - for now only the last word of a sentence
  
  '''
def re_buildcorpus(corpus):  
    for txt in corpus.texts:
        for snt in txt.sentences:
  
            ''' 1- find # of pos, neg, neut words  '''
            wordsofsent = []
            for wordindex in snt.words:
                wordsofsent.append(corpus.words[wordindex])
            
            for tag in polaritytags.values():
                tagcount = countpolaritytags(wordsofsent, tag)
                snt.polaritycount[tag] = tagcount


            '''  2- assign verb - for now only the last word of a sentence !! DUZELT   '''
            if snt.words:
                snt.verb = corpus.words[snt.words[-1]]  
            else:
                snt.verb = Word("-")
             
             
def reportCountryRelatingSentences(corpus):
    for txt in corpus.texts:
        recordtext = ""       
        for snt in txt.sentences:
            if snt.contains2countries: 
                recordtext = "Text: "+txt.txtid+"\n"              
                print "Text, ",txt.txtid
                snt.toscreen(corpus.words)
                recordtext += snt.tostring(corpus.words)
                path = writesentencespath + os.sep + "twocountrysentences.txt"
                IOtools.txttodisc_append(recordtext, path)           
               
'''
 words is a list of Word s. corpus is the corpus containing words of type Word. tag is -1,1 or 0.
'''
def countpolaritytags(words, tag):
    count = 0
    for word in words:
        if word.polarity == tag:
            count = count + 1
    return count


def get_words_ofPOStag(corpus, tag, printt=False):
    taggedwords = []
    for word in corpus.words:
        if word.POStag == tag:
            taggedwords.append(word)
    
    if printt:
        print "Words with tag ",tag
        for w in taggedwords:
            print w.literal
            
    return taggedwords


# returns a list [ ((word i, word j),numofcommonSentences ] which will be used for building 
#  a sentencewise cooccurrence matrix wordlist1,2 is a list of Word instances
#   and also returns the id's of the sentences where the two words co-occur.
def findcooccurrence_sentencewise(wordlist1, wordlist2):
    occurrence = []
    for word1 in wordlist1:
        for word2 in wordlist2:
            numofcommons = 0
            if word1.__cmp__(word2) == 0:
                numofcommons = len(word1.sents)
            else:
                commons = listutils.getintersectionoflists(word1.getsentences(), word2.getsentences())
                numofcommons = len(commons)
            occurrence.append(((str(word1), str(word2)), numofcommons))
    return occurrence


# returns a list [ ((word i, word j),numofcommondocs ] which will be used for building 
#  a docwise cooccurrence matrix wordlist1,2 is a list of Word instances
def findcooccurrence_docwise(wordlist1, wordlist2):
    occurrence = []
    for word1 in wordlist1:
        for word2 in wordlist2:
            numofcommons = 0
            if word1.__cmp__(word2) == 0:
                numofcommons = len(word1.termfreq.keys())
            else:
                commons = listutils.getintersectionoflists(word1.getdocs(), word2.getdocs())
                numofcommons = len(commons)
            occurrence.append(((str(word1), str(word2)), numofcommons))
    return occurrence




def get_magnitudewords_doc_matrix(corpus):
    magwordsdict = keywordhandler.get_keyword_dict("magnitude")
    bigwords = magwordsdict["big"]
    smallwords = magwordsdict["small"]
    
    smallwords = corpus.get_Words_from_words(smallwords)
    bigwords = corpus.get_Words_from_words(bigwords)
    smallbig_incidence = findcooccurrence_sentencewise(smallwords, bigwords)
    
    print "Small words - doc matrix"
    #print get_termdoc_matrix(corpus, smallwords, "smallwords-term-matrix.txt", True)
    print get_docterm_matrix(corpus, smallwords, "smallwords-term-matrix.txt", True)
    print "Small words - doc matrix"
    print get_docterm_matrix(corpus, bigwords, "bigwords-term-matrix.txt", True)
    
    
    
# DIKKAT! RETURNS DOCS AT COLUMNS
# returns the term document matrix of the incidence of the input string terms in the texts of the corpus. terms is a list of strings  
def get_termdoc_matrix(corpus, terms, filename, record = False):
    
    matrix = []    # term x doc incidence matrix
    
    if not isinstance(terms[0], Word): 
        terms_asWords = corpus.get_Words_from_words(terms)
    else:
        terms_asWords = terms
        
    docs_asIDs = [txt.txtid for txt in corpus.texts]
    docs_asIDs.sort()
    
    for term in terms_asWords:
        row = []
        for docid in docs_asIDs:
            if term.frequency.has_key(docid):
                row.append(term.frequency[docid])
            else:
                row.append(0)
        matrix.append(row)
    
    if record:
        outmatrix = []
        outmatrix.append(["      "] + docs_asIDs)
        for i, term in enumerate(terms):
            newrow = [str(term)] + matrix[i]
            outmatrix.append(newrow)
            
        matrixpath = IOtools.ensure_dir(corpus.outputpath+os.sep+"matrix")
        IOtools.todisc_matrix(outmatrix, matrixpath+os.sep+filename)    
     
    return matrix

            
    # map the term(Word) doc-occurrence to a vector
    


    
# returns the term document matrix of the incidence of the input string terms in the texts of the corpus. terms is a list of strings 
# rows are texts as instances and the columns are terms as features.  
def get_docterm_matrix(corpus, terms, filename, record = False):
    
    matrix = []    # term x doc incidence matrix
    
    if not isinstance(terms[0], Word): 
        terms_asWords = corpus.get_Words_from_words(terms)
    else:
        terms_asWords = terms
        
    docs_asIDs = [txt.txtid for txt in corpus.texts]
    docs_asIDs.sort()
    
    for term in terms_asWords:
        row = []
        for docid in docs_asIDs:
            if term.frequency.has_key(docid):
                row.append(term.frequency[docid])
            else:
                row.append(0)
        matrix.append(row)
    
    marray = np.asarray(matrix, dtype = float)
    marray = marray.T
    matrix = marray.tolist()
    
    if record:
        outmatrix = []
        outmatrix.append(["\t"] + terms)
        for i, docid in enumerate(docs_asIDs):
            newrow = [str(docid)] + matrix[i]
            outmatrix.append(newrow)
        matrixpath = IOtools.ensure_dir(corpus.outputpath+os.sep+"matrix")
        IOtools.todisc_matrix(outmatrix, matrixpath+os.sep+filename)    
     
    return matrix

            
    # map the term(Word) doc-occurrence to a vector
   


def get_termterm_matrix(corpus, terms1, terms2):
    return



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
    
    
    
    
    
    
    
''' 8 Mart  
#rootpath = "/home/dicle/Dicle/Tez/dataset/readingtest30/"
rootpath = "/home/dicle/Dicle/Tez/dataset/dataset750/"



corpus = Corpus(rootpath)

starttime = datetime.now()

buildcorpus(corpus)
print "corpus length ",str(len(corpus.words))," words"
print "assigning pos tags"
assignPOStags(corpus)


get_magnitudewords_doc_matrix(corpus)

'''



'''  3 Mart - 1) get all the matrices '''


'''
adjectives = get_words_ofPOStag(corpus, "ADJ")
print "numof adjectives, ",len(adjectives),"  ",adjectives[:-10]
get_docterm_matrix(corpus, adjectives, "adjective-doc-matrix.txt", record = True)

endtime = datetime.now()
passtime = endtime - starttime
print "Elapsed time: ",passtime



#polarity matrix (by word and polarity type)
# polarity bigram matrix
# big matrix, many texts

'''


'''
l = ["yasa", "kuzey", "merhaba", "evet"]

for w in l:
    ind = corpus.searchword(w)
    if ind > -1:
        corpus.words[ind].toscreen()
#print corpus.reportstats()


# 3 Subat. Buyukluk kelimeleri arama.


bigwords = texter.readtextlines(IOtools)

magwords = texter.readtextlines(IOtools.magnitudewords_file)
'''






# 21 Ocak - 3 Subat. Bitti. LSA sonuclari iyi. Daha sonra topic modelling yapmak uzere simdilik dursun.

'''
print "corpus length ",str(len(corpus.words))," words"

print "Adjectives"
adjectives = get_words_ofPOStag(corpus, "ADJ")

print "Sentence occurrence"
for w in adjectives:
    print str(w)," : ",w.sents




print "Adverbs"
adverbs = get_words_ofPOStag(corpus, "ADV")

print "Sentence occurrence"
for w in corpus.words:
    print str(w)," : ",w.sents

print "Sentencewise cooccurrence of Adjectives and Adverbs"    
occurrence = findcooccurrence_sentencewise(adjectives, adverbs)
#occurrence = findcooccurrence_sentencewise(adjectives, adjectives)
for tuplle, commons in occurrence:
    print "[",tuplle[0],", ",tuplle[1], "]: ",commons    
  

print len(occurrence),"  ",len(adjectives)," ",len(adverbs)   


m = len(adverbs)   # number of rows - instances (we now look for, for each adverb, what are the prominent adjectives? (LSI)
n = len(adjectives)  # number of columns - features
matrix = PCA.convert_pairmap2matrix(occurrence, m, n)

for row in matrix:
    print row


pairmap = []
numofselections = 4
t = 0
for i in range(numofselections):
    pairmap.extend(occurrence[t:t+numofselections])
    t = t+n
    

for tuplle, commons in pairmap:
    print "[",tuplle[0],", ",tuplle[1], "]: ",commons
    
matrix2 = PCA.convert_pairmap2matrix(pairmap, numofselections, numofselections)

for row in matrix2:
    print row 

'''


'''
matrix = []
start = 0
for i in range(numofselections):
    row = pairmap[start:start+numofselections]
    rowvalues = []
    for item in row:
        (wordpair, numofcooccurrences) = item
        key = wordpair[0][-3:]+"_"+wordpair[1][-3:]
        rowvalues.append((key,numofcooccurrences))
    matrix.append(rowvalues)
    start = start+numofselections
'''

   
    
    

''' get only the adjectives '''




'''

rootpath = "/home/dicle/Dicle/Tez/dataset/readingtest33/"
#rootpath = "/home/dicle/Dicle/Tez/dataset/dataset1500/"
#rootpath = "/home/dicle/Dicle/Tez/dataset/readingtest150part1/"

corpus = Corpus(rootpath)
buildcorpus(corpus)
print "assigning pos tags"
assignPOStags(corpus)


print "assigning polarity tags"
assignpolarity_onroots(corpus)
print "rebuilding the corpus"
re_buildcorpus(corpus)

hugewordlist = corpus.words

countrycontaining_sentences = search2countries_insentences(corpus)





for sid,tid in countrycontaining_sentences:
    print " ADDED ",sid," ",tid

for sid,tid in countrycontaining_sentences:
    print "In text ",tid
    newtext = corpus.gettext(tid)
    newsentence = newtext.getsentence(sid)
    
    snttxt = ""
    print " sentence ",sid 
    for wordindex in newsentence.words:
        snttxt = snttxt + hugewordlist[wordindex].literal+"  " 
    print "snttxt:  ",snttxt

'''

#printAllSentences(corpus)


#print corpus.sentenceAsText("402969", 3)



'''
print "Country relating sentences: "
print "  # found ",str(len(countrycontaining_sentences))," many sentences in the corpus"
print 
reportCountryRelatingSentences(corpus)
'''

'''
for txt in corpus.texts:
    print "Text: ",txt," ",txt.txtid
    i = 0
    for snt in txt.sentences:
        snttxt = ""
        print " snt ",snt.sntid," i: ",str(i)
        for wordindex in snt.words:
            snttxt = snttxt + corpus.words[wordindex].literal + " "
        print snttxt
        i = i+1
'''


'''

assignPOStags(corpus)

assignpolarity_onroots(corpus)


print "CORPUS"
for word in corpus.words:
    word.toscreen()

## #DEVAM   0n polarity

# no change on hugewordlist! 


hugewordlist = corpus.words

'''

'''
for text in corpus.texts:
    print "in text ",text.txtid
    for snt in text.sentences:
        print "sentence: ",snt.sntid
        snttxt = ""
        pol_vals = ""
        for wordindex in snt.words:
            snttxt = snttxt + hugewordlist[wordindex].literal + " "
            polval = hugewordlist[wordindex].polarity
            p = "n"
            if polval == -1:
                p = "-"
            elif polval == 1:
                p = "+"
            pol_vals = pol_vals + p + "    "
        print snttxt
        print pol_vals
        print snt.words

'''



'''        
        
findbigrams(corpus)


for text in corpus.texts:
    print text.txtid
    IOtools.printdictionary(text.polaritybigramVector)
    

cluster_bigrampolarity(corpus)

writepath = writewordspath+os.sep+"samplewords.txt"
writelist(writepath, corpus.words)

# build feature matrix
# call clustering    
    
    
'''    
    
#DEVAM ##



'''
hugewordlist = assignTFIDF(corpus)    



corpus.assignwords(hugewordlist)


print "CORPUS"
for word in corpus.words:
    word.toscreen()

'''




'''

x = []


#  loop will begin here.



fileids = os.listdir(rootpath)

hugewordlist = []   # will contain distinct Word instances

for fileid in fileids:

    allwords = []    # will contain all words in this text
    alldistinctwords = []
    
    doc_id = fileid.split(".")[0]
    path = rootpath + os.sep + fileid
    lines = readtextlines(path)

    
    for line in lines:
        words = []
        words = line.split()
        words = eliminatepunctuation(words)
        
        allwords = allwords + words
        
        for word in words:
            newword = Word(word)
            x.append(word)
            if not hugewordlist.count(newword):
                hugewordlist.append(newword)
            if not alldistinctwords.count(newword):
                alldistinctwords.append(newword)
    
    print doc_id,": all distinct words:"
    printlist(alldistinctwords)
    
    numofwords = len(allwords)
    for word in alldistinctwords:
        cnt =  allwords.count(word.literal)
        freq = cnt / float(numofwords)
        
        index = hugewordlist.index(word)
        hugewordlist[index].assigntermfreq(freq, doc_id)
        hugewordlist[index].toscreen()
        print "doc - index",doc_id," : ",index
        print word,": ",cnt,", ",freq,", ",str(math.log10(freq)+1.0)
        




####
        
print len(x)," ",len(hugewordlist)


numofdocs = len(fileids)

for word in hugewordlist:
    word.computeinvdocfreq(numofdocs)
    word.computeTFIDF()
    word.toscreen()

writepath = writewordspath+os.sep+"samplewords.txt"
writelist(writepath, hugewordlist)

'''
