'''
Created on Aug 27, 2012

@author: dicle
'''

import math


import IOtools
from stats import numericutils


class Word:

  
    literal = ""
    polarity = 0     # in {-1,0,1}
    frequency = {}   # frequency of the word in docs - {doc_id : freq}
    
    root = ""
    rootPOStag = ""
    
    POStag = ""
    
    # if we consider it global
    termfreq = {}   # {doc_id : freq of this word in doc_id}
    idf = 0.0
    tfidf = {}
    
    docs = []    #  list of docs where this word appears  - most probably identical to termfreq.keys
    sents = []  #  list of sntids-sentences where this word appears
    


    def __init__(self, literal, opinion=0, frequency=0.0):
        self.literal = literal
        self.polarity = opinion
        self.frequency = {}
        self.termfreq = {}
        self.tfidf = {}
        self.sents = []
        
        self.POStag = ""
        self.root = ""
        self.rootPOStag = ""
        

    def tostring(self):
        return self.literal
    
    def tostring_record(self):
        out =  "\n*******"
        out += self.tostring()
        out += " "+self.root+" ~ "+self.rootPOStag
        out += " Polarity: "+str(self.polarity)
        out += "   IDF:"+str(self.idf)
        return out
    
    def toscreen(self):
        print "\n*******"
        print self.tostring()
        print " ",self.root," ~ ",self.rootPOStag
        print " Polarity: ",str(self.polarity)
        print " Term Frequencies:"
        IOtools.printdictionary(self.termfreq)
        print " Frequencies:"
        IOtools.printdictionary(self.frequency)
        print " TF IDF vector:"
        IOtools.printdictionary(self.tfidf)
        print "*******"
        
    
    
     
    def toscreen_simple(self):
        print "\n*******"
        print self.tostring()
        print " ",self.root," ~ ",self.rootPOStag
        print " Polarity: ",str(self.polarity)
        print "*******"
        
    
    
    def assigntermfreq(self, numofoccurrences, doclength, doc_id):
        #freq = round(freq,3)
        if numofoccurrences > 0:
            self.frequency[doc_id] = numofoccurrences
            if doclength == 0:
                freqscale = 0.0
            else: 
                freqscale = numofoccurrences / float(doclength)
            self.termfreq[doc_id] = math.log10(freqscale) + 1.0
        else:
            self.frequency[doc_id] = 0.0
            self.termfreq[doc_id] = 0.0
    
    
    #sentid is the id of the sentence where this word appears
    def insertsentenceid(self, sentid):
        if sentid not in self.sents:
            self.sents.append(sentid)
            
    
    def computeinvdocfreq(self, numofdocs):
        nonzerofreqs = [val for val in self.frequency if val is not 0]
        documentfrequency = len(nonzerofreqs)
        value = numofdocs / float(documentfrequency)
        self.idf = math.log10(value)
    
    
    def computeTFIDF(self):
        #round values
        self.idf = round(self.idf, 4)
        self.termfreq = numericutils.roundvalues(self.termfreq)
        #self.frequency = numericutils.roundvalues(self.frequency)
        
        for key,value in self.termfreq.items():
            self.tfidf[key] = round(value * self.idf, 4)    # may require precaution as idf may not have been computed yet. check for 0.
    
    
    
    # returns the list of docids where this Word occurred
    def getdocs(self):
        return [k for k,v in self.freq.iteritems() if v is not 0]
    
    
    # returns the list of sentences as docid_sntid string where this Word occurred
    def getsentences(self):
        return self.sents
    
    def setPOStag(self, tag):
        self.POStag = tag   

    def setpolarity(self, polarityvalue):
        self.polarity = polarityvalue
    
    def __get__(self):
        return self.literal


    def __cmp__(self, otherword):
        if self.literal > otherword.literal:
            return 1
        elif self.literal == otherword.literal:
            return 0
        else:
            return -1  

    def __str__(self):
        return self.literal
        
            
                
        