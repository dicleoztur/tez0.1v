'''
Created on Aug 27, 2012

@author: dicle
'''

from Sentence import Sentence

class Text:
    
    txtid = ""
    sentences = []     # contain Sentence instances
    sentimentvalue = 0.0
    polaritybigramVector = {}
    clusterid = -1

    def __init__(self, txtid = ""):
        self.txtid = txtid
        self.sentences = []
        self.polaritybigramVector = {}
        self.clusterid = -1
        
    def insertsentence(self, sentence):
        self.sentences.append(sentence)
        
    def assignpolaritybigramvector(self, dct):
        self.polaritybigramVector = dct
    
    def set_clustermembership(self, clindex):
        self.clusterid = clindex
    
    def getsentence(self, sentid):
        if sentid in self.sentences:
            indx = self.sentences.index(sentid)
            return self.sentences[indx]
        else:
            return Sentence()
        
    
    def __cmp__(self, otherID):
        if self.txtid > otherID:
            return 1
        elif self.txtid == otherID:
            return 0
        else:
            return -1   
     

#    def __cmp__(self, other):
#   
#        if self.txtid > other.txtid:
#            return 1
#        elif self.txtid == other.txtid:
#            return 0
#        else:
#            return -1     
     
           
              