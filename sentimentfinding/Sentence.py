'''
Created on Aug 27, 2012

@author: dicle
'''

from Word import Word
import IOtools

class Sentence:
    
    sntid = -1
    words = []
    sentimentvalue = 0.0
    verb = Word("")
    
    polaritycount = {}   # { polaritytag : # of words of type polaritytag } like -1 : 15,1:20 ... 
    
    contains2countries = False

    def __init__(self, sntid = -1):
        self.words = []
        self.sntid = sntid
        self.polaritycount = {}
        self.contains2countries = False

    def insertword(self, word):
        self.words.append(word)
    
    # words is a list containing Word s
    def toscreen(self, wordlist):      
        
        snttxt = ""
        for wordid in self.words:
            newword = wordlist[wordid]
            snttxt = snttxt + newword.literal + " "
           
        print "snt id: ",self.sntid
        print " ",snttxt,"\n"
       
        print " polarity counts: "
        IOtools.printdictionary(self.polaritycount)
        print "__verb: "
        self.verb.toscreen_simple()
        
        
    def tostring(self, wordlist):
        out = ""
        snttxt = ""
        for wordid in self.words:
            newword = wordlist[wordid]
            snttxt = snttxt + newword.literal + " "
        
        out += "snt id: " + str(self.sntid) + "\n"
        out += " " + snttxt + "\n"
       
        out += " polarity counts: \n"
        out += IOtools.dict_tostring(self.polaritycount)
        
        out += "\n__verb: \n "
        out += self.verb.tostring_record()+"\n-------\n\n\n"
        
        return out
    
    def __cmp__(self, otherID):
   
        if self.sntid > otherID:
            return 1
        elif self.sntid == otherID:
            return 0
        else:
            return -1   
  
    
    
