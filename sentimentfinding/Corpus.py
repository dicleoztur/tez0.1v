'''
Created on Sep 3, 2012

@author: dicle
'''

import os
from nltk import FreqDist
import pickle

from Word import Word
from Text import Text
from languagetools import SAKsParser
import IOtools
from txtprocessor import listutils,texter

class Corpus:
    
    corpusname = ""
    words = []    # list of Word instances obtained from all texts
    texts = []    # list of all texts as Text instances
    rootpath = ""  # root path of dataset
    polaritytags = {}
    POStags = []
    outputpath = ""   # path to the results for this corpus
    picklepath = ""
    
    
    def __init__(self, path="", corpusname=""):
        self.corpusname = corpusname
        self.words = []
        self.texts = []
        self.rootpath = path
        self.polaritytags = {-1 : "neg", 0 : "neut", 1 : "pos"}
        self.POStags = SAKsParser.poslabels.keys()
        self.outputpath = IOtools.ensure_dir(IOtools.results_rootpath+os.sep+corpusname+os.sep)
        self.picklepath = IOtools.ensure_dir(self.outputpath+os.sep+"pickle"+os.sep)
        
    def assignwords(self, wordlist):
        self.words.extend(wordlist)
    
    def inserttext(self, text):
        self.texts.append(text)
        
    def assign_polaritysymbols(self, poldct):    
        self.polaritytags = poldct
        
    def set_corpusname(self, corpusname):
        self.corpusname = corpusname
        
    def gettext(self, textid):
        if textid in self.texts:
            indx = self.texts.index(textid)
            return self.texts[indx]
        else:
            return Text()
        
    def sentenceAsText(self, txtid, sntid):
        newtext = self.gettext(txtid)
        newsnt = newtext.getsentence(sntid)
        snttxt = ""
        for wordindex in newsnt.words:
            snttxt = snttxt + self.words[wordindex].literal + " "
        
        return snttxt
    
    # returns the index at the self Word list of the input word which is a string. if not found, returns -1.
    # the input word can be compared against literal forms or root forms of the Words in self list. 
    def searchword(self, newword, literalsearch = True):
        
        if literalsearch:
            for i,w in enumerate(self.words):
                if w.literal == newword:
                    return i 
        else:
            for i,w in enumerate(self.words):
                if w.root == newword:
                    return i 
        return -1
    

    # returns a list of Word instances from the input wordlist which is a list of strings. looks if the input words are in the corpus. 
    # if so, copies the found Words, otherwise creates new words.
    #  useful for fast search. 
    def get_Words_from_words(self, wordlist):
        newWordlist = []   # will contain Word instance of each string in the input wordlist. 
        
        for seekedword in wordlist:
            ind = self.searchword(seekedword)
            if ind < 0:   # seekedword not found in the corpus
                newword = Word(seekedword)
            else:
                newword = self.words[ind]
            newWordlist.append(newword)
        return newWordlist
    
        
    
    def get_words_ofPOStag(self, tag, printt=False):
        taggedwords = []
        for word in self.words:
            if word.POStag == tag:
                taggedwords.append(word)
        
        if printt:
            print "Words with tag ",tag
            for w in taggedwords:
                print w.literal
                
        return taggedwords

    def pickledumpwords(self, start=0, end=0):
        if end == 0:
            end = len(self.words)
        self.picklepath = self.picklepath+"pickle"+str(start)+"-"+str(end)+".txt"
        fwrite = open(self.picklepath, "w")
        pickle.dump(self.words[start:end], fwrite, -1)
        del self.words[0:]    # bismillah!
        fwrite.close() 
    
    def picklegetwords(self, start=0, end=0):
        if end == 0:
            end = len(self.words)
        #path = self.picklepath+"pickle"+str(start)+"-"+str(end)+".txt"
        fread = open(self.picklepath, "r")
        self.words = pickle.load(fread)
        fread.close()
         
    
    def reportstats(self):
        numofdocs = len(self.texts)  
        numofdistinctwords = len(self.words)
        
        totalnumofwords = 0
        totalnumofsentences = 0
        avgsentencelength = 0
        avgdoclength_bysnt = 0
        avgdoclength_byword = 0
        
        snt_lengths = []
        doc_lengths = []
        
        for txt in self.texts:
            doc_lengths.append(len(txt.sentences))
            #numofsentences += len(txt.sentences)
            for snt in txt.sentences:
                snt_lengths.append(len(snt.words))  

        totalnumofsentences = sum(doc_lengths)       
        totalnumofwords = sum(snt_lengths)
        
        avgsentencelength = totalnumofwords / totalnumofsentences
        avgdoclength_bysnt = totalnumofsentences / numofdocs
        avgdoclength_byword = totalnumofwords / numofdocs
        
        postaggedwords_length = {}  # contains *tag* : numofwordswith*tag*
        for postag in self.POStags:
            postaggedwords_length[postag] = len(self.get_words_ofPOStag(postag))
            
        
        output = "# of documents:\t\t\t"+str(numofdocs)
        output += "\n# of distinct words:\t\t"+str(numofdistinctwords)
        output += "\ntotal # of words:\t\t"+str(totalnumofwords)
        output += "\ntotal # of sentences:\t\t"+str(totalnumofsentences)
        output += "\naverage sentence length:\t"+str(avgsentencelength)
        output += "\naverage #of sntnces over docs:\t"+str(avgdoclength_bysnt)
        output += "\naverage # of words over docs:\t"+str(avgdoclength_byword)+"\n"
        for k,v in postaggedwords_length.items():
            output += "\n # of words of POS tag "+k+":\t"+str(v)
        
        IOtools.todisc_txt(output, self.outputpath+os.sep+"corpusstats"+self.corpusname+".txt")
        return output
    


if __name__ == "__main__":
    rootpath = "/home/dicle/Dicle/Tez/dataset/readingtest3030/6"
    corpusname = "6"
    corpus = Corpus(rootpath, corpusname)
    
    fileids = os.listdir(rootpath)    
    
    allwords1 = FreqDist()    # freqdist of Word instances
    allwords2 = FreqDist()   # freqdist of strings
    
    wordlist = []    # will contain list of Word instances
    
    for fileid in fileids:
        path = rootpath + os.sep + fileid
        rawtext = texter.readnewstext(path)
        words = rawtext.split()
        for word in words:
            wordlist.append(Word(word))
            allwords2.inc(word)
            
       
    
    
    print "fd with Word inst."
    listutils.printdictionary(allwords1)
    
    print "fd with str"
    listutils.printdictionary(allwords2)        
    
    
        
        