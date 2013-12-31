'''
Created on May 6, 2013

@author: dicle
'''

import os

import texter
from sentimentfinding import IOtools
from languagetools import SAKsParser

# to make tr subjectivity lexicon (translated from mpqa corpus)


mpqapath = "/home/dicle/Dicle/Tez/languageresources/mpqa/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff"

trpath = "/home/dicle/Dicle/Tez/languageresources/tr_subjectivitylexicon/edit/"
tr_translations = trpath + os.sep + "engtr.txt"

jointlexiconpath = trpath + os.sep + "trsubjectivitylexicon.txt"



types = ["strongsubj", "weaksubj"]
poles = ["positive", "negative", "neutral"]

    


def get_jointlexicon():
    engtr = texter.readtextlines(tr_translations)
    mpqa = texter.readtextlines(mpqapath)
    
    newclues = []
    for tr,en in zip(engtr, mpqa):
        # edit original en words from mpqa
        items = en.split()
        items = filter(lambda x : (not x.startswith('len') and not x.startswith('stem') and len(x)>1), items)
        print en," ###### ",items
        items = [item.split("=")[1] for item in items]
        enclue = " ".join(items)
        
        # edit tr words
        trword = tr.split(":")[1]
        trword = trword.strip().lower()
        s = enclue + "\t" + trword
        newclues.append(s)
    IOtools.todisc_list(jointlexiconpath, newclues)
    
    
def categorize_lexicon():
    lexicon = texter.readtextlines(jointlexiconpath) 
    fname = ""
    
    for line in lexicon:
        items = line.split()
        t = items[0]
        p = items[3]
        fname = t+"_"+p+".txt"
        s = items[1]+" "+items[2]+"\t"+items[4]+"\n"
        IOtools.txttodisc_append(s, trpath+os.sep+fname) 

def lemmatize_lexicon(fname):
    words = texter.readtextlines(trpath+os.sep+fname)
    lemmata = SAKsParser.lemmatize_lexicon(words)
    IOtools.todisc_list(trpath+os.sep+"lemmatized_"+fname, lemmata)
               


if __name__ == "__main__":
    #get_jointlexicon()
    #categorize_lexicon()
    #lemmatize_lexicon("tr_strongsubjective.txt")
    
    l = texter.readtextlines(trpath+"/tr_strongsubjective.txt")
    l = list(set(l))
    l.sort()
    IOtools.todisc_list(trpath+"/tr_strongsubjective_vv.txt", l)
    
        