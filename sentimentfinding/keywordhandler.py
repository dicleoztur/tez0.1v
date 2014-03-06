'''
Created on Feb 3, 2013

@author: dicle
'''

import os

from txtprocessor import texter
from sentimentfinding import IOtools
 

countrypath = "/home/dicle/Dicle/Tez/dataset/country/"
countries = ["syria","turkey"]


magnitude_path = "/home/dicle/Dropbox/Tez/system/data/magnitude/"
extent = ["small", "big"]

polaritybase = "/home/dicle/Dropbox/Tez/system/data/polaritybase/"
poles = ["positive", "negative"]


keywordindices = ["polarity","country","magnitude"]
keywordrootpath = "/home/dicle/Dropbox/Tez/system/keywordbase/"

subjectivitylexicon = keywordrootpath + os.sep + "tr_strongsubjective.txt"

subjverbslist = keywordrootpath + os.sep + "tr_subjectiveverbs-change.txt"

abstractnesslexicon = keywordrootpath + os.sep + "tr_abstractwords-change.txt"



class KeywordIndex:
    '''
    ex.
    KeywordIndex newindex = KeywordIndex(name = "polarity", rootpath = "/home/keywords/", cats = ["positive", "negative"])
    '''
    
    name = ""         # name of this keyword index
    rootpath = ""   # directory of the files that contain categorized keywords for this index. (wordlists-txtfiles for each cats[i])
    cats = []    # categories for this keyword index. 

    def __init__(self, name = "", rootpath = ""):
        self.name = name
        self.rootpath = rootpath
        self.cats = IOtools.getfilenames_of_dir(rootpath)   
        
    
    def __str__(self):
        return self.name
    
    def __get__(self):
        return self.name    
        
        
# complete func. for polarity and  


# returns sorted and smallcased list of keywords located at the given path
def getkeywordlist(path):
    keywords = texter.readtextlines(path)
    keywords = list(set(keywords))
    keywords = [w.lower() for w in keywords]
    keywords.sort()
    return keywords



def get_keyword_dict2(keyword_rootpath, keywordlabels):
    extentdict = {}   # includes the words of the keyword label. -- {'big': ['buyuk','muazzam'..],..} --
    
    for item in keywordlabels:
        path = keyword_rootpath+item+".txt"
        words = getkeywordlist(path)
        extentdict[item] = words
    
    '''
    smallerpath = magnitude_path+os.sep+extent[0]+".txt"
    smallwords = getkeywordlist(smallerpath)
    
    biggerpath = magnitude_path+os.sep+extent[1]+".txt"
    bigwords = getkeywordlist(biggerpath)
    '''
    return extentdict
   

# returns the keywordlabel : keywords dictionary for the given choice which can be 'magnitude', 'polarity' or 'country', 
#  that we store in our system as useful and searchable keyword classes
def get_keyword_dict(choice):
    
    extentdict = {}   # includes the words of the keyword label. -- {'big': ['buyuk','muazzam'..],..}
    
    ####   devam   15.07. geldim 21.52

    if choice not in keywordindices:
        return extentdict
    
    keywordlabel = choice
    path = keywordrootpath + os.sep + choice  + os.sep
    newkeywordindex = KeywordIndex(keywordlabel, path)
    
    for category in newkeywordindex.cats:
        path = newkeywordindex.rootpath+os.sep+category+".txt"
        words = getkeywordlist(path)
        extentdict[category] = words

    return extentdict
   


def get_subjectivity_lexicon():
    subjectivewords = texter.readtextlines(subjectivitylexicon)
    return subjectivewords



def get_subjective_verbs():
    subjectiveverbs = getkeywordlist(subjverbslist)
    return subjectiveverbs

    
def get_abstractwords():
    abstractwords = getkeywordlist(abstractnesslexicon)
    newlist = []
    for w in abstractwords:
        if w.startswith("*") or w.startswith("-"):
            word = w[1:]
        else:
            word = w
        newlist.append(word)
    
    return list(set(newlist))   
             



if __name__ == "__main__":
        
    choice = "country"
    
    print choice," keywords: "
    dct = get_keyword_dict(choice)
    print dct
    
    
    
    keywords = []
    for keywordindex in keywordindices:
        name = keywordindex
        path = keywordrootpath + os.sep + name  + os.sep
        newkeywordindex = KeywordIndex(name, path)
        keywords.append(newkeywordindex)
        
    
    for item in keywords:
        print item.name," ",item.cats   
    
    
    
    '''
    dct = get_keyword_dict(magnitude_path, )
    
    print dct
    print dct["big"]
    '''
    
    