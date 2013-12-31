'''
Created on Apr 24, 2013

@author: dicle
'''
# for reading deyimler.xml

import os
import re
import xml.etree.ElementTree as ET
from BeautifulSoup import BeautifulSoup

from sentimentfinding import IOtools
from txtprocessor import listutils


phrasesxmlin = "/home/dicle/Dicle/Tez/languageresources/deyimler/ADB/"
phrasestxtout = "/home/dicle/Dicle/Tez/languageresources/deyimler/deyimlertxt/"


deyimDOMpath = "entry/name"



def parseXML_phraseslexicon(xmlinfilepath, txtoutfilepath):
    xmlfnames = IOtools.getfilenames_of_dir(xmlinfilepath, removeextension=False)   
    statsstr = "letter       numofphrases"+"\n"
    for fname in xmlfnames:
        letter = fname.split("_")[-1][:-4]    # each fname is of the form "ADB_letter.xml"
        print fname
        path = xmlinfilepath + os.sep  + fname
        tree = ET.parse(path)
        lexiconroot = tree.getroot()
        names = lexiconroot.findall(deyimDOMpath)
        phrases = []
        for name in names:
            phrase = name.text
            phrases.append(phrase.strip().lower())
        outpath = txtoutfilepath + os.sep + letter + ".txt"
        IOtools.todisc_list(outpath, phrases)  
        statsstr += letter+"\t"+str(len(phrases))+"\n"
    IOtools.todisc_txt(statsstr, txtoutfilepath+os.sep+"originalstats.txt")
    




# remove parantheses, create alternatives from the words inside them
def editphrase(s):
    
    removepatterns = [r"\(bir[^)]+\)", r"\(veya[^)]+\)", r"\(kendi[^)]*\)"]
    # remove (bir+)
    
    for p in removepatterns:
        l = re.split(p,s)
        l = filter(lambda x : len(x) != 0, l)
        l = map(lambda x : x.strip(), l)
        s = " ".join(l)
    
    return s
    
    
    '''
    l = re.split(r"\(bir[^)]+\)",s)
    l = filter(lambda x : len(x) != 0, l)
    l = map(lambda x : x.strip(), l)
    s = " ".join(l)
    
    print s
    '''
    
    
    # this is no feasible
    '''
    # generate alternative if the phrase includes (veya WORD)   WORD might have words separated with ',' 
    originalwords = re.split(r"\([^)]+\)",s)
    originalwords = filter(lambda x : len(x) != 0, originalwords)
    originalwords = map(lambda x : x.strip(), originalwords)
    
    allwords = re.split(r"(\([^)]+\))",s)
    allwords = filter(lambda x : len(x) != 0, allwords)
    allwords = map(lambda x : x.strip(), allwords)
    

    templist = []
    templist.extend(allwords)
    for i,w in enumerate(allwords):
        if w not in originalwords:
            allwords[i] = w.strip()[len("(veya "):-1] 
    
    
    allwords = listutils.separateitemsaswords(allwords)
    originalwords = listutils.separateitemsaswords(originalwords)
  
    altwords = list(set(allwords) - set(originalwords))
    #altwords = map(lambda x : x[len("(veya "):-1], altwords)
    
    print allwords," ",originalwords," ",altwords
    newphrases = []
    plist = []
    plist.extend(originalwords)
    for i,w in enumerate(allwords):
        for a in altwords:
            if i < len(allwords)-1:
                if allwords[i+1] == a:
                    print i," ",w,"  ",a
                    plist[originalwords.index(w)] = a
                    newphrases.append(" ".join(plist))
    
    plist = []
    plist.extend(altwords)
    for i,w in enumerate(allwords):    
        for o in originalwords:
            if i > 0:
                if allwords[i-1] == o:
                    print i," ",w,"  ",o
                    print plist
                    #plist[altwords.index(w)] = o
                    #plist[i] = o
                    plist.insert(1, o)
                    #plist[plist.index(w)] = o
                    newphrases.append(" ".join(plist))
    
    if len(newphrases) == 0:
        newphrases.append(s)
    return newphrases
'''

if __name__ == "__main__":
    
    #parseXML_phraseslexicon(phrasesxmlin, phrasestxtout)
    
  
    #IOtools.mergetxtfiles(phrasestxtout, phrasestxtout+os.sep+"tr_phrases.txt")
     
    #s = 'dedigini (veya soyledigini) yapmak (veya etmek)'
    #s = "(birine) dedigini yapmak (veya soylemek) (kendi)"
    s = "dedigini yapmak"
    print editphrase(s) 
    
    
    
    allphrases = IOtools.readtextlines(phrasestxtout+os.sep+"tr_phrases.txt")
    refinedphrases = []
    for p in allphrases:
        refinedphrases.append(editphrase(p))
    
    IOtools.todisc_list(phrasestxtout+os.sep+"tr_phrasesEDIT.txt", refinedphrases)
    
    '''    
    fname = "ADB_a.xml"
    
    path = phrasespath + os.sep + fname
    tree = ET.parse(path)
    root = tree.getroot()
    
    
    print root.findtext("name")
    names = root.findall("entry/name")
    
    for name in names:
        print name.text
    
    
    #root = ET.fromstring(text)
    
    #for name in root.iter('name'):
    for name in root.findall('name'):
        print name
    
    '''
    
    
    '''
    content = BeautifulSoup(path)
    phrase = content.entry.name
    print phrase
    '''
        
        
        
