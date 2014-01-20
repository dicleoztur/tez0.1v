# -*- coding: utf-8 -*-

'''
Created on Jul 30, 2012

@author: dicle
'''

import codecs
import os
import re
import nltk.data


'''  eski:
IDlistPath = "/home/dicle/Dicle/Tez/dataset/idlist/"
itemsPath = "/home/dicle/Dicle/Tez/dataset/newsitems/"
'''

'''  eski eski
IDlistPath = "/home/dicle/Dicle/Tez/datasetv3/idlist/"
itemsPath = "/home/dicle/Dicle/Tez/datasetv3/newsitems/"
'''

crawlrootpath = "/home/dicle/Dicle/Tez/dataset/recrawl_missing/"
IDlistPath = crawlrootpath + os.sep + "idlist" + os.sep
itemsPath = crawlrootpath + os.sep + "newsitems" + os.sep
logPath = crawlrootpath + os.sep + "logs" + os.sep


'''  close on 30 Jun
crawlrootpath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/"
IDlistPath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/idlist/"
itemsPath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/newsitems/"
logPath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/logs/"
'''




# returns the names of the files and dirs in the given directory *path*
def getfilenames_of_dir(path, removeextension=True):
    files = os.listdir(path)
    filenames = []
    for fileitem in files:
        if os.path.isfile(path+os.sep+fileitem):
            if removeextension:
                filename = fileitem.split(".")[0]   # remove extension if any
            else:
                filename = fileitem
            filenames.append(filename)
        
    return filenames

def getfoldernames_of_dir(path):
    files = os.listdir(path)
    foldernames = []
    for fileitem in files:
        if os.path.isdir(path+os.sep+fileitem):
            foldernames.append(fileitem)
    return foldernames



def log_connection_error(resourcename, link, message):
    txt = link + "\n" + message + "\n"
    logpath = ensure_dir(logPath)
    path = logpath + resourcename + ".log"
    todisc_txt(txt, path, m="a")

def ensure_dir(f):
    f = f+os.sep
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return f 


def todisc_list(lst, path):
    f = codecs.open(path,"w", encoding='utf-8')
    for w in lst:
        f.write(w+"\n")
    #f.write("\n".join(lst))
    f.close()


def todisc_txt(txt, path, m="w"):
    f = codecs.open(path, m, encoding='utf-8')
    f.write(txt)
    f.close()

def todisc_list2(lst, path):
    f = open(path,"w")
    f.write("\n".join(lst))
    f.close()


def printlist(lst):
    for w in lst:
        print w



def encodingToutf8(text, encoding):
    text = unicode(text, encoding)
    text = text.encode('utf-8')
    return text  
    

def replaceSpecialChars(text):
    
    p1 = re.compile(r'&(r|l)dquo;')
    text = p1.sub('\"', text)

    p2 = re.compile(r'&(r|l)squo;')
    text = p2.sub('\'', text)  
    
    p3 = re.compile(r'\s+')
    text = p3.sub(' ',text) 
    
    p4 = re.compile(r'</p>?')
    text = p4.sub(' ',text)
    
    p5 = re.compile(r'<?/p>') 
    text = p5.sub(' ',text)
    
    p6 = re.compile(r'<p>?')
    text = p6.sub(' ',text)

    p7 = re.compile(r'<?p>')
    text = p7.sub(' ',text)
    
    p8 = re.compile(r'&quot;')
    text = p8.sub('\'', text)
  
    p9 = re.compile(r'&#039;')
    text = p9.sub("'", text)
    
    return text
    


def splitToSentences(txt):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(txt.strip())
    #print '\n-----\n'.join(sents)
    return sents

       
    