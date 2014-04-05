'''
Created on Feb 6, 2014

@author: dicle
'''

import os
from nltk import ConditionalFreqDist
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


import arrange_N_classes
from txtprocessor import texter
from sentimentfinding import IOtools
from sentimentfinding import keywordhandler
from sentimentfinding import CFDhelpers
from sentimentfinding import plotter
from languagetools import SAKsParser
from corpus import extractnewsmetadata, metacorpus

class Corpus:
    
    corpusname = ""  
    rootpath = ""
    #recordpath = ""
    matrixpath = ""
    
    
    def __init__(self, cname="", rootpath=""):
        
        self.corpusname = cname
        
        self.rootpath = rootpath
        
        #self.recordpath = IOtools.ensure_dir(self.rootpath + os.sep + "record" + os.sep)
        self.matrixpath = IOtools.ensure_dir(self.rootpath + os.sep + "rawfeatures" + os.sep)
        
        '''
        subfolders = ["metadata", "results", "matrix"]
        for subf in subfolders:
            p = IOtools.ensure_dir(self.rootpath+ os.sep + subf+ os.sep)
        '''
    
    
    def setname(self, cname):
        self.corpusname = cname
         
    
    def find_word_matrices(self, newsidlist, processcontent=True, prepend="content"):
        dateroots = []
        datePOStag = []
        
        titleexclamation = [("newsid", "title_exclamation")]
        
        textPOStag = []
        textroots = [] 
        textrootsWpostag = []
        textliterals = []
        
        print prepend, " processing:"
        for newsid in newsidlist:
            print "newsid ",newsid
            filepath = extractnewsmetadata.newsid_to_filepath(newsid)
            content, title, date = extractnewsmetadata.get_news_article2(filepath)
            text = ""
            if processcontent:
                text = content
            else:
                text = title
                if "!" in title:
                    titleexclamation.append((newsid, 1))
                else:
                    titleexclamation.append((newsid, 0))
            
            words = texter.getwords(text)
            lemmata = SAKsParser.lemmatize_lexicon(words)
            for (literal, literalPOS, root, rootPOS) in lemmata:
                
                root = texter.cleanword(root)
                if (len(root) > 0) or (not root.isspace()):
                    #print root,
                    textPOStag.append((newsid, literalPOS))
                    textroots.append((newsid, root))
                    textrootsWpostag.append((newsid, root+" Wpostag "+rootPOS))
                    textliterals.append((newsid, literal+" Wpostag "+literalPOS))
                    dateroots.append((date, root))
                    datePOStag.append((date, literalPOS))
        
          
        cfd_dateroots = ConditionalFreqDist(dateroots)
        cfd_datepostag = ConditionalFreqDist(datePOStag)
        cfd_textpostag = ConditionalFreqDist(textPOStag)
        cfd_textroots = ConditionalFreqDist(textroots)
        cfd_textrootWpostag = ConditionalFreqDist(textrootsWpostag)
        cfd_textliterals = ConditionalFreqDist(textliterals)
        
        print "some id's", cfd_textroots.conditions()
        
        cfd_roottext = ConditionalFreqDist((word, docid) for docid in cfd_textroots.conditions()
                                           for word in list(cfd_textroots[docid])) 
                
        
        # cfd to csv  conditems as cols duzelt:
        csvpath = os.path.join(self.matrixpath, prepend+"-dateroot.csv")
        CFDhelpers.cfd_to_matrix(cfd_dateroots, csvpath)
        
        csvpath = os.path.join(self.matrixpath, prepend+"-datepostag.csv")
        CFDhelpers.cfd_to_matrix(cfd_datepostag, csvpath)
        
        csvpath = os.path.join(self.matrixpath, prepend+"-postagCOUNT.csv")
        CFDhelpers.cfd_to_matrix(cfd_textpostag, csvpath)
        
        termcountcsvpath = os.path.join(self.matrixpath, prepend+"termCOUNT.csv")
        CFDhelpers.cfd_to_matrix(cfd_textroots, termcountcsvpath)
        tfidfcsvpath = os.path.join(self.matrixpath, prepend+"termTFIDF.csv")
        texter.compute_tfidf_ondisc(termcountcsvpath, tfidfcsvpath)
                
        csvpath = os.path.join(self.matrixpath, prepend+"-rootcountindex.csv")
        CFDhelpers.cfd_to_matrix(cfd_roottext, csvpath)
        
        csvpath = os.path.join(self.matrixpath, prepend+"rootWpostagCOUNT.csv")
        CFDhelpers.cfd_to_matrix(cfd_textrootWpostag, csvpath)
        
        csvpath = os.path.join(self.matrixpath, prepend+"literalWpostagCOUNT.csv")
        CFDhelpers.cfd_to_matrix(cfd_textliterals, csvpath)
        
        
        # diger csv'lerden devam   6 Subat 05:42 uyuyuyuyuyuyu
        # kalklaklkalklklaklaklkal 15:32
        
        if not processcontent:
            print "keep exclamation !"
            IOtools.tocsv_lst(titleexclamation, os.path.join(self.matrixpath, prepend+"-exclamation.csv"))
   
   
      
    def extract_corpus_features(self, newsidlist, nostopwords=True):
        
        self.find_word_matrices(newsidlist, processcontent=True, prepend="content")  # read content features
        self.find_word_matrices(newsidlist, processcontent=False, prepend="title")  # read title features



'''

def read_corpus(annotationtype, taggertype, corpusrecordpath=metacorpus.learningdatapath, datasetsize=None):
    
    membersfilepath = metacorpus.get_annotatedtexts_file_path(annotationtype, taggertypes[0])
    membersdf = IOtools.readcsv(membersfilepath)
        
    newsids = membersdf.loc[:, "questionname"].values
    
    if datasetsize is None:
        datasetsize = len(newsids)
        selection = membersdf.copy()
    else:
        newsids = newsids[np.random.choice(len(newsids), datasetsize)]
        selection = membersdf[membersdf["questionname"].isin(newsids)]
    newsids = newsids.tolist()

    corpusname = annotationtype+"-N"+str(datasetsize)  #annotationtype+"-"+taggertype+"-N"+str(datasetsize)
    temppath = os.path.join(corpusrecordpath, annotationtype, str(datasetsize))
    corpuspath = IOtools.ensure_dir(temppath)
      
    answersdf = pd.DataFrame(selection["answer"].values, index=selection["questionname"].values.tolist(), columns=["answer"])
    answerspath = IOtools.ensure_dir(os.path.join(corpuspath, "labels"))
    IOtools.tocsv(answersdf, answerspath+os.sep+taggertype+".csv", keepindex=True)
    
    corpus = Corpus(cname=corpusname, rootpath=corpuspath)
    corpus.extract_corpus_features(newsids)
    return corpuspath


'''


def read_corpus(annotationtype, taggertypes, corpusrecordpath=metacorpus.learningdatapath, datasetsize=None):
    
    membersfilepath = metacorpus.get_annotatedtexts_file_path(annotationtype, taggertypes[0])
    membersdf = IOtools.readcsv(membersfilepath)
        
    newsids = membersdf.loc[:, "questionname"].values
    
    if datasetsize is None:
        datasetsize = len(newsids)
    else:
        newsids = newsids[np.random.choice(len(newsids), datasetsize)]
    newsids = newsids.tolist()

    corpusname = annotationtype+"-N"+str(datasetsize)  #annotationtype+"-"+taggertype+"-N"+str(datasetsize)
    temppath = os.path.join(corpusrecordpath, annotationtype, str(datasetsize))
    corpuspath = IOtools.ensure_dir(temppath)
      
    answerrootpath = IOtools.ensure_dir(os.path.join(corpuspath, "labels"))
    extract_answers(annotationtype, taggertypes, newsids, answerrootpath)
    
    corpus = Corpus(cname=corpusname, rootpath=corpuspath)
    corpus.extract_corpus_features(newsids)
    return corpuspath



def extract_answers(annotationtype, taggertypes, selectednewsids, answerrootpath):    
    
    for taggertype in taggertypes:
        dfpath = metacorpus.get_annotatedtexts_file_path(annotationtype, taggertype)
        df = IOtools.readcsv(dfpath)
        selection = df[df["questionname"].isin(selectednewsids)]
        
        answersdf = pd.DataFrame(selection["answer"].values, index=selection["questionname"].values.tolist(), columns=["answer"])
        
        answervals = answersdf.answer.values.tolist()
        has_unknown = 5 in answervals
        print answervals
        print type(answervals)
        nclasses = len(list(set(answervals)))
        
        #answerrootpath = IOtools.ensure_dir(os.path.join(corpuspath, "labels"))
        
        originalanswersfolder = arrange_N_classes.ensure_nclass_dir(answerrootpath, nclasses)   # record original labels
        originalanswerspath = os.path.join(originalanswersfolder, taggertype+".csv")
        IOtools.tocsv(answersdf, originalanswerspath, keepindex=True)
        
        arrange_N_classes.get_nclass_variations(answerrootpath, taggertype, has_unknown, in_NC=nclasses)
        
        

def read_corpus_from_file(membersfilepath, corpusname, recordpath):
    membersdf = IOtools.readcsv(membersfilepath)
    newsids = membersdf.loc[:, "questionname"].values.tolist()
    corpus = Corpus(cname=corpusname, rootpath=recordpath)
    corpus.extract_corpus_features(newsids)
    
    
if __name__ == "__main__":
        
    annottype = "double"
    taggertypes = ["user"]
    
    recordpath = read_corpus(annotationtype=annottype, taggertypes=taggertypes, datasetsize=30)
    print recordpath
    
    
    
    
    
    
    
    
    


