# -*- coding: utf-8 -*-
'''
Created on Jul 25, 2013

@author: dicle
'''

import numpy as np
import pandas as pd
import os

from sentimentfinding import IOtools


# termfreq * catfreq
# catdocmap = { catlabel: docidlist }
def compute_tfcf_wordcloud(catdocmap, freqdf, path, log=True, smoothlbl="log"):
    words = freqdf.columns.values.tolist()
    cattfdf = pd.DataFrame(index=map(lambda x : smoothlbl+"-"+x, catdocmap.keys()), columns=words)
    
        
    for catlabel, fileidlist in catdocmap.iteritems():
        selectedvectors = freqdf.loc[fileidlist, :].values
        wordweightvector = np.sum(selectedvectors, axis=0, dtype=float)
        categorywisefreq = map(np.count_nonzero, selectedvectors.T[:,])  # count the # of nonzero elements in each column
        if log:
            #tfcatfreq = np.log(wordweightvector * categorywisefreq, dtype=float)
            tfcatfreq = np.log(wordweightvector, dtype=float) * categorywisefreq
            catlabel = smoothlbl+"-"+catlabel
        else:
            tfcatfreq = wordweightvector * categorywisefreq
        weightedwordlist = [word+"\t"+str(weight) for word, weight in zip(words, tfcatfreq) if weight > 0.0]
        #IOtools.todisc_list(path+os.sep+"tfcatf_"+catlabel+"-"+smoothlbl+".txt", weightedwordlist)
        cattfdf.loc[catlabel,:] = categorywisefreq
    return cattfdf


def wordletxt_todisc(words, wordweightvector, path):
    weightedwordlist = [word+"\t"+str(weight) for word, weight in zip(words, wordweightvector)  if weight > 1.0]  # if weight > 0.0]
    IOtools.todisc_list(path, weightedwordlist)       
    

# call geziyakurdi.preparewordle.create_txt_wordle(catdf, negpos, "neg", rpath, weightthreshold=4.0)
def create_txt_wordle(cattf, diffvector, category, rpath, weightthreshold=1.0):
    words = cattf.columns.values.tolist()
    
    diffedwordslist = [(word+" ")*weight for word, weight in zip(words, diffvector) if weight > weightthreshold]
    diffedwordstxt = " ".join(diffedwordslist)
    IOtools.todisc_txt(diffedwordstxt, rpath+os.sep+"w"+str(weightthreshold)+"log"+category+"diff.txt")
    

def get_cat_fileids(fileids):
    #fileids = freqdf.index.values.tolist()
    labels = {}
    labels["pos"] = []
    labels["neg"] = []
    
    for fileid in fileids:
        if fileid.startswith("neg"):
            labels["neg"].append(fileid)
        else:
            labels["pos"].append(fileid) 
    return labels


def get_wordweight_bycat(freqdf, labels, word):
    print "weight of ",word
    for label, fileids in labels.iteritems():
        print label
        for fileid in fileids:
            print fileid[:14]," : ",freqdf.loc[fileid, word]
        
        
           

def catsweighting_towordle(log=True, smoothlbl="logw3"):
    recordpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/wordletest/words/tfcatf/"
    dfpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/wordletest/matrix/"
    
    
    dfname = "matrixdoctermFREQ.csv" 
    freqdf = pd.read_csv(dfpath+os.sep+dfname, index_col=0)
    
    fileids = freqdf.index.values.tolist()
    labels = {}
    labels["pos"] = []
    labels["neg"] = []
    
    for fileid in fileids:
        if fileid.startswith("neg"):
            labels["neg"].append(fileid)
        else:
            labels["pos"].append(fileid)
    
    return compute_tfcf_wordcloud(labels, freqdf, recordpath, log, smoothlbl)
    
    

# fileidlist: names of the files whose words to be included in the wordcloud
# collects the separate files in one (content of file)
def prepare_wordcloud(fileidlist, df, path, filename):
    words = df.columns.values.tolist()
    selectedvectors = df.loc[fileidlist, :].values
    wordweightvector = np.sum(selectedvectors, axis=0)
    
    weightedwordlist = [word+"\t"+str(weight) for word, weight in zip(words, wordweightvector) if weight > 0.0]
    IOtools.todisc_list(path+os.sep+filename, weightedwordlist)
    
    

# outputs the word cloud files of each file in a folder separately
# df is assumed to contain words at columns and docs(as ids) at rows 
def map_wordcollections_towordle(df, recordpath, fileids=[]):
    if not fileids:
        fileids = df.index.values.tolist()

    for fileid in fileids:
        prepare_wordcloud([fileid], df, recordpath, fileid)
        
        

def merge_wordcollections_towordle():
    
    inpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/edit/wordletest/words/"
    collections = ["inlier", "outlier"]
    freqdf = pd.read_csv("/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/edit/wordletest/matrix/matrixdoctermFREQ.csv", index_col=0)
    
    for cname in collections:
        fileids = IOtools.getfilenames_of_dir(inpath+os.sep+cname, removeextension=False)
        prepare_wordcloud(fileids, freqdf, inpath, cname)


def docs_towordle():
    recordpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/wordletest/words/"
    dfpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/wordletest/matrix/"
    
    #1 get wordle's of freq
    dfname = "matrixdoctermFREQ.csv" 
    freqdf = pd.read_csv(dfpath+os.sep+dfname, index_col=0)    
    #fileids = freqdf.index.values.tolist()
    #fileids = [fileid[:-4] for fileid in fileids]
    map_wordcollections_towordle(freqdf, recordpath+os.sep+"wordle_freq")
    
    
    #2 get wordle's of tfidf
    dfname = "matrixdoctermTFIDF.csv" 
    tfidfdf = pd.read_csv(dfpath+os.sep+dfname, index_col=0)
    #fileids = tfidfdf.index.values.tolist()
    #fileids = [fileid[:-4] for fileid in fileids]
    map_wordcollections_towordle(tfidfdf, recordpath+os.sep+"wordle_tfidf")
    
    
    #3 get wordle's of pca'd
    dfname = "doctermPCA.csv"
    pcadf = pd.read_csv(dfpath+os.sep+dfname, index_col=0)
    #fileids = pcadf.index.values.tolist()
    #fileids = [fileid[:-4] for fileid in fileids]
    map_wordcollections_towordle(pcadf, recordpath+os.sep+"wordle_pca")



def cats_towordle():
    recordpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/wordletest/words/"
    dfpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/wordletest/matrix/"
    
    
    dfname = "matrixdoctermFREQ.csv" 
    freqdf = pd.read_csv(dfpath+os.sep+dfname, index_col=0)
    
    dfname = "matrixdoctermTFIDF.csv" 
    tfidfdf = pd.read_csv(dfpath+os.sep+dfname, index_col=0)
    
    dfname = "doctermPCA.csv"
    pcadf = pd.read_csv(dfpath+os.sep+dfname, index_col=0)
    
    fileids = freqdf.index.values.tolist()
    labels = {}
    labels["pos"] = []
    labels["neg"] = []
    
    for fileid in fileids:
        if fileid.startswith("neg"):
            labels["neg"].append(fileid)
        else:
            labels["pos"].append(fileid)
    
    
    for label,lst in labels.iteritems():
        prepare_wordcloud(lst, freqdf, recordpath+os.sep+"wordle_freq", label+"wordle.txt")    
        prepare_wordcloud(lst, tfidfdf, recordpath+os.sep+"wordle_tfidf", label+"wordle.txt") 
        prepare_wordcloud(lst, pcadf, recordpath+os.sep+"wordle_pca", label+"wordle.txt") 
        

if __name__ == "__main__":
    
    #cats_towordle()
    catdf = catsweighting_towordle()
    
    
    '''
    recordpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/wordletest/words/"
    dfpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/wordletest/matrix/"
    
    #1 get wordle's of freq
    dfname = "matrixdoctermFREQ.csv" 
    freqdf = pd.read_csv(dfpath+os.sep+dfname, index_col=0)
    #fileids = freqdf.index.values.tolist()
    #fileids = [fileid[:-4] for fileid in fileids]
    map_wordcollections_towordle(freqdf, recordpath+os.sep+"wordle_freq")
    
    
    #2 get wordle's of tfidf
    dfname = "matrixdoctermTFIDF.csv" 
    tfidfdf = pd.read_csv(dfpath+os.sep+dfname, index_col=0)
    #fileids = tfidfdf.index.values.tolist()
    #fileids = [fileid[:-4] for fileid in fileids]
    map_wordcollections_towordle(tfidfdf, recordpath+os.sep+"wordle_tfidf")
    
    
    #3 get wordle's of pca'd
    dfname = "doctermPCA.csv"
    pcadf = pd.read_csv(dfpath+os.sep+dfname, index_col=0)
    #fileids = pcadf.index.values.tolist()
    #fileids = [fileid[:-4] for fileid in fileids]
    map_wordcollections_towordle(pcadf, recordpath+os.sep+"wordle_pca")
    '''
    
    
    ''' older
    merge_wordcollections_towordle()
    map_wordcollections_towordle()
    '''
    
    
    