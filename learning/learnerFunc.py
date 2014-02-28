'''
Created on Feb 6, 2014

@author: dicle
'''

import pandas as pd
import os
import numpy as np

import matrixhelpers
from sentimentfinding import IOtools, keywordhandler


class Learner:
    
    recordpath = ""
    datamatrixfolder = ""
    
    features = {}
    
    def __init__(self, outpath, matrixpath):
        self.recordpath = outpath
        self.datamatrixfolder = matrixpath
        
        
    def train(self, matrix):
        return
    


class FeatureExtractors:
    features = {}
    inmatrixfolder = ""
    outmatrixfolder = ""
    
    def __init__(self, inmatrix, outmatrix):
        self.features = {}
        self.inmatrixfolder = inmatrix
        self.outmatrixfolder = outmatrix

    @property
    def content_adverbratio(self):
        incsvpath = os.path.join(self.inmatrixfolder, "content-postagCOUNT.csv")
        fname = "content-adverbratio"
        outcsvpath = os.path.join(self.outmatrixfolder, fname+".csv")
        numeratortags = ['ADV']
        denominatortags = ['ADJ', 'Verb']
        rationame = fname
        matrixhelpers.get_featuretags_ratio(incsvpath, outcsvpath, numeratortags, denominatortags, rationame)
        
    
    @property
    def content_adjectiveratio(self):
        incsvpath = os.path.join(self.inmatrixfolder, "content-postagCOUNT.csv")
        fname = "content-adjectiveratio"
        outcsvpath = os.path.join(self.outmatrixfolder, fname+".csv")
        numeratortags = ['ADJ']
        denominatortags = ['Noun']
        rationame = fname
        matrixhelpers.get_featuretags_ratio(incsvpath, outcsvpath, numeratortags, denominatortags, rationame)
    


    @property
    def content_adverbcount(self):
        incsvpath = os.path.join(self.inmatrixfolder, "content-postagCOUNT.csv")
        fname = "content-adverbcount"
        outcsvpath = os.path.join(self.outmatrixfolder, fname+".csv")
        featurename = fname
        postagdf = IOtools.readcsv(incsvpath, keepindex=True)
        adv_count_vect = postagdf.loc[:, "ADV"].values
        countdf = pd.DataFrame(adv_count_vect, index=postagdf.index.values.tolist(), columns=[featurename])
        IOtools.tocsv(countdf, outcsvpath, keepindex=True)
        
    
    @property
    def content_adjectivecount(self):
        incsvpath = os.path.join(self.inmatrixfolder, "content-postagCOUNT.csv")
        fname = "content-adjectivecount"
        outcsvpath = os.path.join(self.outmatrixfolder, fname+".csv")
        featurename = fname
        postagdf = IOtools.readcsv(incsvpath, keepindex=True)
        adj_count_vect = postagdf.loc[:, "ADJ"].values
        countdf = pd.DataFrame(adj_count_vect, index=postagdf.index.values.tolist(), columns=[featurename])
        IOtools.tocsv(countdf, outcsvpath, keepindex=True)
        
        

    @property
    def title_adverbratio(self):
        incsvpath = os.path.join(self.inmatrixfolder, "title-postagCOUNT.csv")
        fname = "title-adverbratio"
        outcsvpath = os.path.join(self.outmatrixfolder, fname+".csv")
        numeratortags = ['ADV']
        denominatortags = ['ADJ', 'Verb']
        rationame = fname
        matrixhelpers.get_featuretags_ratio(incsvpath, outcsvpath, numeratortags, denominatortags, rationame)
        
    
    @property
    def title_adjectiveratio(self):
        incsvpath = os.path.join(self.inmatrixfolder, "title-postagCOUNT.csv")
        fname = "title-adjectiveratio"
        outcsvpath = os.path.join(self.outmatrixfolder, fname+".csv")
        numeratortags = ['ADJ']
        denominatortags = ['Noun']
        rationame = fname
        matrixhelpers.get_featuretags_ratio(incsvpath, outcsvpath, numeratortags, denominatortags, rationame)        


    @property
    def subjectiveverbs_tfidf(self):
        incsvpath = os.path.join(self.inmatrixfolder, "contenttermTFIDF.csv")
        outcsvpath = os.path.join(self.outmatrixfolder, "content-subjverbsTFIDF.csv")
        words = keywordhandler.get_subjective_verbs()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words)
    
    @property
    def subjectiveverbs_count(self):
        incsvpath = os.path.join(self.inmatrixfolder, "contenttermCOUNT.csv")
        outcsvpath = os.path.join(self.outmatrixfolder, "content-subjverbsCOUNT.csv")
        words = keywordhandler.get_subjective_verbs()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words)
    
    @property
    def subjectiveverbs_presence(self):
        incsvpath = os.path.join(self.inmatrixfolder, "contenttermCOUNT.csv")
        outcsvpath = os.path.join(self.outmatrixfolder, "content-subjverbsBINARY.csv")
        words = keywordhandler.get_subjective_verbs()
        maindf = IOtools.readcsv(incsvpath, keepindex=True)
        mainmatrix = maindf.values
        np.place(mainmatrix, mainmatrix > 0, 1)  # map counts to presence values (1 if count > 0 else 0) 
        presencedf = pd.DataFrame(mainmatrix, index=maindf.index.values.tolist(), columns=maindf.columns.values.tolist())
        filtereddf = matrixhelpers.search_words_in_df(presencedf, words)
        IOtools.tocsv(filtereddf, outcsvpath, keepindex=True)
    
    @property
    def abstractwords_tfidf(self):
        incsvpath = os.path.join(self.inmatrixfolder, "contenttermTFIDF.csv")
        outcsvpath = os.path.join(self.outmatrixfolder, "content-abswordsTFIDF.csv")
        words = keywordhandler.get_abstractwords()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words)
    
    @property
    def abstractwords_count(self):
        incsvpath = os.path.join(self.inmatrixfolder, "contenttermCOUNT.csv")
        outcsvpath = os.path.join(self.outmatrixfolder, "content-abswordsCOUNT.csv")
        words = keywordhandler.get_abstractwords()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words)
    
    @property
    def abstractwords_presence(self):
        incsvpath = os.path.join(self.inmatrixfolder, "contenttermCOUNT.csv")
        outcsvpath = os.path.join(self.outmatrixfolder, "content-abswordsBINARY.csv")
        words = keywordhandler.get_abstractwords()
        maindf = IOtools.readcsv(incsvpath, keepindex=True)
        mainmatrix = maindf.values
        np.place(mainmatrix, mainmatrix > 0, 1)  # map counts to presence values (1 if count > 0 else 0) 
        presencedf = pd.DataFrame(mainmatrix, index=maindf.index.values.tolist(), columns=maindf.columns.values.tolist())
        filtereddf = matrixhelpers.search_words_in_df(presencedf, words)
        IOtools.tocsv(filtereddf, outcsvpath, keepindex=True)


    @property
    def title_exclamation(self):
        csvpath = os.path.join(self.inmatrixfolder, "title-exclamation.csv")
        return IOtools.readcsv(csvpath, keepindex=True)


    @property
    def title_subjectiveverbs_tfidf(self):
        incsvpath = os.path.join(self.inmatrixfolder, "titletermTFIDF.csv")
        outcsvpath = os.path.join(self.outmatrixfolder, "title-subjverbsTFIDF.csv")
        words = keywordhandler.get_subjective_verbs()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words)
    
    @property
    def title_subjectiveverbs_count(self):
        incsvpath = os.path.join(self.inmatrixfolder, "titletermCOUNT.csv")
        outcsvpath = os.path.join(self.outmatrixfolder, "title-subjverbsCOUNT.csv")
        words = keywordhandler.get_subjective_verbs()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words)
    
    @property
    def title_subjectiveverbs_presence(self):
        incsvpath = os.path.join(self.inmatrixfolder, "titletermCOUNT.csv")
        outcsvpath = os.path.join(self.outmatrixfolder, "title-subjverbsBINARY.csv")
        words = keywordhandler.get_subjective_verbs()
        maindf = IOtools.readcsv(incsvpath, keepindex=True)
        mainmatrix = maindf.values
        np.place(mainmatrix, mainmatrix > 0, 1)  # map counts to presence values (1 if count > 0 else 0) 
        presencedf = pd.DataFrame(mainmatrix, index=maindf.index.values.tolist(), columns=maindf.columns.values.tolist())
        filtereddf = matrixhelpers.search_words_in_df(presencedf, words)
        IOtools.tocsv(filtereddf, outcsvpath, keepindex=True)
    
    @property
    def title_abstractwords_tfidf(self):
        incsvpath = os.path.join(self.inmatrixfolder, "titletermTFIDF.csv")
        outcsvpath = os.path.join(self.outmatrixfolder, "title-abswordsTFIDF.csv")
        words = keywordhandler.get_abstractwords()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words)
    
    @property
    def title_abstractwords_count(self):
        incsvpath = os.path.join(self.inmatrixfolder, "titletermCOUNT.csv")
        outcsvpath = os.path.join(self.outmatrixfolder, "title-abswordsCOUNT.csv")
        words = keywordhandler.get_abstractwords()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words)
    
    @property
    def title_abstractwords_presence(self):
        incsvpath = os.path.join(self.inmatrixfolder, "titletermCOUNT.csv")
        outcsvpath = os.path.join(self.outmatrixfolder, "title-abswordsBINARY.csv")
        words = keywordhandler.get_abstractwords()
        maindf = IOtools.readcsv(incsvpath, keepindex=True)
        mainmatrix = maindf.values
        np.place(mainmatrix, mainmatrix > 0, 1)  # map counts to presence values (1 if count > 0 else 0) 
        presencedf = pd.DataFrame(mainmatrix, index=maindf.index.values.tolist(), columns=maindf.columns.values.tolist())
        filtereddf = matrixhelpers.search_words_in_df(presencedf, words)
        IOtools.tocsv(filtereddf, outcsvpath, keepindex=True)


class FeatureExtraction:
    features = {}

    def __init__(self, inmatrixfolder, outmatrixfolder):
        
        extractor = FeatureExtractors()   # take matrix or path2matrices
        self.features = {"advratio" : extractor.adverb_ratio,
                "adjratio" : extractor.adjective_ratio
                }
    

    
        

def Experiment(object):
    ename = ""
    datamatrixfolder = ""
    datamatrixname = ""
    
    features_additional = {"content-tfidf" : "pathtotfidf",
                           "abscount" : ""}
    

    def __init__(self):
        self.ename = ""






if __name__ == "__main__":
    
    inmatrix = "/home/dicle/Dicle/Tez/corpusstats/learning/data/random-single-N5/matrix/"
    outmatrix = "/home/dicle/Dicle/Tez/corpusstats/learning/experiments/test1/"
    
    fext = FeatureExtraction(inmatrix, outmatrix)
    
    for k,v in fext.features.iteritems():
        print "k: ",k
        print "v: ",v
    
    
    
    
