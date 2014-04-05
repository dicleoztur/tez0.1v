'''
Created on Feb 12, 2014

@author: dicle
'''


import pandas as pd
import os
import numpy as np

import matrixhelpers, utils
from sentimentfinding import IOtools, keywordhandler
from txtprocessor import listutils
from corpus import metacorpus

    


class FeatureExtractor:
    fname = ""
    inmatrixfolder = ""
    extendedfeaturesfolder = ""
    iscalculated = False
    
    def __init__(self, inmatrix, infilename, outmatrix, featurename):
        self.fname = featurename
        self.inmatrixfolder = inmatrix
        self.extendedfeaturesfolder = outmatrix
        self.iscalculated = False
        
        self.recordpath = os.path.join(self.extendedfeaturesfolder, self.fname+".csv")
        self.inputpath = os.path.join(self.inmatrixfolder, infilename)
    

    def calculate_features(self):
        self.iscalculated = True
        return

    def extract_features(self):
        if not self.iscalculated:
            self.calculate_features()
        return self.recordpath
        '''
        if self.iscalculated:
            return IOtools.readcsv(self.recordpath, keepindex=True)
        else:
            return self.calculate_features()
        '''
    
    @property
    def getfeaturematrixpath(self):
        return self.recordpath

    def __str__(self):
        return self.fname


class content_adverbratio(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath=self.inputpath
        outcsvpath=self.recordpath
        print "calculating"
        numeratortags = ['ADV']
        denominatortags = ['ADJ', 'Verb']
        rationame = self.fname
        matrixhelpers.get_featuretags_ratio(incsvpath, outcsvpath, numeratortags, denominatortags, rationame)
        self.iscalculated = True

  
       
class content_adjectiveratio(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath=self.inputpath
        outcsvpath=self.recordpath
        
        numeratortags = ['ADJ']
        denominatortags = ['Noun']
        rationame = self.fname
        matrixhelpers.get_featuretags_ratio(incsvpath, outcsvpath, numeratortags, denominatortags, rationame) 
        self.iscalculated = True  


class content_adjectivecount(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
      
        postagdf = IOtools.readcsv(incsvpath, keepindex=True)
        adj_count_vect = postagdf.loc[:, "ADJ"].values
        countdf = pd.DataFrame(adj_count_vect, index=postagdf.index.values.tolist(), columns=[self.fname])
        IOtools.tocsv(countdf, outcsvpath, keepindex=True)   
    


class content_adverbcount(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        
        postagdf = IOtools.readcsv(incsvpath, keepindex=True)
        adv_count_vect = postagdf.loc[:, "ADV"].values
        countdf = pd.DataFrame(adv_count_vect, index=postagdf.index.values.tolist(), columns=[self.fname])
        IOtools.tocsv(countdf, outcsvpath, keepindex=True)
        
    
    
class title_adverbratio(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        
        numeratortags = ['ADV']
        denominatortags = ['ADJ', 'Verb']
        matrixhelpers.get_featuretags_ratio(incsvpath, outcsvpath, numeratortags, denominatortags, rationame=self.fname)
        
    

class title_adjectiveratio(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        
        numeratortags = ['ADJ']
        denominatortags = ['Noun']
        matrixhelpers.get_featuretags_ratio(incsvpath, outcsvpath, numeratortags, denominatortags, rationame=self.fname)        


class title_adjectivecount(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
      
        postagdf = IOtools.readcsv(incsvpath, keepindex=True)
        adj_count_vect = postagdf.loc[:, "ADJ"].values
        countdf = pd.DataFrame(adj_count_vect, index=postagdf.index.values.tolist(), columns=[self.fname])
        IOtools.tocsv(countdf, outcsvpath, keepindex=True)   
    


class title_adverbcount(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        
        postagdf = IOtools.readcsv(incsvpath, keepindex=True)
        try:
            adv_count_vect = postagdf.loc[:, "ADV"].values
        except:
            adv_count_vect = np.zeros(postagdf.shape[0])
        countdf = pd.DataFrame(adv_count_vect, index=postagdf.index.values.tolist(), columns=[self.fname])
        IOtools.tocsv(countdf, outcsvpath, keepindex=True)
        
        

class content_subjectiveverbs_tfidf(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        words = keywordhandler.get_subjective_verbs()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words, column_appendix="c")
    


class content_subjectiveverbs_count(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)
    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        words = keywordhandler.get_subjective_verbs()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words, column_appendix="c")
    
    

class content_subjectiveverbs_presence(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        words = keywordhandler.get_subjective_verbs()
        maindf = IOtools.readcsv(incsvpath, keepindex=True)
        mainmatrix = maindf.values
        np.place(mainmatrix, mainmatrix > 0, 1)  # map counts to presence values (1 if count > 0 else 0) 
        presencedf = pd.DataFrame(mainmatrix, index=maindf.index.values.tolist(), columns=maindf.columns.values.tolist())
        filtereddf = matrixhelpers.search_words_in_df(presencedf, words)
        filtereddf = matrixhelpers.column_name_appendixing(filtereddf, appendix="c")
        IOtools.tocsv(filtereddf, outcsvpath, keepindex=True)
    

class content_abstractwords_tfidf(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        words = keywordhandler.get_abstractwords()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words, column_appendix="c")
    

class content_abstractwords_count(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        words = keywordhandler.get_abstractwords()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words, column_appendix="c")
    

class content_abstractwords_presence(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        words = keywordhandler.get_abstractwords()
        maindf = IOtools.readcsv(incsvpath, keepindex=True)
        mainmatrix = maindf.values
        np.place(mainmatrix, mainmatrix > 0, 1)  # map counts to presence values (1 if count > 0 else 0) 
        presencedf = pd.DataFrame(mainmatrix, index=maindf.index.values.tolist(), columns=maindf.columns.values.tolist())
        filtereddf = matrixhelpers.search_words_in_df(presencedf, words)
        filtereddf = matrixhelpers.column_name_appendixing(filtereddf, appendix="c")
        IOtools.tocsv(filtereddf, outcsvpath, keepindex=True)


class content_abstractnessratio(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        abswords = keywordhandler.get_abstractwords()
        matrixhelpers.get_featurewords_ratio(incsvpath, outcsvpath, words=abswords, rationame=self.fname)


class content_subjectivityratio(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        subjverbs = keywordhandler.get_subjective_verbs()
        matrixhelpers.get_featurewords_ratio(incsvpath, outcsvpath, words=subjverbs, rationame=self.fname)
        

class title_exclamation(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        df = IOtools.readcsv(incsvpath, keepindex=True)
        
        # drop duplicates
        df["index"] = df.index
        df = df.drop_duplicates(cols='index', take_last=False)
        del df["index"]
                
        df.index.name = ""
        df.rename(columns={"hasexclamation" : self.fname}, inplace=True)
        df.sort_index(inplace=True)
        IOtools.tocsv(df, outcsvpath, keepindex=True) 



class title_subjectiveverbs_tfidf(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        words = keywordhandler.get_subjective_verbs()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words, column_appendix="t")
    

class title_subjectiveverbs_count(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        words = keywordhandler.get_subjective_verbs()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words, column_appendix="t")
    

class title_subjectiveverbs_presence(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        words = keywordhandler.get_subjective_verbs()
        maindf = IOtools.readcsv(incsvpath, keepindex=True)
        mainmatrix = maindf.values
        np.place(mainmatrix, mainmatrix > 0, 1)  # map counts to presence values (1 if count > 0 else 0) 
        presencedf = pd.DataFrame(mainmatrix, index=maindf.index.values.tolist(), columns=maindf.columns.values.tolist())
        filtereddf = matrixhelpers.search_words_in_df(presencedf, words)
        filtereddf = matrixhelpers.column_name_appendixing(filtereddf, appendix="t")
        IOtools.tocsv(filtereddf, outcsvpath, keepindex=True)


class title_abstractwords_tfidf(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        words = keywordhandler.get_abstractwords()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words, column_appendix="t")
    

class title_abstractwords_count(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        words = keywordhandler.get_abstractwords()
        matrixhelpers.get_featureword_doc_matrix(incsvpath, outcsvpath, words, column_appendix="t")
    

class title_abstractwords_presence(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        words = keywordhandler.get_abstractwords()
        maindf = IOtools.readcsv(incsvpath, keepindex=True)
        mainmatrix = maindf.values
        np.place(mainmatrix, mainmatrix > 0, 1)  # map counts to presence values (1 if count > 0 else 0) 
        presencedf = pd.DataFrame(mainmatrix, index=maindf.index.values.tolist(), columns=maindf.columns.values.tolist())
        filtereddf = matrixhelpers.search_words_in_df(presencedf, words)
        
        # change columns names (words) to indicate they belong to the title
        filtereddf = matrixhelpers.column_name_appendixing(filtereddf, appendix="t")
        IOtools.tocsv(filtereddf, outcsvpath, keepindex=True)


class title_abstractnessratio(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        abswords = keywordhandler.get_abstractwords()
        matrixhelpers.get_featurewords_ratio(incsvpath, outcsvpath, words=abswords, rationame=self.fname)


class title_subjectivityratio(FeatureExtractor):
    
    def __init__(self, inmatrix, infilename, outmatrix):
        featurename = self.__class__.__name__
        FeatureExtractor.__init__(self, inmatrix, infilename, outmatrix, featurename)

    
    def calculate_features(self):
        incsvpath = self.inputpath
        outcsvpath = self.recordpath
        subjverbs = keywordhandler.get_subjective_verbs()
        matrixhelpers.get_featurewords_ratio(incsvpath, outcsvpath, words=subjverbs, rationame=self.fname)
        

class FeaturesPlexer:
    features = []
    inmatrixfolder = ""
    recordfolder = ""
    extendedfeaturesfolder = ""
    #combinedfeaturesfolder = ""

    def __init__(self, inmatrixfolder, outmatrixfolder):
        self.inmatrixfolder = inmatrixfolder
        self.recordfolder = outmatrixfolder
        self.extendedfeaturesfolder = outmatrixfolder
        #self.extendedfeaturesfolder = IOtools.ensure_dir(os.path.join(self.recordfolder, "extendedfeatures"))
        #self.combinedfeaturesfolder = IOtools.ensure_dir(os.path.join(self.recordfolder, "finaldatasets"))
        
        
        # create featureextractor objects
        cadvratio = content_adverbratio(inmatrix=self.inmatrixfolder, infilename="content-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        cadjratio = content_adjectiveratio(inmatrix=self.inmatrixfolder, infilename="content-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        cadjcount = content_adjectivecount(inmatrix=self.inmatrixfolder, infilename="content-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        cadvcount = content_adverbcount(inmatrix=self.inmatrixfolder, infilename="content-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        tadjcount = title_adjectivecount(inmatrix=self.inmatrixfolder, infilename="title-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        tadvcount = title_adverbcount(inmatrix=self.inmatrixfolder, infilename="title-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        tadvratio = title_adverbratio(inmatrix=self.inmatrixfolder, infilename="title-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        tadjratio = title_adjectiveratio(inmatrix=self.inmatrixfolder, infilename="title-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        csubjtfidf = content_subjectiveverbs_tfidf(inmatrix=self.inmatrixfolder, infilename="contenttermTFIDF.csv", outmatrix=self.extendedfeaturesfolder)
        csubjcount = content_subjectiveverbs_count(inmatrix=self.inmatrixfolder, infilename="contenttermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        csubjbinary = content_subjectiveverbs_presence(inmatrix=self.inmatrixfolder, infilename="contenttermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        cabstfidf = content_abstractwords_tfidf(inmatrix=self.inmatrixfolder, infilename="contenttermTFIDF.csv", outmatrix=self.extendedfeaturesfolder)
        cabscount = content_abstractwords_count(inmatrix=self.inmatrixfolder, infilename="contenttermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        cabsbinary = content_abstractwords_presence(inmatrix=self.inmatrixfolder, infilename="contenttermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        cabstractrat = content_abstractnessratio(inmatrix=self.inmatrixfolder, infilename="contenttermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        csubjrat = content_subjectivityratio(inmatrix=self.inmatrixfolder, infilename="contenttermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        texcl = title_exclamation(inmatrix=self.inmatrixfolder, infilename="title-exclamation.csv", outmatrix=self.extendedfeaturesfolder)
        
        tsubjtfidf = title_subjectiveverbs_tfidf(inmatrix=self.inmatrixfolder, infilename="titletermTFIDF.csv", outmatrix=self.extendedfeaturesfolder)
        tsubjcount = title_subjectiveverbs_count(inmatrix=self.inmatrixfolder, infilename="titletermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        tsubjbinary = title_subjectiveverbs_presence(inmatrix=self.inmatrixfolder, infilename="titletermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        tabstfidf = title_abstractwords_tfidf(inmatrix=self.inmatrixfolder, infilename="titletermTFIDF.csv", outmatrix=self.extendedfeaturesfolder)
        tabscount = title_abstractwords_count(inmatrix=self.inmatrixfolder, infilename="titletermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        tabsbinary = title_abstractwords_presence(inmatrix=self.inmatrixfolder, infilename="titletermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        tabsrat = title_abstractnessratio(inmatrix=self.inmatrixfolder, infilename="titletermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        tsubjrat = title_subjectivityratio(inmatrix=self.inmatrixfolder, infilename="titletermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        
        self.features = [cadvratio, cadjratio, cadjcount, cadvcount, tadjcount, tadvcount, tadvratio, 
                         tadjratio, csubjtfidf, csubjcount, csubjbinary, cabstfidf, cabscount, cabsbinary,
                         cabstractrat, csubjrat, texcl, tsubjtfidf, tsubjcount, tsubjbinary, tabstfidf, 
                         tabscount, tabsbinary, tabsrat, tsubjrat]
        
        '''
        cadvratio = content_adverbratio(inmatrix=self.inmatrixfolder, 
                                        infilename="content-postagCOUNT.csv",
                                        outmatrix=self.extendedfeaturesfolder)
                                        
        cadjratio = content_adjectiveratio(inmatrix=self.inmatrixfolder, 
                                        infilename="content-postagCOUNT.csv",
                                        outmatrix=self.extendedfeaturesfolder)
                                        
        '''
        
        self.features.append(cadvratio)
        self.features.append(cadjratio)
        
        
    
    def process_features(self):
        
        for feat in self.features:
            print feat.fname," : ",feat.extract_features()  
        
          
        
        '''
        extractor = FeatureExtractor()   # take matrix or path2matrices
        self.features = {"advratio" : extractor.adverb_ratio,
                "adjratio" : extractor.adjective_ratio
                }
        '''



class FeatureCombiner:
    
    featuremap = {}
    extendedfeaturesfolder = ""
    combinedfeaturesfolder = ""
    
    def __init__(self, extendedfeatsfolder, combinedfeatsfolder):
        self.inmatrixfolder = ""   # we can later attribute it to an abstract outer class
        self.extendedfeaturesfolder = extendedfeatsfolder   # input
        self.combinedfeaturesfolder = combinedfeatsfolder   # output
        
        self.featuremap = {}
        self.get_feature_map()
        
        for k in self.featuremap.keys()[:3]:
            print k,
            featureinstance = self.featuremap[k][0]  #.getfeaturematrixpath()
            print type(featureinstance)
            print featureinstance.getfeaturematrixpath
        
        self.numoffeatures = sum([len(v) for _,v in self.featuremap.iteritems()])  
        self.numofgroups = len(self.featuremap.keys())



    def get_feature_map(self):
        # create featureextractor objects
        cadvratio = content_adverbratio(inmatrix=self.inmatrixfolder, infilename="content-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        cadjratio = content_adjectiveratio(inmatrix=self.inmatrixfolder, infilename="content-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        cadjcount = content_adjectivecount(inmatrix=self.inmatrixfolder, infilename="content-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        cadvcount = content_adverbcount(inmatrix=self.inmatrixfolder, infilename="content-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        tadjcount = title_adjectivecount(inmatrix=self.inmatrixfolder, infilename="title-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        tadvcount = title_adverbcount(inmatrix=self.inmatrixfolder, infilename="title-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        tadvratio = title_adverbratio(inmatrix=self.inmatrixfolder, infilename="title-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        tadjratio = title_adjectiveratio(inmatrix=self.inmatrixfolder, infilename="title-postagCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        csubjtfidf = content_subjectiveverbs_tfidf(inmatrix=self.inmatrixfolder, infilename="contenttermTFIDF.csv", outmatrix=self.extendedfeaturesfolder)
        csubjcount = content_subjectiveverbs_count(inmatrix=self.inmatrixfolder, infilename="contenttermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        csubjbinary = content_subjectiveverbs_presence(inmatrix=self.inmatrixfolder, infilename="contenttermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        cabstfidf = content_abstractwords_tfidf(inmatrix=self.inmatrixfolder, infilename="contenttermTFIDF.csv", outmatrix=self.extendedfeaturesfolder)
        cabscount = content_abstractwords_count(inmatrix=self.inmatrixfolder, infilename="contenttermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        cabsbinary = content_abstractwords_presence(inmatrix=self.inmatrixfolder, infilename="contenttermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        cabsrat = content_abstractnessratio(inmatrix=self.inmatrixfolder, infilename="contenttermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        csubjrat = content_subjectivityratio(inmatrix=self.inmatrixfolder, infilename="contenttermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        texcl = title_exclamation(inmatrix=self.inmatrixfolder, infilename="title-exclamation.csv", outmatrix=self.extendedfeaturesfolder)
        
        tsubjtfidf = title_subjectiveverbs_tfidf(inmatrix=self.inmatrixfolder, infilename="titletermTFIDF.csv", outmatrix=self.extendedfeaturesfolder)
        tsubjcount = title_subjectiveverbs_count(inmatrix=self.inmatrixfolder, infilename="titletermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        tsubjbinary = title_subjectiveverbs_presence(inmatrix=self.inmatrixfolder, infilename="titletermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        tabstfidf = title_abstractwords_tfidf(inmatrix=self.inmatrixfolder, infilename="titletermTFIDF.csv", outmatrix=self.extendedfeaturesfolder)
        tabscount = title_abstractwords_count(inmatrix=self.inmatrixfolder, infilename="titletermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        tabsbinary = title_abstractwords_presence(inmatrix=self.inmatrixfolder, infilename="titletermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        tabsrat = title_abstractnessratio(inmatrix=self.inmatrixfolder, infilename="titletermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        tsubjrat = title_subjectivityratio(inmatrix=self.inmatrixfolder, infilename="titletermCOUNT.csv", outmatrix=self.extendedfeaturesfolder)
        
        # of course we could have gotten file path through storing extendedfolder+classname yet it is not flexible. we might need the instances above later if we first extend then combine for example.
        
        
        # keys are feature groups, corresponding values are featureextract instances
        self.featuremap["cadj"] = [cadjratio, cadjcount]
        self.featuremap["tadj"] = [tadjratio, tadjcount]
        
        self.featuremap["cadv"] = [cadvratio, cadvcount]
        self.featuremap["tadv"] = [tadvratio, tadvcount]   
        
        self.featuremap["cabs"] = [cabstfidf, cabscount, cabsbinary, cabsrat]     
        self.featuremap["tabs"] = [tabstfidf, tabscount, tabsbinary, tabsrat] 
        
        self.featuremap["csubj"] = [csubjtfidf, csubjcount, csubjbinary, csubjrat]
        self.featuremap["tsubj"] = [tsubjtfidf, tsubjcount, tsubjbinary, tsubjrat]
        
        self.featuremap["texcl"] = [texcl]
        
        '''  needs a sort func in classes !
        for k in self.featuremap.keys():
            self.featuremap[k].sort()
        '''
        
    
    
    def exclude_none(self):
        featurecombsmatrix = listutils.get_combination_matrix(self.featuremap)
        '''
        listutils.print_features(self.featuremap, featurecombsmatrix)
        print featurecombsmatrix[:10, :]
        
        mpath = "/home/dicle/Dicle/Tez/corpusstats/learning/data/random-single-N5/combs/comb_all.m"
        np.savetxt(mpath, featurecombsmatrix, fmt='%d', delimiter='\t')
        '''
        #return featurecombsmatrix
        
        self.combine_features(featurecombsmatrix)
    
    
    def exclude_n_features(self, n): 
        return
    
    def exclude_n_groups(self, n):
        return   


    
    # outputs final dataset matrices to the inner 'finaldatasets' folder, to be later read by learning models
    # combmatrix is an np array, possibly read from the 'combs' folder.
    def combine_features(self, combmatrix):
        ncombs, nrows = combmatrix.shape
        
        for i,row in enumerate(combmatrix):
            filename = "comb"+str(i)+"_F"
            featuredflist = []
            for j,featno in enumerate(row):
                groupname = sorted(self.featuremap.keys())[j]
                filename += "_"+str(j)+"-"+str(featno)   # filename = combNO_F_GROUPNO-FEATNO
                                
                extractorinstance = self.featuremap[groupname][featno]
                featurematrixpath = extractorinstance.getfeaturematrixpath
                featurematrix = IOtools.readcsv(featurematrixpath, keepindex=True)
                featuredflist.append(featurematrix)
            
            print filename
            print utils.decode_combcode(filename, self.featuremap)
            datamatrix = pd.concat(featuredflist, axis=1) #, verify_integrity=True) # CLOSED DUE TO THE OVERLAPPING WORDS IN ABS AND SUBJ LISTS
            #datamatrix['index'] = datamatrix.index
            #datamatrix = datamatrix.drop_duplicates(cols='index')
            #del datamatrix['index']
            
            # replace nan and inf cells !! no. work on matrix, not df. better do this change on learning
            #datamatrix[np.isnan(datamatrix)] = 0
            #datamatrix[np.isinf(datamatrix)] = 0
            
            datamatrixpath = self.combinedfeaturesfolder + os.sep + filename + ".csv"
            IOtools.tocsv(datamatrix, datamatrixpath, keepindex=True)
            
    
    def featuremapping_to_datamatrix(self):
        filename = ""
        return               





def get_featurecombinatorial_datasets(datasetpath):
    
    rawdatafolder = os.path.join(datasetpath, "rawfeatures")
    extendeddatafolder = IOtools.ensure_dir(os.path.join(datasetpath, "extendedfeatures"))
    finaldatasetfolder = IOtools.ensure_dir(os.path.join(datasetpath, "finaldatasets"))
    
    feature_extending = FeaturesPlexer(rawdatafolder, extendeddatafolder)
    feature_extending.process_features()
    
    feature_combiner = FeatureCombiner(extendeddatafolder, finaldatasetfolder)
    feature_combiner.exclude_none()
    


if __name__ == "__main__":
  
    datarootpath = metacorpus.learningdatapath
    annotationtype = "double"   # to be a list
    tagger = "user"
    setsize = 150
    datasetpath = os.path.join(datarootpath, annotationtype, str(setsize))
    get_featurecombinatorial_datasets(datasetpath)
    
    
    
    '''
    fulldatasetpath = "/home/dicle/Dicle/Tez/corpusstats/learning/data/random-single-N5/"
    inmatrix = "/home/dicle/Dicle/Tez/corpusstats/learning/data/random-single-N5/rawfeatures/"
    outmatrix = "/home/dicle/Dicle/Tez/corpusstats/learning/data/random-single-N5/extendedfeatures/"
    #outmatrix = "/home/dicle/Dicle/Tez/corpusstats/learning/experiments/test2/"
    
    
    finaldatasetsfolder = "/home/dicle/Dicle/Tez/corpusstats/learning/data/random-single-N5/finaldatasets_test/"
    
    
    fext = FeaturesPlexer(inmatrix, outmatrix)
    
    fext.process_features()
    
    
    
    fcombiner = FeatureCombiner(extendedfeatsfolder=outmatrix, combinedfeatsfolder=finaldatasetsfolder)
    fcombiner.exclude_none()
    combmatrix = np.loadtxt("/home/dicle/Dicle/Tez/corpusstats/learning/data/random-single-N5/combs/combtest.m", dtype=int, delimiter='\t')
    fcombiner.combine_features(combmatrix)
    '''
    
    

    
    
    
    
