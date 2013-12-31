# -*- coding: utf-8 -*-

'''
Created on Jul 5, 2013

@author: dicle
'''



import os
import nltk
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from txtprocessor import texter
from sentimentfinding import IOtools
from sentimentfinding import keywordhandler
from sentimentfinding import CFDhelpers
from sentimentfinding import plotter
from languagetools import SAKsParser
from classification import numericutils, termanalysis
import testlda, testpca

#from stats import classification



class Corpus:
    
    corpusname = ""  
    rootpath = ""
    inputpath = ""
    
    doclabeldct = {}
    
    cfd_DocRoot = nltk.ConditionalFreqDist()
    cfd_RootDoc = nltk.ConditionalFreqDist()
    
    cfd_DateRoot = nltk.ConditionalFreqDist()
    cfd_DatePOStag = nltk.ConditionalFreqDist()
    
    cfd_DocPOStag = nltk.ConditionalFreqDist()
    
    cfd_DocSubjectivity = nltk.ConditionalFreqDist()
    cfd_SubjectivityDoc = nltk.ConditionalFreqDist()
    
    def __init__(self, cname="", recordpath=""):
        
        self.corpusname = cname
        self.rootpath = recordpath
        
        self.doclabeldct = {}
        
        self.cfd_DocRoot = nltk.ConditionalFreqDist()
        self.cfd_RootDoc = nltk.ConditionalFreqDist()
        
        self.cfd_DateRoot = nltk.ConditionalFreqDist()
        self.cfd_DatePOStag = nltk.ConditionalFreqDist()
        
        self.cfd_DocPOStag = nltk.ConditionalFreqDist()
        
        self.cfd_DocSubjectivity = nltk.ConditionalFreqDist()
        self.cfd_SubjectivityDoc = nltk.ConditionalFreqDist()
    
    
    def setname(self, cname):
        self.corpusname = cname
        
    def read_corpus(self, inputpath, recordpath, labelwisefileidlist, plaintext=True, nostopwords=True):
        dateroots = []
        datePOStag = []
        docPOStag = []
        docroots = []
        
        self.rootpath = recordpath 
        self.inputpath = inputpath
        
        if not plaintext:
            for label, fileids in labelwisefileidlist.iteritems():
                for fileid in fileids:
                    
                    docid = fileid
                    
                    path = self.inputtpath + os.sep + label + os.sep + fileid
                    words, date = texter.getnewsitem(path, nostopwords)
                    
                    lemmata = SAKsParser.lemmatize_lexicon(words)
                    for (_, literalPOS, root, _) in lemmata:
                        dateroots.append((date, root))
                        datePOStag.append((date, literalPOS))
                        docroots.append((docid, root))
                        docPOStag.append((docid, literalPOS))
                    self.doclabeldct[docid] = label
                    
            self.cfd_DatePOStag = nltk.ConditionalFreqDist(datePOStag)
            self.cfd_DateWords = nltk.ConditionalFreqDist(dateroots)
            self.cfd_DocPOStag = nltk.ConditionalFreqDist(docPOStag)
            self.cfd_DocRoot = nltk.ConditionalFreqDist(docroots)  
        
        else:
            for label, fileids in labelwisefileidlist.iteritems():
                for fileid in fileids:
                    
                    docid = fileid
                    
                    path = self.inputtpath + os.sep + label + os.sep + fileid
                    
                    words = texter.read_article(path, nostopwords)
                    
                    lemmata = SAKsParser.lemmatize_lexicon(words)
                    for (_, literalPOS, root, _) in lemmata:
                        docroots.append((docid, root))
                        docPOStag.append((docid, literalPOS)) 
                    self.doclabeldct[docid] = label  
            self.cfd_DocPOStag = nltk.ConditionalFreqDist(docPOStag)
            self.cfd_DocRoot = nltk.ConditionalFreqDist(docroots) 
            
        self.cfd_RootDoc = nltk.ConditionalFreqDist((word, fileid) for fileid in self.cfd_DocRoot.conditions()
                                              for word in list(self.cfd_DocRoot[fileid]))   
    
    
    # files, named fileids, contain roots.
    def read_wordlists(self, inputpath, recordpath, labelwisefileidlist):
        self.rootpath = recordpath        
        self.inputpath = inputpath
        
        docroots = []
        
        for label, fileids in labelwisefileidlist.iteritems():
                for fileid in fileids:
                    docid = fileid
                    path = self.inputpath + os.sep + label + os.sep + fileid
                    words = IOtools.readtextlines(path)
                    
                    words = texter.remove_endmarker(words, "(i)")
                    words = texter.remove_endmarker(words, "(ii)")
                    
                    for root in words:
                        docroots.append((docid, root))
        self.cfd_DocRoot = nltk.ConditionalFreqDist(docroots)
        self.cfd_RootDoc = nltk.ConditionalFreqDist((word, fileid) for fileid in self.cfd_DocRoot.conditions()
                                              for word in list(self.cfd_DocRoot[fileid]))
                    
    def get_docterm_matrix(self):
        numofwords = len(self.cfd_RootDoc.conditions())
        numofdocs = len(self.cfd_DocRoot.conditions())
        
        matrix = np.empty((numofdocs, numofwords))
        
        for i,doc in enumerate(self.cfd_DocRoot.conditions()):
            for j,term in enumerate(self.cfd_RootDoc.conditions()):
                matrix[i,j] = self.cfd_DocRoot[doc][term]
        
        doctermframe = pd.DataFrame(matrix, index = self.cfd_DocRoot.conditions(), columns=self.cfd_RootDoc.conditions())
        doctermframe.to_csv(self.rootpath+os.sep+"matrix"+"doctermFREQ.csv") 
        return doctermframe
        
        
    def compute_tfidf(self):
        
        numofwords = len(self.cfd_RootDoc.conditions())
        numofdocs = len(self.cfd_DocRoot.conditions())
        
        matrix = np.empty((numofdocs, numofwords))
        
        for i,doc in enumerate(self.cfd_DocRoot.conditions()):
            for j,term in enumerate(self.cfd_RootDoc.conditions()):
                tf = self.cfd_DocRoot[doc][term]
                idf = math.log(float(numofdocs) / self.cfd_RootDoc[term].N()) 
                matrix[i,j] = tf * idf
                
                # bu cok ciddi bir problem. df is not the total occurrence of a term throughout docs 
                # but it is the total number of docs a term occurred in. corrected above. just realized 24Jul
                #
                # explained superbly in, http://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html
                
                # FreqDist N() fonksiyonu kontrol ettim, sorun yok. item in bulundugu gerceklestigi olay sayisini veriyor,
                # gerceklesme sayisini degil.
        
        matrix = np.around(matrix, decimals=4)
        
        doctermframe = pd.DataFrame(matrix, index = self.cfd_DocRoot.conditions(), columns=self.cfd_RootDoc.conditions()) 
        doctermframe.to_csv(self.rootpath+os.sep+"matrix"+"doctermTFIDF.csv")
        return doctermframe
    
    
    def compute_tfidf2(self, doctermfreq):
        numofdocs, numofwords = doctermfreq.shape
        docs = doctermfreq.index.values.tolist()
        terms = doctermfreq.columns.values.tolist()
        
        matrix = np.empty((numofdocs, numofwords))
        
        for i,doc in enumerate(docs):
            for j,term in enumerate(terms):
                tf = doctermfreq.iloc[i,j]
                df = np.count_nonzero(doctermfreq.iloc[:, j])
                
                idf = math.log(float(numofdocs) / df)
                matrix[i,j] = tf * idf
        
        matrix = np.around(matrix, decimals=4)
        
        doctermframe = pd.DataFrame(matrix, index = docs, columns=terms) 
        doctermframe.to_csv(self.rootpath+os.sep+"matrix"+"doctermTFIDF.csv")
        return doctermframe
    
       
    
    # docs on the x axis, feature values on y axis.
    def plot_features(self):
        figname = "features_" + self.filename
        xLabel = "docs"
        yLabel = "feature values"
        imgoutpath = IOtools.img_output + os.sep + figname + ".png"
        #CFDhelpers.plotcfd_oneline(self.cfd_DocPOStag, figname, xLabel, yLabel, imgoutpath, featuresXaxis=True)        
        plotter.set_plotframe("Feature Values over Docs", xLabel, yLabel)

        #1 plot adj ratio
        self.plot_POStag_features(self.cfd_DocPOStag, ['ADJ'], ['Noun'], label="JJ Ratio", clr='m')
        
        #2 plot adv ratio
        self.plot_POStag_features(self.cfd_DocPOStag, ['ADV'], ['ADJ', 'Verb'], label="ADV Ratio", clr='g')
        
        #3 plot subjective word freq
        yitems = [self.cfd_DocSubjectivity[cond].N() / float(self.cfd_DocRoot[cond].N()) for cond in self.cfd_DocRoot.conditions()]
        plotter.plot_line(self.cfd_DocSubjectivity.conditions(), yitems, linelabel="subjectivity-count", clr="r")
        
        plt.legend()
        plt.savefig(imgoutpath, dpi=100)
        plt.clf()
    
    
    '''  count the roots in subjectivity lexicon '''
    def subjectivity_features(self):
        subjective_keywords = keywordhandler.get_subjectivity_lexicon()
        wordcount = []
        for fileid in self.cfd_DocRoot.conditions():
            for word in list(self.cfd_DocRoot[fileid]):
                if word in subjective_keywords:
                    for i in range(self.cfd_DocRoot[fileid][word]):
                        wordcount.append((fileid, word))
                        
        #self.cfd_DocSubjectivity = nltk.ConditionalFreqDist((fileid, word) for fileid in self.cfd_DocRoots.conditions() 
        #                                                     for word in list(self.cfd_DocRoots[fileid]) if word in subjective_keywords)
        
        self.cfd_DocSubjectivity = nltk.ConditionalFreqDist(wordcount)
        #CFDhelpers.printCFD(self.cfd_DocSubjectivity)
        
        #yitems = [float(val)/numofwords for fileid in self.cfd_DocSubjectivity.conditions() for ]   
        #yitems = [self.cfd_DocSubjectivity[cond].N() for cond in self.cfd_DocSubjectivity.conditions()]
        #plotter.plot_line(self.cfd_DocSubjectivity.conditions(), yitems, linelabel="subjectivity-count", clr="y")
        
    
    def POStag_features(self, cfd, tags1, tags2):
        docpostag_features = CFDhelpers.feature_ratio(cfd, tags1, tags2)   
        return docpostag_features 
    
    def record_all_cfd(self):
        CFDhelpers.recordCFD(self.cfd_DocSubjectivity, "SUBJ-"+self.filename)
        CFDhelpers.recordCFD(self.cfd_DatePOStag, "DatePOStag-"+self.filename)
        CFDhelpers.recordCFD(self.cfd_DateWords, "DateWords-"+self.filename)
        CFDhelpers.recordCFD(self.cfd_DocPOStag, "DocPOStag-"+self.filename)
        CFDhelpers.recordCFD(self.cfd_DocRoot, "DocRoots-"+self.filename)

    def plot_POStag_features(self, cfd, tags1, tags2, label, clr):
        docpostag_features = CFDhelpers.feature_ratio(cfd, tags1, tags2)
        yitems = [val for (_, val) in docpostag_features]
        plotter.plot_line(cfd.conditions(), yitems, label, clr)
        #plt.savefig(IOtools.img_output + os.sep + label.upper() + self.filename + ".png", dpi=100)


#######    corpusfeature class end  ##########


class matrixanalyser():
    '''  in all functions, df is of type dataframe in pandas, containing doc-term incidence matrix,
        additionally having docids as row labels and term literals as column labels.  
        the matrix is usually a tfidf matrix.
    '''
    
    recordpath = ""
    
     
    def __init__(self, path):
        self.recordpath = path
    
    def write_docterm_weight(self, df, rootpath):
        docs = df.index.values.tolist()
        words = df.columns.values.tolist()
        
        for doc in docs:
            docid_termpairs = []
            for word in words:
                val = df.loc[doc, word]
                if val > 0.0:
                    output = word + "\t" + str(val)
                    docid_termpairs.append(output)
            IOtools.todisc_list(rootpath+os.sep+doc, docid_termpairs)
    
    def compute_document_distance(self, df):
        docs = df.index.values.tolist()
        
        distdf = pd.DataFrame(index=docs, columns=docs)
        for doc1 in docs:
            for doc2 in docs:
                u = df.loc[doc1, :].values
                v = df.loc[doc2, :].values
                val = numericutils.euclidean_distance(u, v)
                distdf.loc[doc1, doc2] = round(val, 4)
        
        distdf.to_csv(self.recordpath+os.sep+"distancematrix.csv")
        return distdf
    
    def compute_document_similarity(self, df, filemarker):
        docs = df.index.values.tolist()
               
        simdf = pd.DataFrame(index=docs, columns=docs)
        for i,doc1 in enumerate(docs):
            for j,doc2 in enumerate(docs):
                u = df.loc[doc1, :].values
                v = df.loc[doc2, :].values
                #print i,"   ",j,"      ",type(u),"  ",type(v)
                val = numericutils.cosine_similarity(u, v)
                simdf.loc[doc1, doc2] = round(val, 4)
        simdf.to_csv(self.recordpath+os.sep+"similaritymatrix"+filemarker+".csv")
        return simdf
    
    def get_most_similar_pairs(self, simdf, filemarker):
        docs = simdf.index.values.tolist()
        similarpairs = []
        output = []
        for doc in docs:
            maxval, maxind = numericutils.get2ndmax(simdf.loc[doc].values)
            doc2 = simdf.columns.values[maxind]
            similarpairs.append((doc, doc2, maxval))
            output.append(doc+" # "+doc2+" : "+str(maxval))
        
        IOtools.todisc_list(self.recordpath+os.sep+"similarpairs"+filemarker+".txt", output)
        return similarpairs
    
    def lsa_transform(self, df, dimensions):
        tfidfmatrix = np.around(df.values, 8)     #df.values    
        reduced_tfidfmatrix = termanalysis.lsa_transform(tfidfmatrix, dimensions)
        #reduced_tfidfmatrix = np.around(reduced_tfidfmatrix, 4)
        
        lsadf = pd.DataFrame(reduced_tfidfmatrix, index=df.index.values, columns=df.columns.values)
        lsadf.to_csv(self.recordpath+os.sep+"doctermLSA"+str(dimensions)+".csv")
        
        return lsadf
    
    
    # reduces columns
    def lsa_transform2(self, df):
        tfidfmatrix = df.values
        #tfidfmatrix = tfidfmatrix.T   # due to the specifity of np.svd(): this svd tries to reduce rows
        matrix_summary(tfidfmatrix, "tfidf data matrix")
        
        u, sigma, vt = testlda.apply_svd(tfidfmatrix)
        matrix_summary(u, "U")
        matrix_summary(sigma, "sigma")
        matrix_summary(vt, "Vt")
        
               
        k = testlda.find_optimal_npc(sigma)
        reconstructed_tfidfmatrix = testlda.reconstruct(u, sigma, vt, k)
        
        lsadf = pd.DataFrame(reconstructed_tfidfmatrix, index=df.index.values, columns=df.columns.values[:k])
        lsadf.to_csv(self.recordpath+os.sep+"doctermLSA2"+str(k)+".csv")
        return reconstructed_tfidfmatrix, k
    
    
    def pca_transform(self, df):
        tfidfmatrix = df.values
        tfidfmatrix = tfidfmatrix.T
        
        projectedmatrix = testpca._pca(tfidfmatrix)
        
        del tfidfmatrix
        
        projectedmatrix = projectedmatrix.T
        print type(projectedmatrix[0,0])
        
        pcadf = pd.DataFrame(projectedmatrix, index=df.index.values.tolist(), columns=df.columns.values.tolist())
        pcadf.to_csv(self.recordpath+os.sep+"doctermPCA.csv")
        
        
        
    def get_topical_terms(self, lsadf, N):
        docs = lsadf.index.values.tolist()   
        
        doctopicdct = {} 
        for doc in docs:
            doctopicdct[doc] = [] 
            
        for doc in docs:
            termvector = lsadf.loc[doc, :].values
            termindexpairs = [(j, value) for j,value in enumerate(termvector.tolist())]
            termindexpairs.sort(key=lambda tup : tup[1], reverse=True)
        
            topicindices = [ind for ind, _ in termindexpairs]
            topicindices = topicindices[:N]
            topicdf = lsadf.loc[doc, topicindices]
            
            '''  temporarily close for testing
            p = IOtools.ensure_dir(self.recordpath+os.sep+"topics"+str(dim))
            IOtools.todisc_txt(topicdf.to_string(), p+os.sep+doc)
            '''
            
            terms = topicdf.index.values.tolist()
            doctopicdct[doc] = [(term, topicdf.loc[term]) for term in terms]
        
        return doctopicdct
        
    
    
        
        

    


def main():
    start = datetime.now()
    corpus = Corpus("test")
    rootpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus/"
    labels = ["pos","neg"]
    labelwisepathlist = {}
    
    for label in labels:
        labelwisepathlist[label] = []
    for label in labels:
        labelwisepathlist[label] = IOtools.getfilenames_of_dir(rootpath+os.sep+label, removeextension=False)
    
    corpus.read_corpus(rootpath, labelwisepathlist)
    end = datetime.now()
    
    print "Reading takes: ", str(end-start)
    
    print corpus.cfd_RootDoc["alevi"].N()
    print corpus.cfd_RootDoc.N()
    print len(corpus.cfd_RootDoc.conditions())
        
    print corpus.cfd_DocRoot.N()
    print len(corpus.cfd_DocRoot.conditions())

    df = corpus.compute_tfidf()
    
    end2= datetime.now()
    print "tfidf matrix takes: ",str(end2-end)
    
    return df, corpus



def corpus_construction_fromwords():
    recordpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/edit/wordletest/matrix/"
    inputpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/edit/wordletest/words/"
    labels = ["inlier", "outlier"]
    labelwisepathlist = {}
    
    for label in labels:
        labelwisepathlist[label] = []
    for label in labels:
        labelwisepathlist[label] = IOtools.getfilenames_of_dir(inputpath+os.sep+label, removeextension=False)
    
    corpus = Corpus("wordletest")
    corpus.read_wordlists(inputpath, recordpath, labelwisepathlist)
    doctermfreqdf = corpus.get_docterm_matrix()
    corpus.compute_tfidf2(doctermfreqdf)
    return corpus
    
    
    

def corpus_construction():
    start = datetime.now()
    corpus = Corpus("test")
    rootpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus/"
    labels = ["pos","neg"]
    labelwisepathlist = {}
    
    for label in labels:
        labelwisepathlist[label] = []
    for label in labels:
        labelwisepathlist[label] = IOtools.getfilenames_of_dir(rootpath+os.sep+label, removeextension=False)
    
    corpus.read_corpus(rootpath, labelwisepathlist)
    end = datetime.now()
    
    print "Reading takes: ", str(end-start)
    
    print corpus.cfd_RootDoc["alevi"].N()
    print corpus.cfd_RootDoc.N()
    print len(corpus.cfd_RootDoc.conditions())
        
    print corpus.cfd_DocRoot.N()
    print len(corpus.cfd_DocRoot.conditions())

    df = corpus.compute_tfidf()
    
    end2= datetime.now()
    print "tfidf matrix takes: ",str(end2-end)


def lsa_reconstruction():
    dfpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/pcatests/in/matrixdoctermTFIDF.csv"
    recordpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests/"
    analyser = matrixanalyser(recordpath)
    df = pd.read_csv(dfpath, index_col=0)
    corpus_summary(df)
    _, k = analyser.lsa_transform2(df)
    print k


def pca_reconstruction():
    #dfpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/pcatests/in/matrixdoctermTFIDF.csv"
    recordpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/edit/wordletest/matrix/"
    dfpath = recordpath + os.sep + "matrixdoctermTFIDF.csv"
    #recordpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/"
    
    analyser = matrixanalyser(recordpath)
    df = pd.read_csv(dfpath, index_col=0)
    corpus_summary(df)
    analyser.pca_transform(df)
    

def corpus_summary(df):
    m, n= len(df.columns), len(df.index)
    print "Corpus:","\n ",m," cols, ",n," rows"
    


def matrix_summary(matrix, name):
    print name," shape"
    print matrix.shape
    
if __name__== "__main__":
    
    
    
    # re-construction with new preprocessing addons
    corpus_construction_fromwords()
    
    '''
    # recompute tfidf matrix after repreprocessing
    frpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/edit/wordletest/matrix/"
    corpus = Corpus("wordletest", recordpath=frpath)
    
    doctermfreq = pd.read_csv(frpath+"matrixdoctermFREQ.csv", index_col=0)
    corpus.compute_tfidf2(doctermfreq)
    '''
    
    pca_reconstruction()
    
    
    
    
    
    
    
       
    
    
    
    
    
