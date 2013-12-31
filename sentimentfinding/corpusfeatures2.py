'''
Created on May 14, 2013

@author: dicle
'''

'''
Created on May 5, 2013

@author: dicle
'''

import os
import nltk
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

from txtprocessor import texter, listutils, dateutils
import IOtools
import keywordhandler
from languagetools import SAKsParser
import CFDhelpers
import plotter
from stats import classification



class CorpusFeatures:
    fileids = []
    corpusname = ""
    words = []
    lemmas = []
    filename = ""
    rootpath = ""
    label = ""
    
    cfd_DateWords = nltk.ConditionalFreqDist()
    cfd_DocRoots = nltk.ConditionalFreqDist()
    cfd_DocPOStag = nltk.ConditionalFreqDist()
    cfd_DatePOStag = nltk.ConditionalFreqDist()
    
    cfd_DocSubjectivity = nltk.ConditionalFreqDist()
    cfd_SubjectivityDoc = nltk.ConditionalFreqDist()
    
    def __init__(self, fileidlist, cname, path):
        self.fileids = fileidlist
        self.corpusname = cname
        self.label = cname
        self.lemmas = []
        self.words = []
        self.rootpath = path
        
        self.cfd_DateWords = nltk.ConditionalFreqDist()
        self.cfd_DocRoots = nltk.ConditionalFreqDist()
        self.cfd_DocPOStag = nltk.ConditionalFreqDist()
        self.cfd_DatePOStag = nltk.ConditionalFreqDist()
        
        self.cfd_SubjectivityDoc = nltk.ConditionalFreqDist()
        self.cfd_DocSubjectivity = nltk.ConditionalFreqDist()
    
    
    # 1) bagofwords(roots), 2) adj count, 3) adv count
    def getfeatures(self):
        
        datewords = []
        datePOStag = []
        docPOStag = []
        docroots = []
        
        for fileid in self.fileids:
            print self.rootpath," ",fileid
            path = self.rootpath + os.sep + fileid   # .txt ?
            
            words, date = texter.getnewsitem(path)
            lemmata = SAKsParser.lemmatize_lexicon(words)
            for (_, literalPOS, root, _) in lemmata:
                datewords.append((date, root))
                datePOStag.append((date, literalPOS))
                docroots.append((fileid, root))
                docPOStag.append((fileid, literalPOS))
            
        self.filename = str(len(self.fileids)) + "-".join(self.corpusname.split(os.sep))
        
        self.cfd_DatePOStag = nltk.ConditionalFreqDist(datePOStag)
        self.cfd_DateWords = nltk.ConditionalFreqDist(datewords)
        self.cfd_DocPOStag = nltk.ConditionalFreqDist(docPOStag)
        self.cfd_DocRoots = nltk.ConditionalFreqDist(docroots)    
        
        #CFDhelpers.printCFD(self.cfd_DocPOStag)
        
                
        self.subjectivity_features()
        #CFDhelpers.recordCFD(self.cfd_DocSubjectivity, "SUBJ-"+self.filename)
    
    
    def build_termmatrix(self):
        cfdWordDoc = nltk.ConditionalFreqDist((word, fileid) for fileid in self.cfd_DocRoots.conditions()
                                              for word in list(self.cfd_DocRoots[fileid]))
        
        return cfdWordDoc
    
    
    
    # returns numofdocs X numoffeatures matrix with cells containing values of docI for featureJ
    def build_featurematrix(self):
        # 1-ADJ, 2-ADV, 3-SUBJ
        featurelist = ["ADJ", "ADV", "SUBJ"]
        matrix = {}
        adjfeatures = self.POStag_features(self.cfd_DocPOStag, ["ADJ"], ["Noun"])
        
        
        for doc, jval in adjfeatures:
            matrix[doc] = [jval]
            print " adj:",jval
        advfeatures = self.POStag_features(self.cfd_DocPOStag, ["ADV"], ["ADJ", "Verb"])
        for doc, vval in advfeatures:
            matrix[doc].append(vval)
        
        for doc in self.cfd_DocRoots.conditions():
            sval = self.cfd_DocSubjectivity[doc].N() / float(self.cfd_DocRoots[doc].N())
            matrix[doc].append(sval)
        
        print self.corpusname
        l = [k for k,v in adjfeatures]
        m = [k for k,v in advfeatures]
        n = self.cfd_DocSubjectivity.conditions()
        print "ADJ: ",l
        print "ADV: ",m
        print "SUB: ",n
                        
        return matrix
        
    
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
        yitems = [self.cfd_DocSubjectivity[cond].N() / float(self.cfd_DocRoots[cond].N()) for cond in self.cfd_DocRoots.conditions()]
        plotter.plot_line(self.cfd_DocSubjectivity.conditions(), yitems, linelabel="subjectivity-count", clr="r")
        
        plt.legend()
        plt.savefig(imgoutpath, dpi=100)
        plt.clf()
    
    
    '''  count the roots in subjectivity lexicon '''
    def subjectivity_features(self):
        subjective_keywords = keywordhandler.get_subjectivity_lexicon()
        wordcount = []
        for fileid in self.cfd_DocRoots.conditions():
            for word in list(self.cfd_DocRoots[fileid]):
                if word in subjective_keywords:
                    for i in range(self.cfd_DocRoots[fileid][word]):
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
        CFDhelpers.recordCFD(self.cfd_DocRoots, "DocRoots-"+self.filename)

    def plot_POStag_features(self, cfd, tags1, tags2, label, clr):
        docpostag_features = CFDhelpers.feature_ratio(cfd, tags1, tags2)
        yitems = [val for (_, val) in docpostag_features]
        plotter.plot_line(cfd.conditions(), yitems, label, clr)
        #plt.savefig(IOtools.img_output + os.sep + label.upper() + self.filename + ".png", dpi=100)


#######    corpusfeature class end  ##########




class DataSpace:
    featurematrix = []
    doctermmatrix = []
    corpora = []
    classlabels = []
    spacename = ""     # numoffiles_classtask
    
    def __init__(self):
        self.featurematrix = []
        self.doctermmatrix = []
        self.corpora = []
        self.classlabels = []
        self.spacename = ""
        
    def __getfeaturematrix(self):
        fname = IOtools.picklepath+os.sep+self.spacename+"_featurematrix.p"
        return self.featurematrix, fname
    def __getdoctermmatrix(self):
        fname = IOtools.picklepath+os.sep+self.spacename+"_doctermmatrix.p"
        return self.doctermmatrix, fname
    def __getcorpora(self):
        fname = IOtools.picklepath+os.sep+self.spacename+"_corpora.p"
        return self.corpora, fname
    
    
    def __dumpfeaturematrix(self):
        fname = IOtools.picklepath+os.sep+self.spacename+"_featurematrix.p"
        pickle.dump(self.featurematrix, open(fname, "wb"))
    def __dumpdoctermmatrix(self):
        fname = IOtools.picklepath+os.sep+self.spacename+"_doctermmatrix.p"
        pickle.dump(self.doctermmatrix, open(fname, "wb"))
    def __dumpcorpora(self):
        fname = IOtools.picklepath+os.sep+self.spacename+"_corpora.p"
        pickle.dump(self.corpora, open(fname, "wb"))
    
    '''
    def pdump_object(self, get):
        object, fname = get()
        pickle.dump(object, open(fname, "wb"))
    '''
    def pload_object(self, get):
        _, fname = get()
        return pickle.load(open(fname, "rb"))
    
    def buildcorpus(self, nfile, resourcepath, classlabels, taskname):
        for classlabel in classlabels:
            fileids = []
            p = resourcepath + os.sep + classlabel + os.sep
            fileids.extend(IOtools.getfilenames_of_dir(p, removeextension=False)[:nfile])
            corpus = CorpusFeatures(fileids, classlabel, p)
            corpus.getfeatures()
            self.corpora.append(corpus)
        
        ncat = len(classlabels)
        self.spacename = taskname+"-"+str(nfile*ncat)+"texts"
        
        self.__dumpcorpora()


        
    def build_featurematrix(self):
        for corpus in self.corpora:
            datapoints = corpus.build_featurematrix()
            for k,v in datapoints.iteritems():
                self.featurematrix.append([k]+v+[corpus.label])
        self.record_matrix(self.featurematrix, "featureMATRIX")
        self.__dumpfeaturematrix()
        
    def build_termdocmatrix(self):
        cfdDocTerm = nltk.ConditionalFreqDist()
        self.corpora = self.pload_object(self.__getcorpora)
        
        #docs = []
        labelleddocs = []
        for corpus in self.corpora:
            cfd = corpus.build_termmatrix()
            label = corpus.label
            print label
            for term in cfd.conditions():
                #docs.extend(list(cfd[term]))
                #labelleddocs = [(doc, label) for doc in docs]
                #print list(cfd[term])
                for fileid in list(cfd[term]):
                    cfdDocTerm[term].inc(fileid)
                    labelleddocs.append((fileid, label))
        
        print labelleddocs           
        labelleddocs = list(set(labelleddocs))
        print labelleddocs
        CFDhelpers.recordCFD(cfdDocTerm, self.spacename+"CFDdocterm")
        
        matrix = []
        matrix.append(cfdDocTerm.conditions())
        
        
        for fileid,label in labelleddocs:
            row = []
            for term in cfdDocTerm.conditions():
                numofoccurrences = cfdDocTerm[term][fileid]
                row.append(numofoccurrences)
            self.doctermmatrix.append([fileid]+row+[label])
            matrix.append([fileid]+row+[label])
        
                       
        self.record_matrix(matrix, "DocTermMATRIXn")
        self.record_matrix(self.doctermmatrix, "DocTermMatrix")
        
        self.__dumpdoctermmatrix()
        
        
    
    
    def record_matrix(self, matrix, mname):
        fname = IOtools.matrixpath+os.sep+mname+"-"+self.spacename+"MATRIX.m"
        IOtools.todisc_matrix(matrix, fname)

#######    dataspace class end  ##########     
        

# pickle to make different plots
def buildcorpus(nfile, ncat, resourcename, path):
    resourcepath = path + os.sep + resourcename
    catnames = IOtools.getfoldernames_of_dir(resourcepath)[:ncat]
    
    featurematrix = []
    doctermmatrix = []
    cfdTermDoc = nltk.ConditionalFreqDist()
    
    for catname in catnames:
        fileids = []
        p = resourcepath + os.sep + catname + os.sep
        fileids.extend(IOtools.getfilenames_of_dir(p, removeextension=False)[:nfile])
        corpus = CorpusFeatures(fileids, resourcename+os.sep+catname, p)
        corpus.getfeatures()
        datapoints = corpus.build_featurematrix()
        for k,v in datapoints.iteritems():
            featurematrix.append([k]+v+[resourcename])
            
        corpus.plot_features()
        
        #doc term matrix
        cfd = corpus.build_termmatrix()
        for fileid in cfd.conditions():
            for term in list(cfd[fileid]):
                cfdTermDoc[fileid].inc(term)
    
    IOtools.todisc_matrix(featurematrix, IOtools.results_rootpath+os.sep+"MATRIX"+str(nfile*ncat)+"texts.txt", mode="a")
    #return featurematrix   

def plotcorpus(figname, xLabel, yLabel):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)       
    plt.autoscale(True, axis='both')
    plt.tight_layout(pad=7.0)
    plt.title(figname)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel) 



def dataspace_prepare(nfile, taskname, rootpath, classlabels):
    dataspace = DataSpace()
    #rootpath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/categorize/train/"
    #taskname = "3categorize" 
    dataspace.buildcorpus(nfile, rootpath, classlabels, taskname)
    #dataspace.build_termdocmatrix()
    dataspace.build_featurematrix()
    path = IOtools.picklepath+os.sep+dataspace.spacename+"self.p"
    pickle.dump(dataspace, open(path, "wb"))
    
    return path

if __name__ == "__main__":
    
    start = datetime.now()
    
    '''
    labels = ["dunya", "turkiye", "spor"]
    classlabels, classlabels_decode = classification.classlabel(labels) 
    
    trainpath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/categorize/train/"
    ppath1 = dataspace_prepare(nfile=1500, taskname="3cat-trainn", rootpath=trainpath, classlabels=labels)
    #ptestt = IOtools.picklepath+os.sep+"3categorize-15textsself.p"
    #trainset = pickle.load(open(ppath1, "rb"))
    # plot
    
    testpath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/categorize/test/"
    ppath2 = dataspace_prepare(nfile=150, taskname="3cat-testn", rootpath=testpath, classlabels=labels)
    
    end = datetime.now()
    print "Duration: ",str(end-start)
    '''
    
    
    '''
    trainset = pickle.load(open("3cat-train-600textsself", "rb"))
    testset = pickle.load(open("3cat-train-60textsself", "rb"))
    '''
    
    
    '''
    featmatrix1 = "featureMATRIX-3cat-train-600textsMATRIX.m"
    featmatrix2 = "featureMATRIX-3cat-test-60textsMATRIX.m"    
    
    
    trpath = IOtools.picklepath+os.sep+"3cat-train-600textsself.p"
    tspath = IOtools.picklepath+os.sep+"3cat-test-60textsself.p"
    trainset = pickle.load(open(trpath, "rb"))
    testset = pickle.load(open(tspath, "rb"))
    
    trainset.build_termdocmatrix()
    testset.build_termdocmatrix()
    
    '''
    
    
    # 3 features
    labels = ["dunya", "turkiye", "spor"]
    classlabels, classlabels_decode = classification.classlabel(labels)
    
    '''
    spacename = "3cat-4500texts"
    
    experiment1 = "LDApredict-"
    experiment2 = "NBpredict-"
    trainid = "featureMATRIX-3cat-trainn-4500texts.m"    # get it from the object itself
    testid = "featureMATRIX-3cat-testn-450texts.m"
    
    
    trainX, trainY = classification.prepare_data(IOtools.matrixpath+os.sep+trainid, classlabels)
    testX, testY = classification.prepare_data(IOtools.matrixpath+os.sep+testid, classlabels)
    
    classification.experiment(experiment1+spacename, trainX, trainY, testX, testY, classlabels_decode, classification.LDAclassify, classification.test_LDAclassifier)
    classification.experiment(experiment2+spacename, trainX, trainY, testX, testY, classlabels_decode, classification.naivebayesClassify, classification.test_naivebayesclassifier)
    '''
    
    
    
    # bag of words
    
    spacename = "BOW-3cat-600texts"
    experiment1 = "LDApredict-"
    experiment2 = "NBpredict-"
    trainid = "DocTermMatrix-3cat-train-600textsMATRIX.m"      #"featureMATRIX-3cat-train-600texts.m"    # get it from the object itself
    testid = "DocTermMatrix-3cat-test-60textsMATRIX.m"       #"featureMATRIX-3cat-test-60texts.m"
    
    
    trainX, trainY = classification.prepare_data(IOtools.matrixpath+os.sep+trainid, classlabels, header=True)
    testX, testY = classification.prepare_data(IOtools.matrixpath+os.sep+testid, classlabels, header=True)
    
    classification.experiment(experiment1+spacename, trainX, trainY, testX, testY, classlabels_decode, classification.LDAclassify, classification.test_LDAclassifier)
    classification.experiment(experiment2+spacename, trainX, trainY, testX, testY, classlabels_decode, classification.naivebayesClassify, classification.test_naivebayesclassifier)
    
    
    end = datetime.now()
    print "Duration: ",str(end-start)
    
    
    
    
    
    
    
    
    '''
    # CLOSE ON 14 MAY
    start = datetime.now()
    
    numoffilespercat = 21
    numofcategories = 3
    resourcename = "radikal"
    #rootpath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/newsitems/" 
    rootpath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/testset/"
    resourcenames = ["cumhuriyet", "vakit", "radikal"]
    for resourcename in resourcenames:
        buildcorpus(numoffilespercat, numofcategories, resourcename, rootpath)
    
    end = datetime.now()
    print "Duration: ",str(end-start)
    '''
       






