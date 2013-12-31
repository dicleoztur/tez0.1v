'''
Created on May 5, 2013

@author: dicle
'''

import os
import nltk
from datetime import datetime
import matplotlib.pyplot as plt

from txtprocessor import texter, listutils, dateutils
import IOtools
import keywordhandler
from languagetools import SAKsParser
import CFDhelpers
import plotter

class CorpusFeatures:
    fileids = []
    corpusname = ""
    words = []
    lemmas = []
    filename = ""
    rootpath = ""
    
    cfd_DateWords = nltk.ConditionalFreqDist()
    cfd_DocRoots = nltk.ConditionalFreqDist()
    cfd_DocPOStag = nltk.ConditionalFreqDist()
    cfd_DatePOStag = nltk.ConditionalFreqDist()
    
    cfd_DocSubjectivity = nltk.ConditionalFreqDist()
    cfd_SubjectivityDoc = nltk.ConditionalFreqDist()
    
    def __init__(self, fileidlist, cname, path):
        self.fileids = fileidlist
        self.corpusname = cname
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



if __name__ == "__main__":
    
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
    
       






