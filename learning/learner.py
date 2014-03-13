'''
Created on Mar 3, 2014

@author: dicle
'''

from sklearn import cluster
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from scipy import sparse
from sklearn import metrics

import math
import pandas as pd
import numpy as np
import os


import metaexperimentation, matrixhelpers, arrange_N_classes
from corpus import metacorpus
from sentimentfinding import IOtools




class LearningExperiment:
    
    experimentrootpath = ""
    '''Xpath = ""
    ylabels = None'''
    
    
    def __init__(self, recordpath=metaexperimentation.expscorepath):
        # recordpath : scoreroot + annottype + setsize
        self.experimentrootpath = recordpath
        self.scorefilepath = ""     
        
        self.datapath = ""
        self.labelpath = ""
        
        self.X = None
        self.ylabels = None
        self.labelnames = None
        
        
    def train(self, datamatrix):
        return

    def test(self, originallabels):
        return
    
    
        
    def prepare_experiment(self, Xpath, ypath, erootpath, labelnames=None):
        
        self.datapath = Xpath
        self.labelpath = ypath
        
        self.set_score_folder(erootpath)
        
        yvector = IOtools.readcsv(ypath, True)
        self.ylabels = yvector.answer.values
        yvals = self.ylabels.copy().tolist()
        #print "y vals ",yvals
        #print "vect ", self.ylabels
        if labelnames is None:
            labelnames = ["class "+str(i) for i in list(set(yvals))]

        
        datadf = IOtools.readcsv(Xpath, keepindex=True)
        self.X = datadf.values   
        self.X[np.isnan(self.X)] = 0
        self.X[np.isinf(self.X)] = 0
    
    
    def set_score_folder(self, newpath):
        self.experimentrootpath = newpath
        self.scorefilepath = metaexperimentation.get_scorefilepath(self.experimentrootpath)
        #self.initialize_scorefile()

    
    def reportresults(self, ytrue, ypred, experimentname):
        
        '''
        precision, recall, f1score, _ = metrics.precision_recall_fscore_support(ytrue, ypred)     
        print precision, recall, f1score
        '''
        #print ytrue
        #print ypred     
           
        precision = metrics.precision_score(ytrue, ypred, pos_label=None, average="macro")
        recall = metrics.recall_score(ytrue, ypred, pos_label=None, average="macro")
        f1score = metrics.f1_score(ytrue, ypred, pos_label=None, average="macro")
        accuracy = metrics.accuracy_score(ytrue, ypred)
        
        scoreline = metaexperimentation.csvsep.join(map(lambda x : str(x), [experimentname, precision, recall, f1score, accuracy]))
        IOtools.todisc_txt("\n"+scoreline, self.scorefilepath, mode="a")
        
        modelscorereportpath = os.path.join(self.experimentrootpath, experimentname+".txt")   
        scorereportstr = metrics.classification_report(ytrue, ypred, target_names=self.labelnames)
        IOtools.todisc_txt(scorereportstr, modelscorereportpath)
        

class SVM(LearningExperiment):
    
    methodname = "classification"
   
    kernels = []
    penalty = []
    degress = []   
    
    def __init__(self, erootpath): 
        LearningExperiment.__init__(self, erootpath)
        
        self.methodname = "classification"
        
        self.kernels = ['rbf', 'poly', 'sigmoid', 'linear']   #'rbf',
        self.penalty = [1, 10, 100, 1000]
        self.degrees = range(2, 6)

    
    def apply_algorithms(self, nclasses):
        
        nrows, _ = self.X.shape
        ntest = int(math.ceil(nrows * (metaexperimentation.testpercentage / 100))) 
        print "NTEST ",ntest,nrows
        Xtrain, Xtest = self.X[:-ntest, :], self.X[-ntest:, :]      
        ytrain, ytest = self.ylabels[:-ntest], self.ylabels[-ntest:] 
        
        
        print "apply svm"   
        for k in self.kernels:
            for c in self.penalty:
                for d in self.degrees:
                     
                    clsf = SVC(kernel=k, C=c, degree=d)
                    modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ +"_k-"+k+"_C-"+str(c)+"_d-"+str(d)
                    #experimentname = modelname + "_nc-" + str(nclasses)
                    
                    '''print ytrain,ytrain.shape
                    print list(set(ytrain.tolist()))'''
                    
                    clsf.fit(Xtrain, ytrain)
                    
                    print "...",modelname
                    ytrue, ypred = ytest, clsf.predict(Xtest)
                    self.reportresults(ytrue, ypred, modelname)
        

class NaiveBayes(LearningExperiment):
    
    methodname = "classification"
    models = []
    
    def __init__(self, erootpath):  
        LearningExperiment.__init__(self, erootpath)
        self.models = []
    
    def apply_algorithms(self, nclasses):
        
        nrows, _ = self.X.shape
        ntest = int(math.ceil(nrows * (metaexperimentation.testpercentage / 100))) 
        Xtrain, Xtest = self.X[:-ntest, :], self.X[-ntest:, :]      
        ytrain, ytest = self.ylabels[:-ntest], self.ylabels[-ntest:] 
        
        multinomialnb = MultinomialNB()
        gaussiannb = GaussianNB()
        self.models = [multinomialnb, gaussiannb]
        
        print "apply naive bayes"
        for clsf in self.models:
            
            modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ 
            clsf.fit(Xtrain, ytrain)
            
            print "...",modelname
            ytrue, ypred = ytest, clsf.predict(Xtest)
            self.reportresults(ytrue, ypred, modelname)
         
             

class Clustering(LearningExperiment):
    
    models = []
    methodname = "clustering"
    
    def __init__(self, erootpath):
        LearningExperiment.__init__(self, erootpath)
        
        self.models = []
        
        self.methodname = self.__class__.__name__
    
    
    def apply_algorithms(self, nclusters):
        
        kmeans = cluster.KMeans(n_clusters=nclusters)
        #spectral = cluster.SpectralClustering(n_clusters=nclusters)
        # SPECTRAL CAUSED 'not positive definite matrix' ERROR. TOO STRICT.
        self.models = [kmeans] #, spectral]

    
        print "apply clustering"
        print len(self.models)
        i = 0
        for model in self.models:
        
            modelname = model.__class__.__name__
            experimentname = "_MT-"+self.methodname+"_alg-"+modelname   #+"_nc-"+str(nclusters)
            
            print "...",modelname
            
            #print "shape ", self.X.shape, type(nclusters)
            ytrue, ypred = self.ylabels, model.fit_predict(self.X)
            
            self.reportresults(ytrue, ypred, experimentname)
            print "CLSTR ",i
            i = i + 1


class Experimentation:
    
    datasetname = ""
    outpath = ""   # one experimentation for each datasettype (valid,setsize,features)
    scorefilepath = ""
    
    def __init__(self, experimentrootpath, datasetfolder, datasetname):
        self.datasetfolder = datasetfolder
        self.datasetname = datasetname
        
        self.outpath = experimentrootpath
        
        self.scorefilepath = os.path.join(self.outpath, "algorithms-scores.csv")   # self.datasetname+"-scores.csv")
        self.initialize_scorefile()
    
    
   
    
    def prepare_data(self, taggingtype="random"):
        print "preparing data"
    
        datamatrixcsvpath = os.path.join(self.datasetfolder, self.datasetname+".csv")
    
        print datamatrixcsvpath
        
        ylabelspath = os.path.join(self.datasetfolder, "labels", taggingtype+"-labels.csv")
        ylabels = IOtools.readcsv(csvpath=ylabelspath, keepindex=True)
        ylabels = ylabels.iloc[:, 0].values
    
        print "type(ylabels)  ",type(ylabels)
        #print ylabels
        
        return datamatrixcsvpath, ylabels
    

    

def conduct_experiments2(resultspath):
    
    datafolder = "/home/dicle/Dicle/Tez/corpusstats/learning/data/random-single-N5/finaldatasets_test/"
    datasetname = "feat-00111110000"
    
    datasetnames = IOtools.getfilenames_of_dir(datafolder)
    for datasetname in datasetnames: 
        epath = IOtools.ensure_dir(resultspath+os.sep+datasetname)
        experiment = Experimentation(experimentrootpath=epath, datasetfolder=datafolder, datasetname=datasetname)    
        datamatrixcsvpath, ylabels = experiment.prepare_data()
        
        #clusterer = Clustering(erootpath=epath, datamatrixpath=datamatrixcsvpath, yvector=ylabels)
        #clusterer.apply_algorithms(scorefilepath=experiment.scorefilepath)
       
        svmclassifier = SVM(erootpath=epath, datamatrixpath=datamatrixcsvpath, yvector=ylabels)
        svmclassifier.apply_algorithms(scorefilepath=experiment.scorefilepath)
        


def conduct_experiments(inrootpath=metacorpus.learningdatapath, outrootpath=metaexperimentation.expscorepath):
    annottypes = ["single"]
    setsizes = ["150"]
    taggertypes = ["random"]
    numofcombs = 5
    
    #nclasses = arrange_N_classes.nclasses   # [4,5]
    
    #models = []
    svmclassifier = SVM("")
    clusterer = Clustering("")
    nbclassifier = NaiveBayes("")
    #nbclassifier = MultinomialNB(outrootpath)
    models = [svmclassifier, nbclassifier, clusterer]
    
    
    for annotationtype in annottypes:
        
        sp1 = IOtools.ensure_dir(os.path.join(outrootpath, annotationtype))
        
        for setsize in setsizes:
            
            sp2 = IOtools.ensure_dir(os.path.join(sp1, setsize))
            
            datasetspath = metacorpus.get_datasets_path(annotationtype, setsize)  # finaldatasets
            labelspath = metacorpus.get_labels_path(annotationtype, setsize)
            nclasses = IOtools.getfoldernames_of_dir(labelspath)
                      
            combfilenames = IOtools.getfilenames_of_dir(datasetspath)
            combfilenames = combfilenames[:numofcombs]
            
            for combfile in combfilenames:
            
                Xpath = os.path.join(datasetspath, combfile + ".csv")
                sp3 = IOtools.ensure_dir(os.path.join(sp2, combfile))
                
                for nclass in nclasses:   # count it on labelspath not nclasses
                    
                    #nclabelspath = arrange_N_classes.nclass_label_folder(labelspath, nc)  # get folder path containing nc-grouped labels
                    nclabelspath = os.path.join(labelspath, nclass)
                    nc = nclass.split(metaexperimentation.intrafeatsep)[-1]
                    nc = int(nc)
                    sp4 = IOtools.ensure_dir(os.path.join(sp3, nclass)) #"NC-"+str(nc)))
                    
                    for taggertype in taggertypes:
                        
                        rootscorespath = IOtools.ensure_dir(os.path.join(sp4, taggertype))
                        metaexperimentation.initialize_score_file(rootscorespath)
                        ylabelspath = os.path.join(nclabelspath, taggertype+".csv")
                        
                        for model in models:
                            
                            #labelnames = metacorpus.get_label_names()
                            model.prepare_experiment(Xpath, ylabelspath, rootscorespath, labelnames=None)
                            model.apply_algorithms(nc)
                
                
                '''
                ylabelfiles = IOtools.getfilenames_of_dir(labelspath)
                
                scorespath = os.path.join(sp2, combfile)
                #scorespath = IOtools.ensure_dir(scorespath)  # this to be appended _NC in model class
                # makedirs at each step! don't try creating inner-folders over. OK done. 
                for ylabelfile in ylabelfiles:
                    ylabelpath = os.path.join(labelspath, ylabelfile)
                '''    
    
    
            

def evaluate_performance(resultspath):
    performancepath = os.path.join(resultspath, "performance")  # should be taken from meta experimentation
    datasetcombs = IOtools.getfoldernames_of_dir(os.path.join(resultspath, "scores"))
    
    
    
    
    # best comb per alg
    # best alg per comb
    # worst alg per comb_excld1or2
    # best and worst first 10 settings
    
    # interpreting comb names is important
    # parsing mt and alg names seems easier




# per datasetname  (random/size)
def shell():
    #resultspath = metaexperimentation.rootpath + "/test-N5/"
    conduct_experiments()
    #evaluate_performance(resultspath)
   
   
if __name__ == "__main__":
    shell()
    
    
    
    
    
    
    
    
    

