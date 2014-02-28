'''
Created on Feb 13, 2014

@author: dicle
'''

from sklearn import cluster
from sklearn import naive_bayes
from sklearn.svm import SVC
from scipy import sparse
from sklearn import metrics

import random
import pandas as pd
import numpy as np
import os


import metaexperimentation, matrixhelpers
from sentimentfinding import IOtools




class LearningExperiment:
    
    experimentrootpath = ""
    Xpath = ""
    ylabels = None
    
    def __init__(self, recordpath, datamatrixpath, yvector):
        self.experimentrootpath = recordpath
        self.Xpath = datamatrixpath
        self.ylabels = yvector.copy()
        
    def train(self, datamatrix):
        return

    def test(self, originallabels):
        return
    
    def reportresults(self, ytrue, ypred, experimentname, scorefilepath, labelnames):
        precision = metrics.precision_score(ytrue, ypred)
        recall = metrics.recall_score(ytrue, ypred)
        f1score = metrics.f1_score(ytrue, ypred)
        accuracy = metrics.accuracy_score(ytrue, ypred)
        
        scoreline = metaexperimentation.csvsep.join(map(lambda x : str(x), [experimentname, precision, recall, f1score, accuracy]))
        IOtools.todisc_txt("\n"+scoreline, scorefilepath, mode="a")
        
        selfscorereportpath = os.path.join(self.experimentrootpath, experimentname+".txt")   
        
        scorereportstr = metrics.classification_report(ytrue, ypred, target_names=labelnames)
        IOtools.todisc_txt(scorereportstr, selfscorereportpath)
        

class SVM(LearningExperiment):
    
    methodname = "classification"
   
    kernels = []
    penalty = []
    degress = []   
    
    def __init__(self, erootpath, datamatrixpath, yvector): 
        LearningExperiment.__init__(self, erootpath, datamatrixpath, yvector)
        
        self.methodname = "classification"
        
        self.kernels = ['rbf', 'poly', 'sigmoid', 'linear']   #'rbf',
        self.penalty = [1, 10, 100, 1000]
        self.degrees = range(2, 6)

    
    def apply_algorithms(self, scorefilepath, labelnames=None):
        
        yvals = self.ylabels.copy().tolist()
        print "y vals ",yvals
        if labelnames is None:
            labelnames = ["cluster "+str(i) for i in list(set(yvals))]
        
        nclasses = 3    # we will change it
        datadf = IOtools.readcsv(self.Xpath, keepindex=True)
        X = datadf.values
        
        
        for k in self.kernels:
            for c in self.penalty:
                for d in self.degrees:
                     
                    clsf = SVC(kernel=k, C=c, degree=d)
                    modelname = "_MT-"+clsf.__class__.__name__ +"_k-"+k+"_C-"+str(c)+"_d-"+str(d)
                    experimentname = modelname + "_nc-" + str(nclasses)
                    clsf.fit(X, self.ylabels)
                    
                    print modelname
                    ytrue, ypred = self.ylabels, clsf.predict(X)
                    self.reportresults(ytrue, ypred, experimentname, scorefilepath, labelnames)
        
        
         

class Clustering(LearningExperiment):
       
    Xpath = ""
    ylabels = None
    
    models = []
    methodname = "clustering"
    
    def __init__(self, erootpath, datamatrixpath, yvector):
        LearningExperiment.__init__(self, erootpath, datamatrixpath, yvector)
        
        self.models = []
        
        self.methodname = self.__class__.__name__
    
    
    def apply_algorithms(self, scorefilepath, labelnames=None):
        
        yvals = self.ylabels.copy().tolist()
        print "y vals ",yvals
        if labelnames is None:
            labelnames = ["cluster "+str(i) for i in list(set(yvals))]
        
        nclusters = 3    # we will change it
        kmeans = cluster.KMeans(n_clusters=nclusters)
        spectral = cluster.SpectralClustering(n_clusters=nclusters)
        self.models.append(kmeans)
        self.models.append(spectral)
        
        datadf = IOtools.readcsv(self.Xpath, keepindex=True)
        X = datadf.values
        
        print "y sh ",self.ylabels.shape
        print "X ",X[0]
        
        print "apply clustering"
        for model in self.models:
        
            modelname = model.__class__.__name__
            experimentname = "_MT-"+self.methodname+"_alg-"+modelname+"_nc-"+str(nclusters)
            
            print "...",modelname
            
            ytrue, ypred = self.ylabels, model.fit_predict(X)
            
            self.reportresults(ytrue, ypred, experimentname, scorefilepath, labelnames)
            '''
            precision = metrics.precision_score(ytrue, ypred)
            recall = metrics.recall_score(ytrue, ypred)
            f1score = metrics.f1_score(ytrue, ypred)
            accuracy = metrics.accuracy_score(ytrue, ypred)
            
            scoreline = metaexperimentation.csvsep.join([experimentname, precision, recall, f1score, accuracy])
            IOtools.todisc_txt(scoreline, scorefilepath, mode="a")
            
            selfscorereportpath = os.path.join(self.experimentrootpath, experimentname)            
            if labelnames is None:
                labelnames = ["cluster "+str(i) for i in list(set(self.ylabels))]
            scorereportstr = metrics.classification_report(ytrue, ypred, target_names=labelnames)
            IOtools.todisc_txt(selfscorereportpath, scorereportstr)
            '''



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
    
    
    
    def initialize_scorefile(self):
        header = metaexperimentation.csvsep.join(metaexperimentation.scoresheader)
        IOtools.todisc_txt(header, self.scorefilepath)
    
    
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
    

    

def conduct_experiments(resultspath):
    
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
        


def evaluate_performance(resultspath):
    performancepath = os.path.join(resultspath, "performance")  # should be taken from meta experimentation
    datasetcombs = IOtools.getfoldernames_of_dir(os.path.join(resultspath, "scores"))
    
    
    
    
    # best comb per alg
    # best alg per comb
    # worst alg per comb_excld1or2
    # best and worst first 10 settings
    
    # interpreting comb names is important
    # parsing mt and alg names seems easier


class PerformanceEvaluator:
    experimentspath = ""
    
    def __init__(self, expspath=metaexperimentation.rootpath, basemetric="accuracy"):
        self.experimentspath = expspath
        self.resultspath = os.path.join(self.experimentspath, "performance")
        self.N = 10   # ranking number
        #self.metricname = basemetric
    
    def best_score_per_annottype(self, metricname):
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
            
            annotdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
            
            p1 = os.path.join(scorepath, annottype)
            sizes = IOtools.getfoldernames_of_dir(p1)
            
            for setsize in sizes:
                p2 = os.path.join(p1, setsize)
                datasetnames = IOtools.getfoldernames_of_dir(p2)  # list of combcode_NC names
                
                for dataset in datasetnames:
                    p3 = os.path.join(p2, dataset)
                    combcode, nclasses = metaexperimentation.parse_featsetname(dataset)
                    files = [os.path.join(p3, taggertype+"_"+metaexperimentation.scorefilename+".csv") 
                             for taggertype in metaexperimentation.taggertypes]
                    # we can make following a function taking path and filenames, returning a bigdf sorted.
                    for scorecsvfilepath in files:
                        scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                        
                        rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname])
                        annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values
                        #bigdf = bigdf.append(rankdf)  # bigdf header = metaexperimentation.performanceheader
                    
                    annotdf["featureset"] = combcode
                    annotdf["nclasses"] = nclasses
                
                annotdf["size"] = setsize
            annotdf["annottype"] = annottype  
            annotdf = matrixhelpers.get_first_N_rows(annotdf, self.N, [metricname])  
            bigdf = bigdf.append(annotdf)
            # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["annottype", metricname], ascending=False, inplace=True)
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = "best_score_per_annottype-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
    # best alg is the same. just output alg name.                                                        
                
    
    def best_score_per_size(self):
        
        return
    
    def best_score_per_nclasses(self):
        return
    
    def best_score_per_algorithm(self):
        return
    
    def best_score_per_tagger(self):
        return
    
    '''
    def best_algorithm_per_annottype(self):
        
        return
    
    def best_algorithm_per_size(self):
        
        return
    
    def best_algorithm_per_nclasses(self):
        return
    
    def best_algorithm_per_combination(self):
        return   
    
    def best_algorithm_per_tagger(self):
        return
    '''
    
    def mean_scores(self):
        return


# per datasetname  (random/size)
def shell():
    resultspath = metaexperimentation.rootpath + "/test-N5/"
    conduct_experiments(resultspath)
    evaluate_performance(resultspath)
   
   
if __name__ == "__main__":
    shell()
    
    
    
    
    
    
    
    
    

