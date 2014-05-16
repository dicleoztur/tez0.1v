'''
Created on Mar 3, 2014

@author: dicle
'''

from sklearn import cluster
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing, metrics

import math
import pandas as pd
import numpy as np
import os
import time, multiprocessing


import metaexperimentation, matrixhelpers, arrange_N_classes, utils
from corpus import metacorpus
from sentimentfinding import IOtools
from txtprocessor import listutils



def handler(signum, frame):
    print "    not forever"
    raise Exception("quitting the training")



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
    
    def classify_hardcore(self, Xtrain, ytrain, Xtest, ytest, classifiermodel, modelname):
     
        print "...",modelname," enters:"
        
        sfit = time.time()
        print "   fit starts: "
        classifiermodel.fit(Xtrain, ytrain)        
        efit = time.time()
        fitelapsed = efit - sfit
        print " fit took ",str(fitelapsed / 60)
        
        print "   prediction starts: "                    
        ytrue, ypred = ytest, classifiermodel.predict(Xtest)
        epred = time.time()
        predelapsed = epred - efit
        print " prediction took: ",str(predelapsed / 60)
        
        self.reportresults(ytrue, ypred, modelname)
        
    
    def classify(self, Xtrain, ytrain, Xtest, ytest, classifiermodel, modelname):
        p = multiprocessing.Process(target=self.classify_hardcore,
                                     kwargs={"Xtrain":Xtrain, "ytrain":ytrain,
                                             "Xtest":Xtest, "ytest":ytest,
                                             "classifiermodel":classifiermodel,
                                             "modelname":modelname})
        p.start()
        p.join(2400)   # quit after 40 min.s
        if p.is_alive():
            print "Quit ",modelname
            IOtools.todisc_txt("", self.experimentrootpath+os.sep+"Quit-"+modelname+".txt")
            p.terminate()
            p.join()
        

    

        
        
    '''  with signal does not work. signal time range is global 
    def classify(self, Xtrain, ytrain, Xtest, ytest, classifiermodel, modelname):
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(3) # quit after X seconds
        
        print "...",modelname," enters:"
        try:
            sfit = time.time()
            print "   fit starts: "
            classifiermodel.fit(Xtrain, ytrain)        
            efit = time.time()
            fitelapsed = efit - sfit
            print " fit took ",str(fitelapsed / 60)
            
            print "   prediction starts: "                    
            ytrue, ypred = ytest, classifiermodel.predict(Xtest)
            epred = time.time()
            predelapsed = epred - efit
            print " prediction took: ",str(predelapsed / 60)
            
            self.reportresults(ytrue, ypred, modelname)
            signal.alarm(0)
            
        except Exception, exc:
            print "QUIT: ",modelname 
    '''              
        
    def prepare_experiment(self, Xpath, ypath, erootpath, labelnames=None):
        
        self.datapath = Xpath
        self.labelpath = ypath
        
        #if erootpath:
        self.set_score_folder(erootpath)
        
        yvector = IOtools.readcsv(ypath, True)
        self.ylabels = yvector.answer.values
        yvals = self.ylabels.copy().tolist()
        #print "y vals ",yvals
        #print "vect ", self.ylabels
        if labelnames is None:
            labelnames = ["class "+str(i) for i in list(set(yvals))]

        
        instanceids = yvector.index.values.tolist()
        datadf = IOtools.readcsv(Xpath, keepindex=True)
        datadf = datadf.loc[instanceids, :]
              
        self.X = datadf.values   
        self.X[np.isnan(self.X)] = 0
        self.X[np.isinf(self.X)] = 0
        
        '''  do it inside models
        if normalize:
            self.X = preprocessing.normalize(self.X, axis=0)
        '''
        '''  can't apply standardization as it results in negative entries in the matrix, 
             which is not acceptable in Naive Bayes models but fits well in SVM.
        if standardize:
            self.X = preprocessing.scale(self.X, axis=0)
        '''
    
    
    def get_samples(self):
        return self.X, self.ylabels
    
    def set_score_folder(self, newpath):
        self.experimentrootpath = newpath
        self.scorefilepath = metaexperimentation.get_scorefilepath(self.experimentrootpath)
        IOtools.ensure_dir(os.path.join(self.experimentrootpath, "instances"))
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
        try:
            scorereportstr = metrics.classification_report(ytrue, ypred, target_names=self.labelnames)
        except:
            scorereportstr = "zero division error\n"
        IOtools.todisc_txt(scorereportstr, modelscorereportpath)
        
        # record instances
        path = modelscorereportpath = os.path.join(self.experimentrootpath, "instances", experimentname+".csv")
        iheader = ["ytrue\t ypred"]
        instances = [str(true)+"\t"+str(pred) for (true, pred) in zip(ytrue, ypred)]
        IOtools.todisc_list(path, iheader+instances)
        

class SVM(LearningExperiment):
    
    methodname = "classification"
   
    kernels = []
    penalty = []
    degress = [] 
    
    standardize = False  
    
    def __init__(self, erootpath, standardize=False): 
        LearningExperiment.__init__(self, erootpath)
        
        self.methodname = "classification"
        
        self.kernels = ['rbf', 'poly', 'sigmoid', 'linear']   #'rbf',
        self.penalty = [1, 10, 100, 1000]
        self.degrees = range(2, 6)

        self.standardize = standardize
        
    
    def apply_algorithms(self, nclasses):
        
        nrows, _ = self.X.shape
        ntest = int(math.ceil(nrows * (metaexperimentation.testpercentage / 100))) 
        print "NTEST ",ntest,nrows
        Xtrain, Xtest = self.X[:-ntest, :], self.X[-ntest:, :]      
        ytrain, ytest = self.ylabels[:-ntest], self.ylabels[-ntest:] 
        
        if self.standardize:
            Xtrain = preprocessing.scale(Xtrain)
            Xtest = preprocessing.scale(Xtest)
        
        # train on sigmoid kernel
        clsf = SVC(kernel="sigmoid")
        modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__+"_k-sigmoid" 
        self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)
        
        
        # train on rbf and linear kernels
        for k in ["rbf", "linear"]:
            for c in self.penalty:
                clsf = SVC(kernel=k, C=c)
                modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ +"_k-"+k+"_C-"+str(c)
                self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)
        
        
        # train on poly kernel
        for c in self.penalty:
            for d in self.degrees:         
                clsf = SVC(kernel="poly", C=c, degree=d)
                modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ +"_k-poly"+"_C-"+str(c)+"_d-"+str(d)
                self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)      
                    
        
        '''
        print "apply svm"   
        for k in self.kernels:
            for c in self.penalty:
                for d in self.degrees:
                     
                    clsf = SVC(kernel=k, C=c, degree=d)
                    modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ +"_k-"+k+"_C-"+str(c)+"_d-"+str(d)
                    #experimentname = modelname + "_nc-" + str(nclasses)
                    
                    #print ytrain,ytrain.shape
                    #print list(set(ytrain.tolist()))
                    
                    clsf.fit(Xtrain, ytrain)
                    
                    print "...",modelname
                    ytrue, ypred = ytest, clsf.predict(Xtest)
                    self.reportresults(ytrue, ypred, modelname)
        '''

    def apply_algorithms2(self, Xtrain, ytrain, Xtest, ytest, nclasses):
        
        # train on sigmoid kernel
        clsf = SVC(kernel="sigmoid")
        modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__+"_k-sigmoid" 
        self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)
        
        
        # train on rbf and linear kernels
        for k in ["rbf", "linear"]:
            for c in self.penalty:
                clsf = SVC(kernel=k, C=c)
                modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ +"_k-"+k+"_C-"+str(c)
                self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)
        
        
        # train on poly kernel
        for c in self.penalty:
            for d in self.degrees:         
                clsf = SVC(kernel="poly", C=c, degree=d)
                modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ +"_k-poly"+"_C-"+str(c)+"_d-"+str(d)
                self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)      
         





class NaiveBayes(LearningExperiment):
    
    methodname = "classification"
    models = []
    
    normalize = False
    
    def __init__(self, erootpath, normalize=False):  
        LearningExperiment.__init__(self, erootpath)
        self.models = []
        
        self.normalize = normalize
    
    def apply_algorithms(self, nclasses):
        
        nrows, _ = self.X.shape
        ntest = int(math.ceil(nrows * (metaexperimentation.testpercentage / 100))) 
        Xtrain, Xtest = self.X[:-ntest, :], self.X[-ntest:, :]      
        ytrain, ytest = self.ylabels[:-ntest], self.ylabels[-ntest:] 
        
        if self.normalize:
            Xtrain = preprocessing.normalize(Xtrain, axis=0)
            Xtest = preprocessing.normalize(Xtest, axis=0)
        
        multinomialnb = MultinomialNB()
        gaussiannb = GaussianNB()
        self.models = [multinomialnb, gaussiannb]
        
        print "apply naive bayes"
        for clsf in self.models:
            
            modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ 
            self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)
            '''
            modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ 
            clsf.fit(Xtrain, ytrain)
            print "...",modelname
            ytrue, ypred = ytest, clsf.predict(Xtest)
            self.reportresults(ytrue, ypred, modelname)
            '''

    def apply_algorithms_matrixselect(self, nclasses):
        
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
            self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)
            
            

    def apply_algorithms2(self, Xtrain, ytrain, Xtest, ytest, nclasses):
                
        multinomialnb = MultinomialNB()
        gaussiannb = GaussianNB()
        self.models = [multinomialnb, gaussiannb]
        
        print "apply naive bayes"
        for clsf in self.models:
            
            modelname = "_MT-"+self.methodname+"_alg-"+clsf.__class__.__name__ 
            self.classify(Xtrain, ytrain, Xtest, ytest, clsf, modelname)         
             

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
    


def conduct_cross_validation(k=10,
                             annotationtype="double",
                             combname="comb524_F_0-0_1-1_2-0_3-0_4-0_5-1_6-1_7-0_8-0",
                             agrtype="fullagr",
                             labelfoldername="ALLobj-STGsubj_NC-2",
                             outrootpath="/home/dicle/Dicle/Tez/corpusstats/learning3/crossvalidation"):
    
    svmclassifier = SVM("")
    #clusterer = Clustering("")
    nbclassifier = NaiveBayes("")
    models = [svmclassifier, nbclassifier] #, clusterer]
    
    
    datarootpath = metacorpus.get_datasets_path(annotationtype)
    Xpath = os.path.join(datarootpath, combname+".csv")
    
    labelpath = metacorpus.get_labels_path(annotationtype)
    labelpath = os.path.join(labelpath, agrtype, labelfoldername)
    ylabelspath = os.path.join(labelpath, metacorpus.labelsfilename+".csv")
    
    labelitems = labelfoldername.split(metaexperimentation.interfeatsep)
    unionname = labelitems[0]
    ncstr = labelitems[1]
    nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
    nclasses = int(nc)
           
    outp = IOtools.ensure_dir(os.path.join(outrootpath, combname, unionname))               
    
    print models
    for m in models:
        print m.__class__.__name__    
                    
    for model in models:  
        
        model.prepare_experiment(Xpath, ylabelspath, None, labelnames=None)
        X, y = model.get_samples()
        n, _ = X.shape
        
        teststart = 0
        testsize = int(math.ceil(n * (metaexperimentation.testpercentage / 100))) 
        for _ in range(k):
                                    
            testfinish = teststart + testsize  #(n / k)
            print "\t",model.__class__.__name__," ",teststart,"  ",testfinish,"  testsize:",testsize
            
            Xtrain, ytrain = utils.gettrainset(X, teststart, testfinish), utils.gettrainset(y, teststart, testfinish)
            Xtest, ytest = utils.gettestset(X, teststart, testfinish), utils.gettestset(y, teststart, testfinish)
            foldname = str(teststart)+"-"+str(testfinish)
            teststart = testfinish + 1   # n / k + 1 so that we can cover everything and run spits 10 times
            
            rootscorespath = IOtools.ensure_dir(os.path.join(outp, foldname))
            model.set_score_folder(rootscorespath)
            #metaexperimentation.initialize_score_file(rootscorespath)
            model.apply_algorithms2(Xtrain, ytrain, Xtest, ytest, nclasses)
        
        
        
        

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
        


# made 3 on 5 April
# made x on 11 May
def conduct_experimentsx(inrootpath=metacorpus.learningdatapath, outrootpath=metaexperimentation.expscorepath, scale=False):
    annottypes = ["double"]  #, "single"]
    # simdilik sadece double 0-5 tamam. 5'ten devam. 7 Mayis 11:10
    
    #setsizes = ["150"]
    #taggertypes = ["user"]
    numofcombs = 15
    
    #nclasses = arrange_N_classes.nclasses   # [4,5]
    
    #models = []
    svmclassifier = SVM("")
    clusterer = Clustering("")
    nbclassifier = NaiveBayes("", normalize=scale)
    #nbclassifier = MultinomialNB(outrootpath)
    models = [svmclassifier, nbclassifier, clusterer]
    
    
    for annotationtype in annottypes:
        
        sp1 = IOtools.ensure_dir(os.path.join(outrootpath, annotationtype))
            
        datasetspath = metacorpus.get_datasets_path(annotationtype)  # finaldatasets
        labelspath = metacorpus.get_labels_path(annotationtype)
        
        agreementtypes= IOtools.getfoldernames_of_dir(labelspath)
                  
        '''  select random '''
        combfilenames = IOtools.getfilenames_of_dir(datasetspath)
        combfilenames.sort()
        #combfilenames = combfilenames[:numofcombs]  #make this random
        # random select:
        #combfilenames = listutils.select_random_elements(np.array(combfilenames, dtype=object), numofcombs).tolist() 
        
        ''' process k to m '''
        k = 40
        m = 130
        combfilenames = combfilenames[k:m]
        processedcombs = IOtools.getfoldernames_of_dir(sp1)
        combfilenames = [comb for comb in combfilenames if comb not in processedcombs]
        
        ''' select previously learned '''
        '''combfilenames = IOtools.getfoldernames_of_dir(os.path.join(metacorpus.learningrootpath, 
                                                                 "experiments", 
                                                                 "scores",
                                                                 annotationtype))'''
            
        for i,combfile in enumerate(combfilenames):
        
            print "############  ",combfile,"  ",str(i)
            Xpath = os.path.join(datasetspath, combfile + ".csv")
            sp2 = IOtools.ensure_dir(os.path.join(sp1, combfile))
            
            for agreementtype in agreementtypes:   # count it on labelspath not nclasses
                
                lp1 = os.path.join(labelspath, agreementtype)
                labelunions = IOtools.getfoldernames_of_dir(lp1)
                
                for labelunion in labelunions:
                    
                    lp2 = os.path.join(lp1, labelunion)
                    
                    labelitems = labelunion.split(metaexperimentation.interfeatsep)
                    unionname = labelitems[0]
                    ncstr = labelitems[1]
                    nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
                    nc = int(nc)
                                   
                    rootscorespath = IOtools.ensure_dir(os.path.join(sp2, unionname))
                    metaexperimentation.initialize_score_file(rootscorespath)
                    ylabelspath = os.path.join(lp2, metacorpus.labelsfilename+".csv")
                                    
                    for model in models:
                        
                        #labelnames = metacorpus.get_label_names()
                        model.prepare_experiment(Xpath, ylabelspath, rootscorespath, labelnames=None)
                        model.apply_algorithms(nc)
                
            print "############  ",combfile,"  ",str(i)  



# made 3 on 11 May to add featuregroup hierarchy
def conduct_experiments3(inrootpath=metacorpus.learningdatapath, outrootpath=metaexperimentation.expscorepath, scale=False):
    annottypes = ["single"]  #, "single"]
    # simdilik sadece double 0-5 tamam. 5'ten devam. 7 Mayis 11:10
    
    #setsizes = ["150"]
    #taggertypes = ["user"]
    numofcombs = 15
    
    #nclasses = arrange_N_classes.nclasses   # [4,5]
    
    #models = []
    svmclassifier = SVM("")
    clusterer = Clustering("")
    nbclassifier = NaiveBayes("", normalize=scale)
    #nbclassifier = MultinomialNB(outrootpath)
    models = [nbclassifier, clusterer, svmclassifier]
    
    
    for annotationtype in annottypes:
        
        sp1 = IOtools.ensure_dir(os.path.join(outrootpath, annotationtype))
        
        print "sp1 ",sp1
         
        datasetspath = metacorpus.get_datasets_path(annotationtype)  # finaldatasets
        labelspath = metacorpus.get_labels_path(annotationtype)
        
        agreementtypes= IOtools.getfoldernames_of_dir(labelspath)
                  
    
        feature_metric_comb_lists = utils.get_featuregroupings()  # { featureclassname : { metricname : combcode-filenames } }
                 
        for featureclass, metriccombmap in feature_metric_comb_lists.iteritems():
            
            sp2 = IOtools.ensure_dir(os.path.join(sp1, featureclass))
            
            for metricname, combfilenames in metriccombmap.iteritems():
                
                sp3 = IOtools.ensure_dir(os.path.join(sp2, metricname))
                
                print "metricname ",featureclass,"  ",metricname
                print "b ",len(combfilenames)
                
                processedcombs = IOtools.getfoldernames_of_dir(sp3)
                combfilenames = [comb for comb in combfilenames if comb not in processedcombs]
                print "a ",len(combfilenames)
                print "pr ",len(processedcombs),"  ",sp3
                
                
                for i,combfile in enumerate(combfilenames):
                
                    print "############  ",combfile,"  ",str(i)
                    Xpath = os.path.join(datasetspath, combfile + ".csv")
                    sp4 = IOtools.ensure_dir(os.path.join(sp3, combfile))
                    
                   
                    for agreementtype in agreementtypes:   # count it on labelspath not nclasses
                        
                        lp1 = os.path.join(labelspath, agreementtype)
                        labelunions = IOtools.getfoldernames_of_dir(lp1)
                        
                        for labelunion in labelunions:
                            
                            lp2 = os.path.join(lp1, labelunion)
                            
                            labelitems = labelunion.split(metaexperimentation.interfeatsep)
                            unionname = labelitems[0]
                            ncstr = labelitems[1]
                            nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
                            nc = int(nc)
                                           
                            rootscorespath = IOtools.ensure_dir(os.path.join(sp4, unionname))
                            metaexperimentation.initialize_score_file(rootscorespath)
                            ylabelspath = os.path.join(lp2, metacorpus.labelsfilename+".csv")
                                            
                            for model in models:
                                print Xpath
                                #labelnames = metacorpus.get_label_names()
                                model.prepare_experiment(Xpath, ylabelspath, rootscorespath, labelnames=None)
                                model.apply_algorithms(nc)
                        
                    print "############  ",combfile,"  ",str(i)  
               
               
                     
    
def conduct_experiments(inrootpath=metacorpus.learningdatapath, outrootpath=metaexperimentation.expscorepath, normalize=False):
    annottypes = ["double"]
    setsizes = ["150"]
    taggertypes = ["user"]
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
                            model.prepare_experiment(Xpath, ylabelspath, rootscorespath, labelnames=None, normalize=normalize)
                            model.apply_algorithms(nc)    
            

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
    
    #conduct_cross_validation()
    conduct_experiments3(scale=True)
    #evaluate_performance(resultspath)
   
   
if __name__ == "__main__":
    shell()
    
    
    
    
    
    
    
    
    

