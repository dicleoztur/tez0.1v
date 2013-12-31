'''
Created on Sep 14, 2013

@author: dicle
'''

import os
import numpy as np
import pandas as pd
from datetime import datetime

from sentimentfinding import IOtools
from txtprocessor import texter
import dataspaceV2
import classify

# numofpoints=0 : take all the files in the folder. otherwise, select files randomly.
def get_fileids_infolder(path, numofpoints):
    fileids = IOtools.getfilenames_of_dir(path, removeextension=False)
    
    if numofpoints > 0:
        selected = np.random.randint(0, len(fileids), numofpoints)
        fileids = np.array(fileids)
        fileids = fileids[selected]
    
    for fileid in fileids[:10]:
        print fileid,"  ",texter.getnewsmetadata(path+os.sep+fileid, ["resource"])
    return fileids.tolist()
    
    
    
def get_fileids_perlabel(rootpath, labels, numoffiles):
    
    labelwisefileidlist = {}
    for label in labels:
        labelwisefileidlist[label] = list()
    for label in labels:
        path = rootpath + os.sep + label + os.sep
        labelwisefileidlist[label] = get_fileids_infolder(path, numoffiles)        
    
    return labelwisefileidlist


def build_set(rootpath, numoffiles):
    labels = IOtools.getfoldernames_of_dir(rootpath)
    return get_fileids_perlabel(rootpath, labels, numoffiles)


def makedataspace(inputpath, numoffiles, corpusname, outpath):
    labelwisefileidlist = build_set(inputpath, numoffiles)
         
    corpus = dataspaceV2.Corpus(corpusname, outpath)  
    corpus.read_corpus(inputpath, labelwisefileidlist)
    corpus.get_docterm_matrix()
    corpus.compute_tfidf()
    corpus.extract_features()
    return corpus



def prepare_experiment(inputpath, outputpath, ename, numoftrain, percentagesplit=10, cmetadata=None):
    # dir structure: root->(data->(train,test), results)
    corpuspath = IOtools.ensure_dir(outputpath+os.sep+ename+os.sep+"data"+os.sep)
    resultspath = IOtools.ensure_dir(outputpath+os.sep+ename+os.sep+"results"+os.sep)
    
    if cmetadata is None:
        cmetadata = {}
        cmetadata["train"] = numoftrain
        cmetadata["test"] = int(numoftrain / percentagesplit)
    
    for cname, numoffiles in cmetadata.iteritems():
        inpath = inputpath + os.sep + cname + os.sep
        crootpath = IOtools.ensure_dir(corpuspath + os.sep + cname + os.sep)
        corpusname = cname+str(numoffiles)
        corpus = makedataspace(inpath, numoffiles, corpusname, crootpath)

    return corpuspath, resultspath



def clustering(corpuspath, resultspath, numofclusters):
    
    trainpath = corpuspath + os.sep + "train" + os.sep
    testpath = corpuspath + os.sep + "test" + os.sep
    
    
    # feature hold out!
    featurespaces = IOtools.getfilenames_of_dir(trainpath, removeextension=True)
    for featurespace in featurespaces:
        inpath = trainpath
        procedurename = "kmeans#"+str(numofclusters)+"_"+featurespace
        recordpath = resultspath
        classify.perform_clustering(featurespace, inpath, procedurename, recordpath, numofclusters)


def classification(corpuspath, resultspath):
    trainpath = corpuspath + os.sep + "train" + os.sep
    testpath = corpuspath + os.sep + "test" + os.sep
    
    classifiers = ["naivebayes", "ldac"]
    learner = {}
    learner["naivebayes"] = classify.NBclassifier()
    learner["ldac"] = classify.LDACclassifier()
    # feature hold out!
    featurespaces = IOtools.getfilenames_of_dir(trainpath, removeextension=True)
    
    
    for featurespace in featurespaces: 
        trainset = pd.read_csv(trainpath+os.sep+featurespace+".csv", index_col=0)
        testset = pd.read_csv(testpath+os.sep+featurespace+".csv", index_col=0)     
        for clsfalg in classifiers:
            procedurename = clsfalg+"#_"+featurespace
            recordpath = IOtools.ensure_dir(resultspath + os.sep + procedurename)
            learner[clsfalg].setname(procedurename)
            learner[clsfalg].run(trainset, testset, recordpath)
   
    
    '''
    for featurespace in featurespaces:      
        for classifiername in classifiers:
            procedurename = classifiername+"#_"+featurespace
            recordpath = IOtools.ensure_dir(resultspath + os.sep + procedurename)
            nbclassifier = classify.NBclassifier(procedurename)
            trainset = pd.read_csv(trainpath+os.sep+featurespace+".csv", index_col=0)
            testset = pd.read_csv(testpath+os.sep+featurespace+".csv", index_col=0)
            nbclassifier.run(trainset, testset, recordpath)
    '''        
        
               
    
'''
def classification():
    corpuspath = prepare_experiment(inputpath, outputpath, ename, numoftrain, percentagesplit=10, cmetadata=None)

    trainpath = corpuspath + os.sep + "train" + os.sep
    testpath = corpuspath + os.sep + "test" + os.sep
    model = classifier.train(trainsdf(trainpath))
    predictions = classifier.predict(model, testdf(testpath))
    performance(predictions, testdf)
    plot()
    
''' 

if __name__ == "__main__":
    
    outputpath = "/home/dicle/Dicle/Tez/tests/"
    inputrootpath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/categorize/"
    trialname = "test5s00"
    numoftrain = 500
    
    # DATA PROCESSING
    '''
    start = datetime.now()
    corpuspath, resultspath = prepare_experiment(inputrootpath, outputpath, trialname, numoftrain, percentagesplit=10, cmetadata=None)
    end = datetime.now()
    print "dataspace make: ",str(end-start)
    '''
    
    
    corpuspath = outputpath + os.sep + trialname + os.sep + "data" + os.sep
    resultspath = outputpath + os.sep + trialname + os.sep + "results" + os.sep
    
    
    # CLASSIFICATION
    #classification(corpuspath, resultspath)
    
    
    '''
    # CLUSTERING
    
    end = datetime.now()
    for i in [2,3,4]:
        start = end
        clustering(corpuspath, resultspath, numofclusters=i)
        end = datetime.now()
        print "perform cluster ",i," : ",str(end-start)
    '''    
        
    '''
    sets = ["train", "test"]
    numoftest = 60
    numoftrain = 600
    
    outrootpath = "/home/dicle/Dicle/Tez/tests/"
    outpath = IOtools.ensure_dir(outrootpath+os.sep+"test-"+str(numoftest))
    
    '''
    
    
    
    
    # VISUALIZE
    ''' scatter data points  '''
    
    for settype in ["train", "test"]:
        print settype
        dataset = pd.read_csv(corpuspath + os.sep + settype + os.sep + "featurematrix1.csv", index_col=0)
        featurelist = dataset.columns.values.tolist()
        classify.scatter_data_points(dataset, featurelist, targetfeat="label")
       
        
    '''  scatter predictions 
    path = resultspath + os.sep + "naivebayes#_featurematrix1" +os.sep
    dataset = pd.read_csv(path + os.sep + "naivebayes#_featurematrix1_predictions.csv", index_col=0)
    featurelist = dataset.columns.values.tolist()
    classify.scatter_data_points(dataset, featurelist, targetfeat="predicted")  
    '''
    
    