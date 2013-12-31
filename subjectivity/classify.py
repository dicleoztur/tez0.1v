'''
Created on Sep 14, 2013

@author: dicle
'''

from nltk.cluster import KMeansClusterer, euclidean_distance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from sentimentfinding import plotter
import metrics



from nltk import NaiveBayesClassifier


# dataset is in pd?
class NBclassifier:
    
    classifiername = ""
    outpath = ""
    
    def __init__(self, procedurename=""):
        
        self.classifiername = procedurename
        self.outpath = ""
        #self.classifiername = "NaiveBayes"
        #self.model = None
    
    def setname(self, procedurename):
        self.classifiername = procedurename
    
    # featurelist contains the names of the features
    def nltkfeatureset(self, datapoint, featurelist):
        #features = ["ADJ", "ADV", "SUBJ"]
        featureset = {}
        for f, val in zip(featurelist, datapoint):
            featureset[f] = val
        return featureset
        

    def train(self, X, Y, featurelist):
        trainset = [(self.nltkfeatureset(point, featurelist), label) for (point, label) in zip(X,Y)]
        model = NaiveBayesClassifier.train(trainset)
        return model
    
    
    def test(self, testpoints, testlabels, featurelist, model):
        testset = [(self.nltkfeatureset(point, featurelist), label) for (point, label) in zip(testpoints, testlabels)]
        predictions = []
        for (point, label) in testset:
            predicted = model.classify(point)
            predictions.append(predicted)
        return predictions


    def run(self, trainset, testset, recordpath):
        
        trainpoints = trainset.iloc[:, :-1]
        trainpoints = trainpoints.values
        trainlabels = trainset.iloc[:, -1]
        
        testpoints = testset.iloc[:, :-1]
        testpoints = testpoints.values
        testlabels = testset.iloc[:, -1]
        
        cols = trainset.columns.values.tolist()
        featurelist = cols[:-1]
        
        model = self.train(trainpoints, trainlabels, featurelist)
        predictions = self.test(testpoints, testlabels, featurelist, model)
        
        self.outpath = recordpath
        
        resultspace = testset.copy(deep=True)
        resultspace["predicted"] = np.array(predictions)
        resultspace.to_csv(self.outpath + os.sep + self.classifiername+"_predictions.csv")
        
        classes = list(set(trainlabels))
        measure = metrics.performance(ename=self.classifiername, path=self.outpath, categories=classes)
        measure.report_performance(actuallabels=testlabels, predictedlabels=predictions)
        print " # most informative 3 features:"
        print model.most_informative_features(n=3)
        print 
        
    '''
    def performance(self):
        # compute accuracy
    '''    
          
    
###  end naivebayes





import mlpy
      
class LDACclassifier:
    
    classifiername = ""
    outpath = ""
    
    def __init__(self, procedurename=""):
        self.classifiername = procedurename
        self.outpath = ""
        
    def setname(self, procedurename):
        self.classifiername = procedurename
        
        
    # x is data point and y is class labels (x list of lists, y list)
    # there are 3 classes
    def train(self, x, y):            
        ldac = mlpy.LDAC()
        ldac.learn(x, y)
        
        w = ldac.w()
        print w
        
        b = ldac.bias()
        print self.classifiername, " Bias:\n",b
            
        return ldac
    
    
    def test(self, testpoints, testlabels, model):
        predictions = model.pred(testpoints)
        return predictions 


    def run(self, trainset, testset, recordpath):
        
        trainpoints = trainset.iloc[:, :-1]
        trainpoints = trainpoints.values
        trainlabels = trainset.iloc[:, -1].values.tolist()
        
        testpoints = testset.iloc[:, :-1]
        testpoints = testpoints.values
        testlabels = testset.iloc[:, -1].values.tolist()
        
        # encode classnames in int IDS
        classes = list(set(trainlabels))
        tr_instancelabelIDs, tr_ID2label = labels2IDs(trainlabels, classes)
        ts_instancelabelIDs, ts_ID2label = labels2IDs(testlabels, classes)
        
        print tr_instancelabelIDs[0],"  ",trainlabels[0]
        print tr_ID2label
        
        model = self.train(trainpoints, tr_instancelabelIDs)
        predictions = self.test(testpoints, ts_instancelabelIDs, model)
        
        # decode classIDs back to classnames
        predictions = IDs2labels(ts_instancelabelIDs, ts_ID2label)
        
        self.outpath = recordpath
        
        resultspace = testset.copy(deep=True)
        resultspace["predicted"] = np.array(predictions)
        resultspace.to_csv(self.outpath + os.sep + self.classifiername+"_predictions.csv")
        
        measure = metrics.performance(ename=self.classifiername, path=self.outpath, categories=classes)
        measure.report_performance(actuallabels=testlabels, predictedlabels=predictions)
        
        
   
### end ldac 




def labels2IDs(instancelabels, classes):
    instancelabelIDs = range(len(instancelabels))
    label2ID = {}
    ID2label = {}
    for i,classname in enumerate(classes):
        label2ID[classname] = i
        ID2label[i] = classname
        
    for i,instancelabel in enumerate(instancelabels):
        instancelabelIDs[i] = label2ID[instancelabel]
    
    return instancelabelIDs, ID2label


def IDs2labels(instancelabelIDs, ID2label):
    instancelabels = range(len(instancelabelIDs))
    
    for i,instancelblID in enumerate(instancelabelIDs):
        instancelabels[i] = ID2label[instancelblID]
    
    return instancelabels
    





def kmeans_cluster(datamatrix, numofclusters=3):
    clusterer = KMeansClusterer(numofclusters, euclidean_distance)
    groups = clusterer.cluster(datamatrix, assign_clusters=True, trace=True)
    means = clusterer.means()
    return groups, means


# dataspace is the datapoints.
# clusters contains the cluster assignments by clusterid
# cmeans is the center points of the clusters
def report_clusters(dataspace, clusters, cmeans, recordpath):
    docs = dataspace.index.values.tolist()
    resultspace = dataspace.copy(deep=True)
    resultspace["clusterid"] = np.array(clusters, dtype=int)
    features = resultspace.columns.values.tolist()

    for i,doc in enumerate(docs):
        print doc," : ",resultspace.loc[doc, "label"]," -> ",resultspace.loc[doc, "clusterid"]
    
    
    # analyse by grouping the clusterids
    clusterdocdict = {}
    clusterids = list(set(clusters))
    for i in clusterids:
        clusterdocdict[i] = list()
    for docindex,clusterid in enumerate(clusters):
        clusterdocdict[clusterid].append(docindex)
    
    for clusterid, docitems in clusterdocdict.iteritems():
        print "cluster #",clusterid
        print "  mean: ",cmeans[clusterid]
        for docindex in docitems:
            docname = docs[docindex]
            resource = resultspace.loc[docname, "label"]
            #abstractness = resultspace.loc[docname, "abstractness"]
            print " ", docname," @",resource    #,"  ",abstractness
    
        
    
    resultspace.to_csv(recordpath)     
    return resultspace


def scatter_data_points(clusteredpoints, featurelist, targetfeat="label"):
    classlabels = list(set(clusteredpoints.loc[:,targetfeat].values.tolist()))
    
    '''
    X = clusteredpoints.values
    Y = clusteredpoints.loc[:, "clusterid"]
    '''
    
    colors = ["b", "g", "r"]
    
    fig = plt.figure()
    ax = Axes3D(fig)
    
    for i,lbl in enumerate(classlabels):
        #matrix = clusteredpoints[clusteredpoints.label == label]
        matrix = clusteredpoints[clusteredpoints.loc[:, targetfeat] == lbl]
        ax.plot(matrix.loc[:, featurelist[0]], matrix.loc[:, featurelist[1]], matrix.loc[:, featurelist[2]], 'o', c=colors[i], label=lbl)
    

    ax.set_xlabel(featurelist[0])
    ax.set_ylabel(featurelist[1])
    ax.set_zlabel(featurelist[2])
    
    ax.legend(loc = 'upper left')
    fig.add_axes(ax)
    
    plt.show()
    


#classify.perform_clustering(featurespace, inpath, procedurename, recordpath)
# datapath is the path of the csv file containing data points. "label" field is assumed to be the last column
def perform_clustering(dataspacename, inputpath, experimentname, recordpath, numofclusters):
    datapath = inputpath + os.sep + dataspacename+".csv"
    dataspace = pd.read_csv(datapath, index_col=0)
    matrix = dataspace.values
    featurelist = ["abstractness", "advratio", "adjratio"]
    #### we might select some of the features as well.
    matrix = matrix[:, :-1]
    ########
    
    groups, means = kmeans_cluster(matrix, numofclusters)
    
    recordpath = recordpath + os.sep+ experimentname +"_"+"clustered.csv"
    resultspace = report_clusters(dataspace, groups, means, recordpath)
    
    '''
    # scatter points coloured by label
    scatter_data_points(dataspace, featurelist, targetfeat="label")
    
    # scatter points coloured by clusterid
    scatter_data_points(resultspace, featurelist, targetfeat="clusterid")
   '''


'''
# datapath is the path of the csv file containing data points. "label" field is assumed to be the last column
def perform_clustering(experimentname, rootpath, numofclusters):
    datapath = rootpath + os.sep + experimentname+".csv"
    dataspace = pd.read_csv(datapath, index_col=0)
    matrix = dataspace.values
    featurelist = ["abstractness", "advratio", "adjratio"]
    #### we might select some of the features as well.
    matrix = matrix[:, :-1]
    ########
    
    groups, means = kmeans_cluster(matrix, numofclusters)
    
    recordpath = rootpath + os.sep+ experimentname +"_"+str(numofclusters)+"clustered.csv"
    resultspace = report_clusters(dataspace, groups, means, recordpath)
    
    
    # scatter points coloured by label
    scatter_data_points(dataspace, featurelist, targetfeat="label")
    
    # scatter points coloured by clusterid
    scatter_data_points(resultspace, featurelist, targetfeat="clusterid")
   ''' 


if __name__ == "__main__":
    
    rootpath = "/home/dicle/Dicle/Tez/tests/test30-sept13/results/"
    ename = "featurematrix1"
    perform_clustering(ename, rootpath, numofclusters=2)
    
    