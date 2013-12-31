'''
Created on Sep 15, 2013

@author: dicle
'''

import math
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sentimentfinding import IOtools, plotter



####   text metrics    ####
def compute_tfidf(freqdfpath, tfidfpath):
    doctermfreq = pd.read_csv(freqdfpath, index_col=0)
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
    #doctermframe.to_csv(self.rootpath+os.sep+"matrix"+"doctermTFIDF.csv")
    doctermframe.to_csv(tfidfpath)
    return doctermframe







####    learning metrics    ####


class performance:
    experimentname = ""
    recordpath = ""
    
    classes = []
    
    confusionmatrix = None
    catmetrics = {}
    accuracy = 0.0
    
    
    def __init__(self, ename, path, categories):
        self.experimentname = ename
        self.recordpath = path
        
        self.classes = categories
        
        self.confusionmatrix = pd.DataFrame(np.zeros((len(self.classes), len(self.classes)), dtype=int), index=self.classes, columns=self.classes)
        
        scorenames = ["precision", "recall", "fmeasure"]
        self.catmetrics = pd.DataFrame(np.zeros((len(scorenames), len(self.classes)), dtype=float), index=scorenames, columns=self.classes)
        self.accuracy = 0.0
    
    
    
    def find_confusion_matrix(self, actuallabels, predictedlabels):
           
        for actuallabel,predictedlabel in zip(actuallabels, predictedlabels):
            #print actuallabel,"   ",predictedlabel 
            self.confusionmatrix.loc[actuallabel, predictedlabel] += 1
        
        self.confusionmatrix.to_csv(self.recordpath + os.sep + self.experimentname + "_confmat.csv")
          
        
    def compute_precision(self):
        
        for c in self.classes:     
            truepositive = self.confusionmatrix.loc[c, c]
            numofrelevants = sum(self.confusionmatrix.loc[:, c])
            self.catmetrics.loc["precision", c] = float(truepositive) / numofrelevants 
            
        
    
    def compute_accuracy(self):
        confmat = self.confusionmatrix.values
        self.accuracy = sum(np.diag(confmat)) / float(sum(sum(confmat)))
        
        
        
    def compute_recall(self):
 
        for c in self.classes:
            truepositive = self.confusionmatrix.loc[c, c]
            numofactuals = sum(self.confusionmatrix.loc[c, :])
            self.catmetrics.loc["recall", c] = float(truepositive) / numofactuals 
            
            
        
    def compute_fmeasure(self):
 
        for c in self.classes:
            precision = self.catmetrics.loc["precision", c]
            recall = self.catmetrics.loc["recall", c]
            self.catmetrics.loc["fmeasure", c] = 2 * ((precision * recall) / (precision + recall))
        
    
    
    def plot_scores(self):
        plotter.set_plotframe("scores", "metrics", "value")
                
        xitems = self.catmetrics.index.values.tolist()
        colors = plotter._get_colors(len(self.classes))
        for i,c in enumerate(self.classes):
            yitems = []
            for metricname in xitems:
                yitems.append(self.catmetrics.loc[metricname, c])
            plotter.plot_line(xitems, yitems, linelabel=c, clr=colors[i])
        
        plt.legend()
        plt.savefig(self.recordpath+os.sep+self.experimentname+"_scores.png")
        plt.clf()
        #plt.show()
        
           
    def report_performance(self, actuallabels, predictedlabels):
        self.find_confusion_matrix(actuallabels, predictedlabels)
        
        self.compute_accuracy()
        self.compute_recall()
        self.compute_precision()
        self.compute_fmeasure()
        self.catmetrics.to_csv(self.recordpath+os.sep+self.experimentname+"_scores.csv")
        print self.experimentname," Accuracy: ",self.accuracy
        
        self.plot_scores()
        
# end of performance class 

        
        
        


            
            
            
            





