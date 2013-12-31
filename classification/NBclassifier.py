'''
Created on May 21, 2013

@author: dicle
'''

import os
from nltk import NaiveBayesClassifier, classify


from sentimentfinding import IOtools



class NBclassifier:
    
    classifiername = ""
    model = None
    
    def __init__(self):
        self.classifiername = "NaiveBayes"
        self.model = None
    
    def nltkfeatureset(self, datapoint):
        features = ["ADJ", "ADV", "SUBJ"]
        featureset = {}
        for f, val in zip(features, datapoint):
            featureset[f] = val
        return featureset
        

    def train(self, X, Y):
        trainset = [(self.nltkfeatureset(point), label) for (point, label) in zip(X,Y)]
        self.model = NaiveBayesClassifier.train(trainset)
        return self.model
    
    
    def test(self, testpoints, testlabels):
        testset = [(self.nltkfeatureset(point), label) for (point, label) in zip(testpoints, testlabels)]
        predictions = []
        for (point, label) in testset:
            predicted = self.model.classify(point)
            predictions.append(predicted)
        return predictions
    
    
    def test2(self, testpoints, testlabels, classlabel_decode, nbclassifier, filename):
        testset = [(self.nltkfeatureset(point), label) for (point, label) in zip(testpoints, testlabels)]
        print "TEST SET"
        print testset
        
        out = ""
        
        accuracy = classify.accuracy(nbclassifier, testset)
        out += "Prediction"
        out += "\nNumber of test data: " + str(len(testpoints))
        out += "Accuracy: " + str(accuracy)
        out += "\nPredicted \t Actual \n"
            
        for (point, label) in testset:
            predicted = nbclassifier.classify(point)
            out += classlabel_decode[predicted] + " \t " + classlabel_decode[label] + "\n"
        
        IOtools.todisc_txt(out, IOtools.results_rootpath+os.sep+filename)
        
        
        