'''
Created on May 21, 2013

@author: dicle
'''



import numpy as np
import os
import codecs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sentimentfinding import IOtools, plotter
import classfhelpers


# confusion matrix plot ratios. plot classifier comparison by class label
class Experiment:
    experimentname = ""
    folder = ""
    
    numofdatapoints = 0
    infeatures = []
    allfeatures = {}   #  { featureindex : featurename } where featureindex is the appearence order of the featurename in the matrix 
    classes = []
    
    numoftestcases = 0
    
    confusionmatrix = None
    catmetrics = {}
    accuracy = 0.0
    
    def __init__(self, ename, path, trainpath, N=0, outfeatures=[]):   # outfeatures contain the indices of the features to be excluded from matrix
        self.experimentname = ename
        self.folder = IOtools.ensure_dir(path+os.sep+self.experimentname)
        
        self.numoftestcases = 0
        
        classlabels, featurenames, numofpoints = classfhelpers.get_matrix_metadata(trainpath)
        
        self.classes = classlabels
        self.numofdatapoints = numofpoints
        if N > 0:
            self.numofdatapoints = N
        
        # handle feature choosing
        for i,fname in enumerate(featurenames):
            self.allfeatures[i] = fname
        self.infeatures = list(set(range(len(featurenames))) - set(outfeatures))
        
        self.experimentname += "_N-"+str(self.numofdatapoints)+"_F-"+str(len(outfeatures))+"out_"
        
    
    '''
    def prepare_data(self, matrixpath, N=0, featureindices=None):
        lines = IOtools.readtextlines(matrixpath)
        
        classinfo = lines[0]
        self.classes = classinfo.split(",")
        self.classes = map(lambda x : x.strip(), self.classes)
        classencoding, _ = classfhelpers.classlabelindicing(self.classes)

        header = lines[1]
        items = header.split("\t")
        # HANDLE CHOOSING FEATURES - featureindices, how to pass them as indices or names.. ??? 
        self.features = items[1:-1]
        
        if N == 0:
            self.numofdatapoints = len(lines)-2
        else:
            self.numofdatapoints = N
        
        X = []
        Y = []
        for i in range(2,self.numofdatapoints):
            items = lines[i].split()
            classlabelindicing = classencoding[items[-1]]    # class encoding
            values = [float(val) for val in items[1:-1]]
            X.append(values)
            Y.append(classlabelindicing)
    
        X = np.array(X)
        Y = np.array(Y)    
        
        return X, Y
    '''
    
    def train(self, traindata_path, classifier):
        trainX, trainY = classfhelpers.get_matrix(traindata_path, N=self.numofdatapoints)
        
        if len(self.infeatures) != len(self.allfeatures):
            newX = []
            for i in self.infeatures:
                newX.append(trainX[:,i])
            trainX = np.array(newX)
            trainX = trainX.T
        
        self.experimentname += "C-"+classifier.classifiername+"_"
        predictor = classifier.train(trainX, trainY)
        return predictor
    
    
    # testdata_path yerine direkt matrix almak daha dogru olabilir
    def predict(self, testdata_path, classifier, predictormodel):
        testX, testY = classfhelpers.get_matrix(testdata_path)    #, N=self.numofdatapoints)
        
        if len(self.infeatures) != len(self.allfeatures):
            newX = []
            for i in self.infeatures:
                newX.append(testX[:,i])
            testX = np.array(newX)
            testX = testX.T
            
        predictions = classifier.test(testX, testY, predictormodel)
        self.find_confusion_matrix(testY, predictions)
        
        self.numoftestcases += 1
        self.experimentname += "test"+str(self.numoftestcases)
    
    
    def report_results(self):
        self.compute_precision()
        self.compute_recall()
        self.compute_fmeasure()
        self.compute_accuracy()
        
        IOtools.todisc_matrix(self.confusionmatrix, self.folder+os.sep+self.experimentname+".confmat")
        
        
        
        f = codecs.open(self.folder+os.sep+self.experimentname+".results", "a", encoding='utf8')
        # write report as list not to keep the whole string in memory
        header = "\t" + "\t".join(self.catmetrics.keys()) +"\n"
        f.write(header)
        
        labelencoding, _ = classfhelpers.classlabelindicing(self.classes)    # labeldecoding contains indices
        for c in self.classes:
            i = labelencoding[c]
            line = []
            line.append(c)
            for metricname in self.catmetrics.keys():
                line.append(self.catmetrics[metricname][i])
            line = map(lambda x : str(x), line)
            outstr = "\t".join(line) + "\n"
            f.write(outstr)
        f.write("\nAccuracy: "+str(self.accuracy))
        f.close()
        
    def plot_scores(self):
        plotter.set_plotframe("scores", "metrics", "value")
        labelencoding, _ = classfhelpers.classlabelindicing(self.classes)
        
        xitems = self.catmetrics.keys()
        colors = plotter._get_colors(len(self.classes))
        for c in self.classes:
            i = labelencoding[c]
            yitems = []
            for metricname in xitems:
                yitems.append(self.catmetrics[metricname][i])
            plotter.plot_line(xitems, yitems, linelabel=c, clr=colors[i])
        
        plt.legend()
        plt.savefig(self.folder+os.sep+self.experimentname+"_scores.png")
        plt.clf()
        #plt.show()
            
            
    
    def compute_precision(self):
        labelencoding, _ = classfhelpers.classlabelindicing(self.classes)    # labeldecoding contains indices
        self.catmetrics["precision"] = range(len(self.classes))
        
        for c in self.classes:
            i = labelencoding[c]
            truepositive = self.confusionmatrix[i][i]
            numofrelevants = sum(self.confusionmatrix[:,i])
            self.catmetrics["precision"][i] = float(truepositive) / numofrelevants 
            
        
    
    def compute_accuracy(self):
        self.accuracy = sum(np.diag(self.confusionmatrix)) / float(sum(sum(self.confusionmatrix)))
        
        
    def compute_recall(self):
        labelencoding, _ = classfhelpers.classlabelindicing(self.classes)    # labeldecoding contains indices
        self.catmetrics["recall"] = range(len(self.classes))
        
        for c in self.classes:
            i = labelencoding[c]
            truepositive = self.confusionmatrix[i][i]
            numofactuals = sum(self.confusionmatrix[i,:])
            self.catmetrics["recall"][i] = float(truepositive) / numofactuals 
            
            
        
    def compute_fmeasure(self):
        labelencoding, _ = classfhelpers.classlabelindicing(self.classes)    # labeldecoding contains indices
        self.catmetrics["fmeasure"] = range(len(self.classes))
        
        for c in self.classes:
            i = labelencoding[c]
            precision = self.catmetrics["precision"][i]
            recall = self.catmetrics["recall"][i]
            self.catmetrics["fmeasure"][i] = 2 * ((precision * recall) / (precision + recall))
            
            
      
    
    
    def find_confusion_matrix(self, actuallabels, predictedlabels):
        _, labeldecoding = classfhelpers.classlabelindicing(self.classes)
        self.confusionmatrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)
    
        print labeldecoding
        for actuallabel,predictedlabel in zip(actuallabels, predictedlabels):
            print actuallabel,"   ",predictedlabel 
            self.confusionmatrix[actuallabel, predictedlabel] += 1
        
        
###  end experiment


from nltk import NaiveBayesClassifier, classify

class NBclassifier:
    
    classifiername = ""
    #model = None
    
    def __init__(self):
        self.classifiername = "NaiveBayes"
        #self.model = None
    
    def nltkfeatureset(self, datapoint):
        features = ["ADJ", "ADV", "SUBJ"]
        featureset = {}
        for f, val in zip(features, datapoint):
            featureset[f] = val
        return featureset
        

    def train(self, X, Y):
        trainset = [(self.nltkfeatureset(point), label) for (point, label) in zip(X,Y)]
        model = NaiveBayesClassifier.train(trainset)
        return model
    
    
    def test(self, testpoints, testlabels, model):
        testset = [(self.nltkfeatureset(point), label) for (point, label) in zip(testpoints, testlabels)]
        predictions = []
        for (point, label) in testset:
            predicted = model.classify(point)
            predictions.append(predicted)
        return predictions
    
    

    
###  end naivebayes
        
        

import mlpy
      
class LDACclassifier:
    
    classifiername = ""
    #model = None
    
    def __init__(self):
        self.classifiername = "LDAC"
        #self.model = None
        
         
    # x is data point and y is class labels (x list of lists, y list)
    # there are 3 classes
    def train(self, x, y):
        x1, x2, x3 = [], [], []
        
        labels = list(set(y))
        
        for data,cls in zip(x,y):
            if cls == labels[0]:
                x1.append(data)
            elif cls == labels[1]:
                x2.append(data)
            else:
                x3.append(data)
        
        x1 = np.array(x1)       
        x2 = np.array(x2)
        x3 = np.array(x3)
            
        ldac = mlpy.LDAC()
        ldac.learn(x, y)
        
        w = ldac.w()
        print w
        
        b = ldac.bias()
        print "Bias:\n",b
        
        # plot
        '''
        xx = np.arange(np.min(x[:,0]), np.max(x[:,0]), 0.01)
        yy1 = (xx * (w[1][0] - w[0][0]) + b[1] - b[0]) / (w[0][1] - w[1][1])
        yy2 = (xx * (w[2][0] - w[0][0]) + b[2] - b[0]) / (w[0][1] - w[2][1])
        yy3 = (xx * (w[2][0] - w[1][0]) + b[2] - b[1]) / (w[1][1] - w[2][1])
        
        fig = plt.figure(1)
        
        plotdata = plt.plot(x1[:,0], x1[:,1], 'ob', x2[:,0], x2[:,1], 'or', x3[:,0], x3[:,1], 'og')
        
        plot1 = plt.plot(xx, yy1, '--k')
        plot2 = plt.plot(xx, yy2, '--k')
        plot3 = plt.plot(xx, yy3, '--k')
        '''
        #plt.show()
        
        
        return ldac
    
    
    def test(self, testpoints, testlabels, model):
        predictions = model.pred(testpoints)
        return predictions    
   
### end ldac 


def conduct_experiments(classifierlist, task_data_pairs, rootresultpath):
    experiments = []
    for classifier in classifierlist:
        for task, pathpairs in task_data_pairs.iteritems():
            for pathpair in pathpairs:
                trainpath, testpath = pathpair
                experiment = Experiment("L_"+task, rootresultpath, trainpath)
                model = experiment.train(trainpath, classifier)
                experiment.predict(testpath, classifier, model)
                experiment.report_results()
                experiment.plot_scores()
                experiments.append(experiment)
            
    return experiments

def plot_experiments(experimentlist):
    return

def scatter_data_points(matrixpath, N=0):
    classlabels, featurenames, numofpoints = classfhelpers.get_matrix_metadata(matrixpath)
    if N > 0:
        numofpoints = N
    X, Y = classfhelpers.get_matrix(matrixpath, N=numofpoints)
    
    classlabelencode, classlabeldecode = classfhelpers.classlabelindicing(classlabels)
    colors = ["b", "g", "r"]
    
    labelmatrix = {}
    
    for cname, cindex in classlabelencode.iteritems():
        labelmatrix[cindex] = list()
    
    for cname, cindex in classlabelencode.iteritems():    
        matrix = []
        for (point, label) in zip(X, Y):
            labelmatrix[label] += [point]
    
    fig = plt.figure()
    ax = Axes3D(fig)
    for lbl, matrices in labelmatrix.iteritems():
        print lbl," ",classlabeldecode[lbl]
        
        matrices = np.array(matrices)
        print matrices[:,0],"  ",type(matrices)
        
        ax.plot(matrices[:,0], matrices[:,1], matrices[:,2], 'o', c = colors[lbl], label=classlabeldecode[lbl])
        
        '''
        for matrix in matrices:
            #print matrix[0], matrix[1], matrix[2],"  ",type(matrix[2])
            #ax.scatter(matrix[0], matrix[1], matrix[2], c = colors[lbl], label=classlabeldecode[lbl])
            ax.plot([matrix[0]], [matrix[1]], [matrix[2]], marker='o', c = colors[lbl], label=classlabeldecode[lbl])
        '''

    ax.set_xlabel(featurenames[0])
    ax.set_ylabel(featurenames[1])
    ax.set_zlabel(featurenames[2])
    
    ax.legend(loc = 'upper left')
    fig.add_axes(ax)
    
    plt.show()
    
    '''
    for label, matrix in labelmatrix.iteritems():
        print classlabeldecode[label]," "
        print matrix 
    '''

if __name__ == "__main__":
    
    ldacclassifier = LDACclassifier()
    naivebayesclassifier = NBclassifier()
    classifierlist = []
    classifierlist.append(ldacclassifier)
    classifierlist.append(naivebayesclassifier)
    
    
    tr1 = "/home/dicle/Dicle/Tez/output/CLASSTEST/featureMATRIX-3cat-trainn-4500texts.m"
    ts1 = "/home/dicle/Dicle/Tez/output/CLASSTEST/labelresourcefeatureMATRIX-3cat-testn-450texts.m"    
    tr2 = "/home/dicle/Dicle/Tez/output/CLASSTEST/t600.m"
    ts2 = "/home/dicle/Dicle/Tez/output/CLASSTEST/t60.m"
    trmatrix1 = "/home/dicle/Dicle/Tez/output/CLASSTEST/labelresourcefeatureMATRIX-3cat-trainn-4500texts.m"
    testmatrix1 = "/home/dicle/Dicle/Tez/output/CLASSTEST/labelresourcefeatureMATRIX-3cat-testn-450texts.m"
    trmatrix2 = "/home/dicle/Dicle/Tez/output/CLASSTEST/labelresourcet600.m"
    testmatrix2 = "/home/dicle/Dicle/Tez/output/CLASSTEST/labelresourcet60.m"
    
    task_data_paths = {"resourceclass" : [(trmatrix1, testmatrix1), (trmatrix2, testmatrix2)], "catclass" : [(tr2, ts2), (tr1, ts1)]}
    rootresultpath = IOtools.results_rootpath+os.sep+"classificationresults"
    
    
    #conduct_experiments(classifierlist, task_data_paths, rootresultpath)
    
    scatter_data_points(trmatrix1)   #, N=1000)
    
    
    
    
    
    '''
    experiment1 = Experiment("L_resource", rootresultpath, trmatrix1)
    
    task2 = "L_cat"
    experiment2 = Experiment(task2, rootresultpath, tr2)
    
    ldacmodel = experiment2.train(tr2, ldacclassifier)
    experiment2.predict(ts2, ldacclassifier, ldacmodel)
    experiment2.report_results()
    '''
    
    
    


