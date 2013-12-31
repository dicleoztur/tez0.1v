'''
Created on May 8, 2013

@author: dicle
'''

import numpy as np
import matplotlib.pyplot as plt
import mlpy
import os
import re

import nltk

from sentimentfinding import IOtools
from sentimentfinding import plotter

trainpath = IOtools.results_rootpath+"/MATRIX600texts.txt"
testpath = IOtools.results_rootpath+"/MATRIX63texts.txt"


resourcenames = {"cumhuriyet" : 1, "vakit" : 2, "radikal" : 3}
#resource_decode = {1: "cumhuriyet", 2: "vakit", 3: "radikal" }

# too specialized for MATRIX files 



def classlabelindicing(labels):
    classlabels = {}
    classlabels_decode = {}
    for i,label in enumerate(labels):
        classlabels[label] = i
        classlabels_decode[i] = label
    return classlabels, classlabels_decode



def prepare_data(path, classlabels, header=False):
    
    datapoints = IOtools.readtextlines(path)
    X = []
    Y = []
    if header: datapoints = datapoints[1:]
    for line in datapoints:
        items = line.split()
        classlabelindicing = classlabels[items[-1]]    # class encoding
        values = [float(val) for val in items[1:-1]]
        X.append(values)
        Y.append(classlabelindicing)
    
    X = np.array(X)
    Y = np.array(Y)    
    return X, Y



'''
the format of the input resultfile is
line0 prediction
line1 Number of test data: 
line2 Predicted    Actual
line3 predictedclass    actualclass
...

the input classlabeldecoding is dict containing {classlabelindicing : classno } as decoding
the function extracts the confusion matrix, draws prediction rates by class labels

'''
def classification_results(experimentname, resultfile, classlabeldecoding):
    results = IOtools.readtextlines(IOtools.results_rootpath+os.sep+resultfile)
    
    numofpoints = int(results[1].split(":")[1])
    print results[2]," ",results[3],"  ",numofpoints
    predictions = results[3 : (numofpoints+3)]
    print len(predictions)
    confusionmatrix = np.zeros((len(classlabeldecoding), len(classlabeldecoding)))
    
    for i,prediction in enumerate(predictions):
        #items = prediction.split("\t")
        items = re.split(r"\s+", prediction)
        items = [str(item).strip() for item in items]
        predicted = items[0]
        actual = items[1]
        
        print i,"  ",prediction," ~~ ",items
        
        confusionmatrix[classlabeldecoding[predicted], classlabeldecoding[actual]] += 1
    
    IOtools.todisc_matrix(confusionmatrix.tolist(), IOtools.matrixpath+os.sep+experimentname+"ConfMat.m")
    
    
    # plot confusion matrix
    xitems = [0 for i in range(len(classlabeldecoding))]
    for k,v in classlabeldecoding.iteritems():
        xitems[v] = k
        
    
    classlabeldecoding.keys()
    colors = plotter._get_colors(confusionmatrix.shape[0])
    for k,v in classlabeldecoding.iteritems():
        plotter.plot_line(xitems, confusionmatrix[v, :], linelabel=k, clr=colors[v])
        print xitems," ",k,"  : ",v
    
    
    plotter.plot_line(xitems, confusionmatrix.diagonal().tolist(), linelabel="target", clr="k")
    plt.xlabel("actual")
    plt.ylabel("predicted")
    plt.legend()
    plt.savefig(IOtools.img_output+os.sep+experimentname+"ConfMat.png")
    plt.show()
        

# multiclass data example from http://mlpy.sourceforge.net/docs/3.5/lin_class.html#linear-discriminant-analysis-classifier-ldac
def ldac_classifier():
    np.random.seed(0)
    
    mean1, cov1, n1 = [1, 25], [[1,1],[1,2]], 200  # 200 samples of class 0
    x1 = np.random.multivariate_normal(mean1, cov1, n1)
    y1 = np.zeros(n1, dtype=np.int)
    
    mean2, cov2, n2 = [2.5, 22.5], [[1,0],[0,1]], 300 # 300 samples of class 1
    x2 = np.random.multivariate_normal(mean2, cov2, n2)
    y2 = np.ones(n2, dtype=np.int)
    
    mean3, cov3, n3 = [5, 28], [[0.5,0],[0,0.5]], 200 # 200 samples of class 2
    x3 = np.random.multivariate_normal(mean3, cov3, n3)
    y3 = 2 * np.ones(n3, dtype=np.int)
    
    x = np.concatenate((x1, x2, x3), axis=0)
    y = np.concatenate((y1, y2, y3))
    
    ldac = mlpy.LDAC()
    ldac.learn(x, y)
    
    w = ldac.w()
    print w
    
    b = ldac.bias()
    print "Bias:\n",b
    
    print "X"
    print x
    print "Y"
    print y
    
    # plot
    xx = np.arange(np.min(x[:,0]), np.max(x[:,0]), 0.01)
    yy1 = (xx * (w[1][0] - w[0][0]) + b[1] - b[0]) / (w[0][1] - w[1][1])
    yy2 = (xx * (w[2][0] - w[0][0]) + b[2] - b[0]) / (w[0][1] - w[2][1])
    yy3 = (xx * (w[2][0] - w[1][0]) + b[2] - b[1]) / (w[1][1] - w[2][1])
    
    fig = plt.figure(1)
    
    plotdata = plt.plot(x1[:,0], x1[:,1], 'ob', x2[:,0], x2[:,1], 'or', x3[:,0], x3[:,1], 'og')
    
    plot1 = plt.plot(xx, yy1, '--k')
    plot2 = plt.plot(xx, yy2, '--k')
    plot3 = plt.plot(xx, yy3, '--k')
    
    plt.show()
    

# x is data point and y is class labels (x list of lists, y list)
# there are 3 classes
def LDAclassify(x, y):
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
    xx = np.arange(np.min(x[:,0]), np.max(x[:,0]), 0.01)
    yy1 = (xx * (w[1][0] - w[0][0]) + b[1] - b[0]) / (w[0][1] - w[1][1])
    yy2 = (xx * (w[2][0] - w[0][0]) + b[2] - b[0]) / (w[0][1] - w[2][1])
    yy3 = (xx * (w[2][0] - w[1][0]) + b[2] - b[1]) / (w[1][1] - w[2][1])
    
    fig = plt.figure(1)
    
    plotdata = plt.plot(x1[:,0], x1[:,1], 'ob', x2[:,0], x2[:,1], 'or', x3[:,0], x3[:,1], 'og')
    
    plot1 = plt.plot(xx, yy1, '--k')
    plot2 = plt.plot(xx, yy2, '--k')
    plot3 = plt.plot(xx, yy3, '--k')
    
    #plt.show()
    return ldac


def test_classifier(testdata, testlabels, predictions, resource_decode, recordfilename):
    out = ""  
    out += "Prediction"
    out += "\nNumber of test data: " + str(len(testdata))
    out += "\nActual \t Predicted\n"
    
    errors = 0
    for p,a in zip(predictions, testlabels):
        #print str(p)+" \t "+str(a)
        out += resource_decode[p] + " \t " + resource_decode[a] + "\n"
        if str(p) != str(a):
            errors += 1
    errorrate = float(errors) / len(testdata)
    out += "\nAccuracy: " + str(1-errorrate)
    out += "\nError rate: " + str(errorrate)
    
    IOtools.todisc_txt(out, IOtools.results_rootpath+os.sep+recordfilename)

def test_LDAclassifier(testpoints, testlabels, classlabel_decode, ldac, filename):
    out = ""  
    out += "Prediction"
    out += "\nNumber of test data: " + str(len(testpoints))
    out += "\nPredicted \t Actual \n"
    predictions =  ldac.pred(testpoints)
    errors = 0
    for p,a in zip(predictions, testlabels):
        #print str(p)+" \t "+str(a)
        out += classlabel_decode[p] + " \t " + classlabel_decode[a] + "\n"
        if str(p) != str(a):
            errors += 1
    errorrate = float(errors) / len(testpoints)
    out += "\nError rate: " + str(errorrate)
    
    IOtools.todisc_txt(out, IOtools.results_rootpath+os.sep+filename)
    '''
    print "Prediction"
    print "Actual \t Predicted"
    predictions =  ldac.pred(testpoints)
    errors = 0
    for p,a in zip(predictions, testlabels):
        #print str(p)+" \t "+str(a)
        print resource_decode[p] + " \t " + resource_decode[a]
        if str(p) != str(a):
            errors += 1
    print "Error rate: ", str(float(errors) / len(testpoints))
    '''
    #print ldac.labels()


def nltkfeatureset(datapoint):
    features = ["ADJ", "ADV", "SUBJ"]
    featureset = {}
    for f, val in zip(features, datapoint):
        featureset[f] = val
    return featureset
        

def naivebayesClassify(X, Y):
    trainset = [(nltkfeatureset(point), label) for (point, label) in zip(X,Y)]
    NBclassifier = nltk.NaiveBayesClassifier.train(trainset)
    return NBclassifier

def test_naivebayesclassifier(testpoints, testlabels, classlabel_decode, nbclassifier, filename):
    testset = [(nltkfeatureset(point), label) for (point, label) in zip(testpoints, testlabels)]
    print "TEST SET"
    print testset
    
    out = ""
    
    accuracy = nltk.classify.accuracy(nbclassifier, testset)
    out += "Prediction"
    out += "\nNumber of test data: " + str(len(testpoints))
    out += "Accuracy: " + str(accuracy)
    out += "\nPredicted \t Actual \n"
        
    for (point, label) in testset:
        predicted = nbclassifier.classify(point)
        out += classlabel_decode[predicted] + " \t " + classlabel_decode[label] + "\n"
    
    IOtools.todisc_txt(out, IOtools.results_rootpath+os.sep+filename)
    

def cluster(trainX, testX, testlabels, N=2):
    clusterer = nltk.cluster.KMeansClusterer(N, nltk.cluster.euclidean_distance)
    clusterer.cluster(trainX, assign_clusters=True)
    for point,lbl in zip(testX,testlabels):
        print clusterer.classify(point)," \t ",lbl



def experiment(experimentname, trainX, trainY, testX, testY,  labeldecodes, classifierfunc, testerfunc):
    '''
    trainX, trainY = prepare_data(IOtools.matrixpath+os.sep+trainerid+".m")    
    testX, testY = prepare_data(IOtools.matrixpath+os.sep+testerid+".m")
    '''
    numofdatapoints, numoffeatures = trainX.shape
    classifier = classifierfunc(trainX, trainY)
    
    
    testerfunc(testX, testY, labeldecodes, classifier, experimentname+".txt")
    


# concrete later!
# confusion matrix plot ratios. plot classifier comparison by class label
class Experiment:
    experimentname = ""
    resultfile = ""
    
    def classify(self):
        return
    def classificationresults(self):
        return
   

if __name__ == "__main__":
    
    
    experimentname = "NB4500"
    resultfile = "NBpredict-3cat-4500texts.txt"
    labels = ["dunya", "turkiye", "spor"]
    classlabels, classlabels_decode = classlabelindicing(labels)
    classification_results(experimentname, resultfile, classlabels)
    
    
    experimentname = "LDA4500"
    resultfile = "LDApredict-3cat-4500texts.txt"
    labels = ["dunya", "turkiye", "spor"]
    classlabels, classlabels_decode = classlabelindicing(labels)
    classification_results(experimentname, resultfile, classlabels)
    
    
    
    #ldac_classifier()
    '''
    file_ldaresults = "LDAresults" 
    file_nbresults = "NBresults"
    
    trainX, trainY = prepare_data(trainpath)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX, testY = prepare_data(testpath)
    testX = np.array(testX)
    testY = np.array(testY)
    
    print trainX
    print
    print trainY
    
    # feature holdout
    numofdatapoints, numoffeatures = trainX.shape
    indices = set(range(numoffeatures))
    for i in range(numoffeatures):
        takenfeatures = list(indices - set([i]))
        heldout_trainx = trainX[:,takenfeatures]
        heldout_testx = testX[:,takenfeatures]
        classifier = LDAclassify(heldout_trainx, trainY)
        testout = file_ldaresults+"_"+str(len(testY))+"MATRIX"+"takeout"+str(i)+".txt"
        test_LDAclassifier(heldout_testx, testY, classifier, testout)
        
    '''
    
    '''
    print "max: ",np.min(trainX[:,0])
    classifier = LDAclassify(trainX, trainY)
    
    
    
    testout = "ldacPredict_MATRIX63texts.txt"
    test_LDAclassifier(testX, testY, classifier, testout)
    
    
    nbclassifier = naivebayesClassify(trainX, trainY)
    testout2 = "nbPredict_MATRIX63texts.txt" 
    test_naivebayesclassifier(testX, testY, nbclassifier, testout2)
    '''
    
    # test'te fileid print
    
    #cluster(trainX, testX, testY, N=3)
    
    
    
    
    
    
    
    
    