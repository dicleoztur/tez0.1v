'''
Created on May 21, 2013

@author: dicle
'''

import numpy as np
import mlpy
import os

from sentimentfinding import IOtools

class LDACclassifier:
    
    classifiername = ""
    model = None
    
    def __init__(self):
        self.classifiername = "LDAC"
        self.model = None
        
         
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
        
        self.model = ldac
        return self.model
    
    
    def test(self, testpoints, testlabels):
        predictions = self.model.pred(testpoints)
        return predictions
    
    def test2(self, testpoints, testlabels, classlabel_decode, ldac, filename):
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
        
        
        
        
    
    