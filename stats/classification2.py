'''
Created on May 21, 2013

@author: dicle
'''

import os

from sentimentfinding import IOtools

# confusion matrix plot ratios. plot classifier comparison by class label
class Experiment:
    experimentname = ""
    folder = ""
    
    def __init__(self, ename, path):
        self.experimentname = ename
        self.folder = IOtools.ensure_dir(path+os.sep+self.experimentname)
        
        
        
    def classify(self):
        return
    def classificationresults(self):
        return
   
    def 


if __name__ == "__main__":
    
    rootresultpath = IOtools.results_rootpath+os.sep+"classificationresults"
    experiment4500 = Experiment("4500trainLabelResource", rootresultpath)
    
    
    
    
    