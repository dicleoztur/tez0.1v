'''
Created on Feb 24, 2014

@author: dicle
'''

import os
import pandas as pd
import numpy as np

import metaexperimentation, matrixhelpers
from sentimentfinding import IOtools


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
