'''
Created on Apr 14, 2014

@author: dicle
'''


import os
import pandas as pd
import numpy as np

import metaexperimentation, matrixhelpers, utils
from sentimentfinding import IOtools


class PerformanceEvaluator:
    experimentspath = ""
    
    def __init__(self, expspath=metaexperimentation.experimentsrootpath):
        self.experimentspath = expspath
        self.resultspath = os.path.join(self.experimentspath, "performance")
        self.N = 50   # ranking number
        #self.metricname = basemetric
    
    def best_score_per_annottype(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
            
            annotdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
            
            p1 = os.path.join(scorepath, annottype)
            featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
                        
            for combname in featcombnames:
                
                p2 = os.path.join(p1, combname)
                labelunions = IOtools.getfoldernames_of_dir(p2)
                
                
                for labelunion in labelunions:
                
                    p3 = os.path.join(p2, labelunion)          
                    scorecsvfilepath = p3 + os.sep + metaexperimentation.scorefilename+".csv"
                    scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                    
                    rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname])
                    print rankdf.shape
                    #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                    print " ** ",annotdf.shape
                    rankdf["labelunion"] = labelunion
                    rankdf["featureset"] = combname
                    rankdf["annottype"] = annottype
                    #dflist.append(rankdf)
                    annotdf = annotdf.append(rankdf)
                    print scorecsvfile.shape
                                                       
                            #bigdf = bigdf.append(rankdf)  # bigdf header = metaexperimentation.performanceheader
                        
                        
                        #annotdf["nclasses"] = nclass
                    
                    #annotdf["featureset"] = combname
                
                #annotdf["size"] = setsize
            
            #annotdf["annottype"] = annottype  
            print " * ",annotdf.shape
            
            annotdf = matrixhelpers.get_first_N_rows(annotdf, self.N, [metricname])  
            bigdf = bigdf.append(annotdf)
            # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["annottype", metricname], ascending=False, inplace=True)
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = "best_score_per_annottype-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
    # best alg is the same. just output alg name.                                                        
                
    
    
    
    def best_score_per_labelunion(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
                       
            p1 = os.path.join(scorepath, annottype)
            featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
                        
            for combname in featcombnames:
                
                p2 = os.path.join(p1, combname)                
                labelunions = IOtools.getfoldernames_of_dir(p2)
                            
                for labelunion in labelunions:
                
                    luniondf = pd.DataFrame(columns=metaexperimentation.performanceheader)
                
                    p3 = os.path.join(p2, labelunion)          
                    scorecsvfilepath = p3 + os.sep + metaexperimentation.scorefilename+".csv"
                    scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                                                      
                    scorecsvfilepath = p3 + os.sep + metaexperimentation.scorefilename+".csv"
                    scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                    
                    rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname])
                    print rankdf.shape
                    #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                    print " ** ",luniondf.shape
                    rankdf["labelunion"] = labelunion
                    rankdf["featureset"] = combname
                    rankdf["annottype"] = annottype
                    #dflist.append(rankdf)
                    luniondf = luniondf.append(rankdf)
                    print scorecsvfile.shape
                                                       
                    luniondf = matrixhelpers.get_first_N_rows(luniondf, self.N, [metricname])  
                    bigdf = bigdf.append(luniondf)
                
                # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["labelunion", metricname], ascending=False, inplace=True)
        
        # 14Nisan added new to get top N
        bigdf = bigdf.groupby("labelunion").head(self.N)
        
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = "best_score_per_labelunion-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
 
 
    
    def best_score_per_featureset(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
                       
            p1 = os.path.join(scorepath, annottype)
            featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
                        
            for combname in featcombnames:
                
                p2 = os.path.join(p1, combname)                
                labelunions = IOtools.getfoldernames_of_dir(p2)
                
                combdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
                         
                for labelunion in labelunions:
                
                    p3 = os.path.join(p2, labelunion)          
                    scorecsvfilepath = p3 + os.sep + metaexperimentation.scorefilename+".csv"
                    scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                                                      
                    scorecsvfilepath = p3 + os.sep + metaexperimentation.scorefilename+".csv"
                    scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                    
                    rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname])
                    print rankdf.shape
                    #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                    print " ** ",combdf.shape
                    rankdf["labelunion"] = labelunion
                    rankdf["featureset"] = combname
                    rankdf["annottype"] = annottype
                    #dflist.append(rankdf)
                    combdf = combdf.append(rankdf)
                    print scorecsvfile.shape
                                                       
                combdf = matrixhelpers.get_first_N_rows(combdf, self.N, [metricname])  
                bigdf = bigdf.append(combdf)
                
                # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["featureset", metricname], ascending=False, inplace=True)
        
        # 14Nisan added new to get top N
        bigdf = bigdf.groupby("featureset").head(self.N)
        
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = "best_score_per_featureset-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
        
           
        
    
    def best_score_per_algorithm(self, metricname, scorepath=metaexperimentation.expscorepath):
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
            
            annotdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
            
            p1 = os.path.join(scorepath, annottype)
            featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
                        
            for combname in featcombnames:
                
                p2 = os.path.join(p1, combname)
                labelunions = IOtools.getfoldernames_of_dir(p2)
                
                
                for labelunion in labelunions:
                
                    p3 = os.path.join(p2, labelunion)          
                    scorecsvfilepath = p3 + os.sep + metaexperimentation.scorefilename+".csv"
                    scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                    
                    rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname])
                    print rankdf.shape
                    #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                    print " ** ",annotdf.shape
                    rankdf["labelunion"] = labelunion
                    rankdf["featureset"] = combname
                    rankdf["annottype"] = annottype
                    #dflist.append(rankdf)
                    annotdf = annotdf.append(rankdf)
                    print scorecsvfile.shape
            
                    annotdf = matrixhelpers.get_first_N_rows(annotdf, self.N, [metricname])  
                    bigdf = bigdf.append(annotdf)
            
        bigdf.sort(["algorithm", metricname], ascending=False, inplace=True)
        bigdf = bigdf.groupby("algorithm").head(self.N)
        
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = "best_score_per_algorithm-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
  
  
    
    
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





if __name__ == "__main__":
    
    evaluator = PerformanceEvaluator()
    
    metrics = ["accuracy", "fscore", "precision", "recall"]
    for metric in metrics:
        evaluator.best_score_per_algorithm(metricname=metric)
        evaluator.best_score_per_annottype(metricname=metric)
        evaluator.best_score_per_featureset(metricname=metric)
        evaluator.best_score_per_labelunion(metricname=metric)
        
    #evaluator.best_score_per_annottype(metricname="accuracy")
    #evaluator.best_score_per_labelunion(metricname="fscore")
    #evaluator.best_score_per_featureset(metricname="precision")
    #evaluator.best_score_per_algorithm(metricname="accuracy")
    
    '''
    evaluator.best_score_per_size(metricname="precision")
    evaluator.best_score_per_nclasses(metricname="precision")
    evaluator.best_score_per_dataset(metricname="accuracy")
    evaluator.best_score_per_tagger(metricname="fscore")
    '''
    
    

