'''
Created on Feb 24, 2014

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
            sizes = IOtools.getfoldernames_of_dir(p1)
            
            for setsize in sizes:
                p2 = os.path.join(p1, setsize)
                featcombnames = IOtools.getfoldernames_of_dir(p2)  # list of combcode_NC names
                
                for combname in featcombnames:
                    p3 = os.path.join(p2, combname)
                    #combcode, nclasses = utils.parse_featsetname(dataset)
                    nclasses = IOtools.getfoldernames_of_dir(p3)
                                        
                    for nc in nclasses:
                        
                        p4 = os.path.join(p3, nc)
                        taggings = IOtools.getfoldernames_of_dir(p4)
                        nclass = nc.split(metaexperimentation.intrafeatsep)[1]
                        
                        dflist = []
                        for taggingtype in taggings:
                        
                            p5 = os.path.join(p4, taggingtype)
                            scorecsvfilepath = p5 + os.sep + metaexperimentation.scorefilename+".csv"
                            scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                            
                            rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname])
                            print rankdf.shape
                            #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                            print " ** ",annotdf.shape
                            rankdf["tagger"] = taggingtype
                            rankdf["nclasses"] = nclass
                            rankdf["featureset"] = combname
                            rankdf["size"] = setsize
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
                
    
    def best_score_per_size(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
            
            
            
            p1 = os.path.join(scorepath, annottype)
            sizes = IOtools.getfoldernames_of_dir(p1)
            
            for setsize in sizes:
                
                sizedf = pd.DataFrame(columns=metaexperimentation.performanceheader)
                p2 = os.path.join(p1, setsize)
                featcombnames = IOtools.getfoldernames_of_dir(p2)  # list of combcode_NC names
                
                for combname in featcombnames:
                    p3 = os.path.join(p2, combname)
                    #combcode, nclasses = utils.parse_featsetname(dataset)
                    nclasses = IOtools.getfoldernames_of_dir(p3)
                                        
                    for nc in nclasses:
                        
                        p4 = os.path.join(p3, nc)
                        taggings = IOtools.getfoldernames_of_dir(p4)
                        nclass = nc.split(metaexperimentation.intrafeatsep)[1]
                        
                        dflist = []
                        for taggingtype in taggings:
                        
                            p5 = os.path.join(p4, taggingtype)
                            scorecsvfilepath = p5 + os.sep + metaexperimentation.scorefilename+".csv"
                            scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                            
                            rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname])
                            print rankdf.shape
                            #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                            print " ** ",sizedf.shape
                            rankdf["tagger"] = taggingtype
                            rankdf["nclasses"] = nclass
                            rankdf["featureset"] = combname
                            rankdf["size"] = setsize
                            rankdf["annottype"] = annottype
                            #dflist.append(rankdf)
                            sizedf = sizedf.append(rankdf)
                            print scorecsvfile.shape
                                                       

            
                sizedf = matrixhelpers.get_first_N_rows(sizedf, self.N, [metricname])  
                bigdf = bigdf.append(sizedf)
                # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["size", metricname], ascending=False, inplace=True)
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = "best_score_per_size-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
 
 
        
    
    def best_score_per_nclasses(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
            
            
            
            p1 = os.path.join(scorepath, annottype)
            sizes = IOtools.getfoldernames_of_dir(p1)
            
            for setsize in sizes:                
                
                p2 = os.path.join(p1, setsize)
                featcombnames = IOtools.getfoldernames_of_dir(p2)  # list of combcode_NC names
                
                for combname in featcombnames:
                    p3 = os.path.join(p2, combname)
                    #combcode, nclasses = utils.parse_featsetname(dataset)
                    nclasses = IOtools.getfoldernames_of_dir(p3)
                                        
                    for nc in nclasses:
                        
                        nclassdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
                        
                        p4 = os.path.join(p3, nc)
                        taggings = IOtools.getfoldernames_of_dir(p4)
                        nclass = nc.split(metaexperimentation.intrafeatsep)[1]
                        
                        for taggingtype in taggings:
                        
                            p5 = os.path.join(p4, taggingtype)
                            scorecsvfilepath = p5 + os.sep + metaexperimentation.scorefilename+".csv"
                            scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                            
                            rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname])
                            print rankdf.shape
                            #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                            print " ** ",nclassdf.shape
                            rankdf["tagger"] = taggingtype
                            rankdf["nclasses"] = nclass
                            rankdf["featureset"] = combname
                            rankdf["size"] = setsize
                            rankdf["annottype"] = annottype
                            #dflist.append(rankdf)
                            nclassdf = nclassdf.append(rankdf)
                            print scorecsvfile.shape
                                                       

            
                        nclassdf = matrixhelpers.get_first_N_rows(nclassdf, self.N, [metricname])  
                        bigdf = bigdf.append(nclassdf)
                        # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["nclasses", metricname], ascending=False, inplace=True)
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = "best_score_per_nclasses-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
        
        
    
    def best_score_per_algorithm(self):
        return
  
  
    
    def best_score_per_tagger(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
            
            
            
            p1 = os.path.join(scorepath, annottype)
            sizes = IOtools.getfoldernames_of_dir(p1)
            
            for setsize in sizes:                
                
                p2 = os.path.join(p1, setsize)
                featcombnames = IOtools.getfoldernames_of_dir(p2)  # list of combcode_NC names
                
                for combname in featcombnames:
                    p3 = os.path.join(p2, combname)
                    #combcode, nclasses = utils.parse_featsetname(dataset)
                    nclasses = IOtools.getfoldernames_of_dir(p3)
                                        
                    for nc in nclasses:                        
                                                
                        p4 = os.path.join(p3, nc)
                        taggings = IOtools.getfoldernames_of_dir(p4)
                        nclass = nc.split(metaexperimentation.intrafeatsep)[1]
                        
                        for taggingtype in taggings:
                        
                            taggerdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
                        
                            p5 = os.path.join(p4, taggingtype)
                            scorecsvfilepath = p5 + os.sep + metaexperimentation.scorefilename+".csv"
                            scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                            
                            rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname])
                            print rankdf.shape
                            #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                            print " ** ",taggerdf.shape
                            rankdf["tagger"] = taggingtype
                            rankdf["nclasses"] = nclass
                            rankdf["featureset"] = combname
                            rankdf["size"] = setsize
                            rankdf["annottype"] = annottype
                            #dflist.append(rankdf)
                            taggerdf = taggerdf.append(rankdf)
                            print scorecsvfile.shape
                                                       

            
                            taggerdf = matrixhelpers.get_first_N_rows(taggerdf, self.N, [metricname])  
                            bigdf = bigdf.append(taggerdf)
                            # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["tagger", metricname], ascending=False, inplace=True)
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = "best_score_per_tagger-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
    
    
    
    
    def best_score_per_dataset(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
            
            
            
            p1 = os.path.join(scorepath, annottype)
            sizes = IOtools.getfoldernames_of_dir(p1)
            
            for setsize in sizes:                
                
                p2 = os.path.join(p1, setsize)
                featcombnames = IOtools.getfoldernames_of_dir(p2)  # list of combcode_NC names
                
                for combname in featcombnames:
                    
                    combdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
                    
                    p3 = os.path.join(p2, combname)
                    #combcode, nclasses = utils.parse_featsetname(dataset)
                    nclasses = IOtools.getfoldernames_of_dir(p3)
                                        
                    for nc in nclasses:                        
                                                
                        p4 = os.path.join(p3, nc)
                        taggings = IOtools.getfoldernames_of_dir(p4)
                        nclass = nc.split(metaexperimentation.intrafeatsep)[1]
                        
                        for taggingtype in taggings:
                                                                            
                            p5 = os.path.join(p4, taggingtype)
                            scorecsvfilepath = p5 + os.sep + metaexperimentation.scorefilename+".csv"
                            scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                            
                            rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname])
                            print rankdf.shape
                            #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                            print " ** ",combdf.shape
                            rankdf["tagger"] = taggingtype
                            rankdf["nclasses"] = nclass
                            rankdf["featureset"] = combname
                            rankdf["size"] = setsize
                            rankdf["annottype"] = annottype
                            #dflist.append(rankdf)
                            combdf = combdf.append(rankdf)
                            print scorecsvfile.shape
                                                       

            
                    combdf = matrixhelpers.get_first_N_rows(combdf, self.N, [metricname])  
                    bigdf = bigdf.append(combdf)
                    # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["featureset", metricname], ascending=False, inplace=True)
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = "best_score_per_featureset-"+metricname.upper()
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
    evaluator.best_score_per_annottype(metricname="accuracy")
    evaluator.best_score_per_size(metricname="precision")
    evaluator.best_score_per_nclasses(metricname="precision")
    evaluator.best_score_per_dataset(metricname="accuracy")
    evaluator.best_score_per_tagger(metricname="fscore")
    

