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
    
    def __init__(self, expspath=metaexperimentation.experimentsrootpath, takeworst=False):
        self.experimentspath = expspath
        self.resultspath = IOtools.ensure_dir(os.path.join(self.experimentspath, "performance"))
        self.N = 50   # ranking number
        #self.metricname = basemetric
        self.prefix = "best"
        self.takeworst = takeworst
        if takeworst:
            self.prefix = "worst"
            
    
    def best_score_per_annottype(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
            
            annotdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
            
            p1 = os.path.join(scorepath, annottype)
            #featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
            metricclasses = IOtools.getfoldernames_of_dir(p1)
            
            for metricclass in metricclasses:
                
                p2 = os.path.join(p1, metricclass)
                featcombnames = IOtools.getfoldernames_of_dir(p2)
                   
                for combname in featcombnames:
                    
                    p3 = os.path.join(p2, combname)
                    labelunions = IOtools.getfoldernames_of_dir(p3)
                    
                    
                    for labelunion in labelunions:
                    
                        p4 = os.path.join(p3, labelunion)          
                        scorecsvfilepath = p4 + os.sep + metaexperimentation.scorefilename+".csv"
                        scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                        # drop clustering results as they are useless being not worked on (back validation missing)
                        scorecsvfile = scorecsvfile[np.logical_not(scorecsvfile.algorithm.str.startswith("_MT-Clustering"))]
                        
                        rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname], ascend=self.takeworst)
                        print rankdf.shape
                        #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                        print " ** ",annotdf.shape
                        rankdf["labelunion"] = labelunion
                        rankdf["featureset"] = metricclass + " ** " + combname
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
                
                annotdf = matrixhelpers.get_first_N_rows(annotdf, self.N, [metricname], ascend=self.takeworst)  
                
            bigdf = bigdf.append(annotdf)
            # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["annottype", metricname], ascending=self.takeworst, inplace=True)
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = self.prefix+"_score_per_annottype-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
    # best alg is the same. just output alg name.                                                        
                
    
    
    
    def best_score_per_labelunion(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
                       
            p1 = os.path.join(scorepath, annottype)
            #featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
            metricclasses = IOtools.getfoldernames_of_dir(p1)
            
            for metricclass in metricclasses:
                
                p2 = os.path.join(p1, metricclass)
                featcombnames = IOtools.getfoldernames_of_dir(p2)
                    
                for combname in featcombnames:
                    
                    p3 = os.path.join(p2, combname)                
                    labelunions = IOtools.getfoldernames_of_dir(p3)
                                
                    for labelunion in labelunions:
                    
                        luniondf = pd.DataFrame(columns=metaexperimentation.performanceheader)
                    
                        p4 = os.path.join(p3, labelunion)          
                        scorecsvfilepath = p4 + os.sep + metaexperimentation.scorefilename+".csv"
                        scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                        # drop clustering results as they are useless being not worked on (back validation missing)
                        scorecsvfile = scorecsvfile[np.logical_not(scorecsvfile.algorithm.str.startswith("_MT-Clustering"))]
                        
                        rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname], ascend=self.takeworst)
                        print rankdf.shape
                        #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                        print " ** ",luniondf.shape
                        rankdf["labelunion"] = labelunion
                        rankdf["featureset"] = metricclass + " ** " + combname
                        rankdf["annottype"] = annottype
                        #dflist.append(rankdf)
                        luniondf = luniondf.append(rankdf)
                        print scorecsvfile.shape
                                                           
                        luniondf = matrixhelpers.get_first_N_rows(luniondf, self.N, [metricname], ascend=self.takeworst)  
                        bigdf = bigdf.append(luniondf)
                    
                # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["labelunion", metricname], ascending=self.takeworst, inplace=True)
        
        # 14Nisan added new to get top N
        bigdf = bigdf.groupby("labelunion").head(self.N)
        
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = self.prefix+"_score_per_labelunion-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
 
 
    
    def best_score_per_featureset(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
                       
            p1 = os.path.join(scorepath, annottype)
            #featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
            metricclasses = IOtools.getfoldernames_of_dir(p1)
            
            for metricclass in metricclasses:
                
                p2 = os.path.join(p1, metricclass)
                featcombnames = IOtools.getfoldernames_of_dir(p2)
                     
                for combname in featcombnames:
                    
                    p3 = os.path.join(p2, combname)                
                    labelunions = IOtools.getfoldernames_of_dir(p3)
                    
                    combdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
                             
                    for labelunion in labelunions:
                    
                        p4 = os.path.join(p3, labelunion)          
                        scorecsvfilepath = p4 + os.sep + metaexperimentation.scorefilename+".csv"
                        scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                        # drop clustering results as they are useless being not worked on (back validation missing)                        
                        scorecsvfile = scorecsvfile[np.logical_not(scorecsvfile.algorithm.str.startswith("_MT-Clustering"))]
                                                                                 
                        rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname], ascend=self.takeworst)
                        print rankdf.shape
                        #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                        print " ** ",combdf.shape
                        rankdf["labelunion"] = labelunion
                        rankdf["featureset"] = metricclass + " ** " + combname
                        rankdf["annottype"] = annottype
                        #dflist.append(rankdf)
                        combdf = combdf.append(rankdf)
                        print scorecsvfile.shape
                                                           
                    combdf = matrixhelpers.get_first_N_rows(combdf, self.N, [metricname], ascend=self.takeworst)  
                    bigdf = bigdf.append(combdf)
                
                # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        bigdf.sort(["featureset", metricname], ascending=self.takeworst, inplace=True)
        
        # 14Nisan added new to get top N
        bigdf = bigdf.groupby("featureset").head(self.N)
        
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = self.prefix+"_score_per_featureset-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
        
           
        
    
    def best_score_per_algorithm(self, metricname, scorepath=metaexperimentation.expscorepath):
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        print scorepath
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for annottype in annottypes:
            
            #annotdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
            
            p1 = os.path.join(scorepath, annottype)
            #featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
            
            metricclasses = IOtools.getfoldernames_of_dir(p1)
            
            for metricclass in metricclasses:
                
                p2 = os.path.join(p1, metricclass)
                featcombnames = IOtools.getfoldernames_of_dir(p2)
                           
                for combname in featcombnames:
                    
                    p3 = os.path.join(p2, combname)
                    labelunions = IOtools.getfoldernames_of_dir(p3)
                    
                    
                    for labelunion in labelunions:
                        
                        annotdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
                        
                        p4 = os.path.join(p3, labelunion)          
                        scorecsvfilepath = p4 + os.sep + metaexperimentation.scorefilename+".csv"
                        scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                        # drop clustering results as they are useless being not worked on (back validation missing)
                        scorecsvfile = scorecsvfile[np.logical_not(scorecsvfile.algorithm.str.startswith("_MT-Clustering"))]
                        
                        rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname], ascend=self.takeworst)
                        print rankdf.shape
                        #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                        print " ** ",annotdf.shape
                        rankdf["labelunion"] = labelunion
                        rankdf["featureset"] = metricclass + " ** " + combname
                        rankdf["annottype"] = annottype
                        #dflist.append(rankdf)
                        annotdf = annotdf.append(rankdf)
                        print scorecsvfile.shape
                
                        annotdf = matrixhelpers.get_first_N_rows(annotdf, self.N, [metricname], ascend=self.takeworst)  
                        bigdf = bigdf.append(annotdf)
            
        bigdf.sort(["algorithm", metricname], ascending=self.takeworst, inplace=True)
        bigdf = bigdf.groupby("algorithm").head(self.N)
        
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = self.prefix+"_score_per_algorithm-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
  
  

    def best_score_per_featexc(self, metricname, scorepath=metaexperimentation.expscorepath):
        
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        
        exclusiontypes = IOtools.getfoldernames_of_dir(scorepath)
        
        for featexctype in exclusiontypes:
            
            p0 = os.path.join(scorepath, featexctype)
            
            annottypes = IOtools.getfoldernames_of_dir(p0)
            
            for annottype in annottypes:
                
                annotdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
                
                p1 = os.path.join(p0, annottype)
                #featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
                metricclasses = IOtools.getfoldernames_of_dir(p1)
                
                for metricclass in metricclasses:
                    
                    p2 = os.path.join(p1, metricclass)
                    featcombnames = IOtools.getfoldernames_of_dir(p2)
                       
                    for combname in featcombnames:
                        
                        p3 = os.path.join(p2, combname)
                        labelunions = IOtools.getfoldernames_of_dir(p3)
                        
                        
                        for labelunion in labelunions:
                        
                            p4 = os.path.join(p3, labelunion)          
                            scorecsvfilepath = p4 + os.sep + metaexperimentation.scorefilename+".csv"
                            scorecsvfile = IOtools.readcsv(scorecsvfilepath)
                            # drop clustering results as they are useless being not worked on (back validation missing)
                            #scorecsvfile = scorecsvfile[np.logical_not(scorecsvfile.algorithm.str.startswith("_MT-Clustering"))]
                            
                            rankdf = matrixhelpers.get_first_N_rows(scorecsvfile, int(self.N / 2), [metricname], ascend=self.takeworst)
                            print rankdf.shape
                            #annotdf.loc[:, rankdf.columns.values.tolist()] = rankdf.values.copy()
                            print " ** ",annotdf.shape
                            rankdf["featexc"] = featexctype
                            rankdf["labelunion"] = labelunion
                            rankdf["featureset"] = metricclass + " ** " + combname
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
                    
                    annotdf = matrixhelpers.get_first_N_rows(annotdf, self.N, [metricname], ascend=self.takeworst)  
                    
                bigdf = bigdf.append(annotdf)
                # insert annottype as colname to bigdf. cutbigdf from the first 10.
        
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, "bigdf.csv"))
        bigdf.sort(["featexc", metricname], ascending=self.takeworst, inplace=True)
        #resultantdf = matrixhelpers.get_first_N_rows(bigdf, self.N)
        evaluationname = self.prefix+"_score_per_featureexclusiontype-"+metricname.upper()
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, evaluationname+".csv"))
        
        
  
    def score_stats(self, metricname, scorepath=metaexperimentation.expscorepath):
        bigdf = pd.DataFrame(columns=metaexperimentation.performanceheader)
        
        #scorepath = os.path.join(self.experimentspath, "scores")
        annottypes = IOtools.getfoldernames_of_dir(scorepath)
                
        for annottype in annottypes:
                        
            p1 = os.path.join(scorepath, annottype)
            #featcombnames = IOtools.getfoldernames_of_dir(p1)  # list of combcode_NC names            
            
            metricclasses = IOtools.getfoldernames_of_dir(p1)
            
            for metricclass in metricclasses:
                
                p2 = os.path.join(p1, metricclass)
                featcombnames = IOtools.getfoldernames_of_dir(p2)
                           
                for combname in featcombnames:
                    
                    p3 = os.path.join(p2, combname)
                    labelunions = IOtools.getfoldernames_of_dir(p3)
                    
                    
                    for labelunion in labelunions:
                        
                        p4 = os.path.join(p3, labelunion)          
                        scorecsvfilepath = p4 + os.sep + metaexperimentation.scorefilename+".csv"
                        scorecsvfile = IOtools.readcsv(scorecsvfilepath) 
                        
                        # drop clustering results as they are useless being not worked on (back validation missing)
                        scorecsvfile = scorecsvfile[np.logical_not(scorecsvfile.algorithm.str.startswith("_MT-Clustering"))]
                        
                        scorecsvfile["labelunion"] = labelunion
                        scorecsvfile["featureset"] = metricclass + " ** " + combname
                        scorecsvfile["annottype"] = annottype
                        bigdf = bigdf.append(scorecsvfile)
        
        
        print "bigdf ",bigdf.shape
        print bigdf.columns
        print
        
        IOtools.tocsv(bigdf, os.path.join(self.resultspath, "bigdf.csv"))
        
        groups = ["annottype", "labelunion", "algorithm", "featureset"]
        for factor in groups:
            
            grouped = bigdf.groupby(factor)
            dfgavg = grouped.mean()
            dfgmin = grouped.min()
            dfgmax = grouped.max()
            dfgstd = grouped.std()
            
            
            nrows, _ = dfgavg.shape
            outdf = pd.DataFrame(np.zeros((nrows, 4)), index=dfgavg.index.values.tolist(), columns=["mean", "min", "max", "std"])
            
            outdf["mean"] = dfgavg[metricname]
            outdf["min"] = dfgmin[metricname]
            outdf["max"] = dfgmax[metricname]
            outdf["std"] = dfgstd[metricname]
            
            filename = "stats_"+factor+"-"+metricname.upper()
            IOtools.tocsv(outdf, os.path.join(self.resultspath, filename+".csv"), keepindex=True)
            


 
    
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


def evaluate_featureexcluded_datasets():
    #rootpath = "/home/dicle/Dicle/Tez/corpusstats/learningdata_excludeone/experiments/"
    rootpath = "/home/dicle/Dicle/Tez/corpusstats/learning10/experiments_final_featureexclude/experiments/"
    
    metrics = ["accuracy", "fscore", "precision", "recall"]
    
    scorespath = os.path.join(rootpath, "scores")
    exclusiontypes = IOtools.getfoldernames_of_dir(scorespath)
    
    
    for exclusionname in exclusiontypes:
        inputscorespath = os.path.join(scorespath, exclusionname)
        recordpath = os.path.join(rootpath, exclusionname)
        
        for minmax in [True, False]: 
            evaluator = PerformanceEvaluator(expspath=recordpath, takeworst=minmax)            
            for metric in metrics:
                print
                
                evaluator.best_score_per_algorithm(metricname=metric, scorepath=inputscorespath)
                evaluator.best_score_per_annottype(metricname=metric, scorepath=inputscorespath)
                evaluator.best_score_per_featureset(metricname=metric, scorepath=inputscorespath)
                evaluator.best_score_per_labelunion(metricname=metric, scorepath=inputscorespath)
    

    
    for exclusionname in exclusiontypes:
        rankpath = os.path.join(rootpath, exclusionname)
        inputscorespath = os.path.join(scorespath, exclusionname)
        
        evaluator = PerformanceEvaluator(expspath=rankpath, takeworst=True)
        for metric in metrics:
            evaluator.score_stats(metricname=metric, scorepath=inputscorespath)
            

# get best and worst per exclusion 
def evaluate_featureexcluded_datasets2(rootpath):
    #rootpath = "/home/dicle/Dicle/Tez/corpusstats/learningdata_excludeone/experiments/"
    #rootpath = "/home/dicle/Dicle/Tez/corpusstats/learning10/experiments_final_featureGROUPexclude/"
    
    metrics = ["accuracy", "fscore", "precision", "recall"]
    
    inputscorespath = os.path.join(rootpath, "scores")
    recordpath = os.path.join(rootpath)
    
        
    for minmax in [True, False]: 
        evaluator = PerformanceEvaluator(expspath=recordpath, takeworst=minmax)            
        for metric in metrics:
            print
            
            evaluator.best_score_per_featexc(metricname=metric, scorepath=inputscorespath)
                      


def evaluate_fullsets():
    
    rootpath = "/home/dicle/Dicle/Tez/corpusstats/learning10/"
    #rootpath = "/home/dicle/Dicle/Tez/corpusstats/learning10/exp-weaklabels/"
    #folders = ["learning9", "learning9_svmscale", "learning9_noscale"]
    folders = ["experiments_final_scale"]
    
    metrics = ["accuracy", "fscore", "precision", "recall"]
    
    for foldername in folders:
        
        recordpath = os.path.join(rootpath, foldername)
        inputscorespath = os.path.join(rootpath, foldername, "scores")
        
        for minmax in [True, False]:
            
            evaluator = PerformanceEvaluator(expspath=recordpath, takeworst=minmax)            
            for metric in metrics:
                print
                # done
                evaluator.best_score_per_algorithm(metricname=metric, scorepath=inputscorespath)
                evaluator.best_score_per_annottype(metricname=metric, scorepath=inputscorespath)
                evaluator.best_score_per_featureset(metricname=metric, scorepath=inputscorespath)
                evaluator.best_score_per_labelunion(metricname=metric, scorepath=inputscorespath)

    
        
        evaluator = PerformanceEvaluator(expspath=recordpath, takeworst=minmax)            
        for metric in metrics:
            print metric,"  0000000"
            evaluator.score_stats(metricname=metric, scorepath=inputscorespath)
            



    
                

if __name__ == "__main__":
    
    evaluate_fullsets()
    #evaluate_featureexcluded_datasets2()            
                           
        
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
    
    

