'''
Created on Sep 3, 2014

@author: dicle
'''

import os

from sentimentfinding import IOtools
import learner_selected
import performance_evaluation_crossval



def cross_corpus():
    
    path = "/home/dicle/Dicle/Tez/corpusstats/learning11/crosscorpus_UNequaltest/scores"
    for i in range(1,6):
        p = IOtools.ensure_dir(path + str(i))
         
        learner_selected.split_for_cross_corpus(rootpath=p, equalsize=False)
        performance_evaluation_crossval.evaluate_crosscorpus(scoresroot=p)




def run_cross_validation(outrootpath):
    k = 5
    #scorespath = os.path.join(outrootpath, "scores")
    #learner_selected.conduct_cross_validation_notest(k, outrootpath)
    performance_evaluation_crossval.evaluate_cross_validation(outrootpath)
    



def crawl_folds_for_sets(foldpath, outrootpath):
    folds = IOtools.getfoldernames_of_dir(foldpath)  
    trainitems_fname = "trainitems.csv"
    testitems_fname = "testitems.csv"
    items = {"trainitems.csv" : [], "testitems.csv" : []} 
    for fold in folds:
        p1 = os.path.join(foldpath, fold)
        for fname in items.keys():
            p2 = os.path.join(p1, fname)
            df = IOtools.readcsv(p2, keepindex=True)
            fileids = df.index.values.tolist()
            outpath = os.path.join(outrootpath, "all-"+fname[:-4]+".txt")
            IOtools.todisc_list(outpath, fileids, mode='a')
            items[fname].extend(fileids)
    return items
        
        

def test_sets():
    
    # collect crosscorpus train1-test1
    initpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/5fold_majoritybaseline/scores/double/redef-rat_lex-rat/comb975_F_0-0_1-1_2-1_3-3_4-0_5-1_6-1_7-0_8-3/ALLobj-ALLsubj"
    initoutpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/corpuscheck/inititems/"
    crawl_folds_for_sets(initpath, initoutpath)
    
    crosscorpinpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/crosscorpus_equaltest/scores_equalsize2/redef-rat_lex-rat/ALLobj-ALLsubj/test_with_corpus1/train_corpus1"
    crosscorpoutpath = "/home/dicle/Dicle/Tez/corpusstats/learning11/corpuscheck/crosscorpusitems"
    crawl_folds_for_sets(crosscorpinpath, crosscorpoutpath)
    
    
    
    initpath = ""

if __name__ == "__main__":
    
    '''
    outrootpath="/home/dicle/Dicle/Tez/corpusstats/learning11/5fold_majoritybaseline"
    run_cross_validation(outrootpath)
    '''
    
    cross_corpus()
    
    
    #test_sets()
    
    
    
    
    