'''
Created on Sep 3, 2014

@author: dicle
'''

import os

import learner_selected
import performance_evaluation_crossval




def run_cross_validation(outrootpath):
    k = 5
    #scorespath = os.path.join(outrootpath, "scores")
    learner_selected.conduct_cross_validation_notest(k, outrootpath)
    #performance_evaluation_crossval.evaluate_cross_validation(outrootpath)
    
    
if __name__ == "__main__":
    
    outrootpath="/home/dicle/Dicle/Tez/corpusstats/learning11/5fold_majoritybaseline"
    run_cross_validation(outrootpath)
    
    
    
    
    