'''
Created on May 9, 2014

@author: dicle
'''

import os
import numpy as np

from sentimentfinding import IOtools
from txtprocessor import listutils
from learning import utils




def exclude_col_i(originalmatrix, colno):
    newmatrix = originalmatrix.copy()
    newmatrix = newmatrix[:, colno]
    newmatrix = utils.uniquefy_array(newmatrix)  # remove repeating rows
    return newmatrix  # evaluate the features if val is >-1


if __name__ == "__main__":
    
    featuremap = {}
    
    featuremap["cadj"] = ["cadjratio", "cadjcount"]
    featuremap["tadj"] = ["tadjratio", "tadjcount"]
    
    featuremap["cadv"] = ["cadvratio", "cadvcount"]
    featuremap["tadv"] = ["tadvratio", "tadvcount"]   
    
    featuremap["cabs"] = ["cabstfidf", "cabscount", "cabsbinary", "cabsrat"]     
    featuremap["tabs"] = ["tabstfidf", "tabscount", "tabsbinary", "tabsrat"] 
    
    featuremap["csubj"] = ["csubjtfidf", "csubjcount", "csubjbinary", "csubjrat"]
    featuremap["tsubj"] = ["tsubjtfidf", "tsubjcount", "tsubjbinary", "tsubjrat"]
    
    featuremap["texcl"] = ["texcl"]
    
    for k in featuremap.keys():
        featuremap[k] = sorted(featuremap[k])
        
    
    folder = "/home/dicle/Dicle/Tez/corpusstats/learning4/data/double/excludeone/"
    matrix = listutils.get_combination_matrix(featuremap)
    IOtools.todisc_matrix(matrix, folder+"4096.txt")
    print matrix
    
