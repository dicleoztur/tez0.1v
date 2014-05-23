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




def first_or_last(x, size):
    if x > 0:
        return range(x+1)
    else:
        return range(size-x, size)
        
        


        

def get_exlusion_matrices():
    
    '''
    exclrelevantindicesmapx = {"exclude-content" :    {"redef-cnt_lex-cnt"  :  {["comb1040_F_0-1_1-0_2-0_3-0_4-1_5-0_6-0_7-0_8-0"] : [last 507] },
                                                    "redef-cnt_lex-pres"  :  {["comb2145_F_0-2_1-0_2-0_3-1_4-2_5-0_6-0_7-0_8-1"] : [last 507] },
                                                    "redef-cnt_lex-rat"  :  {["comb195_F_0-0_1-0_2-0_3-3_4-0_5-0_6-0_7-0_8-3"] : [last 4] },
                                                    "redef-cnt_lex-tf"  :  {["comb3250_F_0-3_1-0_2-0_3-2_4-3_5-0_6-0_7-0_8-2"] : [last 507] },
                                                    "redef-rat_lex-cnt"  :  {["comb1820_F_0-1_1-1_2-1_3-0_4-1_5-1_6-1_7-0_8-0"] : [last 507] },
                                                    "redef-rat_lex-pres"  :  {["comb2925_F_0-2_1-1_2-1_3-1_4-2_5-1_6-1_7-0_8-1"] : [last 507] },
                                                    "redef-rat_lex-rat"  :  {["comb975_F_0-0_1-1_2-1_3-3_4-0_5-1_6-1_7-0_8-3"] : [last 5] },
                                                    "redef-rat_lex-tf"  :  {["comb4030_F_0-3_1-1_2-1_3-2_4-3_5-1_6-1_7-0_8-2"] : [last 507] }},
                              
                              "exclude-title" :    {"redef-cnt_lex-cnt"  :  {["comb1040_F_0-1_1-0_2-0_3-0_4-1_5-0_6-0_7-0_8-0"] : [first 506] },
                                                    "redef-cnt_lex-pres"  :  {["comb2145_F_0-2_1-0_2-0_3-1_4-2_5-0_6-0_7-0_8-1"] : [first 506] },
                                                    "redef-cnt_lex-rat"  :  {["comb195_F_0-0_1-0_2-0_3-3_4-0_5-0_6-0_7-0_8-3"] : [first 4] },
                                                    "redef-cnt_lex-tf"  :  {["comb3250_F_0-3_1-0_2-0_3-2_4-3_5-0_6-0_7-0_8-2"] : [first 506] },
                                                    "redef-rat_lex-cnt"  :  {["comb1820_F_0-1_1-1_2-1_3-0_4-1_5-1_6-1_7-0_8-0"] : [first 506] },
                                                    "redef-rat_lex-pres"  :  {["comb2925_F_0-2_1-1_2-1_3-1_4-2_5-1_6-1_7-0_8-1"] : [first 506] },
                                                    "redef-rat_lex-rat"  :  {["comb975_F_0-0_1-1_2-1_3-3_4-0_5-1_6-1_7-0_8-3"] : [first 4] },
                                                    "redef-rat_lex-tf"  :  {["comb4030_F_0-3_1-1_2-1_3-2_4-3_5-1_6-1_7-0_8-2"] : [first 506] }},
                              
                              
                              "exclude-abs" :      {"redef-cnt_lex-cnt"  :  {["comb1040_F_0-1_1-0_2-0_3-0_4-1_5-0_6-0_7-0_8-0"] : [229,230, 231:231+275+1, last 278] },
                                                    "redef-cnt_lex-pres"  :  {["comb2145_F_0-2_1-0_2-0_3-1_4-2_5-0_6-0_7-0_8-1"] : [229,230, 231:231+275+1, last 278] },
                                                    "redef-cnt_lex-rat"  :  {["comb195_F_0-0_1-0_2-0_3-3_4-0_5-0_6-0_7-0_8-3"] : [1,2,3,5,6,7,8] },
                                                    "redef-cnt_lex-tf"  :  {["comb3250_F_0-3_1-0_2-0_3-2_4-3_5-0_6-0_7-0_8-2"] : [229,230, 231:231+275+1, last 278] },
                                                    "redef-rat_lex-cnt"  :  {["comb1820_F_0-1_1-1_2-1_3-0_4-1_5-1_6-1_7-0_8-0"] : [229,230, 231:231+275+1, last 278] },
                                                    "redef-rat_lex-pres"  :  {["comb2925_F_0-2_1-1_2-1_3-1_4-2_5-1_6-1_7-0_8-1"] : [229,230, 231:231+275+1, last 278] },
                                                    "redef-rat_lex-rat"  :  {["comb975_F_0-0_1-1_2-1_3-3_4-0_5-1_6-1_7-0_8-3"] : [1,2,3,5,6,7,8] },
                                                    "redef-rat_lex-tf"  :  {["comb4030_F_0-3_1-1_2-1_3-2_4-3_5-1_6-1_7-0_8-2"] : [229,230, 231:231+275+1, last 278] }}
                              }
    
    
    exclrelevantindicesmapy = {"exclude-content" :    {"redef-cnt_lex-cnt"  :  {["comb1040_F_0-1_1-0_2-0_3-0_4-1_5-0_6-0_7-0_8-0"] : [-1, -1, -1, -1, 1, 0, 0, 0, 0] },
                                                    "redef-cnt_lex-pres"  :  {["comb2145_F_0-2_1-0_2-0_3-1_4-2_5-0_6-0_7-0_8-1"] : [-1, -1, -1, -1, 2, 0, 0, 0, 1] },
                                                    "redef-cnt_lex-rat"  :  {["comb195_F_0-0_1-0_2-0_3-3_4-0_5-0_6-0_7-0_8-3"] : [-1, -1, -1, -1, 0, 0, 0, 0, 3] },
                                                    "redef-cnt_lex-tf"  :  {["comb3250_F_0-3_1-0_2-0_3-2_4-3_5-0_6-0_7-0_8-2"] : [-1, -1, -1, -1, 3, 0, 0, 0, 2] },
                                                    "redef-rat_lex-cnt"  :  {["comb1820_F_0-1_1-1_2-1_3-0_4-1_5-1_6-1_7-0_8-0"] : [-1, -1, -1, -1, 1, 1, 1, 0, 0] },
                                                    "redef-rat_lex-pres"  :  {["comb2925_F_0-2_1-1_2-1_3-1_4-2_5-1_6-1_7-0_8-1"] : [-1, -1, -1, -1, 2, 1, 1, 0, 1] },
                                                    "redef-rat_lex-rat"  :  {["comb975_F_0-0_1-1_2-1_3-3_4-0_5-1_6-1_7-0_8-3"] : [-1, -1, -1, -1, 0, 1, 1, 0, 3] },
                                                    "redef-rat_lex-tf"  :  {["comb4030_F_0-3_1-1_2-1_3-2_4-3_5-1_6-1_7-0_8-2"] : [-1, -1, -1, -1, 3, 1, 1, 0, 2] }},
                              
                              "exclude-title" :    {"redef-cnt_lex-cnt"  :  {["comb1040_F_0-1_1-0_2-0_3-0_4-1_5-0_6-0_7-0_8-0"] : [1, 0, 0, 0, 1, -1, -1, -1, -1, -1] },
                                                    "redef-cnt_lex-pres"  :  {["comb2145_F_0-2_1-0_2-0_3-1_4-2_5-0_6-0_7-0_8-1"] : [2, 0, 0, 1, 2, -1, -1, -1, -1, -1] },
                                                    "redef-cnt_lex-rat"  :  {["comb195_F_0-0_1-0_2-0_3-3_4-0_5-0_6-0_7-0_8-3"] : [0, 0, 0, 3, 0, -1, -1, -1, -1, -1] },
                                                    "redef-cnt_lex-tf"  :  {["comb3250_F_0-3_1-0_2-0_3-2_4-3_5-0_6-0_7-0_8-2"] : [3, 0, 0, 2, 3, -1, -1, -1, -1, -1] },
                                                    "redef-rat_lex-cnt"  :  {["comb1820_F_0-1_1-1_2-1_3-0_4-1_5-1_6-1_7-0_8-0"] : [1, 1, 1, 0, 1, -1, -1, -1, -1, -1] },
                                                    "redef-rat_lex-pres"  :  {["comb2925_F_0-2_1-1_2-1_3-1_4-2_5-1_6-1_7-0_8-1"] : [2, 1, 1, 1, 2, -1, -1, -1, -1, -1] },
                                                    "redef-rat_lex-rat"  :  {["comb975_F_0-0_1-1_2-1_3-3_4-0_5-1_6-1_7-0_8-3"] : [0, 1, 1, 3, 0, -1, -1, -1, -1, -1] },
                                                    "redef-rat_lex-tf"  :  {["comb4030_F_0-3_1-1_2-1_3-2_4-3_5-1_6-1_7-0_8-2"] : [3, 1, 1, 2, 3, -1, -1, -1, -1, -1] }},
                              
                              
                              "exclude-abs" :      {"redef-cnt_lex-cnt"  :  {["comb1040_F_0-1_1-0_2-0_3-0_4-1_5-0_6-0_7-0_8-0"] : [229,230, 231:231+275+1, last 278] },
                                                    "redef-cnt_lex-pres"  :  {["comb2145_F_0-2_1-0_2-0_3-1_4-2_5-0_6-0_7-0_8-1"] : [229,230, 231:231+275+1, last 278] },
                                                    "redef-cnt_lex-rat"  :  {["comb195_F_0-0_1-0_2-0_3-3_4-0_5-0_6-0_7-0_8-3"] : [1,2,3,5,6,7,8] },
                                                    "redef-cnt_lex-tf"  :  {["comb3250_F_0-3_1-0_2-0_3-2_4-3_5-0_6-0_7-0_8-2"] : [229,230, 231:231+275+1, last 278] },
                                                    "redef-rat_lex-cnt"  :  {["comb1820_F_0-1_1-1_2-1_3-0_4-1_5-1_6-1_7-0_8-0"] : [229,230, 231:231+275+1, last 278] },
                                                    "redef-rat_lex-pres"  :  {["comb2925_F_0-2_1-1_2-1_3-1_4-2_5-1_6-1_7-0_8-1"] : [229,230, 231:231+275+1, last 278] },
                                                    "redef-rat_lex-rat"  :  {["comb975_F_0-0_1-1_2-1_3-3_4-0_5-1_6-1_7-0_8-3"] : [1,2,3,5,6,7,8] },
                                                    "redef-rat_lex-tf"  :  {["comb4030_F_0-3_1-1_2-1_3-2_4-3_5-1_6-1_7-0_8-2"] : [229,230, 231:231+275+1, last 278] }}
                              }
    '''
    
    excludeindices = {"exclude-content" : [0, 1, 2, 3],
                     "exclude-title" : [4, 5, 6, 7, 8],
                     "exclude-abs" : [0, 4],
                     "exclude-subj" : [3, 8],
                     "exclude-adj" : [1, 5],
                     "exclude-adv" : [2, 6],
                     "exclude-exclm" : [7]} 
    
    exclrelevantindicesmap = listutils.initialize_dict(keys=excludeindices.keys(), val={})
    for exclusionname in excludeindices.keys():
        combcodemap = utils.get_excluded_columns_combcodes(excludeindices[exclusionname])
        exclrelevantindicesmap[exclusionname] = combcodemap
    
    return exclrelevantindicesmap
    
    
    
    

if __name__ == "__main__":
    
    fdict = get_exlusion_matrices()
    for k in sorted(fdict.keys()):
        print k, " : ", fdict[k]
        
    '''
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
    '''
