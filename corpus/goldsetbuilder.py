# -*- coding: utf-8 -*- 
'''
Created on Feb 5, 2014

@author: dicle
'''

import os
import pandas as pd
import numpy as np
import random

from sentimentfinding import IOtools
import metacorpus



def arrange_evaluations_file(infilename, outfilename, folderpath):
    incsvpath = ""
    indf = IOtools.readcsv(incsvpath)


# extracts two sets from the user annotated db dumped data, doubleevals and singleevals 
def extract_user_annotated_set(pathtodbdump):
    # done on the shell
    return



# doubleannotatedcsv = (questionname, answer1, answer2, answer)
# output two files: fullaggreed4class; halfaggreed2class
def assign_annotator_aggreement(doubleannotated_path, doubleannot_filename):
    csvpath = os.path.join(doubleannotated_path, doubleannot_filename)
    doubleannotatedcsv = IOtools.readcsv(csvpath)
    nrows, _ = doubleannotatedcsv.shape
    
    doubleannotated_full4class = doubleannotatedcsv.loc[:, ["questionname", "answer"]].copy()
    doubleannotated_half2class = doubleannotatedcsv.loc[:, ["questionname", "answer"]].copy()
    
    # get full agreed and half agreed annotations:
    for i in range(nrows):
        answer1 = doubleannotatedcsv.loc[i, "answer1"]
        answer2 = doubleannotatedcsv.loc[i, "answer2"]
    
        if answer1 == answer2:
            doubleannotated_full4class.loc[i, "answer"] = answer1
        if answer1 in [1,2] and answer2 in [1,2]:   # elif?
            doubleannotated_half2class.loc[i, "answer"] = 12
        elif answer1 in [3,4] and answer2 in [3,4]:
            doubleannotated_half2class.loc[i, "answer"] = 34
    
    
    # filtrate non-agreeing rows:
    doubleannotated_full4class = doubleannotated_full4class[doubleannotated_full4class["answer"] > 0]
    csvpath1 = os.path.join(doubleannotated_path, "doubleannotated_fullagr4class.csv")
    IOtools.tocsv(doubleannotated_full4class, csvpath1)
    
    doubleannotated_half2class = doubleannotated_half2class[doubleannotated_half2class["answer"] > 0]
    csvpath2 = os.path.join(doubleannotated_path, "doubleannotated_halfagr2class.csv")
    IOtools.tocsv(doubleannotated_half2class, csvpath2)
    
    
    


def get_randomly_annotated_set(incsvfilename, outcsvfilename,
                               incsvfolder=metacorpus.userannotatedpath, outcsvfolder=metacorpus.randomannotatedpath, 
                               randomchoicevalues=metacorpus.subjectivity_label_values.keys()):
    df = IOtools.readcsv(os.path.join(incsvfolder, incsvfilename))  # df cols: questionname,userid,answer
    randomdf= df.copy()
    numofrows, _ = randomdf.values.shape
    subjvalues = randomchoicevalues
    
    randomanswers = [random.choice(subjvalues) for _ in range(numofrows)]
    randomdf.loc[:, "answer"] = randomanswers
    
    # extra: assign 5 of the rows the value 5 for the answer 'no idea, ambiguous'
    notknowingrows = random.sample(range(numofrows), 5)
    
    '''
    for _ in range(5):
        randindex = random.randint(0, numofrows-1)
        while randindex in notknowingrows:
            randindex = random.randint(0, numofrows-1)
        notknowingrows.append(randindex)
    '''        
    #notknowingrows = [random.randint(0, numofrows-1) for _ in range(5)]  # be careful with this 5 number it is subject to change for the sake of statistical validity
    randomdf.loc[notknowingrows, "answer"] = 5
    
    IOtools.tocsv(randomdf, os.path.join(outcsvfolder, outcsvfilename))


if __name__ == "__main__":
    
    '''
    # create randomly answered sets
    # for single annotated
    get_randomly_annotated_set(incsvfilename=metacorpus.singleantcsv, 
                               outcsvfilename="randomly-"+metacorpus.singleantcsv)
    
    #for double annotated
    get_randomly_annotated_set(incsvfilename=metacorpus.doubleantcsv, 
                               outcsvfilename="randomly-"+metacorpus.doubleantcsv)
    '''
    
    # get agreement for final answers
    doubleannotated_path = "/home/dicle/Dicle/Tez/corpusstats/clusterable/userannotated/"
    doubleannot_filename = "doubleAnnotated700rows.csv"
    assign_annotator_aggreement(doubleannotated_path, doubleannot_filename)
    
    
    
    
    # measure interannotator agreement with various metrics
    
    
    
    
    
    
    
    
    