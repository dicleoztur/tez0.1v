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
    return



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
    
    # create randomly answered sets
    # for single annotated
    get_randomly_annotated_set(incsvfilename=metacorpus.singleantcsv, 
                               outcsvfilename="randomly-"+metacorpus.singleantcsv)
    
    #for double annotated
    get_randomly_annotated_set(incsvfilename=metacorpus.doubleantcsv, 
                               outcsvfilename="randomly-"+metacorpus.doubleantcsv)
    
    # measure interannotator agreement with various metrics
    
    
    
    
    
    
    
    
    