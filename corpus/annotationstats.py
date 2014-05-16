'''
Created on Apr 7, 2014

@author: dicle
'''

import os
import pandas as pd
import numpy as np
from nltk import FreqDist

from sentimentfinding import IOtools
from learning import utils
import metacorpus


shuffledpath = "/home/dicle/Dicle/Tez/corpusstats/learning3/labelshuffling/"

def get_label_counts(annottype, agreementtype, labelrootfolder=metacorpus.annotatedrootpath, statsfolder=metacorpus.annotationstatsfolder):

    dfpath = metacorpus.get_annotatedtexts_file_path(annottype, agreementtype)
    labeldf = IOtools.readcsv(dfpath)
    #nrows, _ = labeldf.shape

    labeldist = FreqDist(labeldf["answer"].values.tolist())
    
    outpath = os.path.join(statsfolder, annottype+"_"+agreementtype+"-labeloccurrences.csv")
    IOtools.todisc_freqdist(outpath, labeldist)



def print_label_count(labels):
    labelvals = list(set(labels))
    fdist = FreqDist(labels)
    for lval in labelvals:
        percentage = round((fdist[lval] / float(len(labels))), 3) * 100
        print "\t",lval," : ",fdist[lval],"  ",percentage


def label_counts_per_split(rootpath, mix=False, scount=0):
    annotagrtype = {"double" : ["fullagr", "halfagr"], "single" : ["halfagr"]}
    
    
    print "SCOUNT ",[scount]*10
    for annottype, agrtypes in annotagrtype.iteritems():
        print annottype," --- >"
        for agrtype in agrtypes:
            lp1 = os.path.join(rootpath, annottype, "labels", agrtype)
            labelunions = IOtools.getfoldernames_of_dir(lp1)
            
            for lunion in labelunions:
                
                print lunion," :::::: "
                lp2 = os.path.join(lp1, lunion, "labels.csv")
                ldf = IOtools.readcsv(lp2, True)
                labels = ldf["answer"].values.tolist()
                
                
                if mix:
                    ids = ldf.index.values.tolist()
                    np.random.shuffle(ids)
                    labels = ldf.loc[ids, "answer"].values.tolist()                  
                    
                    matrix = np.empty((len(ids), 2), dtype=object)
                    matrix[:, 0] = ids
                    matrix[:, 1] = labels
                    
                    shuffledldf = pd.DataFrame(labels, index=ids, columns=["answer"])                    
                    mixpath = IOtools.ensure_dir(os.path.join(shuffledpath+str(scount), annottype, agrtype, lunion))
                    mixpath = os.path.join(mixpath, "labels.csv")
                    IOtools.tocsv(shuffledldf, mixpath, keepindex=True)
                    #labels = ldf.loc[ids, "answer"].values.tolist()
                    #np.random.shuffle(labels)
                
                ntest = utils.get_ntest(len(labels))
                
                ltrain = labels[:-ntest]
                print "TRAIN ----"
                print_label_count(ltrain)
                
                ltest = labels[-ntest:]
                print "TEST -----"
                print_label_count(ltest)
                print "------------"
            print "--------------------------"
                
                

if __name__ == "__main__":
    
    # get stats for each train-test set of each label union in each agrtype and annottype
    label_counts_per_split(rootpath='/home/dicle/Dicle/Tez/corpusstats/learning3/data/', mix=True, scount=5)
    
    
    #get all stats   - 7 April
    '''
    annotagrtype = {"double" : ["fullagr", "halfagr"], "single" : ["fullagr"]}
    #agrtypes = ["fullagr", "halfagr"]
    
    for annottype, agrtypes in annotagrtype.iteritems():
        for agreementtype in agrtypes:
            get_label_counts(annottype, agreementtype)
            
    '''        
            
            