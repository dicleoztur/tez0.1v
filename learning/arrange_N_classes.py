'''
Created on Mar 8, 2014

@author: dicle
'''

import os
import numpy as np
import pandas as pd
import random

from corpus import metacorpus
import metaexperimentation
from sentimentfinding import IOtools


nclasses = [2,3,4,5]

# 1,2 -> "s"; 3,4 -> "o"; 5 -> rand("s","o")
def get_2_classes(labelrootpath, taggertype, in_NC=5):
    
    out_NC = 2
    if in_NC <= out_NC:
        return
    
    originallabelspath = os.path.join(labelrootpath, "NC"+metaexperimentation.intrafeatsep+str(in_NC), taggertype+".csv")
    labeldf = IOtools.readcsv(originallabelspath, keepindex=True)
    
    outlabelspath = os.path.join(ensure_nclass_dir(labelrootpath, out_NC), taggertype+".csv")
    
    labelvector = labeldf.values
    labelvector = np.array(labelvector, dtype=object)
    
    # replace values  12->"sub"; 34->"obj"
    labelvector[labelvector == 1] = 12
    labelvector[labelvector == 2] = 12
    labelvector[labelvector == 3] = 34
    labelvector[labelvector == 4] = 34
    
    for i,_ in enumerate(labelvector):
        if labelvector[i] == 5:
            labelvector[i] = random.choice([12, 34])
    
    twolabeldf = pd.DataFrame(labelvector, columns=labeldf.columns.values.tolist(), index=labeldf.index.values.tolist())
    IOtools.tocsv(twolabeldf, outlabelspath, keepindex=True)
        

# 1,2 -> "s"; 2,3 -> "o"; 5 -> "u"
def get_3_classes(labelrootpath, taggertype, in_NC=5):  
    
    out_NC = 3
    if in_NC <= out_NC:
        return
    
    originallabelspath = os.path.join(labelrootpath, "NC"+metaexperimentation.intrafeatsep+str(in_NC), taggertype+".csv")
    labeldf = IOtools.readcsv(originallabelspath, keepindex=True)
    
    outlabelspath = os.path.join(ensure_nclass_dir(labelrootpath, out_NC), taggertype+".csv")
    
    labelvector = labeldf.values
    labelvector = np.array(labelvector, dtype=object)
    
    # replace values
    labelvector[labelvector == 1] = 12
    labelvector[labelvector == 2] = 12
    labelvector[labelvector == 3] = 34
    labelvector[labelvector == 4] = 34
    #labelvector[labelvector == 5] = "u"
    
    threelabeldf = pd.DataFrame(labelvector, columns=labeldf.columns.values.tolist(), index=labeldf.index.values.tolist())
    IOtools.tocsv(threelabeldf, outlabelspath, keepindex=True)


# 5 -> rand(1,2,3,4)
def get_4_classes(labelrootpath, taggertype, in_NC=5): 
    
    out_NC = 4
    if in_NC <= out_NC:
        return
    print "y"
    originallabelspath = os.path.join(labelrootpath, "NC"+metaexperimentation.intrafeatsep+str(in_NC), taggertype+".csv")
    labeldf = IOtools.readcsv(originallabelspath, keepindex=True)
    
    outlabelspath = os.path.join(ensure_nclass_dir(labelrootpath, out_NC), taggertype+".csv")
    
    labelvector = labeldf.values
    labelvector = np.array(labelvector, dtype=object)
    
    # replace values
    for i,_ in enumerate(labelvector):
        if labelvector[i] == 5:
            labelvector[i] = random.choice([1,2,3,4])
    
    fourlabeldf = pd.DataFrame(labelvector, columns=labeldf.columns.values.tolist(), index=labeldf.index.values.tolist())
    IOtools.tocsv(fourlabeldf, outlabelspath, keepindex=True)
    
    

def nclass_label_folder(rootpath, nc):
    return os.path.join(rootpath, "NC"+metaexperimentation.intrafeatsep+str(nc))    
    

def ensure_nclass_dir(labelspath, nclass):
    return IOtools.ensure_dir(nclass_label_folder(labelspath, nclass))


def ensure_nclass_dirs(outrootpath, maxnc):
    nclasslist = range(2, maxnc)
    for nc in nclasslist:
        IOtools.ensure_dir(nclass_label_folder(outrootpath, nc))


def get_nclass_variations(outrootpath, taggertype, has_unknown, in_NC):
    get_2_classes(outrootpath, taggertype, in_NC)
    if has_unknown:
        get_3_classes(outrootpath, taggertype, in_NC)
    get_4_classes(outrootpath, taggertype, in_NC)


if __name__ == "__main__":
    
    nclasses = [2,3,4]
    outrootpath = "/home/dicle/Dicle/Tez/corpusstats/learning2/karalama/labels/"
    
    for nc in nclasses:
        IOtools.ensure_dir(nclass_label_folder(outrootpath, nc))
    
    taggertype = "random"
    
    #get_2_classes(outrootpath, taggertype)
    #get_3_classes(outrootpath, taggertype)
    #get_4_classes(outrootpath, taggertype, 4)
        
    get_nclass_variations(outrootpath, taggertype, in_NC=5)
    
    
    
    
    
    
    
    
    
    
    