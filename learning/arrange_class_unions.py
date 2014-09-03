'''
Created on Apr 7, 2014

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



# outputs the instances whose values are strongly subjective or strongly objective
def get_StgObj_StgSubj_class(originallabelspath, outfolder, stgobjval=4, stgsubjval=1):
    out_NC = 2
    
    labeldf = IOtools.readcsv(originallabelspath)   #, keepindex=True)
    labelvector = labeldf["answer"].values
    labelvector = np.array(labelvector, dtype=object)


    if stgobjval not in labelvector or stgsubjval not in labelvector:
        #print "VAL NOT FOUND"
        return
    
    # drop rows if answer not in [1,4]

    stronglabelsdf = labeldf[labeldf["answer"].isin([stgobjval, stgsubjval])]
    outpath = os.path.join(ensure_unionclass_dir(outfolder, "STGobj-STGsubj", out_NC), metacorpus.labelsfilename + ".csv")
    IOtools.tocsv(stronglabelsdf, outpath)


# outputs the instances whose values are weakly subjective or weakly objective
def get_WkObj_WkSubj_class(originallabelspath, outfolder, weakobjval=3, weaksubjval=2):
    out_NC = 2
       
    labeldf = IOtools.readcsv(originallabelspath)   #, keepindex=True)
    labelvector = labeldf["answer"].values
    labelvector = np.array(labelvector, dtype=object)

    if weakobjval not in labelvector or weaksubjval not in labelvector:
        return
    
    # drop rows if answer not in [1,4]

    weaklabelsdf = labeldf[labeldf["answer"].isin([weakobjval, weaksubjval])]
    outpath = os.path.join(ensure_unionclass_dir(outfolder, "WKobj-WKsubj", out_NC), metacorpus.labelsfilename + ".csv")
    IOtools.tocsv(weaklabelsdf, outpath)

   

# 1,2 -> "s"; 4 -> "o"; 5 -> ignore
def get_StgObj_AllSubj_class(originallabelspath, outfolder, stgobjval=4, in_NC=5):
    
    out_NC = 2
    if in_NC <= out_NC:
        return

    labeldf = IOtools.readcsv(originallabelspath, keepindex=True)
    labeldf = labeldf[labeldf["answer"].isin([4, 1, 2])]

    labelvector = labeldf.values
    labelvector = np.array(labelvector, dtype=object)
    
    if stgobjval not in labelvector:
        return
    
    # replace values  12->"sub"; 34->"obj"
    labelvector[labelvector == 1] = 12
    labelvector[labelvector == 2] = 12
    labelvector[labelvector == 4] = 34
        
    outpath = os.path.join(ensure_unionclass_dir(outfolder, "STGobj-ALLsubj", out_NC), metacorpus.labelsfilename + ".csv")
    twolabeldf = pd.DataFrame(labelvector, columns=labeldf.columns.values.tolist(), index=labeldf.index.values.tolist())
    IOtools.tocsv(twolabeldf, outpath, keepindex=True)



'''  22 Temmuz - on writing results: why adding weak labels increases performance? '''
# 1,2 -> "s"; 4 -> "o"; 5 -> ignore
def get_WkObj_AllSubj_class(originallabelspath, outfolder, wkobjval=3, in_NC=5):
    
    out_NC = 2
    if in_NC <= out_NC:
        return

    labeldf = IOtools.readcsv(originallabelspath, keepindex=True)
    labeldf = labeldf[labeldf["answer"].isin([wkobjval, 1, 2])]

    labelvector = labeldf.values
    labelvector = np.array(labelvector, dtype=object)
    
    if wkobjval not in labelvector:
        return
    
    # replace values  12->"sub"; 34->"obj"
    labelvector[labelvector == 1] = 12
    labelvector[labelvector == 2] = 12
    labelvector[labelvector == 3] = 34
        
    outpath = os.path.join(ensure_unionclass_dir(outfolder, "WKobj-ALLsubj", out_NC), metacorpus.labelsfilename + ".csv")
    twolabeldf = pd.DataFrame(labelvector, columns=labeldf.columns.values.tolist(), index=labeldf.index.values.tolist())
    IOtools.tocsv(twolabeldf, outpath, keepindex=True)

'''  22 Temmuz '''
# 2 -> "s"; 3,4 -> "o"; 5 -> rand("s","o")
def get_AllObj_WkSubj_class(originallabelspath, outfolder, wksubjval=2, in_NC=5):
    
    out_NC = 2
    if in_NC <= out_NC:
        return

    labeldf = IOtools.readcsv(originallabelspath, keepindex=True)
    labeldf = labeldf[labeldf["answer"].isin([wksubjval, 3, 4])]

    labelvector = labeldf.values
    labelvector = np.array(labelvector, dtype=object)
    
    if wksubjval not in labelvector:
        return
    
    # replace values  12->"sub"; 34->"obj"
    labelvector[labelvector == wksubjval] = 12
    labelvector[labelvector == 3] = 34
    labelvector[labelvector == 4] = 34
        
    outpath = os.path.join(ensure_unionclass_dir(outfolder, "ALLobj-WKsubj", out_NC), metacorpus.labelsfilename + ".csv")
    twolabeldf = pd.DataFrame(labelvector, columns=labeldf.columns.values.tolist(), index=labeldf.index.values.tolist())
    IOtools.tocsv(twolabeldf, outpath, keepindex=True)
    


# outputs the instances whose values are strongly subjective or strongly objective
def get_WkObj_StgSubj_class(originallabelspath, outfolder, wkobjval=3, stgsubjval=1):
    out_NC = 2
    
    labeldf = IOtools.readcsv(originallabelspath)   #, keepindex=True)
    labelvector = labeldf["answer"].values
    labelvector = np.array(labelvector, dtype=object)


    if wkobjval not in labelvector or stgsubjval not in labelvector:
        #print "VAL NOT FOUND"
        return
    
    # drop rows if answer not in [1,4]

    stronglabelsdf = labeldf[labeldf["answer"].isin([wkobjval, stgsubjval])]
    outpath = os.path.join(ensure_unionclass_dir(outfolder, "WKobj-STGsubj", out_NC), metacorpus.labelsfilename + ".csv")
    IOtools.tocsv(stronglabelsdf, outpath)


# outputs the instances whose values are strongly subjective or strongly objective
def get_StgObj_WkSubj_class(originallabelspath, outfolder, stgobjval=4, wksubjval=2):
    out_NC = 2
    
    labeldf = IOtools.readcsv(originallabelspath)   #, keepindex=True)
    labelvector = labeldf["answer"].values
    labelvector = np.array(labelvector, dtype=object)


    if stgobjval not in labelvector or wksubjval not in labelvector:
        #print "VAL NOT FOUND"
        return
    
    # drop rows if answer not in [1,4]

    stronglabelsdf = labeldf[labeldf["answer"].isin([stgobjval, wksubjval])]
    outpath = os.path.join(ensure_unionclass_dir(outfolder, "STGobj-WKsubj", out_NC), metacorpus.labelsfilename + ".csv")
    IOtools.tocsv(stronglabelsdf, outpath)




# 1 -> "s"; 3,4 -> "o"; 5 -> rand("s","o")
def get_AllObj_StgSubj_class(originallabelspath, outfolder, stgsubjval=1, in_NC=5):
    
    out_NC = 2
    if in_NC <= out_NC:
        return

    labeldf = IOtools.readcsv(originallabelspath, keepindex=True)
    labeldf = labeldf[labeldf["answer"].isin([1, 3, 4])]

    labelvector = labeldf.values
    labelvector = np.array(labelvector, dtype=object)
    
    if stgsubjval not in labelvector:
        return
    
    # replace values  12->"sub"; 34->"obj"
    labelvector[labelvector == 1] = 12
    labelvector[labelvector == 3] = 34
    labelvector[labelvector == 4] = 34
        
    outpath = os.path.join(ensure_unionclass_dir(outfolder, "ALLobj-STGsubj", out_NC), metacorpus.labelsfilename + ".csv")
    twolabeldf = pd.DataFrame(labelvector, columns=labeldf.columns.values.tolist(), index=labeldf.index.values.tolist())
    IOtools.tocsv(twolabeldf, outpath, keepindex=True)


# 1,2 -> "s"; 3,4 -> "o"; 5 -> rand("s","o")
def get_AllObj_AllSubj_class(originallabelspath, outfolder, in_NC=5):
    
    out_NC = 2
    if in_NC <= out_NC:
        return

    labeldf = IOtools.readcsv(originallabelspath, keepindex=True)
    
    outpath = os.path.join(ensure_unionclass_dir(outfolder, "ALLobj-ALLsubj", out_NC), metacorpus.labelsfilename + ".csv")
    
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
    IOtools.tocsv(twolabeldf, outpath, keepindex=True)



def get_EachObj_EachSubj_class(originallabelspath, outfolder, foldername="EACHobj-EACHsubj"):
    
    labeldf = IOtools.readcsv(originallabelspath, keepindex=True)
    
    labelvector = labeldf["answer"].values
    out_NC = len(list(set(list(labelvector.tolist()))))
 
    '''
    for i,_ in enumerate(labelvector):
        if labelvector[i] == 5:
            labelvector[i] = random.choice([12, 34])
    '''
    
    outpath = os.path.join(ensure_unionclass_dir(outfolder, foldername, out_NC), metacorpus.labelsfilename + ".csv")
    IOtools.tocsv(labeldf, outpath, keepindex=True)




def unionclass_label_folder(rootpath, unionname, nc):
    return os.path.join(rootpath, unionname+metaexperimentation.interfeatsep+"NC"+metaexperimentation.intrafeatsep+str(nc))    
    

def ensure_unionclass_dir(labelspath, unionname, nclass):
    return IOtools.ensure_dir(unionclass_label_folder(labelspath, unionname, nclass))



def arrange_class_union_variations(originallabelspath, outfolder):
    get_StgObj_StgSubj_class(originallabelspath, outfolder)
    get_EachObj_EachSubj_class(originallabelspath, outfolder)
    get_AllObj_StgSubj_class(originallabelspath, outfolder)
    get_StgObj_AllSubj_class(originallabelspath, outfolder)
    get_WkObj_WkSubj_class(originallabelspath, outfolder)
    get_AllObj_AllSubj_class(originallabelspath, outfolder)



def add_weak_combs():
    originallabelspath = "/home/dicle/Dicle/Tez/corpusstats/annotatedtexts/"
    annottype = "single"
    agrtype = "halfagr"
    
    inpath = os.path.join(originallabelspath, annottype, annottype+"_"+agrtype+"-evals.csv")
    outpath = "/home/dicle/Dicle/Tez/corpusstats/weaklabels/single/"
    
    get_WkObj_AllSubj_class(inpath, outpath, wkobjval=3, in_NC=5)
    get_AllObj_WkSubj_class(inpath, outpath)
    get_StgObj_WkSubj_class(inpath, outpath)
    get_WkObj_StgSubj_class(inpath, outpath)
    
    

if __name__ == "__main__":
    
    add_weak_combs()
    
    ''' closed 22 temmuz
    f = "/home/dicle/Dicle/Tez/corpusstats/learning2/karalama/labels2/"
    annottypes = ["double", "single"]
    agrtypes = ["halfagr", "fullagr"]
    
    annottype = "double"
    agrtype = "fullagr"
    inpath = os.path.join(f, annottype, annottype + "_" + agrtype + "-evals.csv")
    #get_StgObj_StgSubj_class(origlabelpath=inpath, outfolder=f, stgobjval=4, stgsubjval=1)
    
    get_EachObj_EachSubj_class(inpath, outfolder=f)
    #get_AllObj_StgSubj_class(inpath, outfolder=f, in_NC=5)
    
    #print metacorpus.get_annotatedtexts_file_path(annotationtype="single", agreementype="halfagr")
    '''
    
    
    
    '''
    nclasses = [2,3,4]
    outrootpath = "/home/dicle/Dicle/Tez/corpusstats/learning2/karalama/labels/"
    
    for nc in nclasses:
        IOtools.ensure_dir(nclass_label_folder(outrootpath, nc))
    
    taggertype = "random"
    
    #get_2_classes(outrootpath, taggertype)
    #get_3_classes(outrootpath, taggertype)
    #get_4_classes(outrootpath, taggertype, 4)
        
    get_nclass_variations(outrootpath, taggertype, in_NC=5)
    '''
    
    
    
    
    
    
    
    
    
    