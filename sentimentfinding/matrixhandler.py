'''
Created on May 20, 2013

@author: dicle
'''


import os

from sentimentfinding import  IOtools
from txtprocessor import texter


datapath = "/home/dicle/Dropbox/Tez/system/data/categorize/"

def add_resource_label(matrixpath, datasetname, replacelabel=False, headers=True):
    matrixlines = IOtools.readtextlines(matrixpath)  # 1st item=fileid, lastitem=filecat.
    
    newmatrix = []
    
    if headers:
        matrixlines = matrixlines[2:]
    
    for instance in matrixlines:
        items = instance.split()
        fileid = items[0]
        print instance,
        path = datapath+os.sep+datasetname
        foldernames = IOtools.getfoldernames_of_dir(datapath+os.sep+datasetname)
        #print foldernames
        for folder in foldernames:
            allfileids = IOtools.getfilenames_of_dir(path+os.sep+folder, removeextension=False)
            #print allfileids
            if fileid in allfileids:
                newspath = path+os.sep+folder+os.sep+fileid
                resourcename = texter.getnewsmetadata(newspath, ["resource"])["resource"]
                #print "## ",resourcename,"  ",type(instance),"  ~~ ",instance
                
                if replacelabel: items = items[:-1]
                newmatrix.append(items +[resourcename])
                break
    
    return newmatrix




if __name__ == "__main__":
    
    '''
    matrixpath = IOtools.matrixpath
    
    m1 = "featureMATRIX-3cat-testn-450texts.m"
    m2 = "featureMATRIX-3cat-trainn-4500texts.m"
    '''
    
    matrixpath = "/home/dicle/Dicle/Tez/output/CLASSTEST/"
    m1 = "t600.m"
    m2 = "t60.m"
    
    
    newmatrix1 = add_resource_label(matrixpath+os.sep+m1, "train", replacelabel=True)
    print newmatrix1
    
    IOtools.todisc_matrix(newmatrix1, matrixpath+os.sep+"labelresource"+m1)
    
    '''
    newmatrix2 = add_resource_label(matrixpath+os.sep+m2, "test", replacelabel=True)
    IOtools.todisc_matrix(newmatrix2, matrixpath+os.sep+"labelresource"+m2)
    '''
    
    
    
    
    
    
    