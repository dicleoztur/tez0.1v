'''
Created on Feb 14, 2014

@author: dicle
'''

import re
import os

from corpus import metacorpus
from sentimentfinding import IOtools

csvsep = "\t"
intrafeatsep = "-"
interfeatsep = "_"


#rootpath = "/home/dicle/Dicle/Tez/corpusstats/learning/experiments/"
exprfoldername = "experiments"
scoresfoldername = "scores"
perffoldername = "performance"
experimentsrootpath = os.path.join(metacorpus.learningrootpath, exprfoldername)
expscorepath = os.path.join(experimentsrootpath, scoresfoldername)
expperfpath = os.path.join(experimentsrootpath, perffoldername) 


testpercentage = 15.0  # percentage of test set

# run at the top of learning
def initialize_experiments_folder():
    IOtools.ensure_dir(expscorepath)
    IOtools.ensure_dir(expperfpath)
    




scoresheader = ["algorithm", "precision", "recall", "fscore", "accuracy"]
performanceheader = ["annottype", "size", "tagger", "nclasses", "algorithm", "featureset", 
                     "precision", "recall", "fscore", "accuracy"]

scorefilename = "algorithms-scores"

taggertypes = ["random", "user"]



def get_rootscorepath():
    return IOtools.ensure_dir(expscorepath)

def get_rootperformancepath():
    return IOtools.ensure_dir(expperfpath)

def get_scorefilepath(scorefolder):
    return os.path.join(scorefolder, scorefilename+".csv")

def initialize_score_file(scorefolder):
    scorefilepath = get_scorefilepath(scorefolder)
    header = csvsep.join(scoresheader)
    IOtools.todisc_txt(header, scorefilepath)  
    
    
'''
# combNO_F_groupNO-featNO -> [csubjtfidf, tabscount,texcl,..]
def decode_combcode(combcode):
    return
def encode_combname(combname):
    return

# featsetname = _combNO_F_groupNO-featNO_NC-nclasses
def parse_featsetname(featsetname):
    featset_pattern = re.compile(r'_\d{1,3}\-\d{1,3}')
    groupnofeatnopairs = featset_pattern.findall(featsetname)
    combcode = "".join(groupnofeatnopairs)
    
    nclasses = featsetname.split('_NC-')[-1]
    
    return combcode, nclasses
    '''
    
if __name__ == "__main__":
    print taggertypes

