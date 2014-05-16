'''
Created on Feb 24, 2014

@author: dicle
'''


import re
import math
import numpy as np

import metaexperimentation
from txtprocessor import listutils
from sentimentfinding import IOtools




# returns value arrays sorted where each item is str
def get_featuremap_sortedstr():
    featuremap = {}
    
    featuremap["cabs"] = ["content_abstractnessratio", "content_abstractwords_count", "content_abstractwords_presence", "content_abstractwords_tfidf"]
    featuremap["cadj"] = ["content_adjectivecount", "content_adjectiveratio"] 
    featuremap["cadv"] = ["content_adverbcount", "content_adverbratio"]
    featuremap["csubj"] = ["content_subjectiveverbs_count", "content_subjectiveverbs_presence", "content_subjectiveverbs_tfidf", "content_subjectivityratio"] 
    featuremap["tabs"] = ["title_abstractnessratio", "title_abstractwords_count", "title_abstractwords_presence", "title_abstractwords_tfidf"]
    featuremap["tadj"] = ["title_adjectivecount", "title_adjectiveratio"]
    featuremap["tadv"] = ["title_adverbcount", "title_adverbratio"]
    featuremap["texcl"] = ["title_exclamation"] 
    featuremap["tsubj"] = ["title_subjectiveverbs_count", "title_subjectiveverbs_presence", "title_subjectiveverbs_tfidf", "title_subjectivityratio"]
        
    for k in featuremap.keys():
        featuremap[k] = sorted(featuremap[k])
    
    return featuremap



# returns the dict of combcodes (measure : combcodes) finding the corresponding combcodes of the given 
# groupnames by the given measurelist 
def encode_selected_feature_measures(groupnames, measurenames):
    featuremap = get_featuremap_sortedstr()
    groupnos = [featgroup_index_byname(featuremap, groupname) for groupname in groupnames]
    
    
    
    
    # initialize measurename-colname-index mapper
    mname_fname_cellval = listutils.initialize_dict(measurenames, val={})
    for k in mname_fname_cellval.keys():
        mname_fname_cellval[k] = listutils.initialize_dict(groupnames, val=0)
    
    # update cell values in mname_fname_cellval map
    for mname in measurenames:
        
        for fname in groupnames:
            #groupno = featgroup_index_byname(featuremap, fname)
            measurenos = featuremap[fname]
            

            for i,realmeasurename in enumerate(measurenos):
                if realmeasurename.endswith(mname):
                    mname_fname_cellval[mname][fname] = i
    
    
    combmatrix = listutils.get_combination_matrix(featuremap)
    
    # select map's corresponding rows from the combmatrix
    mnamecomblistmap = listutils.initialize_dict(measurenames, val=[])
    for mname, fname_mindexmap in mname_fname_cellval.iteritems():
        selection = combmatrix.copy()
        for fname, cellval in fname_mindexmap.iteritems():
            colno = featgroup_index_byname(featuremap, fname)
            selection = selection[selection[:, colno] == cellval]
        mnamecomblistmap[mname] = selection.tolist()
        
        
    # encode combcodelists to combnames
    mnamecombnamemap = listutils.initialize_dict(mnamecomblistmap.keys(), val=[])
    print mnamecombnamemap.keys()
    for mname, combnames in mnamecomblistmap.iteritems():
        combcodes = []
        for combname in combnames:
            rowno = np.where(np.all(combname == combmatrix, axis=1))[0].tolist()[0]
            combcode = encode_combname(combname, rowno)
            #print rowno," 9 ",combcode
            combcodes.append(combcode)
        
        mnamecombnamemap[mname] = combcodes
        
            
        
    
    return mnamecombnamemap
    
    '''
    # find the indices of the given feature measures
    measures = featuremap[groupnames[0]].copy()
    measurenos = []

                
                
    
    for featuregroupname in groupnames:
        groupno = featgroup_index_byname(featuremap, featuregroupname)
        measures = featuremap[featuregroupname].copy()
        
        measurecombcodemap = {}
    '''


def get_redefinitionfeature_combs():
    groups = ["cadj", "cadv", "tadj", "tadv"]
    measures = ["count", "ratio"]
    
    return encode_selected_feature_measures(groups, measures)


def get_lexiconfeature_combs():
    groups = ["cabs", "csubj", "tabs", "tsubj"]
    measures = ["count", "presence", "ratio", "tfidf"]
    
    return encode_selected_feature_measures(groups, measures)


def get_featuregroupings():
    featuremetric_combs = {}
    
    groups = ["cadj", "cadv", "tadj", "tadv"]
    measures = ["count", "ratio"]
    
    featuremetric_combs["redefinition_metrics"] = encode_selected_feature_measures(groups, measures)
    
    groups = ["cabs", "csubj", "tabs", "tsubj"]
    measures = ["count", "presence", "ratio", "tfidf"]
    featuremetric_combs["lexicon_metrics"] = encode_selected_feature_measures(groups, measures)
    
    return featuremetric_combs



def uniquefy_array(a):
    b = a.ravel().view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))
    _, unique_idx = np.unique(b, return_index=True)
    return a[np.sort(unique_idx)]



# combNO_F_groupNO-featNO -> [csubjtfidf, tabscount,texcl,..]
def decode_combcode(combcode, combitemsmap, classname=True):
    featset_pattern = re.compile(r'_\d{1,3}\-\d{1,3}')
    groupnofeatnopairs = featset_pattern.findall(combcode)
    
    #combitems_name_map = listutils.initialize_dict(sorted(combitemsmap.keys()), val=[])
    combitems_name_map = {}
    for k in sorted(combitemsmap.keys()):
        combitems_name_map[k] = []
    
    for indexedpair in groupnofeatnopairs:
        pairliteral = indexedpair[1:]
        pairitems = pairliteral.split("-")
        pairitems = map(lambda x : int(x), pairitems)
        groupno = pairitems[0]
        featno = pairitems[1]
        
        groupname = featgroup_name_byindex(combitemsmap, groupno)
        featname = combitemsmap[groupname][featno]
        if classname:
            featname = featname.__class__.__name__
        
        combitems_name_map[groupname].append(featname)
        
    
    return combitems_name_map
    #return "".join(groupnofeatnopairs), groupnofeatnopairs


# combitems_name_map is a dict converted from combcode to names
def tostr_combitems_name_map(combitems_name_map):
    s = ""
    for k in sorted(combitems_name_map.keys()):
        s += str(k) + " : "+str(combitems_name_map[k]) + "\n"
    return s


def tostr_decoded_combcode(combcode, combitemsmap, classname=True):
    combitems_name_map = decode_combcode(combcode, combitemsmap, classname)
    return tostr_combitems_name_map(combitems_name_map)


# combname is a list of measureno values in combmatrix
def encode_combname(combname, rowno):
    filename = "comb"+str(rowno)+"_F"
    for j,featno in enumerate(combname):
        filename += "_"+str(j)+"-"+str(featno)   # filename = combNO_F_GROUPNO-FEATNO
    return filename


# featsetname = _combNO_F_groupNO-featNO_NC-nclasses
def parse_featsetname(featsetname):
    featset_pattern = re.compile(r'_\d{1,3}\-\d{1,3}')
    groupnofeatnopairs = featset_pattern.findall(featsetname)
    combcode = "".join(groupnofeatnopairs)
    
    nclasses = featsetname.split('_NC-')[-1]
    
    return combcode, nclasses



def featgroup_name_byindex(featmap, ind):
    l = sorted(featmap.keys())
    return l[ind]

def featgroup_index_byname(featmap, name):
    l = sorted(featmap.keys())
    try:
        return l.index(name)
    except:
        return -1


def print_features(featmap, matrix):
    print "features"
    _, ncols = matrix.shape
    
    for i in range(ncols):
        groupname = featgroup_name_byindex(featmap, i)
        print groupname," : ",
        groupvect = matrix[:, i]
        for featno in groupvect:
            if featno > -1:
                print featmap[groupname][featno],", ",
            else:
                print "group excluded"
        print



def gettrainset(array, teststart, testfinish):
    return np.append(array[:teststart], array[testfinish:], axis=0)

def gettestset(array, teststart, testfinish):
    return array[teststart:testfinish]


def get_ntest(nrows):
    return int(math.ceil(nrows * (metaexperimentation.testpercentage / 100)))

if __name__ == "__main__":
    
    '''
    s = "comb0_F_0-0_1-0_2-0_3-0_4-0"  #_5-0_6-0_7-0_8-0"
    combitemsmap = {"c_subj" : ["tfidf", "binary"],
                    "t_abs" : ["count", "tfidf"],
                    "c_adj" : ["count", "ratio"],
                    "t_excl" : ["excl"],
                    "t_adv" : ["ratio"]
                    }
    
    print decode_combcode(s, combitemsmap)
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
   
    path = "/home/dicle/Dicle/Tez/corpusstats/learning4/data/double/finaldatasets_be4namerec/"
    combs = IOtools.getfilenames_of_dir(path)
    combs.sort()
    print len(combs)
    for comb in combs[:20]:
        print comb," : ",
        dct = decode_combcode(comb, featuremap, False)
        print tostr_combitems_name_map(dct)
        print
        
    print featgroup_name_byindex(featuremap, ind=4)    
    
    