# -*- coding: utf-8 -*-
'''
Created on Feb 24, 2014

@author: dicle
'''


import re
import math
import numpy as np
import os

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


# returns the metric-mixed combs (512+64)
def get_featuregroupings():
    featuremetric_combs = {}
    
    groups = ["cadj", "cadv", "tadj", "tadv"]
    measures = ["count", "ratio"]
    
    featuremetric_combs["redefinition_metrics"] = encode_selected_feature_measures(groups, measures)
    
    groups = ["cabs", "csubj", "tabs", "tsubj"]
    measures = ["count", "presence", "ratio", "tfidf"]
    featuremetric_combs["lexicon_metrics"] = encode_selected_feature_measures(groups, measures)
    
    return featuremetric_combs


# returns the metric-fixed 8 combs
def get_relevant_featuregroupings():
    featuremetric_combs = {}
    
    featuremetric_combs["redef-rat_lex-pres"] = "2925"
    featuremetric_combs["redef-rat_lex-cnt"] = "1820"
    featuremetric_combs["redef-rat_lex-rat"] = "975"
    featuremetric_combs["redef-rat_lex-tf"] = "4030"
    
    featuremetric_combs["redef-cnt_lex-pres"] = "2145"
    featuremetric_combs["redef-cnt_lex-cnt"] = "1040" 
    featuremetric_combs["redef-cnt_lex-rat"] = "195"
    featuremetric_combs["redef-cnt_lex-tf"] = "3250"
    
    # get complete combcodes
    featuremap = get_featuremap_sortedstr()
    combmatrix = listutils.get_combination_matrix(featuremap)
    for groupname, rowno in featuremetric_combs.iteritems():
        rowno = int(rowno[-4:])
        metricvaluelist = combmatrix[rowno,:].tolist()
        combcode = encode_combname(metricvaluelist, rowno)
        featuremetric_combs[groupname] = [combcode]
        
        #print rowno,"  :  ",metricvaluelist
        
        '''
        rp = "/home/dicle/Dicle/Tez/corpusstats/learning9/experiments/scores/"
        annots = ["double", "single"]
        for a in annots:
            p = os.path.join(rp, a, groupname)
            IOtools.ensure_dir(p)
        ''' 
    
    return featuremetric_combs


# returns the metric-fixed 8 combs making -1 the indices given in the parameter
def get_excluded_columns_combcodes(excludeindices):
    featuremetric_combs = {}
    
    featuremetric_combs["redef-rat_lex-pres"] = "2925"
    featuremetric_combs["redef-rat_lex-cnt"] = "1820"
    featuremetric_combs["redef-rat_lex-rat"] = "975"
    featuremetric_combs["redef-rat_lex-tf"] = "4030"
    
    featuremetric_combs["redef-cnt_lex-pres"] = "2145"
    featuremetric_combs["redef-cnt_lex-cnt"] = "1040" 
    featuremetric_combs["redef-cnt_lex-rat"] = "195"
    featuremetric_combs["redef-cnt_lex-tf"] = "3250"
    
    # get complete combcodes
    featuremap = get_featuremap_sortedstr()
    combmatrix = listutils.get_combination_matrix(featuremap)
    for groupname, rowno in featuremetric_combs.iteritems():
        rowno = int(rowno[-4:])
        metricvaluelist = combmatrix[rowno,:]
        metricvaluelist[excludeindices] = -1
        combcode = encode_combname(metricvaluelist, rowno)
        combrowdct = {combcode : metricvaluelist.tolist()}
        featuremetric_combs[groupname] = combrowdct
        
        #print rowno,"  :  ",metricvaluelist
        
        '''
        rp = "/home/dicle/Dicle/Tez/corpusstats/learning9/experiments/scores/"
        annots = ["double", "single"]
        for a in annots:
            p = os.path.join(rp, a, groupname)
            IOtools.ensure_dir(p)
        ''' 
    
    return featuremetric_combs




def get_excluded_features_map():    
    excludeindices = {"exclude-content" : [0, 1, 2, 3],
                     "exclude-title" : [4, 5, 6, 7, 8],
                     "exclude-abs" : [0, 4],
                     "exclude-subj" : [3, 8],
                     "exclude-adj" : [1, 5],
                     "exclude-adv" : [2, 6],
                     "exclude-exclm" : [7]} 
    
    exclrelevantindicesmap = listutils.initialize_dict(keys=excludeindices.keys(), val={})
    for exclusionname in excludeindices.keys():
        combcodemap = get_excluded_columns_combcodes(excludeindices[exclusionname])
        exclrelevantindicesmap[exclusionname] = combcodemap
    
    return exclrelevantindicesmap    




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
        if featno == -1:
            featno = "X"
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



def gettrainset(array, trainstart, trainfinish):
    start = min(trainstart, trainfinish)
    finish = max(trainstart, trainfinish)
    return np.append(array[:start], array[finish:], axis=0)

def gettestset(array, teststart, testfinish):
    start = min(teststart, testfinish)
    finish = max(teststart, testfinish)
    return array[start:finish]


def get_ntest(nrows):
    return int(math.ceil(nrows * (metaexperimentation.testpercentage / 100.0)))


def get_nsplit(nrows, rate):
    return int(math.ceil(nrows * (rate / 100.0)))



# converts labelfileiddict = { label : [fileid]} to X and y 
def dict2matrix(labelfileiddict, Xpath):
    
    y = []
    
    datadf = IOtools.readcsv(Xpath, keepindex=True)
    testinstanceids = []
    
    for label, fileids in labelfileiddict.iteritems():
        testinstanceids.extend(fileids)
        y.extend([label] * len(fileids))
    
    X = datadf.loc[testinstanceids, :].values
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    y = np.array(y)

    return X,y

# converts fileidlabelpairs = [(fileid, label)] to X (feature matrix) and y (label array)
def tuple2matrix(fileidlabelpairs, Xpath):
    
    fileids = []
    y = []
    for fileid,label in fileidlabelpairs:
        fileids.append(fileid)
        y.append(label)
    y = np.array(y)
    
    datadf = IOtools.readcsv(Xpath, keepindex=True)
    datadf = datadf.loc[fileids, :]
    
    X = datadf.values   
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    
    # check for fileid order
    count = 0
    instanceids = datadf.index.values.tolist()
    for a,b in zip(instanceids, fileids):
        if a == b:
            count += 1
    if count != len(instanceids):
        print "FAAAAAAAAAAAAALLLLLLLLLLLLLSSSSSSSSSEEEEEEE"
    
    return X, y



def conduct_cross_validation_notest(k=10,
                             #annotationtype="double",
                             #combname="comb524_F_0-0_1-1_2-0_3-0_4-0_5-1_6-1_7-0_8-0",
                             #agrtype="fullagr",
                             #labelfoldername="ALLobj-STGsubj_NC-2",
                             outrootpath="/home/dicle/Dicle/Tez/corpusstats/learning11/experiments_10fold_scale/scores/"):
    
    
    annottypes = ["double", "single"]
    # simdilik sadece double 0-5 tamam. 5'ten devam. 7 Mayis 11:10
    
    #setsizes = ["150"]
    #taggertypes = ["user"]
      
    #nclasses = arrange_N_classes.nclasses   # [4,5]
    
    #models = []
    svmclassifier = SVM("", standardize=True)
    
    nbclassifier = NaiveBayes("", normalize=True)
    #nbclassifier = MultinomialNB(outrootpath)
    models = [nbclassifier, svmclassifier]
    
    
    for annotationtype in annottypes:
        
        sp1 = IOtools.ensure_dir(os.path.join(outrootpath, annotationtype))
        
        print "sp1 ",sp1
         
        datasetspath = metacorpus.get_datasets_path(annotationtype)  # finaldatasets
        labelspath = metacorpus.get_labels_path(annotationtype)
        
        agreementtypes= IOtools.getfoldernames_of_dir(labelspath)
        #agreementtypes = ["fullagr"]      
    
        feature_metric_comb_lists = utils.get_relevant_featuregroupings()  # { featureclasses : [combcode]}
                 
        for featureclass, combfilenames in feature_metric_comb_lists.iteritems():
            
            sp2 = IOtools.ensure_dir(os.path.join(sp1, featureclass))
                            
            print "metricname ",featureclass
            #print "b ",len(combfilenames)
            
            processedcombs = IOtools.getfoldernames_of_dir(sp2)
            combfilenames = [comb for comb in combfilenames if comb not in processedcombs]
            #print "a ",len(combfilenames)
            #print "pr ",len(processedcombs),"  ",sp2
            
            
            for i,combfile in enumerate(combfilenames):
            
                print "############  ",combfile,"  ",str(i)
                Xpath = os.path.join(datasetspath, combfile + ".csv")
                sp3 = IOtools.ensure_dir(os.path.join(sp2, combfile))
                
               
                for agreementtype in agreementtypes:   # count it on labelspath not nclasses
                    
                    lp1 = os.path.join(labelspath, agreementtype)
                    labelunions = IOtools.getfoldernames_of_dir(lp1)
                    '''labelunions = ["ALLobj-WKsubj_NC-2",
                                   "STGobj-WKsubj_NC-2",
                                   "WKobj-ALLsubj_NC-2",
                                   "WKobj-STGsubj_NC-2"]'''
                    
                    for labelunion in labelunions:
                        
                        lp2 = os.path.join(lp1, labelunion)
                        
                        labelitems = labelunion.split(metaexperimentation.interfeatsep)
                        unionname = labelitems[0]
                        ncstr = labelitems[1]
                        nc = ncstr.split(metaexperimentation.intrafeatsep)[-1]
                        nc = int(nc)
                                       
                        sp4 = IOtools.ensure_dir(os.path.join(sp3, unionname))
                        
                        
                        ylabelspath = os.path.join(lp2, metacorpus.labelsfilename+".csv")
                        y = IOtools.readcsv(ylabelspath, True)                
                        labelitems = y.groupby("answer").groups  # labelitems = {label : [newsid]}
                        
                        countlabels = listutils.initialize_dict(labelitems.keys(), val=0.0)
                        
                        '''  skip test division
                        # record test instances for guranteeing
                        testinstances = listutils.initialize_dict(labelitems.keys(), val=[])
                        traininstances = listutils.initialize_dict(labelitems.keys(), val=[])
                        
                        for label, instanceids in labelitems.iteritems():
                            ntest = utils.get_nsplit(len(instanceids), metaexperimentation.testpercentage)
                            testinstances[label] = instanceids[-ntest:]
                            traininstances[label] = instanceids[:-ntest]
                        
                        IOtools.todisc_json(os.path.join(sp4, "test_instances"), testinstances)
                        IOtools.todisc_json(os.path.join(sp4, "train_instances"), traininstances)
                        '''
                        
                        checktrs = []
                        checktss = []
                        
                        intersectstr = []
                        intersectsts = []
                             
                        validstart = 0
                        for foldno in range(k):
                            # both will contain (fileid, label) 
                            trainitems = []  
                            testitems = []
                               
                            for label, fileids in labelitems.iteritems():
                                nvalid = utils.get_nsplit(len(fileids), metaexperimentation.validationpercentage)
                                #ntest = utils.get_nsplit(len(fileids), metaexperimentation.testpercentage)
                                
                                '''
                                print " LABEL: ",label
                                print "  nvalid: ",nvalid,"  ntest: ",ntest
                                '''
                                
                                instanceids = fileids   #fileids[:-ntest]
                                validstart = (foldno * (nvalid + 1)) % len(fileids)
                                validfinish = (validstart + nvalid) % len(fileids)
                                trainids = utils.gettrainset(instanceids, validstart, validfinish)  # fileids to be included in the train set
                                testids = utils.gettestset(instanceids, validstart, validfinish)  # fileids to be included in the test set
                                trainitems.extend([(fileid, label) for fileid in trainids])
                                testitems.extend([(fileid, label) for fileid in testids])

                                '''
                                print "    ntrain: ",len(trainids)
                                print "    ntestset: ",len(testids)
                                
                                
                                if len(trainids) <= len(testids):
                                    print "*******  ",foldno,labelunion, label
                                '''
                                
                                # check file collision. completed and closed 12:43
                                '''
                                coltr = listutils.getintersectionoflists(checktrs, trainids)
                                colts = listutils.getintersectionoflists(checktss, testids)
                                
                                intersectstr.extend(coltr)
                                intersectsts.extend(colts)
                                '''
                    
                            '''
                            print i," ----- ",
                            print "  intersect-train: ",intersectstr,"  ** intersect-test : ",intersectsts
                            print
                            '''
                                
                            foldpath = IOtools.ensure_dir(os.path.join(sp4, "fold-"+str(foldno)))
                            
                            metaexperimentation.initialize_score_file(foldpath)
                            
                            IOtools.tocsv_lst(trainitems, os.path.join(foldpath, "trainitems.csv"))
                            IOtools.tocsv_lst(testitems, os.path.join(foldpath, "testitems.csv"))
                            
                            Xtrain, ytrain = utils.tuple2matrix(trainitems, Xpath)
                            Xtest, ytest = utils.tuple2matrix(testitems, Xpath)
                            
                            # classify
                            for model in models:
                                model.set_score_folder(foldpath)
                                model.apply_algorithms2(Xtrain, ytrain, Xtest, ytest)
                            
                            # random and frequency classification for baseline comparison
                            experimentname = "random"
                            distinctlabels = list(set(ytest))
                            ypred = [random.choice(distinctlabels) for _ in range(len(ytest))]
                            models[0].reportresults(ytest, ypred, experimentname)
                            
                            '''
                            experimentname = "frequency"
                            labelcount = [ytest.tolist().count(label) for label in distinctlabels]
                            ind = np.argmax(labelcount)
                            maxoccurringlabel = distinctlabels[ind]
                            ypred = [maxoccurringlabel] * len(ytest)
                            models[0].reportresults(ytest, ypred, experimentname)
                            '''
     

if __name__ == "__main__":
    
    fdict = get_relevant_featuregroupings()
    print fdict
    '''
    fdict = get_excluded_features_map()
    for k in sorted(fdict.keys()):
        print k, " : ", fdict[k]
    '''
    
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
    '''
    