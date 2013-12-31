# -*- coding: utf-8 -*- 
'''
Created on Dec 24, 2013

@author: dicle
'''

# ismi collection analysis olmalı

import pandas as pd
import numpy as np
import os
from nltk import ConditionalFreqDist

import metacorpus
import extractnewsmetadata
from sentimentfinding import IOtools
from sentimentfinding import CFDhelpers
from sentimentfinding import plotter
from visualization import cfdplotter

'''
J1- get newsmetadata list
J2- prune the list and record back (remove numofwords<6 or date=None)
J3- get resource-cat-date stats (numfoftexts, numofcats over days, resources, cat)
J4- get daily-monthly news density for each cat and resource


'''

csvsep = metacorpus.csvsep


def get_resourcecatmap(corpuspath=metacorpus.rawcorpuspath):
    resourcecatmap = {}
    
    resources = IOtools.getfoldernames_of_dir(corpuspath)
    for resource in resources:
        path = os.path.join(corpuspath, resource)
        cats = IOtools.getfoldernames_of_dir(path)
        resourcecatmap[resource] = []
        
        for cat in cats:
            resourcecatmap[resource].append(resource+"-"+cat)
    
    return resourcecatmap
        

# 1- get newsmetadata list
def get_newsmetadata(corpuspath=metacorpus.rawcorpuspath, resources=None, 
                     csvfilefolder=metacorpus.statspath, csvfilename=metacorpus.metafilename):
    if resources is None:
        resources = get_resourcecatmap(corpuspath).keys()
    extractnewsmetadata.main(corpuspath, resources, csvfilefolder, csvfilename)




# 2- prune the list and record back (remove numofwords<6 or date=None)
'''  the fileids recorded in the file rawcsvname will be removed if the number of words in that fileid is less than leastwords_threshold.
     remaining fileids (above threshold) will be recorded back in the file prunedcsvname.
'''  
def prune_newsmetadata(rawcsvfolder=metacorpus.statspath, rawcsvname=metacorpus.metafilename, prunedcsvfolder=metacorpus.statspath, 
                       prunedcsvname=metacorpus.prunedmetafilename, leastwords_threshold=6):
    rawpath = os.path.join(rawcsvfolder, rawcsvname)
    rawmetadf = pd.read_csv(rawpath, sep=csvsep, header=0, encoding='utf-8')
    
    prunedmetadf = rawmetadf[rawmetadf["numofwords"] >= leastwords_threshold]
    #prunedmetadf = prunedmetadf[prunedmetadf["date"] is not " "]
    
    prunedpath = os.path.join(prunedcsvfolder, prunedcsvname)
    prunedmetadf.to_csv(prunedpath, index=False, header=True, sep=csvsep, encoding='utf-8')
    



#J4
# we are given a file containing resource 



def visualize_monthly_news_stats(csvfolder=metacorpus.statspath, csvname=metacorpus.prunedmetafilename,
                                 imgoutpath=metacorpus.imgfolder,
                                 rescatmap=metacorpus.resourcecategorymap):
    colldf = IOtools.readcsv(csvfolder+os.sep+csvname)
    
    numoftexts, numofcols = colldf.values.shape
    
    
    # daily news counts for resources
    cfddailyresourcecount = ConditionalFreqDist((colldf.loc[i,"date"], colldf.loc[i,"resource"].strip()) for i in range(numoftexts))
    CFDhelpers.cfd2csv(cfddailyresourcecount, csvfolder+os.sep+"cfddailyresourcecount2.csv", colnames=['date','resource','count'])
    cfdresourcesdaycount = ConditionalFreqDist((resource, day) for day in cfddailyresourcecount.conditions() for resource in list(cfddailyresourcecount[day]))
    
    
    # daily news counts for categories
    cfddailycategorycount = ConditionalFreqDist((colldf.loc[i,"date"], 
                                                 "-".join(map(lambda x : str(x).strip(), [colldf.loc[i, "resource"], colldf.loc[i, "category"]]))) for i in range(numoftexts)) 
    CFDhelpers.cfd2csv(cfddailycategorycount, csvfolder+os.sep+"cfddailycategorycount2.csv", ["date", "category", 'count'])
    cfdcatsdaycount = ConditionalFreqDist((category, date) for date in cfddailycategorycount.conditions() for category in list(cfddailycategorycount[date]))

    
    
    # visualize monthly   --- assuming the dates are of the form yyyy-mm-dd -we did it so while recording
    
    cfdmonthlyresourcecount = ConditionalFreqDist((colldf.loc[i,"date"][:-3], colldf.loc[i,"resource"].strip()) for i in range(numoftexts))
    CFDhelpers.cfd2csv(cfdmonthlyresourcecount, csvfolder+os.sep+"cfdmonthlyresourcecount.csv", colnames=['month','resource','count'])
    cfdresourcesmonthcount = ConditionalFreqDist((resource, month) for month in cfdmonthlyresourcecount.conditions() for resource in list(cfdmonthlyresourcecount[month]))
    imgpath = IOtools.ensure_dir(os.path.join(imgoutpath, "resourcebasednewscount"))
    visualize_monthly_cfd(cfd=cfdmonthlyresourcecount, figuretitle="Monthly news count for each resource", ylabel="news published", imgoutpath=imgpath)



    
    cfdmonthlycategorycount = ConditionalFreqDist((colldf.loc[i,"date"][:-3], 
                                                   "-".join(map(lambda x : str(x).strip(), [colldf.loc[i, "resource"], colldf.loc[i, "category"]]))) 
                                                  for i in range(numoftexts)) 
    CFDhelpers.cfd2csv(cfdmonthlycategorycount, csvfolder+os.sep+"cfdmonthlycategorycount.csv", ["month", "category", 'count'])
    #cfdcatsmonthcount = ConditionalFreqDist((category, month) for month in cfdmonthlycategorycount.conditions() for category in list(cfdmonthlycategorycount[month]))
    
    imgpath = IOtools.ensure_dir(os.path.join(imgoutpath, "categorybasednewscount"))
    for canoniccatname, rescatnames in rescatmap.iteritems():
        monthresourcepairs = []
        for rescat in rescatnames:
            resourcename = rescat.split("-")[0]
            for month in cfdmonthlycategorycount.conditions():
                numofoccurrences = cfdmonthlycategorycount[month][rescat]
                #print resourcename," had ",numofoccurrences," times texts in :",rescat," during ",month
                for i in range(numofoccurrences):
                    monthresourcepairs.append((month, resourcename))
        cfdmonthlyresourcecount_percat = ConditionalFreqDist(monthresourcepairs) 
            
        print canoniccatname,resourcename," * ",rescat," : ",len(cfdmonthlyresourcecount_percat.conditions()),"  ",cfdmonthlyresourcecount_percat.N()
        figuretitle = "Monthly news count of each resource over category "+canoniccatname.upper()
        visualize_monthly_cfd(cfdmonthlyresourcecount_percat, figuretitle, ylabel="news published", imgoutpath=imgpath)
   





def visualize_monthly_cfd(cfd, figuretitle, ylabel, imgoutpath):
    cfd_reverse = ConditionalFreqDist((entity, month) for month in cfd.conditions() for entity in list(cfd[month]))
    
    months = cfd.conditions()
    months.sort()
    
    barlabels = cfd_reverse.conditions()
    #print months
    print barlabels
    
    
    yitemsmatrix = []
    
    for entity in barlabels:
        row = []
        for month in months:
            row.append(cfd[month][entity])
        yitemsmatrix.append(row)
    
    
    if len(barlabels) == 0 or len(yitemsmatrix) == 0:
        return
    
    yitemsmatrix = np.array(yitemsmatrix)
    #yitemsmatrix = yitemsmatrix.T
    print yitemsmatrix.shape
    
    colors = plotter.get_n_colors(len(barlabels))
    months = map(lambda x : str(x), months)
    
    
    # partition the figure in case x axis gets too large by the number of months
    numofxitems = 5
    numoffigures = (len(months) / numofxitems ) + 1
    for i in range(numoffigures):
        matrix = yitemsmatrix[:, (i*numofxitems) : ((i+1)*numofxitems)] 
        print matrix
        xlabels = months[(i*numofxitems) : ((i+1)*numofxitems)]
        # save fig. pass img path with i
        figurename = figuretitle + " "+ str(i)
        cfdplotter.multiplebargraphs(barlabels, matrix.tolist(), colors, figurename, xlabels, ylabel, imgpath=imgoutpath)
    


'''
# develop    
def gettemporalstatsofresource(resourcename):
    cfdmonthlyresourcecount = CFDhelpers.csv2csv()
    for month in cfdmonthlyresourcecount.conditions():
        print cfdmonthlyresourcecount[month][resourcename]   


# for reporting crawl-back
# develop
# filesfolder contains many newsitem.txt
def gettimerange(filesfolder):
    return timerange (min-max), numoffilesinfolder

'''


if __name__ == "__main__":
    
    get_newsmetadata()
    prune_newsmetadata()
    visualize_monthly_news_stats()
    
    
    
    
    '''
    rescatmap = get_resourcecatmap(corpuspath="/home/dicle/Dicle/Tez/dataset/recrawl_missing/newsitems/")
    resources = rescatmap.keys()
    print rescatmap
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    