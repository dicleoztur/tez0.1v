'''
Created on Dec 20, 2013

@author: dicle
'''

import os
import pandas as pd
import xml.etree.ElementTree as ET
import random

from sentimentfinding import IOtools
from txtprocessor import dateutils,texter
from processing import Crawling2
import metacorpus



csvsep = metacorpus.csvsep
_metafilepath = os.path.join(metacorpus.statspath, metacorpus.metafilename)


def preparemetadatafile(csvfilepath=_metafilepath):
    header = csvsep.join(metacorpus.statsheader)
    print csvfilepath
    IOtools.todisc_txt(header+"\n", csvfilepath)
    


def gettagdata(content, tag):
    marker1 = "<"+tag+">"
    marker2 = "</"+tag+">"
    data = Crawling2.extractitem(marker1, marker2, content)
    if tag == "date":
        data = dateutils.parsewrittendate(data)
    return data.strip()


# returns the title and content of the newsitem located in filepath
def get_news_article(filepath):
    wholetext = texter.readtxtfile(filepath)
    content = gettagdata(wholetext, "ttxtt")
    title = gettagdata(wholetext, "title")
    return title,content

# returns the title and content of the newsitem located in filepath
def get_news_article2(filepath):
    wholetext = texter.readtxtfile(filepath)
    content = gettagdata(wholetext, "ttxtt")
    title = gettagdata(wholetext, "title")
    date = gettagdata(wholetext, "date")
    return content, title, date


def getmetadata_fromtxt(filepath):
    print filepath
    wholetext = texter.readtxtfile(filepath)
    
    newsid = gettagdata(wholetext, "id")
    resource = gettagdata(wholetext, "resource")
    category = gettagdata(wholetext, "cat")
    date = gettagdata(wholetext, "date")
    content = gettagdata(wholetext, "ttxtt")
    words = texter.getwords(content, nostopwords=False)
    numofwords = len(words)
    numofchars = len(content)
    
    metadata = csvsep.join([newsid, resource, category, date, str(numofwords), str(numofchars)])
    return metadata+"\n"



def getmetadata_fromxml(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    newsid = root.find("id").text
    resource = root.find("resource").text
    category = root.find("cat").text
    date = root.find("date").text
    date = dateutils.parsewrittendate(date)
    content = root.find("ttxtt").text
    words = texter.getwords(content, nostopwords=False)
    numofwords = len(words)
    numofchars = len(content)
    
    metadata = csvsep.join([newsid, resource, category, date, str(numofwords), str(numofchars)])
    return metadata+"\n"



def recordnewsmetadata_crawltxt(corpuspath=metacorpus.rawcorpuspath, resourcefolders=metacorpus.resources, csvfilepath=_metafilepath):
      
    for resource in resourcefolders:
        xp1 = IOtools.ensure_dir(os.path.join(corpuspath, resource))  # replicate the folder hierarchy into the xml folder as well
        categories = IOtools.getfoldernames_of_dir(xp1)
        
        for cat in categories:
            xp2 = IOtools.ensure_dir(os.path.join(xp1, cat))
            filenames = IOtools.getfilenames_of_dir(xp2, removeextension=False)
            
            for filename in filenames:
                filepath = xp2 + os.sep + filename 
                metadataline = getmetadata_fromtxt(filepath)    #metadataline = getmetadata_fromtxt(filepath+".txt") 
                #print csvfilepath               
                IOtools.todisc_txt(metadataline, csvfilepath, mode="a")
        
            print "finished "+resource+"/"+cat


# newsid: resourcename-normalizedcatname-textid
def newsid_to_filepath(newsid):
    items = newsid.split("-")
    if len(items) != 3:
        return None
    resourcename = items[0]
    catname = items[1]
    textid = items[2]
    
    trcatname = metacorpus.resourcecategorymap2[catname][resourcename]
    for altcatname in trcatname:   # some resources have multiple category names mapped to englishcatname(catname in the newsid)
        catname = altcatname
        folderpath = os.path.join(metacorpus.rawcorpuspath,resourcename, catname)
        if IOtools.folder_contains_file(folderpath, textid):
            filepath = os.path.join(metacorpus.rawcorpuspath,resourcename, catname, textid+metacorpus.textfileextension)
            return filepath
    return None

        

def main(corpuspath=metacorpus.rawcorpuspath, resources=metacorpus.resources, csvfilefolder=metacorpus.statspath, csvfilename=metacorpus.metafilename):
    csvfilepath = os.path.join(csvfilefolder, csvfilename)
    preparemetadatafile(csvfilepath)   
    recordnewsmetadata_crawltxt(corpuspath, resources, csvfilepath)
    
     

if __name__ == "__main__":
    #preparemetadatafile()
    #recordnewsmetadata_crawltxt()
    
    #print metacorpus.metafilepath
    newsids = IOtools.readcsv(os.path.join(metacorpus.randomannotatedpath, "randomly-"+metacorpus.singleantcsv))
    newsids = newsids.loc[random.sample(range(newsids.shape[0]), 10), "questionname"].values.tolist()
    for newsid in newsids:
        print newsid," : ",newsid_to_filepath(newsid)
        
        
        
        
        


    
    