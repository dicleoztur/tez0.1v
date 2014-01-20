'''
Created on Dec 20, 2013

@author: dicle
'''

import os
import pandas as pd
import xml.etree.ElementTree as ET

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
    data = Crawling2.extractitem(marker1, marker2, content).lower()
    if tag == "date":
        data = dateutils.parsewrittendate(data)
    return data.strip()


# returns the title and content of the newsitem located in filepath
def get_news_article(filepath):
    wholetext = texter.readtxtfile(filepath)
    content = gettagdata(wholetext, "ttxtt")
    title = gettagdata(wholetext, "title")
    return title,content



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



def main(corpuspath=metacorpus.rawcorpuspath, resources=metacorpus.resources, csvfilefolder=metacorpus.statspath, csvfilename=metacorpus.metafilename):
    csvfilepath = os.path.join(csvfilefolder, csvfilename)
    preparemetadatafile(csvfilepath)   
    recordnewsmetadata_crawltxt(corpuspath, resources, csvfilepath)
    
     

if __name__ == "__main__":
    #preparemetadatafile()
    #recordnewsmetadata_crawltxt()
    
    print metacorpus.metafilepath
    
    
    