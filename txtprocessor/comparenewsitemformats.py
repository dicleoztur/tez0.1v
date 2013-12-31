# -*- coding: utf-8 -*-
'''
Created on Dec 20, 2013

@author: dicle
'''
 
import os
import time
import xml.etree.ElementTree as ET
import codecs

from sentimentfinding import IOtools
import dateutils, listutils
from processing import Crawling2

def readtxtfile(path):
    f = codecs.open(path,encoding='utf8')
    rawtext = f.read()
    f.close()
    return rawtext


def getnewsmetadata_usual(newspath, tags):
    wholetext = readtxtfile(newspath)
    tagdata = {}
   
    for tag in tags:
        marker1 = "<"+tag+">"
        marker2 = "</"+tag+">"
       
        data = Crawling2.extractitem(marker1, marker2, wholetext).lower()
        if tag == "date":
            data = dateutils.parsewrittendate(data)
        tagdata[tag] = data
    return tagdata


def getnewsmetadata_xml(newspath, tags):
    tagdata = {}
    
    tree = ET.parse(newspath)
    itemroot = tree.getroot()
      
    for tg in tags:
        item = itemroot.find(tg)
        print tg,item.text
        if item is not None:
            tagdata[tg] = item.text 
    return tagdata




if __name__ == "__main__":
    path = "/home/dicle/Dicle/Tez/corpusstats/formatsamples/"
    filename = "251134"
    xpath = os.path.join(path, "xml", filename+".xml")
    upath = os.path.join(path, "usual", filename+".txt")

    
    tags = ["id","cat","ttxtt","author"]
    
    startU = time.time()
    print "usual: ",getnewsmetadata_usual(upath, tags)
    
    startX = time.time()
    print "xml: ",getnewsmetadata_xml(xpath, tags)
    endX = time.time()

    print "Usual took: ",str(startX - startU)
    print "XML took: ",str(endX - startX)
    
    
    
