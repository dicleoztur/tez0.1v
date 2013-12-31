'''
Created on Aug 6, 2012

@author: dicle
'''

import IO

import os


class NewsResource:
    
    name = ""
    rootlink_id = ""
    rootlink_item = ""
    
    newsidpath = ""
    
    idlimit1 = ""
    idlimit2 = ""
    
    idpatternInLink1 = ""
    idpatternInLink2 = ""
    
    markerTitle1 = ""
    markerTitle2 = ""
    
    markerDate1 = ""
    markerDate2 = ""
    
    markerAuthor1 = ""
    markerAuthor2 = ""

    markerText1 = ""
    markerText2 = ""
    
    encoding = ""
    
    
    def __init__(self, name, rootlink_id, rootlink_item, idlimit1, idlimit2, idpatternInLink1, idpatternInLink2, markerTitle1, markerTitle2, markerText1, markerText2, markerDate1="", markerDate2="", markerAuthor1="", markerAuthor2=""):
        self.name = name
        self.rootlink_id = rootlink_id
        self.rootlink_item = rootlink_item
        self.newsidpath = IO.IDlistPath+os.sep+self.name+os.sep
        self.newsidpath = IO.ensure_dir(self.newsidpath)
        self.idlimit1 = idlimit1
        self.idlimit2 = idlimit2
        self.markerTitle1 = markerTitle1
        self.markerTitle2 = markerTitle2
        self.markerText1 = markerText1
        self.markerText2 = markerText2
        self.idpatternInLink1 = idpatternInLink1
        self.idpatternInLink2 = idpatternInLink2
        self.markerDate1 = markerDate1
        self.markerDate2 = markerDate2
        self.markerAuthor1 = markerAuthor1
        self.markerAuthor2 = markerAuthor2
        
        
    def tostring(self):
        outstr = ""
        outstr += self.name+" : "+self.rootlink_item
        return outstr
        
    def toscreen(self):
        print self.tostring()
        
        
    def setEncoding(self, encoding):
        self.encoding = encoding
        
    
    
    


        