'# -*- coding: <utf-8>-*-'

'''
Created on Jul 23, 2012

@author: dicle
'''


import urllib2
import re
import os
import nltk

from NewsItem import NewsItem
#from IO import *
import IO




'''  Radikal '''





def extractitem(marker1, marker2, rawhtml):
    start = rawhtml.find(marker1)
    end = rawhtml.find(marker2)
    print start," ",end
    item = rawhtml[start+len(marker1)+1 : end]
    
    return item



def getnewsitem(rawhtml):
    markerTitle1 = '<title>'
    markerTitle2 = '</title>'
    title = extractitem(markerTitle1, markerTitle2, rawhtml)
    
    markerText1 = '<div id="metin2" class="fck_li">'
    markerText2 = '<div class="IndexKeywordsHeader"'    # veya 'id="hiddenTitle"'
    text = extractitem(markerText1, markerText2, rawhtml)
    text = nltk.clean_html(text)
    
    encoding=rawhtml.headers['content-type'].split('charset=')[-1]
    print encoding
    #text = unicode(text)
    return NewsItem(title, "", text, "")
    
    



def getnewsitem2(url):
    
    f = urllib2.urlopen(url)
    rawhtml = f.read()
    #rawhtml = rawhtml.encode('iso-8859-9')
    f.close()
    encoding = f.headers['content-type'].split('charset=')[-1]

    
    markerTitle1 = '<title>'
    markerTitle2 = '</title>'
    title = extractitem(markerTitle1, markerTitle2, rawhtml)
    title = IO.encodingToutf8(title, encoding)
    title = title.split("/")[0]
    title = IO.replaceSpecialChars(title)
    
    markerText1 = '<div id="metin2" class="fck_li">'
    markerText2 = '<div class="IndexKeywordsHeader"'    # veya 'id="hiddenTitle"'
    text = extractitem(markerText1, markerText2, rawhtml)
    text = nltk.clean_html(text)
    
    text = IO.encodingToutf8(text, encoding)
    text = IO.replaceSpecialChars(text)
    
    return NewsItem(title, "", text, "")
    




def retrieveNewsIDs(link):    
    content = readhtml(link)
    pattern1 = r";ArticleID=[0-9]{6,9}"
    pattern2 = r'[0-9]{6,9}'
        
    #preprocess to get the raw list of news links  
    
    limit1 = "<div class=\"cat-news\"><ol";
    limit2 = "var Geri = 'Geri'";
    item = extractitem(limit1,limit2,content)
    
    # extract ids
    list1 = re.findall(pattern1,item)
    #print list1
    listfinal = []
    for rawID in list1:
        newsid = re.findall(pattern2, rawID)[0]      #buradan bug cikabilir
        listfinal.append(newsid)

    listfinal = list(set(listfinal))
    return listfinal



def readhtml(url):
    f = urllib2.urlopen(url)
    rawhtml = f.read()
    #rawhtml = rawhtml.encode('iso-8859-9')
    f.close()
    
    return rawhtml






url = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalKategoriTumuV3&CategoryID=81&PAGE=1"
IDlist = retrieveNewsIDs(url)
print "List: ",IDlist


start = 1
numOfPages = 2
rooturl = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalKategoriTumuV3&CategoryID=81&PAGE="
IDlist = []
for i in range(start,start+numOfPages):
    url = rooturl + str(i)
    IDlist = IDlist + retrieveNewsIDs(url)

IDlist = list(set(IDlist))
path = IO.IDlistPath+os.sep+"dunya_newsIDs"+str(start)+"-"+str(numOfPages)+".txt"
IO.todisc_list(IDlist, path)



rootlink_news = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalDetayV3&ArticleID="
for newsid in IDlist:
    newslink = rootlink_news + str(newsid)
    extraction = getnewsitem2(newslink)
    extraction.toConsole()



'''
rawhtml = readhtml(url)


newsitem = getnewsitem(rawhtml)

newsitem.toConsole()  
    

# radikal

# category: dunya
link = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalKategoriV3&CategoryID=81"



numOfItems = 100

for i in range(numOfItems):
    print "item ",i
    
    
''' 