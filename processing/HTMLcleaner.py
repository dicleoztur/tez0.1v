'# -*- coding: <utf-8>-*-'

'''
Created on Jul 19, 2012

@author: dicle
'''


#RADIKAL crawler

import nltk
from urllib import urlopen


def todisc(outstr, rootpath):
    f = open(rootpath+"-content.txt","w+")
    f.write(outstr)
    f.close()



def readhtml(url):
    rawhtml = urlopen(url).read()
    text = nltk.clean_html(rawhtml)
    return text, rawhtml


def gettitle(rawhtml):
    


print "HTMLLLLLLLLLLL"
url = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalDetayV3&ArticleID=1094627&CategoryID=77"

#url = "http://www.skylit.com/javamethods/faqs/javaindos.html"
out = readhtml(url)

print "RAW HTML"
rawhtml = out[1]
print rawhtml


print "TEXT"
txt = out[0]
print txt
