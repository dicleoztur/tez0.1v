'# -*- coding: <utf-8>-*-'

'''
Created on Jul 6, 2012

@author: dicle
'''

from BeautifulSoup import BeautifulSoup
from urllib import urlopen

import re
import nltk



def readhtml(url):
    rawhtml = urlopen(url).read()
    text = nltk.clean_html(rawhtml)
    return text, rawhtml

    




if __name__ == "__main__":
    
    html_doc = """ <html><head><title>The Dormouse's story</title></head>
    <p class="title"><b>The Dormouse's story</b></p>
    <p class="story">Once upon a time there were three little sisters; and their names were
    <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
    <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
    <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
    and they lived at the bottom of a well.</p>
    <p class="story">...</p>
    """
    
    soup = BeautifulSoup(html_doc)
    
    #print soup.prettify()

    print "0", soup.contents[0]

    print "1", soup.contents[1]
    
    links = soup.a
    print links.renderContents()
    
    
    print "HTMLLLLLLLLLLL"
    url = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalDetayV3&ArticleID=1093505&CategoryID=81"
    
    out = readhtml(url)
    
    rawhtml = out[1]
    print rawhtml
    
    reader = BeautifulSoup(rawhtml)
    print reader.contents[1]
