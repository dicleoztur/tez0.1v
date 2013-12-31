'''
Created on Aug 6, 2012

@author: dicle
'''
import urllib2
import re


url = "http://www.radikal.com.tr/Radikal.aspx?aType=RadikalDetayV3&ArticleID=1096386&CategoryID=77"
url2 = "http://www.cumhuriyet.com.tr/?hn=356940"
url3 = "http://www.yenisafak.com.tr/Dunya/?t=06.08.2012&i=400358&k=c11"

f = urllib2.urlopen(url3)
rawhtml = f.read()
#rawhtml = rawhtml.encode('iso-8859-9')
f.close()
encoding = f.headers['content-type'].split('charset=')[-1]

print encoding
print f.headers['content-type']


out = ""
if out == "":
    print "xxxxx"
    

title = "jfkshfk - ytuywuq"
title = title.split("\/ || \-")[0]

p = "[/-]"
print re.split(p, title)[0]


print title