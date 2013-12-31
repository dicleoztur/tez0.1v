'# -*- coding: <utf-8>-*-'

'''
Created on Jul 23, 2012

@author: dicle
'''

import re


text = "ve yerel haberler, Dünyadan gelişmeler <title> Emniyet Genel Müdürlüğüne sanal saldırı </title> <script src=/D/j/1/share.js?v=3 type=text/javascript></script>"

items = []
items = text.split()

start = items.index('<title>')
end = items.index('</title>')

title = ' '.join(items[start+1:end])
print title




start = text.find('<title>')
end = text.find('</title>')
print text[start+8:end]


txt = ""
start = "x"

pattern1 = r"\&i=[0-9]{5,9}"
pattern2 = r'[0-9]{5,9}'


url = 'ef="?t=06.08.2012&i=400498&k=m2">Geçitlerin kapatılması insani krize yol'
x = re.findall(pattern1,url)
z = re.findall(pattern2, "i=400498")
print "URL:",x," ",z 




z = "İran     &rsquo;da zoru  &ldquo; nlu"
#z = z.replace("&rsquo;", "\"")


p = re.compile( '&(r|l)(s|d)quo;')
z = p.sub( '\"', z)
print z

z = z.replace(r"\s+", " ")
print z





p4 = re.compile(r'</p>?')
p5 = re.compile(r'<?/p>') 
p6 = re.compile(r'<p>?')
text = "z çok güzel güle güle kullanın</p>"   
print p4.sub(" ",text) 



s = "Suriye sınırında PKK bayrağı  / Dünya / Radikal İnternet  (dha)"
t = s.split("\\")
print t[0]



p7 = re.compile(r'\(w+\)')
print p7.sub(' ',s)
