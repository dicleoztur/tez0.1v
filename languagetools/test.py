'''
Created on Sep 6, 2012

@author: dicle
'''

import re


s = 'git[Verb]+[Pos]-mAk[Noun+Inf1+A3sg+Pnon]+[Nom]'

morphemes = re.split(r'\][+-]\[', s)

for morpheme in morphemes:
    print morpheme

  
rootal = re.split(r'\[', morphemes[0])
print morphemes[0], " ", rootal
root = rootal[0]
rootPOS = rootal[1]

rootinfo = (root, rootPOS)
print "Root: ",rootinfo
