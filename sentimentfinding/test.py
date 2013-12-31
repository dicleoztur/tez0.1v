'''
Created on Aug 27, 2012

@author: dicle
'''

from Word import Word


def f4(seq): 
    # order preserving
    noDupes = []
    [noDupes.append(i) for i in seq if not noDupes.count(i)]
    return noDupes


l = []


l.append(Word('iddia'))
l.append(Word('bilgisayar'))
l.append(Word('internet'))
l.append(Word('istek'))
l.append(Word('internet'))

# s = list(set(l))



print "set"

s = f4(l)

    
s = sorted(s)

strrr= "hello"


set1 = set(l)

l2 = l + ['merhaba']

set2 = set(l2)
x = set1 - set2
print x