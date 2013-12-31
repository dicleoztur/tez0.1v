'''
Created on Aug 30, 2012

@author: dicle
'''
import os
import codecs
import pickle

import IOtools





def readpolarityfiles(path):
    f = codecs.open(path,"r", encoding='utf8')
    lines = f.readlines()
    lines = [line.strip().split("#")[0] for line in lines]
    f.close()
    return lines



wordspath = "/home/dicle/Dicle/Tez/dataset/output/samplewords.txt"
polpath = "/home/dicle/Dicle/Tez/polaritybase"


words = IOtools.readtextlines(wordspath)

positivewords = readpolarityfiles(polpath+os.sep+"positive.txt")
negativewords = readpolarityfiles(polpath+os.sep+"negative.txt")


'''
posset = set(positivewords)
negset = set(negativewords)
wordset = set(words)

nonpositive = wordset - posset

nonnegative = wordset - negset

'''


print str(len(words))," ",str(len(positivewords))

poss = [w for w in words if w in positivewords]
negs = [w for w in words if w in negativewords]

print "\nPositive words"
IOtools.printlist(poss)

print "\nNegative words"
IOtools.printlist(negs)


print str(len(poss))," ",str(len(negs))





path1 = polpath+os.sep+"pozitif.txt"
path2 = polpath+os.sep+"negatif.txt"
'''
IOtools.writelist(path1, positivewords)
IOtools.writelist(path2, negativewords)
'''