'''
Created on Jan 25, 2014

@author: dicle
'''


import os

from sentimentfinding import IOtools



def main(folder):
    singlejsonpath = os.path.join(folder, "singleannotation.json")
    doublejsonpath = os.path.join(folder, "doubleannotation.json")
    
    singleas = IOtools.readjson(singlejsonpath)
    doubleas = IOtools.readjson(doublejsonpath)
    
    
    print "Single assignments:"
    traversesingles(singleas)
    print "Double assignments:"
    traversedoubles(doubleas)
    


def traversesingles(dct):
    for userid, resourcedist in dct.iteritems():
        for resourcename, catdist in resourcedist.iteritems():
            for catname, monthdist in catdist.iteritems():
                for month, ntexts in monthdist.iteritems():
                    print "+ ",
                print



    
    
    