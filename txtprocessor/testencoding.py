# -*- coding: utf-8 -*-
'''
Created on Jan 1, 2014

@author: dicle
'''

import os
import codecs
import re

from sentimentfinding import IOtools

replacements = {"Ä±"  : "ı",
                "Ã¶"  : "ö",
                "Ã¼"  : "ü",
                "Ä" : "ğ",
                "Å" : "ş",
                "Ã§" : "ç",
                #===============================================================
                "Ä°" : "İ",
                "Ã" : "Ö",
                "Ã" : "Ü",
                # "" : "Ğ",
                "Å" : "Ş",
                # "" : "Ç",
                #===============================================================
                "Â" : "",
                "â" : '"'
                }



def todisc_txt(txt, path, m="w", enc='utf-8'):
    f = codecs.open(path, m, encoding=enc)
    f.write(txt)
    f.close()
    
def readtxtfile(path):
    f = codecs.open(path,encoding='utf8')
    rawtext = f.read()
    f.close()
    return rawtext    


def replacetrchars(text):
  
    for fromchar, tochar in replacements.iteritems():
        print fromchar, tochar
        text = text.replace(fromchar, tochar)
    return text

def fixencoding(path, fname):
    inp = os.path.join(path, fname+".txt")
    try:
        return readtxtfile(inp)
    except UnicodeDecodeError:
        f = codecs.open(inp, encoding='iso-8859-9')
        rawtext = f.read()
        f.close()
        rawtext = replacetrchars(rawtext)
        outp = os.path.join(path, "fixed", fname+".txt")
        #rawtext = rawtext.encode("utf-8").decode("iso-8859-9")
        
        
        
        todisc_txt(rawtext, outp)   #, enc='iso-8859-9')
        
        text = readtxtfile(outp)
        text = replacetrchars(text)
        return text
    

 


def main():
    p1 = "/home/dicle/Dicle/Tez/dataset/dataset_onur/newsitems/solhaber/"
    
    folders = IOtools.getfoldernames_of_dir(p1)
    for foldername in folders:
        p2 = os.path.join(p1, foldername)
        IOtools.ensure_dir(os.path.join(p2, "fixed"))
        filenames = IOtools.getfilenames_of_dir(p2, removeextension=True)
        for fname in filenames:
            fixencoding(p2, fname)

    


if __name__ == "__main__":
    
    main()
    
    
    
    