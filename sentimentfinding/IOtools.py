# -*- coding: utf-8 -*-
'''
Created on Aug 29, 2012

@author: dicle
'''

import pandas as pd
import codecs
import os


countrypath = "/home/dicle/Dicle/Tez/dataset/country/"
countries = ["syria","turkey"]


magnitudewords_path = "/home/dicle/Dropbox/Tez/system/data/magnitude/"
magnitudewords_file = magnitudewords_path+"magnitudewords_tr.txt"


results_rootpath = "/home/dicle/Dicle/Tez/output/"
img_output = "/home/dicle/Dicle/Tez/output/img/"

picklepath = "/home/dicle/Dicle/Tez/output/pickle/"

matrixpath = results_rootpath+os.sep+"matrix/"

# returns the names of the files and dirs in the given directory *path*
def getfilenames_of_dir(path, removeextension=True):
    files = os.listdir(path)
    filenames = []
    for fileitem in files:
        if os.path.isfile(path+os.sep+fileitem):
            if removeextension:
                filename = fileitem.split(".")[0]   # remove extension if any
            else:
                filename = fileitem
            filenames.append(filename)
        
    return filenames

def getfoldernames_of_dir(path):
    files = os.listdir(path)
    foldernames = []
    for fileitem in files:
        if os.path.isdir(path+os.sep+fileitem):
            foldernames.append(fileitem)
    return foldernames


# merges all the txt files located in rootpath and records the new file in onefilepath
def mergetxtfiles(rootpath, onefilepath):
    filenames = getfilenames_of_dir(rootpath, False)
    
    for fname in filenames:
        lines = readtextlines(rootpath + os.sep + fname)
        todisc_list(onefilepath, lines, mode="a")
        

# ensures if the directory given on *f* exists. if not creates it.
def ensure_dir(f):
    #d = os.path.dirname(f)
    if not os.path.exists(f):
        os.makedirs(f)
    return f 


def printdictionary(dct):
    for key,value in dct.items():
        print key," : ",value

def dict_tostring(dct):
    out = ""
    for key,value in dct.items():
        out += " "+str(key)+" : "+str(value)+"\n"
    return out


# matrix is a list of lists
def todisc_matrix(matrix, path, mode="w"):
    f = codecs.open(path, mode, encoding='utf8')
    for row in matrix:
        outstr = ""
        for item in row:
            outstr = outstr + str(item) + "\t"
        f.write(outstr+"\n")
    f.close()

def todisc_txt(txt, path, mode="w"):
    f = codecs.open(path, mode, encoding='utf8')
    f.write(txt)
    f.close()
    

def txttodisc_append(txt, path):
    f = codecs.open(path,"a", encoding='utf8')
    f.write(txt)
    f.close()

def todisc_list(path, lst, mode="w"):
    f = codecs.open(path, mode, encoding='utf8')
    
    i = 0
    for item in lst:
        f.write(item.__str__()+"\n")
        #print str(i)," - ",item.__str__()
        i = i+1
    f.close()

def wordletxt_todisc(words, wordweightvector, path):
    weightedwordlist = [word+"\t"+str(weight) for word, weight in zip(words, wordweightvector) if weight > 0.0]
    #todisc_list(path, weightedwordlist) 
    f = codecs.open(path, mode="w", encoding='utf8')
    print f
    for item in weightedwordlist:
        print type(item),"  ",item.decode("utf-8")
        f.write(item.decode("utf-8")+"\n")
    f.close()
    
    

def todisc_freqdist(outpath, fdist):
    weightedlist = [item+"\t"+str(fdist[item]) for item in list(fdist)]
    todisc_list(outpath, weightedlist)
    

def readtxtfile(path):
    f = codecs.open(path,encoding='utf8')
    rawtext = f.read()
    f.close()
    return rawtext

def readtextlines(path):
    f = codecs.open(path,"r", encoding='utf8')
    lines = f.readlines()
    lines = [line.strip() for line in lines if not line.isspace()]
    f.close()
    return lines

def printlist(lst):
    for w in lst:
        print w
        

def readcsv(csvpath):
    return pd.read_csv(csvpath, sep="\t",header=0, encoding='utf-8')
 

def tocsv(df, csvpath):
    df.to_csv(csvpath, index=False, header=True, sep="\t", encoding='utf-8')


if __name__ == "__main__":
    corpuspath = "/home/dicle/Dicle/Tez/dataset/dataset750/"
    rootpath = corpuspath
    folders = getfoldernames_of_dir(corpuspath)
    print folders
    for foldername in folders:
        rootpath = corpuspath + os.sep + foldername + os.sep
        print rootpath
        
    
        
