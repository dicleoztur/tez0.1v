# -*- coding: utf-8 -*- 
'''
Created on Dec 20, 2013

@author: dicle
'''

import os



# META COLLECTION

#rawcorpuspath = "/home/dicle/Dicle/Tez/dataset/dataset_merged/newsitems/"   #"/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/newsitems/"
#rawcorpuspath = "/home/dicle/Dicle/Tez/corpusstats/samples/"
rawcorpuspath = "/home/dicle/Dicle/Tez/dataset/dataset_diclecs/newsitems/"

xmlcorpuspath = "/home/dicle/Dicle/Tez/dataset/xmldataset/"

resources = ["cumhuriyet", "vakit", "radikal"]  # ["radikal", "cumhuriyet", "vakit"]

newsitemtags = ['id', 'link', 'resource', 'date', 'author', 'title', 'cat', 'ttxtt']

fileext = ".txt"






# META STATS 
statspath = "/home/dicle/Dicle/Tez/corpusstats/dataset_diclecs/" #/dateenhanced/"
metafilename = "corpusbigstats.csv"
prunedmetafilename = "corpusbigstats-pruned.csv"
#metafilepath = statspath + os.sep + metafilename
statsheader = ["newsid", "resource", "category", "date", "numofwords", "numofchars"]
csvsep = "\t"
relevanttags = ["id", "resource", "cat", "date", "ttxtt"]

imgfolder = statspath + os.sep + "img" + os.sep

'''
resourcecategorymap = {"cumhuriyet" : ["siyaset", "ekonomi", "dünya", "spor", "türkiye"],
                       "radikal" : ["politika", "ekonomi", "dünya", "spor", "türkiye"],
                       "vakit" : ["siyaset", "ekonomi", "dünya", "spor", "güncel"]}

'''
resourcecategorymap = {"politics" : ["cumhuriyet-siyaset", "radikal-politika", "vakit-siyaset"],
                       "economy"  : ["cumhuriyet-ekonomi", "radikal-ekonomi", "vakit-ekonomi"],
                       "world"    : ["cumhuriyet-dunya", "radikal-dunya", "vakit-dunya"],
                       "sports"   : ["cumhuriyet-spor", "radikal-spor", "vakit-spor"],
                       "turkey"   : ["cumhuriyet-turkiye", "radikal-turkiye", "vakit-guncel"]}








