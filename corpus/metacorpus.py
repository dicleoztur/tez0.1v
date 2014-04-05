# -*- coding: utf-8 -*- 
'''
Created on Dec 20, 2013

@author: dicle
'''

import os



# META COLLECTION

rawcorpuspath = "/home/dicle/Dicle/Tez/dataset/dataset_merged/newsitems/"   #"/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/newsitems/"
#rawcorpuspath = "/home/dicle/Dicle/Tez/corpusstats/samples/"
#rawcorpuspath = "/home/dicle/Dicle/Tez/dataset/dataset_diclecs/newsitems/"
#rawcorpuspath = "/home/dicle/Dicle/Tez/dataset/recrawl_missing/newsitems/"

xmlcorpuspath = "/home/dicle/Dicle/Tez/dataset/xmldataset/"

resources = ["cumhuriyet", "vakit", "radikal"]  # ["radikal", "cumhuriyet", "vakit"]

newsitemtags = ['id', 'link', 'resource', 'date', 'author', 'title', 'cat', 'ttxtt']

textfileextension = ".txt"






# META STATS 
statspath = "/home/dicle/Dicle/Tez/corpusstats/dataset_merged"       #recrawled/" #/dateenhanced/"
metafilename = "corpusbigstats.csv"
prunedmetafilename = "corpusbigstats-pruned.csv"
prunedmetafilepath = statspath + os.sep + metafilename
statsheader = ["newsid", "resource", "category", "date", "numofwords", "numofchars"]
csvsep = "\t"
relevanttags = ["id", "resource", "cat", "date", "ttxtt"]

imgfolder = statspath + os.sep + "img" + os.sep

'''
resourcecategorymap = {"cumhuriyet" : ["siyaset", "ekonomi", "dünya", "spor", "türkiye"],
                       "radikal" : ["politika", "ekonomi", "dünya", "spor", "türkiye"],
                       "vakit" : ["siyaset", "ekonomi", "dünya", "spor", "güncel"]}

'''

'''
resourcecategorymap = {"politics" : ["cumhuriyet-siyaset", "radikal-politika", "vakit-siyaset"],
                       "economy"  : ["cumhuriyet-ekonomi", "radikal-ekonomi", "vakit-ekonomi"],
                       "world"    : ["cumhuriyet-dunya", "radikal-dunya", "vakit-dunya"],
                       "sports"   : ["cumhuriyet-spor", "radikal-spor", "vakit-spor"],
                       "turkey"   : ["cumhuriyet-turkiye", "radikal-turkiye", "vakit-guncel"]}


'''

# added sol
resourcecategorymap = {"politics" : ["radikal_politika", "vakit_siyaset", "solhaber_devlet-ve-siyaset"],
                       "economy"  : ["radikal_ekonomi", "vakit_ekonomi", "solhaber_ekonomi", "solhaber_sonuncu-kavga"],
                       "world"    : ["radikal_dunya", "vakit_dunya", "solhaber_dunyadan", "solhaber_enternasyonal-gundem"],
                       "turkey"   : ["radikal_turkiye", "vakit_guncel", "solhaber_kent-gundemleri", "solhaber_soldakiler"]}


resourcecategorymap2 = {"economy" : {"radikal" : ["ekonomi"], 
                                     "vakit" : ["ekonomi"],
                                     "solhaber": ["ekonomi", "sonuncu-kavga"]},
                        "politics" : {"radikal" : ["politika"],
                                      "vakit" : ["siyaset"],
                                      "solhaber" : ["devlet-ve-siyaset"]},
                        "world" : {"radikal" : ["dunya"],
                                   "vakit" : ["dunya"],
                                   "solhaber" : ["dunyadan", "enternasyonal-gundem"]},
                        "turkey" : {"radikal" : ["turkiye"],
                                    "vakit" : ["guncel"],
                                    "solhaber" : ["kent-gundemleri", "soldakiler"]}
                        }

'''
resourcecategorymap = {"politics" : ["radikal-politika", "vakit-siyaset", "solhaber-devlet-ve-siyaset"],
                       "economy"  : ["radikal-ekonomi", "vakit-ekonomi", "solhaber-ekonomi", "solhaber-sonuncu-kavga"],
                       "world"    : ["radikal-dunya", "vakit-dunya", "solhaber-dunyadan", "solhaber-enternasyonal-gundem"],
                       "sports"   : ["radikal-spor", "vakit-spor"],
                       "turkey"   : ["radikal-turkiye", "vakit-guncel", "solhaber-kent-gundemleri", "solhaber-soldakiler"]}

'''


# CORPUS INFO

datestart = "2012-07-01"
dateend = "2013-04-01"



# LEARNABLE SETS
annotatedrootpath = "/home/dicle/Dicle/Tez/corpusstats/annotatedtexts/"   #clusterable/"
annotateddbdumpcsvpath = "/home/dicle/Dicle/Tez/corpusstats/clusterable/rawdata/"

'''##
userannotatedpath = annotatedrootpath + os.sep + "userannotated" +os.sep
randomannotatedpath = annotatedrootpath + os.sep + "randomannotated" +os.sep
singleantcsv = "singleannotated.csv"
doubleantcsv = "doubleannotated.csv"
##'''

annotationstatsfolder = "/home/dicle/Dicle/Tez/corpusstats/annotationstats/"


annotationtypes = ["single", "double"]
taggertypes = ["random", "user"]



subjectivity_basic_labels = {12: "subjective",
                             34: "objective"}


subjectivity_label_values = {1: "Fazla taraflı / manipülatif",
                             2: "Taraflı",
                             3: "Tarafsız denebilir",
                             4: "Tamamen tarafsız"}


def encode_subjectivity_labels(literallabel):
    literal2value = {"Fazla taraflı / Manipülatif" : 1,
                                 "Taraflı / Subjektif" : 2,
                                 "Tarafsız denebilir" : 3,
                                 "Tamamen tarafsız / Tamamen objektif" : 4,
                                 "Karar veremedim / Belirsiz" : 5}
    return literal2value[literallabel.encode("utf8")]
    



# EXPERIMENTS
learningrootpath = "/home/dicle/Dicle/Tez/corpusstats/learning2/"
learningdatapath = learningrootpath + os.sep + "data" + os.sep
learningexperimentspath = learningrootpath + os.sep + "experiments" + os.sep




# utils
labelfoldername = "labels"
finaldatasetfoldername = "finaldatasets"
def get_annotatedtexts_folder_path(annottype):
    folderpath = os.path.join(annotatedrootpath, annottype)
    return folderpath # if asked to write, ensure_dir where asked


def get_annotatedtexts_file_path(annotationtype, taggertype):
    filename = annotationtype + "_" + taggertype + "-evals.csv"
    filepath = get_annotatedtexts_folder_path(annotationtype) + os.sep + filename
    return filepath


# combined feature datasets of one annottype(sing-doub) and setsize
def get_datasets_path(annotationtype, setsize, datarootpath=learningdatapath):
    path = os.path.join(datarootpath, annotationtype, setsize, finaldatasetfoldername)
    return path


def get_labels_path(annotationtype, setsize, datarootpath=learningdatapath):
    path = os.path.join(datarootpath, annotationtype, setsize, labelfoldername)
    return path














