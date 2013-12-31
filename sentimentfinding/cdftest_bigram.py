# -*- coding: utf-8 -*- 
'''
Created on Apr 8, 2013

@author: dicle
'''




import os
import nltk
import pylab as plt
from numpy import *
import numpy as np
import random
from datetime import datetime

from txtprocessor import texter, listutils, dateutils
import IOtools
import keywordhandler
from languagetools import SAKsParser



# cat leri kaynagin kendi folder indan okumak daha iyi.
resourcecategories = {"radikal"   : ["dunya","turkiye","hayat","ekonomi","politika","spor","sinema"],
                      "vakit"     : ["guncel","siyaset","dunya","ekonomi"],
                      "cumhuriyet": ["dunya","turkiye","siyaset","ekonomi","kultur-sanat","cevre","saglik","bilim-teknik","yasam","spor"]}
'''
#cat_radikal = { 81:"dunya", 77:"turkiye", 80:"ekonomi", 41:"hayat", 78:"politika", 84:"spor" }     
cat_radikal = { 41:"hayat", 80:"ekonomi" }   #sinema: 120 bitti.   
#cat_radikal = { 78: "politika"}

cat_habervaktim = { 3:"guncel", 4:"siyaset", 5:"dunya", 6:"ekonomi", 7:"kultur-sanat", 8:"aile-yasam", 10:"bilim", 11:"saglik", 19:"spor", 20:"egitim", 23:"medya" }
cat_cumhuriyet = {6:"siyaset", 7:"turkiye", 8:"dunya", 9:"ekonomi", 12:"kultur-sanat", 17:"spor", 20:"yasam", 18:"bilim-teknik", 19:"saglik", 21:"cevre" }
'''


###############   Analyse a resource visually by its bigrams and words over days and months     ##########################
########################################################################################################################
def analyseresource(inresource, rootpath, numoffiles=0):
    for category in resourcecategories[inresource]:
        resource = inresource
        cat = category
        
        #rootpath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/newsitems/"     #vakit/siyaset/"
        newspath = rootpath + os.sep + resource + os.sep + cat + os.sep
        
        
        fileids = os.listdir(newspath)
        if numoffiles == 0:
            numoffiles = len(fileids)   #30
        fileids = fileids[:numoffiles]
        
    
        #===========================================================================
        # 
        # ''' doc by word adjacency list '''
        # cfdDocWord = nltk.ConditionalFreqDist((fileid[:-4], word) for fileid in fileids for word in texter.getnewsitem(newspath+os.sep+fileid)[0])
        # 
        # ''' word by doc as adjacency list '''
        # cfdWordDoc = nltk.ConditionalFreqDist((word, fileid[:-4]) for fileid in fileids for word in texter.getnewsitem(newspath+os.sep+fileid)[0])
        # 
        # '''  date by docid  '''
        # #cfdDateDocID = nltk.ConditionalFreqDist((date, fileid[:-4]) for fileid in fileids for _, date in texter.getnewsitem(newspath+os.sep+fileid))
        # # # of files at some date d : cfdDateDocID[d].N()
        # 
        # 
        
        filetitle = resource+"_"+cat+"_"+str(numoffiles)+"texts_"
         
        ''' date by word as adjacency list '''
        datewordpairs = []
        for fileid in fileids:
            
            path = newspath + os.sep + fileid
            words, date = texter.getnewsitem(path)
            print fileid, ": ",date," ",len(words)," ",type(words)
            for word in words:
                datewordpairs.append((date,word))
        cfdDateWord = nltk.ConditionalFreqDist(datewordpairs)
        
        ''' word by date '''
        #cfdWordDate = nltk.ConditionalFreqDist((word, date) for date,word in datewordpairs)
        
        
        '''   word by month   '''
        cfdMonthWord = nltk.ConditionalFreqDist((date[3:], (word,)) for date in cfdDateWord.conditions() 
                                                for word in list(cfdDateWord[date]))
        recordCFD(cfdMonthWord, "monthword_"+filetitle)
        
        mxitems = dateutils.sortdatelist(cfdMonthWord.conditions())
        bargraph_freqitem(cfdMonthWord, mxitems, "monthlyfrequentwords_"+filetitle, "Months", "TheMostFrequentWord") 
         
    
        
        
        '''   date by bigrams '''
        datebigrampairs = []
        for fileid in fileids:
            path = newspath + os.sep + fileid
            words, date = texter.getnewsitem(path)
            bigramlist = nltk.bigrams(words)
            #bigramlist = [" ".join(bigram) for bigram in bigramlist]
            #print fileid, ": ",date," ",len(words)," ",type(words)
            for bigram in bigramlist:
                datebigrampairs.append((date,bigram))
        cfdDateBigram = nltk.ConditionalFreqDist(datebigrampairs)
        recordCFD(cfdDateBigram, "datebigram-"+filetitle)
    
        '''    
        #  bar graph for bigrams by dates
        filetitle = resource+"_"+cat+"_"+str(numoffiles)+"texts_"   
        dxitems = dateutils.sortdatelist(cfdDateBigram.conditions())
        dyitems = [cfdDateBigram[date].N() for date in dxitems]
        bargraph_cfd2(dxitems, dyitems, filetitle+"DateBar", "Days", "Bigrams")
        '''
        
        #  bar graph for bigrams by months  
        cfdMonthBigram = nltk.ConditionalFreqDist((date[3:], word) for date in cfdDateBigram.conditions() 
                                                for word in list(cfdDateBigram[date]))
        recordCFD(cfdMonthWord, "monthbigram-"+filetitle)
        
        mxitems = dateutils.sortdatelist(cfdMonthBigram.conditions())
        
        bargraph_freqitem(cfdMonthBigram, mxitems, "monthlyfrequentbigrams_"+filetitle, "Months", "TheMostFrequentBigram")
##################################################################################################################################       
        


''' the paraneter cfd is a conditional frequency distribution of docs over words  (word, docs)  '''
def get_word_cooccurrences(cfd):
    wordpairs = [(w1,w2) for i,w1 in enumerate(cfd.conditions()) for j,w2 in enumerate(cfd.conditions()) if i < j]
    pairscommonslist = []
    
    for (w1,w2) in wordpairs:
        docs1 = list(cfd[w1])
        docs2 = list(cfd[w2])
        commondocs = listutils.getintersectionoflists(docs1, docs2)
        numofcooccurrences = len(commondocs)
        if numofcooccurrences != 0:
            pairscommonslist.append(((w1, w2), numofcooccurrences))
    return pairscommonslist


''' selects the cfd of each word of typekeywords from maincfd which is a distribution of words over docs (doc, words)  ''' 
def get_cfd_oftype(typekeywords, maincfd):
    cfd = nltk.ConditionalFreqDist( (fileid, word) for fileid in maincfd.conditions() for word in list(maincfd[fileid]) if word in typekeywords)
    return cfd

def get_cfd_ofpostag(postag, docslemmatized):
    cfd_postag = nltk.ConditionalFreqDist((fileid, literal) 
                                          for (fileid,lemmas) in docslemmatized 
                                          for (literal, literalPOS, root, rootPOS) in lemmas if literalPOS == postag)
    return cfd_postag


'''takes a list of lemmatized quadruples [(literal, literalPOS, root, rootPOS)] of docs,
    returns a distribution of postags like verb : n, adj : m,...   '''
def get_cfd_bypostags(docslemmatized):
    cfd_postags = nltk.ConditionalFreqDist((literalPOS, fileid)
                                           for (fileid, lemmas) in docslemmatized
                                           for (literal, literalPOS, root, rootPOS) in lemmas)
    return cfd_postags


#########
'''  stats     '''

def get_cond_with_max_outcome(cfd): 
    maxcond = ""
    maxvalue = -1
    for cond in cfd.conditions():
        outcomevalue = cfd[cond].max()
        if outcomevalue > maxvalue:
            maxcond = cond
            maxvalue = outcomevalue 
    return maxcond, outcomevalue 
    
####



#############
'''   graphs   '''
def bargraph_postags(cfdpostags, figname):
    postags = cfdpostags.conditions()
    postagfreqvalues = [cfdpostags[postag].N() for postag in cfdpostags.conditions()]
        
    plt.bar(arange(len(postags)), postagfreqvalues, align='center')
    plt.xticks(arange(len(postags)), postags)
    plt.xlabel("POS tags")
    plt.ylabel("Number of cumulative occurrences in docs")
    plt.title("POS tag occurrence in docs")
    plt.savefig(IOtools.results_rootpath+os.sep+figname+".png")


def bargraph_cfd(xitems, yitems, figname, xLabel, yLabel):
    #fig = plt.figure(dpi=100)
    
    plt.bar(arange(len(xitems)), yitems, align='center', facecolor='gray')
    plt.autoscale(True, axis='both')
    plt.tight_layout(pad=7.0)
    plt.xticks(arange(len(xitems)), xitems, rotation=90)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(figname)
    plt.savefig(IOtools.results_rootpath+os.sep+figname+".png") 



def bargraph_cfd2(xitems, yitems, figname, xLabel, yLabel, barlabels=None):
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    plt.bar(arange(len(xitems)), yitems, align='center')
    plt.autoscale(True, axis='both')
    plt.tight_layout(pad=7.0)
    plt.xticks(arange(len(xitems)), xitems, rotation=90)
    if barlabels:
        ypos = max(yitems)/2   # assuming yitems contains numeric values
        print "YITEMS", yitems
        print "ypos", ypos
        for i,label in enumerate(barlabels):
            #label = map(lambda s : s.encode('utf-8'), label)
            s = " ".join(label)
            plt.text(i, ypos, s, rotation=90, va='center', ha='center', color='purple')   # assuming barlabel contains lists or tuples   #va='center', ha='center',
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(figname)
    plt.savefig(IOtools.results_rootpath+os.sep+figname+".png",dpi=100) 
    plt.show()


def bargraph_foroneword(cfd, xitems, specword, figname, xLabel, yLabel):
    yitems2 = [cfd[cond][specword] for cond in xitems]
    bargraph_cfd2(xitems, yitems2, figname, xLabel, yLabel)

def bargraph_freqitem(cfd, xitems, figname, xLabel, yLabel, numofitems=0):
    yitemtriples = [(cfd[cond].max(), cfd[cond][cfd[cond].max()], cfd[cond].N()) for cond in xitems]  # contains (maxcondition, maxcondvalue, numofinstances) triples
    if numofitems == 0:
        yitems = [float(value) / numofwords for (_, value, numofwords) in yitemtriples]
    else:
        yitems = [float(value) / numofitems for (_, value, _) in yitemtriples]
    barlabels = [label+(" ("+str(numofoccurrences)+")",) for (label,numofoccurrences,_) in yitemtriples]
    bargraph_cfd2(xitems, yitems, figname, xLabel, yLabel, barlabels)




def plot_cfd(cfd, xitems, figname, xLabel, yLabel):   #xitems, yitems, figname, xLabel, yLabel, plotlabels=None):
    linelabels = [item for cond in xitems for item in list(cfd[cond])]
    linelabels = list(set(linelabels))   #uniquefy
    linelabels.sort()
    
    yitemslist = []
    for line in linelabels:
        yitems = []
        for cond in xitems:
            yitems.append(cfd[cond][line])
        yitemslist.append(yitems)
    
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    
    print linelabels
    print len(yitemslist)," labels: ",len(linelabels)
    i=0
    colors = ['r', 'b', 'g', 'c', 'y', 'k', 'm']
    
    cm = plt.get_cmap('gist_rainbow')
    numofcolors = len(linelabels)
    plt.gca().set_color_cycle([cm(1.*j/numofcolors) for j in range(numofcolors)])
    for yitems, linelabel in zip(yitemslist, linelabels):    
        plt.plot(arange(len(xitems)), yitems, color=cm(1.*i/numofcolors), label=linelabel)  #color=colors[i],
        i = i + 1
    
    plt.legend()
        
    plt.autoscale(True, axis='both')
    plt.tight_layout(pad=7.0)
    plt.xticks(arange(len(xitems)), xitems, rotation=90)
    '''
    if barlabels:
        ypos = max(yitems)/2   # assuming yitems contains numeric values
        print "YITEMS", yitems
        print "ypos", ypos
        for i,label in enumerate(barlabels):
            #label = map(lambda s : s.encode('utf-8'), label)
            s = " ".join(label)
            plt.text(i, ypos, s, rotation=90, va='center', ha='center', color='purple')   # assuming barlabel contains lists or tuples   #va='center', ha='center',
            '''
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(figname)
    plt.savefig(IOtools.results_rootpath+os.sep+figname+".png",dpi=100) 
    plt.show()



##########






def printCFD(cfd):
    for cond in cfd.conditions():
        totaloccurrence = cfd[cond].N()
        print cond," : "
        print "cumulative occ. ",totaloccurrence
        print "the most occ. element. ",cfd[cond].max()
        
        for item in list(cfd[cond]):
            print "\t",item," : ",cfd[cond][item]
    print "# of conds: ",len(cfd.conditions())




def recordCFD(cfd, filename):
    outstr = "\n"
    for cond in cfd.conditions():
        totaloccurrence = cfd[cond].N()
        outstr = outstr + "\n" + str(cond) + " :\n"
        outstr = outstr + "  cumulative occ. " + str(totaloccurrence)+"\n"
        outstr = outstr + "  the most occ. element. " + str(cfd[cond].max())+"\n"
        
        itemstr = "occurrences:\n"
        for item in list(cfd[cond]):
            itemstr += "\t" + str(item) + " : " + str(cfd[cond][item]) + "\n"
        
        outstr = outstr + itemstr   
    outstr = outstr + "# of conds: " + str(len(cfd.conditions()))
    IOtools.todisc_txt(outstr, IOtools.results_rootpath + os.sep + filename + ".txt")



        

if __name__ == "__main__":
    
    #counting words by genre
    '''
    cfd = nltk.ConditionalFreqDist((genre, word) for genre in brown.categories() for word in brown.words(categories=genre))
    #cfd.tabulate()
    print cfd.N()
    print len(cfd['news'])
    #cfd.plot()
    '''
    #rootpath = "/home/dicle/Dicle/Tez/dataset/dataset1500/"
    #rootpath = "/home/dicle/Dicle/Tez/dataset/dataset750/5/"
    
    rootpath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/temp/newsitems/"     #vakit/siyaset/"
    resource = "vakit"
    cat = "guncel"
    
    #analyseresource(resource, rootpath, numoffiles=5)
    
    newspath = rootpath + os.sep + resource + os.sep + cat + os.sep
    fileids = os.listdir(newspath)
    numoffiles = 500 #len(fileids)   #30
    fileids = fileids[:numoffiles]
    
    
    starttime = datetime.now()
#    ''' (root,word) by doc'''
    # (literal, literalPOS, root, rootPOS) = SAKsParser.lemmatizeword(word)
    '''
    cfd_roots = nltk.ConditionalFreqDist((root, fileid[:-4])
                                        for fileid in fileids
                                        for (literal, literalPOS, root, rootPOS) in SAKsParser.lemmatize_lexicon(texter.getnewsitem(rootpath+os.sep+fileid)))
    '''
    '''
    cfd_roots = nltk.ConditionalFreqDist((root, fileid) for fileid in cfd.conditions() 
                                        for (literal, literalPOS, root, rootPOS) in SAKsParser.lemmatize_lexicon(list(cfd[fileid])))
    '''



    filetitle = resource+"_"+cat+"_"+str(numoffiles)+"texts_"
    '''    day by word roots    '''    
    dailyroots = []
    dailyadjectives = []
    dailypostags = []
    postags = ["ADJ","ADV","Verb","Noun"]
    for fileid in fileids:
        path = newspath + os.sep + fileid
        words, date = texter.getnewsitem(path)
        doclemmas = SAKsParser.lemmatize_lexicon(words)
        for (_, literalPOS, root, _) in doclemmas:
            dailyroots.append((date, (root.decode('utf-8'),)))    # decoding'e dikkat
            dailypostags.append((date, literalPOS))
            '''if literalPOS == "ADJ":
                print literalPOS
                dailyadjectives.append((date, (literalPOS,)))
            '''
    
    cfdDatePOStag = nltk.ConditionalFreqDist(dailypostags)
    '''
    mxitems = dateutils.sortdatelist(cfdDatePOStag.conditions())
    plot_cfd(cfdDatePOStag, mxitems, "dailyfrequentPOStagsPlot_"+filetitle, "Days", "TheMostFrequentPOStag")
    '''
    
    cfdMonthPOStag = nltk.ConditionalFreqDist((date[3:], (word,)) for date in cfdDatePOStag.conditions() 
                                            for word in list(cfdDatePOStag[date]))
    recordCFD(cfdMonthPOStag, "monthpostag_"+filetitle)
    
    mxitems = dateutils.sortdatelist(cfdMonthPOStag.conditions())
    plot_cfd(cfdMonthPOStag, mxitems, "monthlyPOStagcounts_"+filetitle, "Months", "POStag_count") 
    
    '''
    recordCFD(cfdDatePOStag,"DailyPOStags_"+filetitle)
    
    mxitems = dateutils.sortdatelist(cfdDatePOStag.conditions())
    bargraph_freqitem(cfdDatePOStag, mxitems, "dailyfrequentPOStags_"+filetitle, "Days", "TheMostFrequentPOStag")
    '''
            
    '''        
    cfdDateADJ = nltk.ConditionalFreqDist(dailyadjectives)
    recordCFD(cfdDateADJ,"DailyAdjectives_"+filetitle)
    
    mxitems = dateutils.sortdatelist(cfdDateADJ.conditions())
    bargraph_freqitem(cfdDateADJ, mxitems, "dailyfrequentADJs_"+filetitle, "Days", "TheMostFrequentAdj")
    '''
            
    '''
    cfdDateRoot = nltk.ConditionalFreqDist(dailyroots)
    recordCFD(cfdDateRoot,"DailyRoots_"+filetitle)
    
    mxitems = dateutils.sortdatelist(cfdDateRoot.conditions())
    bargraph_freqitem(cfdDateRoot, mxitems, "dailyfrequentROOTs_"+filetitle, "Days", "TheMostFrequentRoot") 
    '''
            
            
    
    #  kalanlar:
    #   postag frequency by days and months
    #   deyimler
    #   root bigram
    #   select bigram or word if it is noun | adj | verb
    
    '''
    docslemmatized = [ (fileid, SAKsParser.lemmatize_lexicon(list(cfd[fileid]))) for fileid in cfd.conditions() ]
    endtime = datetime.now()
    ptime = endtime-starttime                                   
    for (fileid, lemmas) in docslemmatized[:5]:
        print fileid, ":  ",len(lemmas)," - ",lemmas
    
    cfdpostags = get_cfd_bypostags(docslemmatized)
    '''
    
    
#    
#    
#    rootpath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/newsitems/"     #vakit/siyaset/"
#    newspath = rootpath + os.sep + resource + os.sep + cat + os.sep
#    
#    
#    fileids = os.listdir(newspath)
#    numoffiles = len(fileids)   #30
#    fileids = fileids[:numoffiles]
#    
#    starttime = datetime.now()
#    #===========================================================================
#    # 
#    # ''' doc by word adjacency list '''
#    # cfdDocWord = nltk.ConditionalFreqDist((fileid[:-4], word) for fileid in fileids for word in texter.getnewsitem(newspath+os.sep+fileid)[0])
#    # 
#    # ''' word by doc as adjacency list '''
#    # cfdWordDoc = nltk.ConditionalFreqDist((word, fileid[:-4]) for fileid in fileids for word in texter.getnewsitem(newspath+os.sep+fileid)[0])
#    # 
#    # '''  date by docid  '''
#    # #cfdDateDocID = nltk.ConditionalFreqDist((date, fileid[:-4]) for fileid in fileids for _, date in texter.getnewsitem(newspath+os.sep+fileid))
#    # # # of files at some date d : cfdDateDocID[d].N()
#    # 
#    # 
#    
#    filetitle = resource+"_"+cat+"_"+str(numoffiles)+"texts_"
#     
#    ''' date by word as adjacency list '''
#    datewordpairs = []
#    for fileid in fileids:
#        
#        path = newspath + os.sep + fileid
#        words, date = texter.getnewsitem(path)
#        print fileid, ": ",date," ",len(words)," ",type(words)
#        for word in words:
#            datewordpairs.append((date,word))
#    cfdDateWord = nltk.ConditionalFreqDist(datewordpairs)
#    
#    ''' word by date '''
#    cfdWordDate = nltk.ConditionalFreqDist((word, date) for date,word in datewordpairs)
#    endtime = datetime.now()
#    print "Elapsed time for cfd's: ",str(endtime-starttime)
#    
#    '''   word by month   '''
#    cfdMonthWord = nltk.ConditionalFreqDist((date[3:], (word,)) for date in cfdDateWord.conditions() 
#                                            for word in list(cfdDateWord[date]))
#    recordCFD(cfdMonthWord, "monthword_"+filetitle)
#    
#    mxitems = dateutils.sortdatelist(cfdMonthWord.conditions())
#    bargraph_freqitem(cfdMonthWord, mxitems, "monthlyfrequentwords_"+filetitle, "Months", "TheMostFrequentWord") 
#     
#
#    
#    
#    '''   date by bigrams '''
#    datebigrampairs = []
#    for fileid in fileids:
#        path = newspath + os.sep + fileid
#        words, date = texter.getnewsitem(path)
#        bigramlist = nltk.bigrams(words)
#        #bigramlist = [" ".join(bigram) for bigram in bigramlist]
#        #print fileid, ": ",date," ",len(words)," ",type(words)
#        for bigram in bigramlist:
#            datebigrampairs.append((date,bigram))
#    cfdDateBigram = nltk.ConditionalFreqDist(datebigrampairs)
#    
#    
#    
#    
#    
#    '''
#    print "datewordpairs:"
#    for i in datebigrampairs:
#        k,v = i
#        print k," ",v
#    ''' 
#    
#    #recordCFD(cfdDateBigram, "datebigram-"+filetitle)
#
#    #  bar graph for bigrams by dates
#    filetitle = resource+"_"+cat+"_"+str(numoffiles)+"texts_"   
#    dxitems = dateutils.sortdatelist(cfdDateBigram.conditions())
#    dyitems = [cfdDateBigram[date].N() for date in dxitems]
#    #bargraph_cfd2(dxitems, dyitems, filetitle+"DateBar", "Days", "Bigrams")
#    
#    
#    #  bar graph for bigrams by months  
#    cfdMonthBigram = nltk.ConditionalFreqDist((date[3:], word) for date in cfdDateBigram.conditions() 
#                                            for word in list(cfdDateBigram[date]))
#     
#    
#    #recordCFD(cfdMonthWord, "monthbigram-"+filetitle)
#    
#    mxitems = dateutils.sortdatelist(cfdMonthBigram.conditions())
#    myitems = [cfdMonthBigram[month].max() for month in mxitems]
#    bargraph_freqitem(cfdMonthBigram, mxitems, "monthlyfrequentbigrams_"+filetitle, "Months", "TheMostFrequentBigram")
#    
#    
    
    
    # plot bar graph as days in bins and numofwords in the days as y values
    '''
    filetitle = resource+"_"+cat+"_"+str(numoffiles)+"texts_"   
    xitems = cfdDateWord.conditions()
    yitems = [cfdDateWord[date].N() for date in cfdDateWord.conditions()]
    bargraph_cfd2(xitems, yitems, filetitle+"DateBar", "Days", "NumOfWords")
    recordCFD(cfdDateWord, filetitle+"DateWords")
    '''
    #===========================================================================
    # 
    # '''   daily graphs   '''
    # filetitle = resource+"_"+cat+"_"+str(numoffiles)+"texts_"   
    # dxitems = dateutils.sortdatelist(cfdDateWord.conditions())
    # dyitems = [cfdDateWord[date].N() for date in dxitems]
    # bargraph_cfd2(dxitems, dyitems, filetitle+"DateBar", "Days", "NumOfWords")
    # recordCFD(cfdDateWord, filetitle+"DateWords")
    # 
    # 
    # 
    # '''   monthly graphs   '''
    # cfdMonthWord = nltk.ConditionalFreqDist((date[3:], word) for date in cfdDateWord.conditions() 
    #                                        for word in list(cfdDateWord[date]))
    # recordCFD(cfdMonthWord, filetitle+"MonthWords")
    # mxitems = dateutils.sortdatelist(cfdMonthWord.conditions())
    # myitems = [cfdMonthWord[month].N() for month in mxitems]
    # bargraph_cfd2(mxitems, myitems, filetitle+"MonthBar", "Months", "NumOfWords")
    # 
    # 
    # specialwords = [u"kürt", u"ermeni", u"süryani", u"alevi"]
    # for specword in specialwords:
    #    #yitems2 = [cfdDateWord[date][specword] for date in xitems]
    #    #bargraph_cfd2(xitems, yitems2, specword+"_"+filetitle+"DateOneWordBar", "Days", "Num of occurrences of "+specword) 
    #    bargraph_foroneword(cfdDateWord, dxitems, specword, specword+"_"+filetitle+"DateOneWordBar", "Days", "Num of occurrences of "+specword)
    #    bargraph_foroneword(cfdMonthWord, mxitems, specword, specword+"_"+filetitle+"MonthOneWordBar", "Months", "Num of occurrences of "+specword)
    # 
    # # word record test
    # d = "18/11/2012"
    # for i,word in enumerate(list(cfdDateWord[d])):
    #    print i," ",word
    #    
    #    
    #===========================================================================
    
    
    
    #cfdWordDate.plot()
    #bargraph_postags(cfdDateWord, "Date bar")
    
    #printCFD(cfdDateWord)


##  31 Mart'ta kapattım düzenliyorum.
#
#    fileids = os.listdir(rootpath) 
#    #fileids = fileids[:30]
#    print "Num of files: ",len(fileids)
#    
#    starttime = datetime.now() 
#    ''' doc by word as adjacency list '''
#    cfd = nltk.ConditionalFreqDist((fileid[:-4], word) for fileid in fileids for word in texter.getnewsitem(rootpath+os.sep+fileid))
#    endtime = datetime.now()
#    print "completed cfd ",str(starttime-endtime)," sn"
#    #bargraph_postags(cfd, "doc by word")
#    
#    ''' word by doc '''
#    #cfd2 = nltk.ConditionalFreqDist((word, fileid[:-4]) for fileid in fileids for word in texter.getnewsitem(rootpath+os.sep+fileid))
#     
#    
#    ''' (root,word) by doc'''
#    # (literal, literalPOS, root, rootPOS) = SAKsParser.lemmatizeword(word)
#    '''
#    cfd_roots = nltk.ConditionalFreqDist((root, fileid[:-4])
#                                        for fileid in fileids
#                                        for (literal, literalPOS, root, rootPOS) in SAKsParser.lemmatize_lexicon(texter.getnewsitem(rootpath+os.sep+fileid)))
#    '''
#    '''
#    cfd_roots = nltk.ConditionalFreqDist((root, fileid) for fileid in cfd.conditions() 
#                                        for (literal, literalPOS, root, rootPOS) in SAKsParser.lemmatize_lexicon(list(cfd[fileid])))
#    '''
#    
#    
#    
#    docslemmatized = [ (fileid, SAKsParser.lemmatize_lexicon(list(cfd[fileid]))) for fileid in cfd.conditions() ]
#    endtime2 = datetime.now()
#    ptime = endtime2-endtime
#    print "completed lemmatization ",ptime," sn"
#    '''cfd_roots = nltk.ConditionalFreqDist((root, fileid) for (fileid,lemmas) in docslemmatized 
#                                        for (literal, literalPOS, root, rootPOS) in lemmas)    
#       '''                                     
#    for (fileid, lemmas) in docslemmatized[:5]:
#        print fileid, ":  ",len(lemmas)," - ",lemmas
#    
#    cfdpostags = get_cfd_bypostags(docslemmatized)
#    endtime3 = datetime.now()
#    ptime = endtime3 - endtime2
#    print "completed postag-doc cfd ",ptime," sn"
#    endtime4 = datetime.now()
#    bargraph_postags(cfdpostags, "postags bar graph")
#    pitme = endtime4-endtime3
#    print "completed bar graph ", ptime, "sn" 
#    
#    
#    ''' postag by doc   '''
#'''   
#    postag = "ADJ"
#    cfd_adj = get_cfd_ofpostag(postag, docslemmatized)
#    print "Adjective distribution:"
#    #printCFD(cfd_adj)
#'''     
#     
#   
   
   
    
    
#
## MAGWORD ANALYSIS   
#    magwordsdict = keywordhandler.get_keyword_dict("magnitude")
#    bigwords = magwordsdict["big"]
#    smallwords = magwordsdict["small"] 
#    ''' doc by magword '''
#    cfd_smallwords = nltk.ConditionalFreqDist((fileid[:-4], word) for fileid in fileids for word in texter.getnewsitem(rootpath+os.sep+fileid)
#                                              if word in smallwords) 
#    
#    cfd_bigwords = nltk.ConditionalFreqDist((fileid[:-4], word) for fileid in fileids for word in texter.getnewsitem(rootpath+os.sep+fileid)
#                                              if word in bigwords)
#       
#    #printCFD(cfd_smallwords)
#    
#    filename = "1500texts_bigwordscfd"
#    
#    bigword = u"çok"
#    print bigword
#    y = []
#    x = []
#    for i,cond in enumerate(cfd_bigwords.conditions()):
#        #print cond, ": ",cfd_bigwords[cond][u"çok"]
#        y.insert(i, cfd_bigwords[cond][bigword])
#        x.insert(i, cond)
#    
#
#    # standard graph
#    plot(y, 'k+-')
#    xticks(range(1,len(x)+1), x)
#    xlim(1, len(x))
#    xlabel('Docs')
#    ylabel('Frequency of word %s' %bigword)
#    title('Change of word %s throughout docs' %bigword)
#    show()
#    
#    
#    # specgram
#    Fs = 4
#    NFFT = 8
#    specgram(y, NFFT, Fs, noverlap=0)
#    xlabel('Docs')
#    ylabel('Frequency of word %s' %bigword)
#    title('Change of word %s throughout docs' %bigword)
#    show()    
#    







#    WORDLE
#
#    N = 5   # numofdocs
#    files = fileids[:N]
#    ''' doc by word as adjacency list '''
#    cfd = nltk.ConditionalFreqDist((fileid[:-4], word) for fileid in files for word in texter.getnewsitem(rootpath+os.sep+fileid))
#     
#    ''' word by doc '''
#    cfd2 = nltk.ConditionalFreqDist((word, fileid[:-4]) for fileid in files for word in texter.getnewsitem(rootpath+os.sep+fileid))
#    
#    
#    figure(dpi=100)
#    # firstly determine maximum box size 
#    rotation = 0
#    fontsize = 5
#    
#    maxcond, maxratio = get_cond_with_max_outcome(cfd2)
#    maxwordlength = 15   # predictive
#    boxhypotenus = maxwordlength * fontsize * maxratio + fontsize
#    
#    W = len(cfd2.conditions())
#    N = int(math.sqrt(W)) + 1
#    steplength = boxhypotenus   # suppose rotation is 45
#    numofaxispoints = steplength*(N+1)
#    xaxrange, yaxrange = arange(0, numofaxispoints, steplength),arange(0, numofaxispoints, steplength)
#    xticks(xaxrange, rotation=-80)
#    yticks(yaxrange)
#    
#    plot(xaxrange, yaxrange, 'w+-', label='wordle for 30 texts')
#    
#    grid()
#    
#    
#    yaxisrange = yaxrange + steplength/2
#    locationmatrix = np.zeros([len(xaxrange), len(yaxrange)])
#    
#    for cond in cfd2.conditions():
#        occupied = True
#        while(occupied):
#            # generate random position
#            xpos = random.choice(xaxrange)
#            ypos = random.choice(yaxisrange)
#            
#            # eliminate collision
#            xind = xpos / 155          
#            yind = (ypos - steplength/2) / 155
#            if locationmatrix[xind][yind] == 0:
#                occupied = False
#                locationmatrix[xind][yind] = 1
#        
#        word = cond
#        ratio = cfd2[cond].N()
#        boxlen = len(word) * fontsize*ratio
#        print str(cond)," : ",ratio," - box: ", boxlen, " x, y:",xpos," , ",ypos
#        text(x=xpos, y=ypos, s=word, rotation=random.randint(-30,30)*ratio, fontsize=fontsize*ratio)
#    
#    
#    print W," ",N," ",boxhypotenus*N
#    print "num of words: ", W
#    print "max box size: ", boxhypotenus
#    print "steplength: ",steplength
#    print "num of axis points: ",numofaxispoints
#    print xaxrange / 2
#    savefig(IOtools.results_rootpath + os.sep + "5texts3"+".png", dpi=250)
##    show()
#    
    
    
    
    
    
'''
    print "word cooccurrences:"
    l = get_word_cooccurrences(cfd2)
    for (pair, commons) in l:
        print pair," : ",commons
    '''
    
'''
    print "Small words cond."
    printCFD(cfd_smallwords)
    
    
    
    printCFD(cfd2)
    
    l1 = cfd2.conditions()
    l2 = list(set(l1))
    
    print len(l1),"  ",len(l2)
    '''
    
    
    
'''
    for fileid in cfd.conditions():
        print fileid," :  ",cfd[fileid]['suriye'] 
    print list(cfd[cfd.conditions()[10]])
    
    
    #cfd.plot(samples=['suriye','şam','türkiye','ankara','savaş','barış','saldırı'])
    
    keywords = ['suriye','şam','türkiye','ankara','savaş','barış','saldırı']
    cfd2 = nltk.ConditionalFreqDist((fileid, (w1, w2)) 
                                    for fileid in cfd.conditions()[:10] 
                                    for w1 in list(cfd[fileid]) 
                                    for w2 in keywords)

    '''
    
    