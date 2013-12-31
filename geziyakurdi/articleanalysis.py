# -*- coding: utf-8 -*- 

'''
Created on Jul 1, 2013

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
from sentimentfinding import IOtools
from sentimentfinding import keywordhandler
from languagetools import SAKsParser


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


'''takes a list of lemmatized quadruples [(literal, literalPOS, root, rootPOS)] of odcs,
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
        outcomevalue = cfd[cond].N()
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
    
    plt.bar(arange(len(xitems)), yitems, align='center')
    plt.autoscale(True, axis='both')
    plt.tight_layout(pad=7.0)
    plt.xticks(arange(len(xitems)), xitems, rotation=90)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(figname)
    plt.savefig(IOtools.results_rootpath+os.sep+figname+".png") 



def bargraph_cfd2(xitems, yitems, figname, xLabel, yLabel):
    plt.figure()
    plt.bar(arange(len(xitems)), yitems, align='center')
    plt.autoscale(True, axis='both')
    plt.tight_layout(pad=7.0)
    plt.xticks(arange(len(xitems)), xitems, rotation=90)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(figname)
    plt.savefig(IOtools.results_rootpath+os.sep+figname+".png") 
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


def merge_word_lists(indirectory, outdirectory, outfilename):
    fileids = IOtools.getfilenames_of_dir(indirectory, removeextension=False)
    

    allwords = []    
    for fileid in fileids:
        words = IOtools.readtextlines(indirectory+os.sep+fileid)
        allwords.extend(words)
    IOtools.todisc_list(outdirectory+os.sep+outfilename+".txt", allwords)
    
    fdist = nltk.FreqDist(allwords)
    IOtools.todisc_freqdist(outdirectory+os.sep+"weighted-"+outfilename+".txt", fdist)
    '''
    weightedwordlist = [word+"\t"+str(fdist[word]) for word in list(fdist)]
    IOtools.todisc_list(outdirectory+os.sep+"weighted-"+outfilename+".txt", weightedwordlist)
    '''



def intersect_word_lists(list1, list2, outdir, outfilename):
    intersect = []
    allwords = []
    allwords.extend(list1)
    allwords.extend(list2)
    
    
    
    for w1 in list1:
        if w1 in list2:
            intersect.append(w1)
    IOtools.todisc_list(outdir+os.sep+outfilename+".txt", intersect)
    
    fdist = nltk.FreqDist(intersect)
    fdist2 = nltk.FreqDist(list2)
    for item in list(fdist2):
        fdist.inc(item, fdist2[item])
    IOtools.todisc_freqdist(outdir+os.sep+"weighted-"+outfilename+".txt", fdist)
    return intersect


def diff_word_lists(list1, list2, outdir, outfilename):
    l = list(set(list1) - set(list2))
    IOtools.todisc_list(outdir+os.sep+outfilename+".txt", l)
    
    fdist = nltk.FreqDist(l)
    IOtools.todisc_freqdist(outdir+os.sep+"weighted-"+outfilename+".txt", fdist)
    return l
    

def document_similarity(list1, list2):
    return




def getwordsandlemmasfromfile():
    rootpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/"
    
    corpuspath = rootpath + os.sep + "texts/"
    outwordspath = rootpath + os.sep + "weightedwords/"
    
    fileids = IOtools.getfilenames_of_dir(corpuspath, removeextension=False)
    
    
    
    for fileid in fileids:
        txt = texter.readtxtfile(corpuspath+os.sep+fileid)
        
        marker = "Haziran 2013"
        mark = txt.find(marker)    # skip metadata
        txt = txt[mark+len(marker):]
        
        words = texter.getwords(txt)
        lemmatuples = SAKsParser.findrootsinlexicon(words)
        roots = [root for _,root,_ in lemmatuples]
        
        fdwords = nltk.FreqDist(words)
        
        fdroots = nltk.FreqDist(roots)
        
        weightedwords = [word+"\t"+str(fdwords[word]) for word in list(fdwords)]
        weightedroots = [root+"\t"+str(fdroots[root]) for root in list(fdroots)]
               
        IOtools.todisc_list(outwordspath+os.sep+"lemma"+os.sep+fileid, weightedwords)
        IOtools.todisc_list(outwordspath+os.sep+"root"+os.sep+fileid, weightedroots)
    
def cfd_analysis():
    
    
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
    
    resource = "radikal" #"vakit"
    cat = "politika"   # "guncel"
    
    rootpath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/radikal_KasimEylul2012/newsitems/"
    rootpath = "/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/newsitems/"     #vakit/siyaset/"
    newspath = rootpath + os.sep + resource + os.sep + cat + os.sep
    
    
    fileids = os.listdir(newspath)
    numoffiles = len(fileids)   #30
    fileids = fileids[:numoffiles]
    
    starttime = datetime.now()
    
    ''' doc by word adjacency list '''
    cfdDocWord = nltk.ConditionalFreqDist((fileid[:-4], word) for fileid in fileids for word in texter.getnewsitem(newspath+os.sep+fileid)[0])
    
    ''' word by doc as adjacency list '''
    cfdWordDoc = nltk.ConditionalFreqDist((word, fileid[:-4]) for fileid in fileids for word in texter.getnewsitem(newspath+os.sep+fileid)[0])
    
    ''' date by word as adjacency list '''
    datewordpairs = []
    for fileid in fileids:
        path = newspath + os.sep + fileid
        words, date = texter.getnewsitem(path)
        print fileid, ": ",date," ",len(words)," ",type(words)
        for word in words:
            datewordpairs.append((date,word))
    
    print "datewordpairs:"
    '''
    for i in datewordpairs:
        k,v = i
        print k," ",v
       ''' 
    cfdDateWord = nltk.ConditionalFreqDist(datewordpairs)
    #cfdDateWord = nltk.ConditionalFreqDist((date, word) for )
    
    ''' word by date '''
    worddatepairs = listutils.fliptuplepairs(datewordpairs)
    cfdWordDate = nltk.ConditionalFreqDist(worddatepairs)
    endtime = datetime.now()
    print "Elapsed time for cfd's: ",str(endtime-starttime)
    
    # find bigram freq
    
    # plot bar graph as days in bins and numofwords in the days as y values
    filetitle = resource+"_"+cat+"_"+str(numoffiles)+"texts_"
       
    #xitems = cfdDateWord.conditions()
    #yitems = [cfdDateWord[date].N() for date in cfdDateWord.conditions()]
    ''' sort dates '''
    xitems = dateutils.sortdatelist(cfdDateWord.conditions())
    yitems = [cfdDateWord[date].N() for date in xitems]
    
    bargraph_cfd2(xitems, yitems, filetitle+"DateBar", "Days", "NumOfWords")
    recordCFD(cfdDateWord, filetitle+"DateWords")
    
    
    specialwords = [u"kürt", u"ermeni", u"süryani", u"alevi", u"açlık", u"cezaevi"]
    for specword in specialwords:
        yitems2 = [cfdDateWord[date][specword] for date in cfdDateWord.conditions()]
        bargraph_cfd2(xitems, yitems2, specword+"_"+filetitle+"DateOneWordBar", "Days", "Num of occurrences of "+specword)
    
    
    
    
    
       

if __name__ == "__main__":
    
    
    #getwordsandlemmasfromfile()
    
    
    # intersect and differ categories
    indirectory = "/home/dicle/Dicle/Tez/geziyakurdiproject/words/merged/"
    outdirectory = indirectory
    poswords = IOtools.readtextlines(indirectory+os.sep+"posRoot.txt")
    negwords = IOtools.readtextlines(indirectory+os.sep+"negRoot.txt")
    
    i = intersect_word_lists(poswords, negwords, outdirectory, "posneg_intersect")
    pn = diff_word_lists(poswords, negwords, outdirectory, "pos-neg_diff")
    np = diff_word_lists(negwords, poswords, outdirectory, "neg-pos_diff")
    
    # report 
    numofcommons = len(set(i))
    numofpos = len(set(poswords))
    numofneg = len(set(negwords))
    numofposdiff= len(set(pn))
    numofnegdiff = len(set(np))
    print "POS ratio"
    print "numofcommon / numofPOS", str(float(numofcommons) / numofpos)
    print "\nNEG ratio"
    print "numofcommon / numofNEG", str(float(numofcommons) / numofneg)
    
    
    '''
    # merge categories
    
    indirectory = "/home/dicle/Dicle/Tez/geziyakurdiproject/words/merged/root/"
    outdirectory = "/home/dicle/Dicle/Tez/geziyakurdiproject/words/merged/"
    outfilename = "fullsetRoot"
    merge_word_lists(indirectory, outdirectory, outfilename)
    
    
    
    1- pos
    2- neg
    3- full
    
    '''
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
    
    