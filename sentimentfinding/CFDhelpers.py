'''
Created on May 5, 2013

@author: dicle
'''

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk

import IOtools
import plotter


def printCFD(cfd):
    for cond in cfd.conditions():
        totaloccurrence = cfd[cond].N()
        print cond," : "
        print "cumulative occ. ",totaloccurrence
        print "the most occ. element. ",cfd[cond].max()
        
        for item in list(cfd[cond]):
            print "\t",item," : ",cfd[cond][item]
    print "# of conds: ",len(cfd.conditions())


def cfd2csv(cfd, csvpath, colnames=[]):
    matrix = []
    
    for cond in cfd.conditions():
        for item in list(cfd[cond]):
            value = cfd[cond][item]
            matrix.append([cond, item, value])
    
    matrix = np.array(matrix, dtype=object)
    df = pd.DataFrame(matrix, columns=colnames)
    header = False
    if colnames:
        header = True
    df.to_csv(csvpath, header=header, index=False, sep='\t', encoding='utf-8')
        

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
    


# returns ratio of two feature values  (to be used for getting adj / noun and adv / (adj+verb) ratios     
def feature_ratio(cfd, numerator_features, denominator_features):
    condfeature_ratios = []
    for cond in cfd.conditions():
        
        numeratorval = 0
        for f in numerator_features:
            numeratorval = numeratorval + cfd[cond][f]
        denominatorval = 0 
        for f in denominator_features:
            denominatorval = denominatorval + cfd[cond][f]  
        
        if denominatorval == 0:
            value = 0
        else:
            value = float(numeratorval) / denominatorval
        condfeature_ratios.append((cond, value)) 
    return condfeature_ratios 


def keyword_count(maincfd, keywords):
    cfd = nltk.ConditionalFreqDist( (fileid, word) for fileid in maincfd.conditions() for word in list(maincfd[fileid]) if word in keywords)
    return cfd


def plotcfd_oneline(cfd, figname, xLabel, yLabel, imgoutpath, featuresXaxis=True):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)       
    plt.autoscale(True, axis='both')
    plt.tight_layout(pad=7.0)
    plt.title(figname)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel) 
    
    if featuresXaxis:
        colors = plotter.get_n_colors(len(cfd.conditions()))
        
        xitems = [item for cond in cfd.conditions() for item in list(cfd[cond])]
        xitems = list(set(xitems))
        plt.xticks(np.arange(len(xitems)), xitems, rotation=90)
        for cond, clr in zip(cfd.conditions(), colors):
            yitems = [float(cfd[cond][item])/cfd[cond].N() for item in xitems]
            
            linelabel = cond
            plotter.plot_line(xitems, yitems, linelabel, clr)
        
    else:    # conditions are X axis
        xitems = cfd.conditions()
    
    plt.legend(loc=2, borderaxespad=0.)  
     
    plt.savefig(imgoutpath+".png",dpi=100)
    plt.clf()
    
      

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
        plt.plot(np.arange(len(xitems)), yitems, color=cm(1.*i/numofcolors), label=linelabel)  #color=colors[i],
        i = i + 1
    
    plt.legend()
        
    plt.autoscale(True, axis='both')
    plt.tight_layout(pad=7.0)
    plt.xticks(np.arange(len(xitems)), xitems, rotation=90)
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
    #plt.show()
    



    

if __name__ == "__main__":
    xs = np.arange(10)
    for i in range(5):
        ys = np.random.random_sample(15)
        plot_cfd()
    
    
    
    
    
    
    
    
    
    
    
    
      

