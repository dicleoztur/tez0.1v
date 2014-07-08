'''
Created on May 26, 2014

@author: dicle
'''

import os
import numpy as np

from sentimentfinding import IOtools



def csv2latextable_featset(inpath, outpath, filename, metricname):
    
    header = "\\begin{table}[h] \n \
\\begin{center} \n \
\\begin{tabular}{|p{5cm}|p{2cm}|p{2cm}|p{2cm}|} \n \
\\hline  \\bf feature-combined dataset name & \\bf mean "+ metricname +" & \\bf minimum "+ metricname +" & \\bf maximum "+ metricname +"   \\\ \\hline"
    
    footer = "\\end{tabular} \n \
\\end{center} \n \
\\caption{\\label{featset-"+metricname[:4]+"-stats} Mean, maximum and minimum "+metricname+" results for 8 feature-measure-combined datasets } \n \
\\end{table}"
    
    ip1 = os.path.join(inpath, filename+".csv")
    df = IOtools.readcsv(ip1, keepindex=True)
    nrows, ncols = df.shape
    rowids = df.index.values.tolist()
    
    out = header+"\n"
    for rowid in rowids:
        featset = rowid.split("**")[0].strip()
        featset = "\\verb|"+featset+"|"
        out += featset + " & "
        #np.round(a, decimals, out)
        mean = df.loc[rowid, "mean"]
        min = df.loc[rowid, "min"]
        max = df.loc[rowid, "max"]
        stats = map(lambda x : str(round(x, 5)), [mean, min, max])
        statsstr = " & ".join(stats)
        out += statsstr + " \\\ \hline " + "\n"
    
    out += footer
    IOtools.todisc_txt(out, os.path.join(outpath, filename+".txt"))



def csv2latextable_labelunion(inpath, outpath, filename, metricname):
    
    header = "\\begin{table}[h] \n \
\\begin{center} \n \
\\begin{tabular}{|p{5cm}|p{2cm}|p{2cm}|p{2cm}|} \n \
\\hline  \\bf label union name & \\bf mean "+ metricname +" & \\bf minimum "+ metricname +" & \\bf maximum "+ metricname +"   \\\ \\hline"
    
    footer = "\\end{tabular} \n \
\\end{center} \n \
\\caption{\\label{labelun-"+metricname[:4]+"-stats} Mean, maximum and minimum "+metricname+" results for 7 label-combined datasets } \n \
\\end{table}"
    
    ip1 = os.path.join(inpath, filename+".csv")
    df = IOtools.readcsv(ip1, keepindex=True)
    nrows, ncols = df.shape
    rowids = df.index.values.tolist()
    
    out = header+"\n"
    for rowid in rowids:
        featset = rowid
        
        out += featset + " & "
        #np.round(a, decimals, out)
        mean = df.loc[rowid, "mean"]
        min = df.loc[rowid, "min"]
        max = df.loc[rowid, "max"]
        stats = map(lambda x : str(round(x, 5)), [mean, min, max])
        statsstr = " & ".join(stats)
        out += statsstr + " \\\ \hline " + "\n"
    
    out += footer
    IOtools.todisc_txt(out, os.path.join(outpath, filename+".txt"))


def csv2latextable_algorithm(inpath, outpath, filename, metricname):
    
    header = "\\begin{table}[h] \n \
\\begin{center} \n \
\\begin{tabular}{|p{9cm}|p{2cm}|p{2cm}|p{2cm}|} \n \
\\hline  \\bf algorithm \& parameters & \\bf mean "+ metricname +" & \\bf minimum "+ metricname +" & \\bf maximum "+ metricname +"   \\\ \\hline"
    
    footer = "\\end{tabular} \n \
\\end{center} \n \
\\caption{\\label{alg-"+metricname[:4]+"-stats} Mean, maximum and minimum "+metricname+" results for 27 learning models } \n \
\\end{table}"
    
    ip1 = os.path.join(inpath, filename+".csv")
    df = IOtools.readcsv(ip1, keepindex=True)
    nrows, ncols = df.shape
    rowids = df.index.values.tolist()
    
    out = header+"\n"
    for rowid in rowids:
        featset = rowid[4:]
        featset = "\\verb|"+featset+"|"
        
        out += featset + " & "
        #np.round(a, decimals, out)
        mean = df.loc[rowid, "mean"]
        min = df.loc[rowid, "min"]
        max = df.loc[rowid, "max"]
        stats = map(lambda x : str(round(x, 5)), [mean, min, max])
        statsstr = " & ".join(stats)
        out += statsstr + " \\\ \hline " + "\n"
    
    out += footer
    IOtools.todisc_txt(out, os.path.join(outpath, filename+".txt"))





def featset_results_table():
    path = "/home/dicle/Dicle/Tez/corpusstats/learning9_svmscale/experiments/performance/getltxcodes/"
    filename = "stats_featureset-"
    metrics = ["accuracy", "fscore", "precision", "recall"]

    for metricname in metrics:
        fname = filename + metricname.upper() 
        csv2latextable_featset(path, path, fname, metricname=metricname)   


def labelunion_results_table():
    path = "/home/dicle/Dicle/Tez/corpusstats/learning9_svmscale/experiments/performance/getltxcodes/"
    filename = "stats_labelunion-"
    metrics = ["accuracy", "fscore", "precision", "recall"]

    for metricname in metrics:
        fname = filename + metricname.upper() 
        csv2latextable_labelunion(path, path, fname, metricname=metricname) 


def algorithm_results_table():
    path = "/home/dicle/Dicle/Tez/corpusstats/learning9_svmscale/experiments/performance/getltxcodes/"
    filename = "stats_algorithm-"
    metrics = ["accuracy", "fscore", "precision", "recall"]

    for metricname in metrics:
        fname = filename + metricname.upper() 
        csv2latextable_algorithm(path, path, fname, metricname=metricname) 
        

if __name__ == "__main__":
    
    algorithm_results_table()
    
    
    
    
    