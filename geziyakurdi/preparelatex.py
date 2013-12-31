# -*- coding: utf-8 -*-
'''
Created on Aug 1, 2013

@author: dicle
'''

import pandas as pd
from sentimentfinding import IOtools


def metadata_tabular():
    rpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/wordletest/words/temp/"
    metadf = pd.read_csv(rpath+"/metadocs.csv", index_col=None, sep="\t")
    
    print metadf.loc[0,"Author"]
    metadf = metadf.sort(["Polarity", "Date", "Author"], ascending=[False, True, True])
    v = metadf.iloc[0,:]
    print v.loc["Author"],v.loc["Resource"]
    
    header = "\\begin{tabular}{l  | c | c | c | c } \n \
kategori & yazar & başlık & tarih & yayın \\\\ \n \
\\hline \\hline \n"

    end = "\\end{tabular}"
    outltx = ""
    numofdocs, fields = metadf.shape
    for i in range(numofdocs):
        row = metadf.iloc[i,:]
        cat = row.loc["Polarity"]
        cat = "\\textbf{"+cat+"}"
        author = row.loc["Author"]
        title = row.loc["Title"]
        link = row.loc["Link"]
        date = row.loc["Date"]
        resource = row.loc["Resource"]
        
        title = "\\href{"+link+"}{"+title+"}"
        date = "\\textit{"+date+"}"
        resource = "@"+resource
        
        s = " & ".join([cat, author, title, date, resource])
        outltx = outltx + s + "\\\\ \n \\hline \n"
    
    outltx = header + outltx + end
    IOtools.todisc_txt(outltx, rpath+"docswordle_tableLaTeX.txt")
        

def img_ref_captions():
    rpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/wordletest/words/temp/"
    metadf = pd.read_csv(rpath+"/metadocs.csv", index_col=None, sep="\t")
    
    header = '\\begin{figure}[ht] \n \
\subfigure[frequency weighted word cloud]{ \n \
 \includegraphics[width=3.5in]{pics_docs/freq_'
 
    middle1 = '.png}} \n \
\quad \
\subfigure[tfidf weighted word cloud]{ \n \
 \includegraphics[width=3in]{pics_docs/tfidf_'
 
    middle2 = ".png}} \n \
                \caption{ " 

    end = "\end{figure}"
           
    outltx = ""
    numofdocs, fields = metadf.shape
    for i in range(numofdocs):
        filename = metadf.loc[i, "filename"]
        author = metadf.loc[i, "Author"]
        title = metadf.loc[i, "Title"]
        link = metadf.loc[i, "Link"]
        date = metadf.loc[i, "Date"]
        resource = metadf.loc[i, "Resource"]
        
        caps_link = "\href{" + link + "}"
        caps_a = "{\\textit{" + author + "}, " + title + ", \\textit{" + date + "} - @" + resource + "} }\n"
        figtxt = header + filename + middle1 + filename + middle2 + caps_link + caps_a + end
        
        outltx = outltx + figtxt + "\n\n"
    
    IOtools.todisc_txt(outltx, rpath+"docswordle_figLaTeX.txt")


if __name__ == "__main__":
    #img_ref_captions()
    metadata_tabular()
    
    