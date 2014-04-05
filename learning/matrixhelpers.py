'''
Created on Feb 10, 2014

@author: dicle
'''

import numpy as np
import pandas as pd
import os


from sentimentfinding import IOtools, keywordhandler

# df is the matrix container dataframe
# tags1: list of colnames whose values in the matrix will add up to be numerator
# tags2: list of colnames whose values in the matrix will add up to be denominator
# returns the ratio of tags1 over tags2 as a new df
def feature_ratio_over_df(df, tags1, tags2, rationame):
    pointnames = df.index.values.tolist()
    ratiodf = pd.DataFrame(np.zeros(len(pointnames)), index=pointnames, columns=[rationame])
    
    for item in pointnames:
        numeratorval = 0
        for tag1 in tags1:
            tagoccr = 0
            try:
                tagoccr = df.loc[item, tag1]
            except:
                tagoccr = 0
            numeratorval += tagoccr
        
        denominatorval = 0
        for tag2 in tags2:
            tagoccr = 0
            try:
                tagoccr = df.loc[item, tag2]
            except:
                tagoccr = 0
            denominatorval += tagoccr
            #denominatorval += df.loc[item, tag2]
        
        print tags1," / ",tags2," ratio ",item,"\n ",numeratorval," / ",denominatorval
        if denominatorval == 0:
            featureratio = 0.0
        else:
            featureratio = float(numeratorval) / denominatorval
        
        featureratio = round(featureratio, 4)
        ratiodf.loc[item, rationame] = featureratio
    
    return ratiodf


def get_featuretags_ratio(incsvpath, outcsvpath, numeratortags, denominatortags, rationame):
    indf = IOtools.readcsv(incsvpath, keepindex=True)
    outdf = feature_ratio_over_df(df=indf, tags1=numeratortags, tags2=denominatortags, rationame=rationame)
    IOtools.tocsv(outdf, outcsvpath, keepindex=True)


def search_words_in_df(df, words):
    rows = df.index.values.tolist()
    indfwords = df.columns.values.tolist()
    indfwords.sort()
    wordsfiltereddf = pd.DataFrame(np.zeros((len(rows), len(words))), index=rows, columns=words)

    print "df shape"
    print " ",df.shape
    print "filt shape"
    print " ",wordsfiltereddf.shape

    for word in words:
        if word in indfwords:
            #print "FOUND: ",word
            wordvector = df.loc[:, word].values
            #print word, "  shape ",wordvector.shape," v ",wordsfiltereddf.shape," vv ",wordsfiltereddf.loc[:, word].shape
            #print wordvector
            wordsfiltereddf.loc[:, word] = wordvector
            
        '''    
        else:
            wordsfiltereddf.loc[:, word] = np.zeros(len(rows))'''
    return wordsfiltereddf


def get_featureword_doc_matrix(incsvpath, outcsvpath, words, column_appendix=None):
    maindf = IOtools.readcsv(incsvpath, keepindex=True)
    filtereddf = search_words_in_df(maindf, words)
    if column_appendix:
        filtereddf = column_name_appendixing(filtereddf, appendix=column_appendix)
    IOtools.tocsv(filtereddf, outcsvpath, keepindex=True) 


def get_featurewords_ratio(incsvpath, outcsvpath, words, rationame):
    mainwordscountdf = IOtools.readcsv(incsvpath, keepindex=True)
    numofdocs, _ = mainwordscountdf.shape
    docwordcount = np.array([np.count_nonzero(mainwordscountdf.values[i, :]) for i in range(numofdocs)])
       
    
    searchedwordsdf = search_words_in_df(mainwordscountdf, words)
    searchedwordsmatrix = searchedwordsdf.values
    wordsfreq = np.zeros(numofdocs)
    for i in range(numofdocs):
        wordsfreq[i] = np.sum(searchedwordsmatrix[i, :]) / docwordcount[i]        
    wordsfreq = np.around(wordsfreq, decimals=4)
    
    wordsfreqdf = pd.DataFrame(wordsfreq, index=mainwordscountdf.index.values.tolist(), columns=[rationame])
    IOtools.tocsv(wordsfreqdf, outcsvpath, keepindex=True)
    

def get_first_N_rows(scorecsvfile, N, conditioncols, ascend=False):
    sorteddf = scorecsvfile.sort(conditioncols, ascending=ascend)
    nrows, _ = sorteddf.shape
    N = min(N, nrows)
    return sorteddf.iloc[: N, :]
    
    


def column_name_appendixing(df, appendix):
    column_replacement = {}
    words = df.columns.values.tolist()
    for w in words:
        column_replacement[w] = w + "*" + appendix
    return df.rename(columns=column_replacement)      

if __name__ == "__main__":
    
    '''
    csvpath = "/home/dicle/Dicle/Tez/corpusstats/learning/data/random-single-N5/matrix/content-postagCOUNT.csv"
    postagdf = IOtools.readcsv(csvpath, keepindex=True)
    
    outfolder = "/home/dicle/Dicle/Tez/corpusstats/learning/experiments/test1/"
    ratname = "adverbratio"
    exdf = feature_ratio_over_df(postagdf, tags1=['ADV'], tags2=['ADJ', 'Verb'], rationame=ratname)
    print exdf
    IOtools.tocsv(exdf, outfolder+os.sep+ratname+".csv", keepindex=True)
    '''
    outfolder = "/home/dicle/Dicle/Tez/corpusstats/learning/experiments/test1/"
    csvpath = "/home/dicle/Dicle/Tez/corpusstats/learning/data/random-single-N5/matrix/contenttermCOUNT.csv"
    #tfidfdf = IOtools.readcsv(csvpath, keepindex=True)
    #print "LEN ",len(tfidfdf.columns.values.tolist())
    abswords = keywordhandler.get_abstractwords()
    
    '''
    absdf = search_words_in_df(tfidfdf, abswords)
    IOtools.tocsv(absdf, outfolder+"/abstfidf.csv", keepindex=True)
    '''
    #get_featurewords_ratio(incsvpath=csvpath, outcsvpath=outfolder+"/absratio.csv", words=abswords, rationame="abstractness")
    
    outfolder = "/home/dicle/Dicle/Tez/corpusstats/learning/data/random-single-N5/titletfidfsearchtest/"
    incsvpath = "/home/dicle/Dicle/Tez/corpusstats/learning2/data/single/30/rawfeatures/contenttermTFIDF.csv"
    tdf = IOtools.readcsv(incsvpath, keepindex=True)
    tabstdf = search_words_in_df(tdf, abswords)
    IOtools.tocsv(tabstdf, outfolder+"/testabs5.csv", True)
    
    print "len abs words",len(abswords)
    print tabstdf.shape
    
    cols = tabstdf.columns.values.tolist()
    print cols[0]
    print tabstdf.iloc[:,0]
    print
    print tabstdf.loc[:, cols[300]].values

    
          
    