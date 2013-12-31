'''
Created on Jul 10, 2013

@author: dicle
'''

import os
import numpy as np
import pandas as pd

from sentimentfinding import IOtools, plotter
import documentsimilarity
from classification import numericutils
import testpca



def matrix_valueshift_rows(rpath, matname):
    matdf = pd.read_csv(rpath+os.sep+matname+".csv", index_col=0)
    matrix = matdf.values
    
    m, n = matrix.shape
    
    for i in range(m):
        matrix[i,:] = matrix[i,:] - np.min(matrix[i,:])
    
    shiftedmatdf = pd.DataFrame(matrix, index=matdf.index.values.tolist(), columns=matdf.columns.values.tolist())
    shiftedmatdf.to_csv(rpath+os.sep+matname+"Shift.csv")
    return shiftedmatdf
        

def compare_topical_terms(doctopic1, doctopic2, doclist, path):
        
    out = ""
    for doc in doclist:
        out += "\n\n"+doc[:13]+"\n"
        for w1,w2 in zip(doctopic1[doc], doctopic2[doc]):
                l1, v1 = w1
                l2, v2 = w2
                out += l1+" ,"+str(round(v1,4))+"\t"+l2+" ,"+str(round(v2,4))+"\n"
    
    IOtools.todisc_txt(out, path)    
    
     
    

def batch_lsa(df, rootpath, dimvals, N=15):
    docs = df.index.values.tolist()
    analyser = documentsimilarity.matrixanalyser(rootpath)
    
    tfdoctopic = analyser.get_topical_terms(df, N)
    for dim in dimvals:
        lsadf = analyser.lsa_transform(df, dimensions=dim)
        doctopic = analyser.get_topical_terms(lsadf, N)
        
        path = rootpath+os.sep+"compare"+str(dim)+"D.txt"
        compare_topical_terms(doctopic, tfdoctopic, docs, path)
        

def main_lsa():
    rpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus/revisedgetwords/normalised/"
    analyser = documentsimilarity.matrixanalyser(rpath)
    
    dim_start = 10
    dim_end = 31
    steps = 1
    interval = range(dim_start, dim_end, steps)
    
    df = pd.read_csv(rpath+os.sep+"normtfidfmatrix.csv", index_col=0)
    
    simdf = pd.read_csv(rpath+os.sep+"similaritymatrix.csv", index_col=0)
    analyser.get_most_similar_pairs(simdf)
    
    batch_lsa(df, rpath, interval)


def matrix_analysis():
    recordpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/"
    analyser = documentsimilarity.matrixanalyser(recordpath)
    
    simdf_orig = pd.read_csv(recordpath+os.sep+"similaritymatrix_orig.csv", index_col=0)
    simdf_pca = pd.read_csv(recordpath+os.sep+"similaritymatrix_pca.csv", index_col=0)
    
    simorig = simdf_orig.values
    simpca = simdf_pca.values
    
    # to overcome colorbar range, make diagonals 0
    for i in range(len(simorig)):
        simorig[i,i] = 0.0
        simpca[i,i] = 0.0
    
    plotter.set_plotframe("similarity matrix", "docs", "docs")
    testpca.visualize_matrix(simorig, recordpath+os.sep+"img/simorig", xlabels=range(len(simorig)))
    plotter.set_plotframe("similarity matrix", "docs", "docs")
    testpca.visualize_matrix(simpca, recordpath+os.sep+"img/simpca", xlabels=range(len(simpca)))



def main_over():
    rpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus/revisedgetwords/normalised/"
    
    nonorm_df = pd.read_csv(rpath+os.sep+"matrixdoctermTFIDF.csv", index_col=0)
    
    analyser = documentsimilarity.matrixanalyser(rpath)
    
    # normalize
    tfidfmatrix = nonorm_df.values
    normtfidfmatrix = numericutils.row_normalize(tfidfmatrix)
    
    
    df = pd.DataFrame(normtfidfmatrix, index=nonorm_df.index.values, columns=nonorm_df.columns.values)
    df.to_csv(rpath+os.sep+"normtfidfmatrix.csv")
    
    distdf = analyser.compute_document_distance(df)
    simdf = analyser.compute_document_similarity(df)
    
    batch_lsa(df, rpath, range(5,50,5))


def compute_similarity():
    #main_lsa()
    
    '''  22 Temmuz 2013  17:20   
    # corpus construction, dimensionality reduction done, recorded.
    
    dfpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/pcatests/in/matrixdoctermTFIDF.csv"
    pcadfpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/doctermPCA.csv"
    recordpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/"
    analyser = documentsimilarity.matrixanalyser(recordpath)
    '''
    
    
    recordpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/edit/wordletest/matrix/"
    dfpath = recordpath + os.sep + "matrixdoctermTFIDF.csv"
    pcadfpath = recordpath + os.sep + "doctermPCA.csv"
    analyser = documentsimilarity.matrixanalyser(recordpath)
    
    
    df = pd.read_csv(dfpath, index_col=0)
    pcadf = pd.read_csv(pcadfpath, index_col=0)
    docs = df.index.values.tolist()
    
    # 1- similariy comparison
    
    simdf_orig = analyser.compute_document_similarity(df, "_orig") 
    simdf_pca = analyser.compute_document_similarity(pcadf, "_pca")
    
    analyser.get_most_similar_pairs(simdf_orig, "_orig")
    analyser.get_most_similar_pairs(simdf_pca, "_pca")
    
    
    # 2- topical terms
    N = 25
    tfdoctopic = analyser.get_topical_terms(df, N)
    pcadoctopic = analyser.get_topical_terms(pcadf, N)
    
    compare_topical_terms(tfdoctopic, pcadoctopic, docs, recordpath+os.sep+"comparetopicaltermsN25.txt")
    
    
    
    # 3- shift pca matrix because of the mighty negative values in it
    matrix_valueshift_rows(recordpath, "similaritymatrix_pca")




if __name__ == "__main__":
    

    
    #compute_similarity()
    
    
    
    #matrix_analysis()
    
    