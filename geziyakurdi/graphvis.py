'''
Created on Jul 23, 2013

@author: dicle
'''

import numpy as np
import pandas as pd
import networkx as nx
import pylab as plt
import datetime
import random

from sentimentfinding import plotter
from classification import numericutils

# mostly for similartity or distance matrices


x_axislength = 1200
y_axislength = 1000

def name_nodes(filename):
    '''  in this function,
    the filename is assumed to be in the format,
    ##   polarityDate-author_ArticleTitle.txt   ##
    where date is like 4Haz with 4 or 5 chars
    and polarity is pos or neg.
    '''
    items = filename.split("-")
        
    attr, name = items[0], items[1]
    polarity = attr[:3]
    date = attr[3:]
        
    items = name.split("_")
    author = items[0]
    title = items[1]
    title = title[:-4]
    
    # bunlari dict'e atalim.
    
    return name[:-4], polarity, date



def uniquefy_edgelist(edges):
    newedgelist = []
    n = len(edges)
    if n > 0:
        newedgelist.append(edges[0])
        for i in range(1,n):
            if edges[i] not in newedgelist:
                newedgelist.append(edges[i])
    return newedgelist

monthdict = {"May" : 5, "Haz" : 6, "Tem" : 7, "Agu" : 8}  # devami gelir


def arrange_dates_to_position(nodedatedict, polarnodedict):
    year = 2013
    
    datenodedict = {}
    datenodes = []
    dates = []
    for node,date in nodedatedict.iteritems():
        day = int(date[:-3])
        month = date[-3:]
        month = monthdict[month]
        newdate = datetime.date(year, month, day)
        newdate = newdate.strftime('%d.%m.%Y')
        datenodes.append((newdate, node))
        dates.append(newdate)
    
    for date in dates:
        datenodedict[date] = []
    for k,v in datenodes:
        datenodedict[k].append(v)
        
    dates = datenodedict.keys()
    dates.sort()
    
    nodepositiondict = {}
    intervalx = x_axislength / len(dates)
    for date, nodes in datenodedict.iteritems():
        ind = dates.index(date)
        xpos = ind * intervalx
        intervaly = x_axislength / len(nodes)
        for i,node in enumerate(nodes):
            ypos = random.randint(10, intervaly-10) + i*intervaly
            if polarnodedict[node] < 0:
                ypos = 0 - ypos
            #print node," : ",date," . ",ypos
            nodepositiondict[node] = (xpos, ypos)
       
    return nodepositiondict,dates


'''
def arrange_dates_to_position(nodedatedict):
    year = 2013
    
    datenodedict = {}
    datenodes = []
    dates = []
    for node,date in nodedatedict.iteritems():
        day = int(date[:-3])
        month = date[-3:]
        month = monthdict[month]
        newdate = datetime.date(year, month, day)
        newdate = newdate.strftime('%d.%m.%Y')
        datenodes.append((newdate, node))
        dates.append(newdate)
    
    for date in dates:
        datenodedict[date] = []
    for k,v in datenodes:
        datenodedict[k].append(v)
        
    dates = datenodedict.keys()
    dates.sort()
    
    nodepositiondict = {}
    intervalx = x_axislength / len(dates)
    for date, nodes in datenodedict.iteritems():
        ind = dates.index(date)
        xpos = ind * intervalx
        intervaly = x_axislength / len(nodes)
        for i,node in enumerate(nodes):
            ypos = random.randint(10, intervaly-10) + i*intervaly
            print node," : ",date," . ",ypos
            nodepositiondict[node] = (xpos, ypos)
       
    return nodepositiondict,dates
'''


polaritydict = {"pos" : 1, "neg" : -1, "neu": 1}
edgecolordict= {"y" : "opponent connection", "g" : "pos connection", "r" : "neg connection"}

def build_graph(matpath, imgpath, directed=False):
    matrixdf = pd.read_csv(matpath, index_col=0)
    
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    nodelist = matrixdf.index.values.tolist()
    n = len(nodelist)
    
    posnodes = []
    negnodes = []
    edgelists = []
    nodedatedict = {}
    polarnodedict = {}
    
    for i in range(n):
        median = np.median(matrixdf.values[i,:])
        for j in range(n):
        #for j in range(i+1,n):
            source, pol1, date1 = name_nodes(nodelist[i])
            target, pol2, date2 = name_nodes(nodelist[j])
            val = round(matrixdf.iloc[i,j], 4)
            nodedatedict[source] = date1
            nodedatedict[target] = date2
            '''
            G.add_edge(source, target, weight=val, color="green")
            edgelists.append(([(source, target)], val))
            
            if pol1 == "pos":
                posnodes.append(source)
            else:
                negnodes.append(source)
                
            if pol2 == "pos":
                posnodes.append(target)
            else:
                negnodes.append(target)
            
            '''
            if val >= median:
                p1 = polaritydict[pol1]
                p2 = polaritydict[pol2]
                if p1 + p2 == 0:
                    clr = "y"
                elif p1+p2 > 0:
                    clr = "g"
                else:
                    clr = "r"
                    
                G.add_edge(source, target, weight=val)
                edgelists.append(([(source, target)], val, clr))
                
                if p1 > 0:
                    posnodes.append(source)
                else:
                    negnodes.append(source)
                    
                if p2 > 0:
                    posnodes.append(target)
                else:
                    negnodes.append(target)
    
    for i in edgelists:
        print i
    print "b: ",str(len(edgelists))        
    edgelists = uniquefy_edgelist(edgelists) 
    print "a: ",str(len(edgelists)) 
                 
    posnodes = list(set(posnodes))
    negnodes = list(set(negnodes))
    print "p: ",str(len(posnodes)),"  n: ",str(len(negnodes))
    for n in posnodes:
        polarnodedict[n] = 1
    for n in negnodes:
        polarnodedict[n] = -1
        
    # start drawing
    #fig = plotter.set_plotframe("Document similarity relations", " ", " ")       
    im = plt.imread("/home/dicle/Desktop/imgedit2/background7.png")
    implot = plt.imshow(im, origin='lower', extent=[-350, 3500, -75, 2350])
    fig = plt.gcf()
    dpi = 100
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    #plt.autoscale(True, axis='both')
    
    #position=nx.spring_layout(G)   # should be by date
    plt.xlim(4000)
    position = nx.spectral_layout(G, scale=im.shape[1])
    '''
    position, dates = arrange_dates_to_position(nodedatedict, polarnodedict)
    plt.xticks(np.arange(len(dates))*(x_axislength/len(dates)), dates, rotation=45)
    '''
    for k,v in position.iteritems():
        print k," ",v
 
    # pos nodes of color red, neg nodes of color yellow
    nx.draw_networkx_nodes(G,pos=position, nodelist=posnodes, node_size=150, node_color="g", linewidths=0, label="pos")
    nx.draw_networkx_nodes(G,pos=position, nodelist=negnodes, node_size=150, node_color="r", linewidths=0, label="neg")
    
    for edgeitem in edgelists:
        edge, weight, clr = edgeitem
        edgesize= 35 * weight
        nx.draw_networkx_edges(G, pos=position, edgelist=edge, width=edgesize, edge_color=clr, label=edgecolordict[clr])
        
    nx.draw_networkx_labels(G, position, font_size=10, font_family='sans-serif')      
    
    #plt.axis('off')
    ax = plt.gca()
    ax.set_xlim([-1000,4000])
    #ax.invert_yaxis()
    #ax.yaxis.set_visible(False)
    #implot = plt.imshow(im, origin='lower')
    plt.savefig(imgpath, dpi=dpi)
    #plt.legend()
    plt.show()





def build_graph_mostsimilars(matpath, imgpath):
    matrixdf = pd.read_csv(matpath, index_col=0)
    
    G = nx.DiGraph()
    
    nodelist = matrixdf.index.values.tolist()
    n = len(nodelist)
    
    posnodes = []
    negnodes = []
    edgelists = []
    nodedatedict = {}
    polarnodedict = {}
    
    for i in range(n):
        
        val, j = numericutils.get2ndmax(matrixdf.values[i,:])
        val = round(val, 4)
        
        source, pol1, date1 = name_nodes(nodelist[i])
        target, pol2, date2 = name_nodes(nodelist[j])
        nodedatedict[source] = date1
        nodedatedict[target] = date2

        p1 = polaritydict[pol1]
        p2 = polaritydict[pol2]
        
        if p2 > 0:   # edge is coloured by target oohhh asimilasyon
            clr = "g"
        else:
            clr = "r"
            
        G.add_edge(source, target, weight=val)
        edgelists.append(([(source, target)], val, clr))
        
        if p1 > 0:
            posnodes.append(source)
        else:
            negnodes.append(source)
            
        if p2 > 0:
            posnodes.append(target)
        else:
            negnodes.append(target)
    
               
    edgelists = uniquefy_edgelist(edgelists) 
    
                 
    posnodes = list(set(posnodes))
    negnodes = list(set(negnodes))
    print "p: ",str(len(posnodes)),"  n: ",str(len(negnodes))
    for n in posnodes:
        polarnodedict[n] = 1
    for n in negnodes:
        polarnodedict[n] = -1
        
    # start drawing
    fig = plotter.set_plotframe("Document similarity relations", " ", " ")   
    '''    if with background
    im = plt.imread("/home/dicle/Desktop/imgedit2/background7.png")
    implot = plt.imshow(im, origin='lower', extent=[-350, 3500, -75, 2350])
    
    fig = plt.gcf()
    dpi = 100
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    #plt.autoscale(True, axis='both')
    plt.xlim(4000)
    '''
       
    #position=nx.spring_layout(G)   # should be by date
    #position = nx.spectral_layout(G, scale=im.shape[1])
    position, dates = arrange_dates_to_position(nodedatedict, polarnodedict)
    plt.xticks(np.arange(len(dates))*(x_axislength/len(dates)), dates, rotation=45)
    
    for k,v in position.iteritems():
        print k," ",v
 
    # pos nodes of color red, neg nodes of color yellow
    nx.draw_networkx_nodes(G,pos=position, nodelist=posnodes, node_size=150, node_color="g", linewidths=0, label="pos")
    nx.draw_networkx_nodes(G,pos=position, nodelist=negnodes, node_size=150, node_color="r", linewidths=0, label="neg")
    
    print "xxxxxx ",len(edgelists)
    
    for edgeitem in edgelists:
        edge, weight, clr = edgeitem
        print edge," ",weight
        edgesize= 35 * weight
        nx.draw_networkx_edges(G, pos=position, edgelist=edge, width=edgesize, edge_color=clr, label=edgecolordict[clr])
        
    nx.draw_networkx_labels(G, position, font_size=12, font_family='sans-serif')      
    
    #plt.axis('off')
    ax = plt.gca()
    #ax.set_xlim([-1000,4000])
    #ax.invert_yaxis()
    ax.yaxis.set_visible(False)
    #implot = plt.imshow(im, origin='lower')
    plt.savefig(imgpath, dpi=100, edgecolor='none')   #, facecolor="y")
    #plt.legend()
    plt.show()





'''
def build_graph(matpath, imgpath):
    matrixdf = pd.read_csv(matpath, index_col=0)
    
    G = nx.Graph()
    
    nodelist = matrixdf.index.values.tolist()
    n = len(nodelist)
    
    posnodes = []
    negnodes = []
    edgelists = []
    
    for i in range(n):
        for j in range(i+1,n):
            source, pol1 = name_nodes(nodelist[i])
            target, pol2 = name_nodes(nodelist[j])
            val = round(matrixdf.iloc[i,j], 4)
            
            G.add_edge(source, target, weight=val, color="green")
            edgelists.append(([(source, target)], val))
            
            if pol1 == "pos":
                posnodes.append(source)
            else:
                negnodes.append(source)
            if pol2 == "pos":
                posnodes.append(target)
            else:
                negnodes.append(target)
    
    # start drawing
    plotter.set_plotframe("Document similarity relations", " ", " ")       
    #position=nx.spring_layout(G)   # should be by date
    position = nx.spectral_layout(G)
 
    # pos nodes of color red, neg nodes of color yellow
    nx.draw_networkx_nodes(G,pos=position, nodelist=posnodes, node_size=150, node_color="r")
    nx.draw_networkx_nodes(G,pos=position, nodelist=negnodes, node_size=150, node_color="y")
    
    for edgeitem in edgelists:
        edge, weight = edgeitem
        edgesize= 50 * weight
        nx.draw_networkx_edges(G, pos=position, edgelist=edge, width=edgesize, edge_color="g")
        
    nx.draw_networkx_labels(G, position, font_size=8, font_family='sans-serif')      
    
    plt.axis('off')
    plt.savefig(imgpath)
    plt.show()
    '''

def create_mostsimilars_graph():
    #inpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/edit/"
    inpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/wordletest/matrix/"
    outpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/edit/wordletest/matrix/"
    g_orig = "similaritymatrix_orig"
    #g_pca = "similaritymatrix_pca"
    g_pca = "similaritymatrix_pcaShift"
    #build_graph(inpath+g_orig+".csv", inpath+g_orig)
    build_graph_mostsimilars(inpath+g_pca+".csv", inpath+"mostsimilar-"+g_pca)
    #build_graph_mostsimilars(inpath+g_orig+".csv", inpath+"mostsimilar-"+g_orig)
      

if __name__ == "__main__":
    
    '''
    #inpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/edit/"
    inpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/edit/wordletest/matrix/"
    outpath = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus2/ldatests22Temmuz/edit/wordletest/graphvisuals/"
    g_orig = "similaritymatrix_orig"
    #g_pca = "similaritymatrix_pca"
    g_pca = "similaritymatrix_pcaShift"
    #build_graph(inpath+g_orig+".csv", inpath+g_orig)
    build_graph(inpath+g_pca+".csv", inpath+g_pca)
    '''
    create_mostsimilars_graph()
    
    
    
    
    
    
    
    
    