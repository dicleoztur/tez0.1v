'''
Created on Jan 18, 2013

@author: dicle
'''

import numpy as np
import dateutils



def uniquefylist(lst): 
    # order preserving
    noDupes = []
    [noDupes.append(i) for i in lst if not noDupes.count(i)]
    return noDupes
 

def getintersectionoflists(list1, list2):
    intersection = []
    for item1 in list1:
        if item1 in list2:
            intersection.append(item1)
    intersection = uniquefylist(intersection)
    numofcommons = len(intersection)
    return intersection


# separates list elements "w1 w2" to [w1,w2] and inserts them back to the list
def separateitemsaswords(inlist):
    temp = []
    for w in inlist:
        l = w.split()
        temp.extend(l)
    return temp



def printlist(lst):
    for w in lst:
        print w
        

def printdictionary(dct):
    for key,value in dct.items():
        print key," : ",value


''' takes a list of pairs [(a,x)] and returns a list of tuples as the reverse of the input pairs [(x,a)]  '''
def fliptuplepairs(tuplelist):
    flippedlist = []
    for item in tuplelist:
        k,v = item
        flippedlist.append((v,k))
    return flippedlist
                    



def initialize_dict(keys, val=0):
    dct = {}
    for key in keys:
        dct[key] = val
    return dct


def update_dct(updater, updatee):
    for k,v in updater.iteritems():
        updatee[k] += v


# f = lambda x,thr : x > thr
def filter_dict(dct, f, threshold):
    newdct = {}
    for k,v in dct.iteritems():
        if f(v,threshold):
            newdct[k] = v
    return newdct 

def filter_dict2(dct, threshold=0):
    newdct = {}
    for k,v in dct.iteritems():
        if v > threshold:
            newdct[k] = v
    return newdct           



# m is a dict = {"featureGROUPname" : [members]}
# returns the combination mapping of all members as matrix(nxm) where 
# n is the number of combinations and m=len(m.keys())
def get_combination_matrix(m):
    ls= [m[k] for k in sorted(m.keys())]
    
    ncombs = 1
    for _,v in m.iteritems():
        ncombs *= len(v)
    ncols = len(m.keys())
    matrix = np.zeros((ncombs, ncols), dtype=int)
    
    for i,items in enumerate(ls):
        lengths = [len(x) for x in ls[i+1:]] or [1]
        fillrange = reduce(lambda x,y : x*y, lengths) 
        print "col ",i,"  fillrange: ",fillrange," lenself: ",len(items)
        nrepeat = int(ncombs / (fillrange * len(items)))
        print "nrepeat ",nrepeat 
        for z in range(nrepeat):
            for j,_ in enumerate(items):
                start = j*fillrange + z*(len(items)*fillrange)
                print "\t z: ",z," start",start
                print
                matrix[start : (fillrange+start), i] = j
    return matrix
    

def featgroup_name_byindex(featmap, ind):
    l = sorted(featmap.keys())
    return l[ind]
def featgroup_index_byname(featmap, name):
    l = sorted(featmap.keys())
    try:
        return l.index(name)
    except:
        return -1


def print_features(featmap, matrix):
    print "features"
    _, ncols = matrix.shape
    
    for i in range(ncols):
        groupname = featgroup_name_byindex(featmap, i)
        print groupname," : ",
        groupvect = matrix[:, i]
        for featno in groupvect:
            if featno > -1:
                print featmap[groupname][featno],", ",
            else:
                print "group excluded"
        print
            

def exclude_n_featuregroups(matrix, groupindices):
    newmatrix = matrix.copy()
    newmatrix[:, groupindices] = -1
    return newmatrix


numofprint = lambda lst,lstname : "num of "+lstname+" : "+str(len(lst))

           
if __name__ == "__main__":
    
    m = {"l1" : ['1087449_1', '1087449_5', '1087449_9'],
    "l2" : ['1087449_1', '1087449_5', '1087449_10', '1087449_22', '1087449_23']}
    
    matrix = get_combination_matrix(m)
    print matrix
    print matrix.shape
    
    print_features(m, matrix)
    
    print exclude_n_featuregroups(matrix, [0])
    
    
    '''
    l1 = ['1087449_1', '1087449_5', '1087449_9', '1087449_22', '1087449_23']
    l2 = ['1087449_1', '1087449_5', '1087449_10']
    for w1,w2 in zip(l1,l2):
        print w1," ",w2
    
    ll1 = [1,2]
    ll2 = [8,9,0]
    print getintersectionoflists(ll1, ll2)   
    print dateutils.dateformat
    
    tuplelist = [(5,"a"),(8,"b")]
    print fliptuplepairs(tuplelist)
    '''
