'''
Created on Jan 18, 2013

@author: dicle
'''


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



numofprint = lambda lst,lstname : "num of "+lstname+" : "+str(len(lst))

           
if __name__ == "__main__":
    
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
    
