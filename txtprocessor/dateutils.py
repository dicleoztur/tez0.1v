# -*- coding: utf-8 -*- 
'''
Created on Mar 31, 2013

@author: dicle
'''


# 22 aralik 2013. No change format to get iso format yyyy-mm-dd
'''
date format is intended to be dd/mm/yyyy
'''
# -*- coding: utf-8 -*- 
'''
Created on Mar 31, 2013

@author: dicle
'''


'''
date format is intended to be dd/mm/yyyy
'''

import re

dateformat = {u"ocak":"01", u"şubat":"02", u"mart":"03", u"nisan":"04", u"mayıs":"05", u"haziran":"06", u"temmuz":"07", u"ağustos":"08",
              u"eylül":"09", u"ekim":"10", u"kasım":"11", u"aralık":"12"}

datesep = "-"


# dlist is a list of "dd/mm/yyyy" strings. returns sorted list of "dd/mm/yyyy" strings
def sortdatelist(dlist):
    dlist2 = map(lambda s : s.split(datesep), dlist)  # dlist is [["dd","mm","yyyy"]]
    #dlist2 = [map(lambda x : int(x), item) for item in dlist]   # dlist2 is [[dd,mm,yyyy]] - integers
    #print "dlist ",dlist
    #print "dlist2 ",dlist2
    map(lambda x : x.reverse(), dlist2)  # dlist2 is [[yyyy,mm,dd]]
    dlist2.sort()
    map(lambda x : x.reverse(), dlist2)
    dlist2 = [datesep.join(item) for item in dlist2]
    return dlist2


# dlist is a list of [dd,mm,yyyy] int lists. returns sorted list of [dd,mm,yyyy] int lists.
def sortdatelist2(dlist):
    map(lambda x : x.reverse(), dlist2)  # dlist2 is [[yyyy,mm,dd]]
    dlist2.sort()
    map(lambda x : x.reverse(), dlist2)
    return dlist2



# dlist is a list of [dd,mm,yyyy] int lists. returns sorted list of "dd/mm/yyyy" strings
def sortdatelist3(dlist):
    map(lambda x : x.reverse(), dlist2)  # dlist2 is [[yyyy,mm,dd]] 
    dlist2.sort()
    map(lambda x : x.reverse(), dlist2)
    dlist3 = [datesep.join(map(lambda x : str(x), item)) for item in dlist2]
    return dlist3


# dlist is a list of "dd/mm/yyyy" strings. the function returns a list of (day,month,year) integer tuples.
def getdateitems(dlist):
    dtuples = []
    for date in dlist:
        items = dlist.split(date.sep)
        

def parsewrittendate3(wdate):
    items = wdate.split()
    
    if wdate.isspace() or len(items) == 0:
        return ""
    #print wdate," ",items
    
    if len(items) <= 2 and len(items) >= 1:    # format is dd/mm/yyyy hh:mm
        ndate = items[0]
    elif len(items) > 2:                # format is day month year (hh:mm)? 
                                        # other types like dd.mm.yyyy not handled                                  
        month = items[1]
        if len(month) > 2:
            items[1] = dateformat[month]
            
        ndate = datesep.join(items[:3])
    
    return ndate.strip()



# returns date as a list of [dd,mm,yyyy] where the elements are int.
def parsewrittendate(wdate):
    items = wdate.lower().split()
    date = ""
    if wdate.isspace() or len(items) == 0:
        return date
    #print wdate," ",items
    
    if len(items) <= 2 and len(items) >= 1:    # format is dd/mm/yyyy hh:mm
        datelist = [i for i in re.split(r"\.|\/", items[0])]
        datelist = datelist[::-1]
        date = datesep.join(datelist)       
    elif len(items) > 2:                # format is day month year (hh:mm)?                                                                        
        month = items[1]
        if len(month) > 2:
            items[1] = dateformat[month]
        day = items[0]
        if len(day) == 1:
            items[0] = "0"+day  
        
        datelist = items[:3]  
        datelist = datelist[::-1]   # reverse to get iso format dates 
        date = datesep.join(datelist)
    
    return date        


def parsewrittendate2(wdate):
    items = wdate.split()
    date = []
    if wdate.isspace() or len(items) == 0:
        return date
    #print wdate," ",items
    
    if len(items) <= 2 and len(items) >= 1:    # format is dd/mm/yyyy hh:mm
        date = [int(i) for i in re.split(r"\.|\/", items[0])]       
    elif len(items) > 2:                # format is day month year (hh:mm)?                                                                        
        month = items[1]
        if len(month) > 2:
            items[1] = dateformat[month]
            
        date = [int(i) for i in items[:3]]
    
    return date[::-1]        # reverse to get iso format dates   #return date


def getmonth():
    return

def getday():
    return

def getyear():
    return



if __name__ == "__main__":
    d = "3 Mart 2013 Cumartesi 01:25".lower()
    print parsewrittendate(d)


    dlist = ["21/02/2013", "21/03/2012", "20/02/2011", "09/10/2010", "05/05/2009", "04/03/2011", "19/03/2013", "25/07/2008", "24/09/2007", "01/11/2013"]
    dlist2 = [parsewrittendate2(d) for d in dlist]
    print dlist2
    print sortdatelist3(dlist2)
    
    print sortdatelist(dlist)

'''
>>> dlist = map(lambda s : s.split(datesep), dlist)>>> dlist
[['21', '02', '2013'], ['21', '03', '2012'], ['20', '02', '2011'], ['09', '10', '2010'], ['05', '05', '2009'], ['04', '03', '2011'], ['19', '03', '2013'], ['25', '07', '2008'], ['24', '09', '2007'], ['01', '11', '2013']]
>>> dlist2 = [map(lambda x : int(x), item) for item in dlist]
>>> dlist2
[[21, 2, 2013], [21, 3, 2012], [20, 2, 2011], [9, 10, 2010], [5, 5, 2009], [4, 3, 2011], [19, 3, 2013], [25, 7, 2008], [24, 9, 2007], [1, 11, 2013]]
>>> dlist2 = [tuple(map(lambda x : int(x), item)) for item in dlist]


>>> dlist5
[(2013, 11, 1), (2013, 3, 19), (2013, 2, 21), (2012, 3, 21), (2011, 3, 4), (2011, 2, 20), (2010, 10, 9), (2009, 5, 5), (2008, 7, 25), (2007, 9, 24)]
>>> dlist6 = map(lambda x : list(x), dlist5)
>>> map(lambda x : x.reverse(), dlist6)
[None, None, None, None, None, None, None, None, None, None]
>>> dlist7 = map(lambda x : tuple(x), dlist6)
>>> dlist7
[(1, 11, 2013), (19, 3, 2013), (21, 2, 2013), (21, 3, 2012), (4, 3, 2011), (20, 2, 2011), (9, 10, 2010), (5, 5, 2009), (25, 7, 2008), (24, 9, 2007)]
>>> dlist7.reverse()
'''
    
    
    