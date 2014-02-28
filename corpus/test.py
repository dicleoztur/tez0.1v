'''
Created on Dec 25, 2013

@author: dicle
'''



cfdmonthlycategorycount = ConditionalFreqDist((colldf.loc[i,"date"][:-3], "-".join(map(lambda x : str(x).strip(), [colldf.loc[i, "resource"], colldf.loc[i, "category"]]))) for i in range(numoftexts)) 


catcfdmap = {}
for canoniccatname, rescatnames in rescatmap.iteritems():
    monthresourcepairs = []
    for rescat in rescatnames:
        resourcename = rescat.split("-")[0]
        for month in cfdmonthlycategorycount.conditions():
            numofoccurrences = cfdmonthlycategorycount[month][rescat]
            print resourcename," had ",numofoccurrences," times texts in :",rescat," during ",month
            for i in range(numofoccurrences):
                monthresourcepairs.append((month, resourcename))
    cfdmonthlyresourcecount_percat = ConditionalFreqDist(monthresourcepairs)    
    catcfdmap[canoniccatname] = cfdmonthlyresourcecount_percat

 