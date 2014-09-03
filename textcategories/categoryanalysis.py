'''
Created on Aug 11, 2014

@author: dicle
'''

import os


from txtprocessor import listutils
from sentimentfinding import IOtools

# annotation classes vs prediction results



def is_in_category(textid, catname):
    items = textid.split("-")
    textcat = items[1]
    
    if catname == textcat:
        return 1
    else:
        return 0

def is_of_source(textid, sourcename):
    items = textid.split("-")
    textsource = items[0]
    
    if sourcename == textsource:
        return 1
    else:
        return 0


# dct = { key : count}
def dctcounts_to_percentage(dct): 
    newdct = dct.copy()
    
    total = sum(dct.values())
    for k in dct.keys():
        newdct[k] = round((float(dct[k]) / total) * 100.0, 2)
    return newdct
    

# attributelist: sources or categories
# has_attribute(textid, attr): returns 1 if textid has attr, 0 otherwise. 
def get_inlabel_stats(attributelist, labeltextdct, has_attribute, percentage=True):
        
    labels = labeltextdct.keys()
    
    labelattrcountdct = listutils.initialize_dict(keys=labels, val={})
    
    for label in labels:
        
        count_attrs = listutils.initialize_dict(keys=attributelist, val=0)
        
        textids = labeltextdct[label]
        for attr in attributelist:
            for textid in textids:
                count_attrs[attr] += has_attribute(textid, attr)
        
        labelattrcountdct[label] = count_attrs
    
    if percentage:
        for label in labelattrcountdct.keys():
            labelattrcountdct[label] = dctcounts_to_percentage(labelattrcountdct[label])
    
    return labelattrcountdct

   

if __name__ == "__main__":
    
    sources = ["radikal", "solhaber", "vakit"]
    categories = ["world", "economy", "politics", "turkey"]
    
    jsonpath = "/home/dicle/Dicle/Tez/corpusstats/learning10/experiments_5fold_scale/scores/double/redef-rat_lex-rat/comb975_F_0-0_1-1_2-1_3-3_4-0_5-1_6-1_7-0_8-3/STGobj-ALLsubj/"
    fname = "test_instances"
    labeltextdct = IOtools.readjson(os.path.join(jsonpath, fname))
    
    scounts = get_inlabel_stats(sources, labeltextdct, is_of_source, False)
    print scounts
    
    ccounts = get_inlabel_stats(categories, labeltextdct, is_in_category, False)
    print ccounts
    
    # convert jsons to csv. record them at learning_for_vis with the name of the test set
    
    
