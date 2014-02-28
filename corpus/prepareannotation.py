'''
Created on Jan 3, 2014

@author: dicle
'''

import random


nsingle = 2  # number of texts that will be annotated by only person p within p's coding set
noverlaps = 3  # number of texts that will be annotated by two persons within one coder-set
ntexts = nsingle + noverlaps

ncoders = 4   # should be even

totaltexts = (ncoders / 2)*noverlaps + ncoders*nsingle

users = [1,2,3,4]
tids = range(1, totaltexts)   # designated textids
##############


months = range(1,11)
nmonths = len(months)

cats = ["c1", "c2", "c3", "c4"]
ncats = len(cats)

resources = ["rad", "vak", "sol"]
nresources = len(resources)
sizecol = 250  # number of news in a cat of a res in a month

selected_texts = []  # contains [(newsid-res-cat)]

def initialize_dict(keys):
    dct = {}
    for key in keys:
        dct[key] = 0
    return dct


def update_dct(updater, updatee):
    for k,v in updater.iteritems():
        updatee[k] += v


def select_n_from_list(n, lst):
    countdct = initialize_dict(lst)
    
    lstsize = len(lst)
    mleap = n % lstsize
    mtimes = n / lstsize
    
    lstselected = random.sample(lst, mleap)
    
    for item in lst:
        countdct[item] += mtimes
    for selecteditem in lstselected:
        countdct[selecteditem] += 1
    
    return lstselected, countdct


def favour_not_selected(overalldistdct, nselections):
    return

# select n items from a collection of size (sizecol*nmonths*ncats*nresources) with a fair contribution from all res, cats and months
def justice_selection(n):
    # distribute to resources firstly
    '''
    resleap = n % nresources
    restimes = n / nresources
    
    selres2 = random.sample(resources, resleap)
       for res in resources:
        countresources[res] += restimes
    for selectedres in resselected2:
        countresources[selectedres] += resleap
    
    countmonths = initialize_dict(months)
    countresources = initialize_dict(resources)
    countcats = initialize_dict(cats)
    '''
    
    # overall distributions
    distmonths = initialize_dict(months)
    distresources = initialize_dict(resources)
    distcats = initialize_dict(cats)
    
    _, countresources = select_n_from_list(n, resources)
    update_dct(countresources, distresources)

    for res,rangecat in countresources.iteritems():
        
        catselected2, countcats = select_n_from_list(rangecat, cats)
        update_dct(countcats, distcats)
        print res.upper()
        print "\t totaltexts:",rangecat
        print "\t selected cats: ",catselected2,countcats
        
        #for cat,rangemonth in countcats.iteritems():
        for cat,rangemonth in countcats.iteritems():
            monthselected2, countmonths = select_n_from_list(rangemonth, months)
            update_dct(countmonths, distmonths)
            print "\t",cat.upper()
            print "\t\t selected months: ", monthselected2,countmonths
            
            #for month, rangetext in countmonths.iteritems():
            for month,rangetext in countmonths.iteritems():
                print "\t\t month",month,  
                print "\t selected  ",rangetext," texts"
                # select from corpus.csv rangetext*
                

    print distresources
    print distcats
    print distmonths


def get_row_from_corpuscsv(newsid, resourcename, catname, corpusname):
    return 
    

def generate_question_table():
    return


def generate_evalutation_table():
    return


def get_user_text_distributions(questionscsv="", usercsv="", evaluationcsv=""):
    users = range(1, ncoders+1)
    
    for user in users:
        singtexts = justice_selection(nsingle)   # will return textids as (newsid-res-cat) # handle selected_texts here
        
    
    

def main():
    justice_selection(10)


if __name__ == "__main__":
    main()
        
    
    


