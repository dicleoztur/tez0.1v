# -*- coding: utf-8 -*-

'''
Created on Jan 25, 2014

@author: dicle
'''




import random
import string
import os
from numpy import mean
from datetime import date,datetime
from dateutil import relativedelta,parser
import csv


from txtprocessor.listutils import *   # to use initialize_dict, update_dict
import metacorpus,extractnewsmetadata
from sentimentfinding import IOtools



class AnnotationBuilder():
    nsingle = 2  # number of texts that will be annotated by only person p within p's coding set
    noverlaps = 3  # number of texts that will be annotated by two persons within one coder-set
    ntexts = nsingle + noverlaps
    
    ncoders = 4   # should be even
    
    totaltexts = (ncoders / 2)*noverlaps + ncoders*nsingle
    
    users = [1,2,3,4]
    #tids = range(1, totaltexts)   # designated textids
    ##############
    
    
    months = range(1,11)
    nmonths = len(months)
    
    cats = ["c1", "c2", "c3", "c4"]
    ncats = len(cats)
    
    resources = ["rad", "vak", "sol"]
    nresources = len(resources)
    sizecol = 250  # number of news in a cat of a res in a month
    
    selected_texts = []  # contains [(newsid-res-cat)]
    
    corpuspath = ""
    outfolder = ""

    
    
    '''
    ntexts: total number of texts that will be annotaed per user 
    psingle: percentage of texts that will be annotated by only one user (percentage in the set of one annotator) like 2/3
    poverlaps: percentage of texts that will be annotated by two users (per pair)
    nusers: number of users to get involved in the annotation task
    
    months: months of the texts in the annotation set
    cats: categories of the texts in the annotation set
    resources: resources of the texts in the annotation set
    
    corpuspath: path to the csvfile containing the text metadata in the corpus
    
    '''
    def __init__(self, ntexts, psingle, poverlaps, nusers, months, resources, cats, corpuspath, outfolder):
        
        fold = float(psingle + poverlaps)
        
        self.ntexts = ntexts  # per user
        self.nsingle = int((self.ntexts / fold) * psingle)
        self.noverlaps = int((self.ntexts / fold) * poverlaps)
        self.ncoders = nusers
        
        self.coders = self.generate_usernames(self.ncoders)
        
        self.totaltexts = (self.ncoders / 2)*self.noverlaps + self.ncoders*self.nsingle #self.ntexts * self.ncoders
        
        self.months = months
        self.nmonths = len(self.months)
        
        self.cats = cats
        self.ncats = len(self.cats)
        
        self.resources = resources
        self.nresources = len(self.resources)
        
        self.corpuspath = corpuspath
        self.outfolder = outfolder
        
        self.selected_texts = []   # to assure no collision in the texts during random selection 
        
        self.singles_jsonpath = self.outfolder + os.sep + "singleannotation.json"
        self.singles_csvpath = self.outfolder + os.sep + "singleannotation.csv"
        self.doubles_jsonpath = self.outfolder + os.sep + "doubleannotation.json"
        self.doubles_csvpath = self.outfolder + os.sep + "doubleannotation.csv"
        
        '''
        self.distmonths = {}
        self.distresources = {}
        self.distcats = {}
        '''
    
    
    
    def toscreen(self):
        
        print numofprint(self.coders, "coders")
        print "coder names: ",self.coders
        
        print numofprint(self.cats, "cats")
        print numofprint(self.months, "months")
        
        print "numoftexts per coder: ",self.ntexts
        print "total num of distinct texts annotatable: ",self.totaltexts
    
    
    def generate_usernames(self, nusers):
        usernames = ["".join(random.sample(string.ascii_letters, random.randint(5,9))) for _ in range(nusers)]
        return usernames
    
    
    
    def select_n_from_list(self, n, lst):
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
        
        
    '''
    def select_n_from_list(self, n, uppercount):
        
        
        # lst = filter(lambda x :)
        
        meanval = int(mean(uppercount.values()))
        favoureditems = filter_dict(uppercount, f=lambda x,y: x<=y, threshold=meanval)
        lst = favoureditems.keys()
        
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
    '''
    
    def favour_not_selected(self, overalldistdct, nselections):
        return
    
    '''
    # select n items from a collection of size (sizecol*nmonths*ncats*nresources) with a fair contribution from all res, cats and months
    def justice_selection(self, n):
        
        # overall distributions
        fairselect = Selection()
        fairselect.initialize(self.months, self.resources, self.cats)
        
        assignments = {}

        
        #_, countresources = self.select_n_from_list(n, self.resources)
        _, countresources = self.select_n_from_list(n, initialize_dict(self.resources))
        update_dct(countresources, fairselect.distresources)
        
        catdistribution_overall = initialize_dict(self.cats)
    
        for res,rangecat in countresources.iteritems():
            
            catselected2, countcats = self.select_n_from_list(rangecat, catdistribution_overall)
            update_dct(countcats, fairselect.distcats)
            update_dct(countcats, catdistribution_overall)
            monthdisribution_overresource = initialize_dict(self.months)
            
            print res.upper()
            print "\t totaltexts:",rangecat
            print "\t selected cats: ",catselected2,countcats
            
            #assign_cats = initialize_dict(self.cats, val={})
            assign_cats = {}
            
            for cat,rangemonth in countcats.iteritems():
                monthselected2, countmonths = self.select_n_from_list(rangemonth, monthdisribution_overresource)
                assign_months = filter_dict2(countmonths)
                if assign_months:    
                    assign_cats[cat] = assign_months
                
                update_dct(countmonths, monthdisribution_overresource)
                update_dct(countmonths, fairselect.distmonths)
                print "\t",cat.upper()
                print "\t\t selected months: ", monthselected2,countmonths
                
                #for month, rangetext in countmonths.iteritems():
                for month,rangetext in countmonths.iteritems():
                    print "\t\t month",month,  
                    print "\t selected  ",rangetext," texts"
                    # select from corpus.csv rangetext*
            
            if assign_cats:       
                assignments[res] = assign_cats       
    
        fairselect.toscreen_dists()
        return fairselect, assignments   #assignment is in json form    
    '''
       
    # select n items from a collection of size (sizecol*nmonths*ncats*nresources) with a fair contribution from all res, cats and months
    def justice_selection(self, n):
        
        # overall distributions
        fairselect = Selection()
        fairselect.initialize(self.months, self.resources, self.cats)
        
        assignments = {}

        
        #_, countresources = self.select_n_from_list(n, self.resources)
        _, countresources = self.select_n_from_list(n, initialize_dict(self.resources))
        update_dct(countresources, fairselect.distresources)
        
        for res,rangecat in countresources.iteritems():
            
            catselected2, countcats = self.select_n_from_list(rangecat, self.cats)
            update_dct(countcats, fairselect.distcats)
            
                        
            print res.upper()
            print "\t totaltexts:",rangecat
            print "\t selected cats: ",catselected2,countcats
            
            #assign_cats = initialize_dict(self.cats, val={})
            assign_cats = {}
            
            for cat,rangemonth in countcats.iteritems():
                monthselected2, countmonths = self.select_n_from_list(rangemonth, self.months)
                assign_months = filter_dict2(countmonths)
                if assign_months:    
                    assign_cats[cat] = assign_months
                
                update_dct(countmonths, fairselect.distmonths)
                print "\t",cat.upper()
                print "\t\t selected months: ", monthselected2,countmonths
                
                #for month, rangetext in countmonths.iteritems():
                for month,rangetext in countmonths.iteritems():
                    print "m",
                    #print "\t\t month",month,  
                    #print "\t selected  ",rangetext," texts"
                    # select from corpus.csv rangetext*
            
            if assign_cats:       
                assignments[res] = assign_cats       
    
        fairselect.toscreen_dists()
        return fairselect, assignments   #assignment is in json form
    
    
    
    def get_row_from_corpuscsv(self, newsid, resourcename, catname, corpusname):
        return 
        
    
    def generate_user_table(self, tablefolder, userlist=None):
        if userlist is None:
            userlist = self.coders
        
        usertable = [("uid", "uname", "lastevaluationid")]
        for i,u in enumerate(userlist):
            usertable.append((i, u, -1))
        
        tpath = os.path.join(tablefolder, "users.csv")
        IOtools.tocsv_lst(usertable, tpath)
        '''
        with open(tpath, "w") as f:
            csvwriter = csv.writer(f, delimiter="\t")
            csvwriter.writerows(usertable)
        '''
        
    # read jsons select from csvs
    def generate_question_table(self, tablefolder, textindices):
        df = IOtools.readcsv(self.corpuspath)
        
        questiontable = [("qid", "qname", "qtitle", "qcontent")]
        
        for i,textindex in enumerate(textindices):
            resourcename = df.loc[textindex, "resource"]
            orgcatname = df.loc[textindex, "originalcatname"]
            catname = df.loc[textindex, "category"]
            textid = df.loc[textindex, "newsid"]
            textid = str(textid).split(".")[0]
            print textindex, type(textindex), textid, type(textid)
            questionname = resourcename + "-" + catname + "-" + textid
            
            filepath = os.path.join(metacorpus.rawcorpuspath, resourcename, orgcatname, textid+metacorpus.itemext)
            title, content = extractnewsmetadata.get_news_article(filepath)
            questiontable.append((i, questionname, title, content))
         
        tpath = os.path.join(tablefolder, "questions.csv")
        IOtools.tocsv_lst(questiontable, tpath)
        
    
    
    def generate_evalutation_table(self, tablefolder, usertextpairs):
        evaluationtable = [("eid", "userid", "questionname", "answer", "isanswered", "qorder")]
        
        userids = [userid for userid,_ in usertextpairs]
        uqorderdct = initialize_dict(userids)
        
        for i,(uid,qname) in enumerate(usertextpairs):
            qorder = uqorderdct[uid]
            evaluationtable.append((i, uid, qname, -100, 0, qorder))
            uqorderdct[uid] = uqorderdct[uid] + 1
            
        tpath = os.path.join(tablefolder, "evaluations.csv")
        IOtools.tocsv_lst(evaluationtable, tpath)
        

    '''
    def generate_evalutation_table(self, tablefolder, usertextpairs):
        evaluationtable = [("eid", "userid", "questionname", "answer", "isanswered")]
        
        for i,(uid,qname) in enumerate(usertextpairs):
            evaluationtable.append((i, uid, qname, -100, 0))
            
        tpath = os.path.join(tablefolder, "evaluations.csv")
        IOtools.tocsv_lst(evaluationtable, tpath)
    '''       
    
    def generate_tables(self, singlejsonpath=None, doublejsonpath=None):
        
        if singlejsonpath is None:
            singlejsonpath=self.singles_jsonpath
        if doublejsonpath is None:
            doublejsonpath=self.doubles_jsonpath
            
        selectedindices = []
        user_textid = []
        df = IOtools.readcsv(self.corpuspath)
        df["annotator"] = ""
        
        # singles  
        single_assignments = IOtools.readjson(singlejsonpath)
        for userid, resourcedist in single_assignments.iteritems():
            for resourcename, catdist in resourcedist.iteritems():
                for catname, monthdist in catdist.iteritems():
                    for month, ntexts in monthdist.iteritems():
                        dfx = df[(df["resource"]==resourcename) & (df["category"]==catname) & (df["date"].str.startswith(month))]
                        for _ in range(ntexts):
                            #print resourcename,catname,month
                            #print dfx.values.shape,"  <- ",df.values.shape
                            randomindex = random.choice(dfx.index.values.tolist())
                            while randomindex in selectedindices:
                                randomindex = random.choice(dfx.index.values.tolist())
                            selectedindices.append(randomindex)  
                            newsid = df.loc[randomindex, "newsid"]
                            newsid = str(int(newsid))
                            textid = resourcename + "-" + catname + "-" + str(newsid)
                            user_textid.append((userid, textid))
                            df.loc[randomindex, "annotator"] = userid
        
        # record these newstexts to a csv 
        dfx = df.loc[selectedindices, :]
        IOtools.tocsv(dfx, self.singles_csvpath) 
        
        
        # doubles
        df["annotator2"] = ""
        selectedindices2 = []
        double_assignments = IOtools.readjson(doublejsonpath)
        for pairname, resourcedist in double_assignments.iteritems():
        #for i in range(0,self.ncoders,2):
            for resourcename, catdist in resourcedist.iteritems():
            #for resourcename, catdist in resourcedist.iteritems():
                for catname, monthdist in catdist.iteritems():
                    for month, ntexts in monthdist.iteritems():
                        dfx = df[(df["resource"]==resourcename) & (df["category"]==catname) & (df["date"].str.startswith(month))]
                        for _ in range(ntexts):
                            randomindex = random.choice(dfx.index.values.tolist())
                            while randomindex in selectedindices:
                                randomindex = random.choice(dfx.index.values.tolist())
                            selectedindices.append(randomindex)  
                            selectedindices2.append(randomindex) 
                            newsid = df.loc[randomindex, "newsid"]
                            newsid = str(int(newsid))
                            textid = resourcename + "-" + catname + "-" + str(newsid)
                            
                            '''user1 = self.coders[i]
                            user2 = self.coders[i+1]
                            user_textid.append((user1, textid))
                            user_textid.append((user2, textid)) '''
                            userids = pairname.split("-")
                            user1 = str(userids[0])
                            user2 = str(userids[1])
                            user_textid.append((user1, textid))
                            user_textid.append((user2, textid))
                            df.loc[randomindex, "annotator"] = user1          
                            df.loc[randomindex, "annotator2"] = user2  
    
        dfx = df.loc[selectedindices2, :]
        IOtools.tocsv(dfx, self.doubles_csvpath) 
        IOtools.tocsv(df, os.path.join(self.outfolder, "corpusstats_annotatable.csv"))
                      
    
        tablespath = IOtools.ensure_dir(os.path.join(self.outfolder,"tables"))
        # write user table
        self.generate_user_table(tablespath)
        
        # write question table
        self.generate_question_table(tablespath, selectedindices)
        
        # write evaluation table
        self.generate_evalutation_table(tablespath, user_textid)
        
        
    
    # write jsons
    def get_user_text_distributions(self):
        #users = range(1, self.ncoders+1)
        
        # 1- get single-annotation list
        # 2- get double-annotation list
        
        usertextassignment = {}
        singleannot_distribution = Selection()
        singleannot_distribution.initialize(self.months, self.resources, self.cats)
        for i,user in enumerate(self.coders):
            oneuser_distribution, assignment = self.justice_selection(self.nsingle)   # will return textids as (newsid-res-cat) # handle selected_texts here
            usertextassignment[i] = assignment
            singleannot_distribution.update_selection(oneuser_distribution)
        
        # record userassign. and distribution
        #self.singles_jsonpath = os.path.join(self.outfolder, "singleannotation_assignments.txt")
        IOtools.todisc_json(self.singles_jsonpath, usertextassignment, ind=5)
        singleannot_distribution.todisc(os.path.join(self.outfolder, "singleannotation_distribution.txt"))
        
        
        
        # BURADA numberofdoubleannotatabletexts sayısında bug var. (self.ncoders/2)*self.noverlaps olmalı.
        #doubleannot_distribution, textassignments = self.justice_selection(self.ncoders * self.noverlaps)
        
        
        # ve daha önemli bir bug ortaya çıktı. her pair için ntotaloverlaps kere text alıyor! noverlaps kee almalıydı.
        pairtextassignment = {}
        doubleannot_distribution = Selection()
        doubleannot_distribution.initialize(self.months, self.resources, self.cats)
        for i in range(0,self.ncoders,2):
            pair_distribution, assignment = self.justice_selection(self.noverlaps)
            pairname = str(i)+"-"+str(i+1)
            pairtextassignment[pairname] = assignment
            doubleannot_distribution.update_selection(pair_distribution)
        
        #self.doubles_jsonpath = os.path.join(self.outfolder, "doubleannotation_assignments.txt")
        IOtools.todisc_json(self.doubles_jsonpath, pairtextassignment, ind=5)
        doubleannot_distribution.todisc(os.path.join(self.outfolder, "doubleannotation_distribution.txt"))
        
        
        
    
    def main(self):
        #print self.justice_selection(10)
        self.get_user_text_distributions()

class Selection():
    
    distmonths = {}
    distresources = {}
    distcats = {}
    
    def __init__(self):
        self.distmonths = {}
        self.distresources = {}
        self.distcats = {}
    
    def initialize(self, months, resources, cats):
        self.distmonths = initialize_dict(months)
        self.distresources = initialize_dict(resources)
        self.distcats = initialize_dict(cats)
    
    def toscreen_dists(self):
        print "resource distribution: ",self.distresources
        print "month distribution: ",self.distmonths
        print "category distribution: ",self.distcats
    
    def todisc(self, path):
        IOtools.todisc_txt(IOtools.dict_tostring(self.distresources), path, mode="w")
        IOtools.todisc_txt(IOtools.dict_tostring(self.distcats), path, mode="a")
        IOtools.todisc_txt(IOtools.dict_tostring(self.distmonths), path, mode="a")
    
    # updater is another selection
    def update_selection(self, updater):  
        # update_dct(updater, updatee)
        update_dct(updater.distmonths, self.distmonths)
        update_dct(updater.distcats, self.distcats)
        update_dct(updater.distresources, self.distresources)




def get_months(mstart=metacorpus.datestart, nmonths=10):
    m1 = parser.parse(mstart)

    months = []
    for _ in range(nmonths):
        months.append(m1)
        m1 = m1 + relativedelta.relativedelta(months=1)
    
    months = [date.isoformat(m)[:-3] for m in months]
    return months
    
    

if __name__ == "__main__":
    
    months = get_months()
    print months
    
    outf = IOtools.ensure_dir("/home/dicle/Dicle/Tez/corpusstats/annotatable/distsfull_10-1-0_50/")
    testbuilder = AnnotationBuilder(ntexts = 10, 
                                    psingle = 0, 
                                    poverlaps = 2, 
                                    nusers = 50, 
                                    months = months, 
                                    resources = ["radikal", "vakit", "solhaber"], 
                                    cats = ["turkey", "world", "economy", "politics"], 
                                    corpuspath="/home/dicle/Dicle/Tez/corpusstats/annotatable/catnormalized_corpusstats.csv",
                                    outfolder=outf
                                    )
    testbuilder.get_user_text_distributions()
    testbuilder.generate_tables()
    testbuilder.toscreen() 
    
    #testbuilder.main()   # get_user_text_distributions()



