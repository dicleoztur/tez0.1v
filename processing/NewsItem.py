'''
Created on Jul 23, 2012

@author: dicle
'''
import os
import IO


class NewsItem:
    id = ""
    title = ""
    date = ""
    text = ""
    author = ""
    resource = ""
    link = ""
    category = ""

    def __init__(self, id, newtitle, newdate, newtext, newresource, newauthor= "", link = ""):
        self.id = id
        self.title = newtitle.strip()
        self.date = newdate.strip()
        self.text = newtext.strip()
        self.resource = newresource.strip()
        self.author = newauthor.strip()
        self.link = link
        self.category = ""
        
    
    def setcategory(self, catname):
        self.category = catname
    
    def toString(self):
        out = self.title+"\n"
        out += self.id+"\n"
        out += "on "+self.date+"\n"
        out += "in "+self.resource+"\n"
        out += "by "+self.author+"\n"
        out += self.text[:20]+"\n"
        
        return out
    
    def toString2(self):
        out = "<id> "+ self.id + " </id>\n"
        out += "<link> " + self.link + " </link>\n"
        out += "<resource> " + self.resource + " </resource>\n"
        out += "<date> " + self.date + " </date>\n"
        out += "<author> " + self.author + "</author>\n\n"
        out += "<title>" + self.title + "</title>\n"
        out += "<cat>" + self.category + "</cat>\n"
        out += "\n<ttxtt>\n " + self.text + "\n</ttxtt>\n"
        
        return out
    
    
    def toConsole(self):
        print self.toString()
        # temp
        #print "SENTENCES:"
        #IO.splitToSentences(self.text)
    
    '''    
    def toDisc(self, path):
        sentences = IO.splitToSentences(self.text)
        path = path + os.sep + self.id + ".txt"
        IO.todisc_list(sentences, path)    
    
    '''   
    def toDisc(self, path):
        path = path + os.sep + self.id + ".txt"
        outtxt = self.toString2()
        IO.todisc_txt(outtxt, path)
        
        