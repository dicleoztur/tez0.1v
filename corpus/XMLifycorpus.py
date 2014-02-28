'''
Created on Dec 20, 2013

@author: dicle
'''

import os

from sentimentfinding import IOtools
import metacorpus

rawcorpuspath = metacorpus.rawcorpuspath #"/home/dicle/Dicle/Tez/dataset/23Mart-enlarge/newsitems/"
xmlcorpuspath = metacorpus.xmlcorpuspath   #"/home/dicle/Dicle/Tez/dataset/xmldataset/"

resourcefolders = metacorpus.resources

tags = metacorpus.newsitemtags


# XML-related info
headxml = "<?xml version='1.0' encoding='utf-8'?>\n \
<!DOCTYPE note SYSTEM 'Newsitem.dtd'>\n \
<newsitem>"
footxml = "</newsitem>"
fromextension = ".txt"
toextension = ".xml"



''' ve xml buyuk kazigini atar, kendinden vazgecirir. ne guzel olacakti! ampersand'i kabul etmiyor icerikte. delice bir tercih bu, nasıl bir sartlanis! nasil bir siirlenis bu da iste basbaya bir fonksiyon secimi.. 
'''


def crawlandmakexmlcorpus():
    
    for resource in resourcefolders:
        p1 = os.path.join(rawcorpuspath, resource)
        xp1 = IOtools.ensure_dir(os.path.join(xmlcorpuspath, resource))  # replicate the folder hierarchy into the xml folder as well
        categories = IOtools.getfoldernames_of_dir(p1)
        for cat in categories:
            p2 = os.path.join(p1,cat)
            xp2 = IOtools.ensure_dir(os.path.join(xp1, cat))
            txtfiles = IOtools.getfilenames_of_dir(p2, removeextension=True)
            
            for filename in txtfiles:
                txtpath = p2 + os.sep + filename + fromextension
                xmlpath = xp2 + os.sep + filename + toextension
                txtcontent = IOtools.readtxtfile(txtpath)
                xmlcontent = headxml + "\n" + txtcontent + "\n" + footxml
                IOtools.todisc_txt(xmlcontent, xmlpath)



if __name__ == "__main__":
    crawlandmakexmlcorpus()
                
                
    
    
    