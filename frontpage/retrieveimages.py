'''
Created on Jun 27, 2013

@author: dicle
'''

from datetime import datetime, date, timedelta
import os
import urllib

from sentimentfinding import IOtools

# http://t24.com.tr/media/papers/zoom/sabah_2013-06-25.jpg




rootdir = "/home/dicle/Dicle/Tez/frontpageproject/"
rootimgdir = "/home/dicle/Dicle/Tez/frontpageproject/pageimages/"



def get_brands(fname):
    f = open(fname, "r")
    lines = f.readlines()
    brands = []
    for line in lines:
        brand = line.split("_")[0]
        brands.append(brand.strip())
    
    brands = list(set(brands))
    IOtools.todisc_list(rootdir+os.sep+"brands.txt", brands)
    return brands


def save_images_onebrand(brand, seedday=26, seedmonth=5, seedyear=2013, intervaldays=32):
    rootlink = "http://t24.com.tr/media/papers/zoom/"
    
    seeddate = datetime(seedyear, seedmonth, seedday)
        
   
    for i in range(intervaldays):
        seeddate = seeddate + timedelta(days=1)
        strdate = seeddate.strftime('%Y-%m-%d')
        imgname = brand + "_" + strdate + ".jpg"
        link = rootlink + "/" + imgname
        dir = IOtools.ensure_dir(rootimgdir + os.sep + brand)
        urllib.urlretrieve(link, dir + os.sep + imgname)
    



def trydates(brand, seedday=26, seedmonth=6, seedyear=2013, intervaldays=1):
    rootlink = "http://t24.com.tr/media/papers/zoom/"
    
    seeddate = datetime(seedyear, seedmonth, seedday)
        
   
    for i in range(intervaldays):
        seeddate = seeddate + timedelta(days=1)
        strdate = seeddate.strftime('%Y-%m-%d')
        imgname = brand + "_" + strdate + ".jpg"
        print strdate, "   ", imgname
        '''
        link = rootlink + "/" + imgname
        dir = IOtools.ensure_dir(rootimgdir + os.sep + brand)
        urllib.urlretrieve(link, dir + os.sep + imgname)
        '''

def save_images(brandnames):
    for brand in brandnames:
        save_images_onebrand(brand)


if __name__ == "__main__":
    brands = IOtools.readtextlines(rootdir + os.sep + "brandsselected.txt")
    save_images(brands)
    #trydates("")



 
    