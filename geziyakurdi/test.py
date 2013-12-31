'''
Created on Jul 7, 2013

@author: dicle
'''

import numpy as np
import pandas as pd
from datetime import datetime


def f1():
    p = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus/test/"
    A = np.random.randint(0, 10, size=360000).reshape(600, 600)
    names = [a for i in range(100) for a in 'abcdef']
    df = pd.DataFrame(A, index=names, columns=names)
    df.to_csv(p+'df.csv', index=True, header=True, sep=' ')


def f2():
    p = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus/test/"
    A = np.random.randint(0, 10, size=360000).reshape(600, 600)
    names = [a for i in range(100) for a in 'abcdef']
    np.savetxt(p+"df.txt", A)


def fback():
    p = "/home/dicle/Dicle/Tez/geziyakurdiproject/corpus/test/"
    data = pd.read_csv(p+"df.csv")
    print type(data)
    print data
    

if __name__ == "__main__":
    start = datetime.now()

    f1()
    
    end = datetime.now()
    print str(end-start)
    print
    
    f2()
    
    end2 = datetime.now()
    print str(end2-end)
    
    fback()
    
    
    
    