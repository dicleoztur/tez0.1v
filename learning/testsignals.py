'''
Created on May 16, 2014

@author: dicle
'''


import time
import signal



def handler(signum, frame):
    print "not forever"
    print signum
    print frame
    #signal.alarm(0)
    raise Exception("done")
    


def loop_forever():
    while True:
        print "sec"
        time.sleep(1)


def caller():
    try:
        loop_forever()
    except Exception, exc:
        print exc," quit"
    print "in caller"

if __name__ == "__main__":
  
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(5)
    
    
    for i in range(4):
        caller()
        
    
    
      
        
        