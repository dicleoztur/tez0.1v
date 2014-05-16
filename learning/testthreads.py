'''
Created on May 16, 2014

@author: dicle
'''
import multiprocessing
import time

# bar
def bar(x):
    for i in range(x):
        print "Tick"
        time.sleep(1)


def threader(a):
        # Start bar as a process
    #a = 100
    p = multiprocessing.Process(target=bar,  kwargs={"x":a})
    p.start()

    # Wait for 10 seconds or until process finishes
    p.join(5)

    # If thread is still active
    if p.is_alive():
        print "running... let's kill it..."

        # Terminate
        p.terminate()
        p.join()
        
        
if __name__ == '__main__':

    for _ in range(4):
        threader(100)
        
        