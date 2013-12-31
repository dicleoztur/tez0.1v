'''
Created on Mar 8, 2013

@author: dicle
'''
from datetime import datetime


start = datetime.now()
print 'hello'
end = datetime.now()
delta = end-start
print type(delta)

print delta