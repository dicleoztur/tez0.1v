'''
Created on Mar 20, 2013

@author: dicle
'''

import matplotlib.pyplot as plt
import numpy as np

x = np.array([0,1,2,3])
y = np.array([20,21,22,23])
my_xticks = ['John','Arnold','Mavis','Matt']
plt.xticks(x, my_xticks)
plt.plot(x, y)
plt.show()