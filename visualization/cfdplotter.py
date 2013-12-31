'''
Created on Dec 24, 2013

@author: dicle
'''

import os
import numpy as np
import matplotlib.pyplot as plt



# yitemslist is a list of lists where each row contains values of a different bar (rows->bars)
def multiplebargraphs(barlabels, yitemslist, colors, figurelabel, xlabels, ylabel, imgpath=None):
    plt.clf()
    numofbars = len(yitemslist)
    numofxlocations = len(yitemslist[0])
    
    xlocrange = np.arange(numofxlocations)
    barwidth = 0.25
    
    print "numofxlocations: ", numofxlocations
    print "xlocrange: ",xlocrange
    
    fig, ax = plt.subplots()
    
    fig.set_size_inches(18.5,10.5)
    plt.autoscale(True, axis='both')
    plt.tight_layout(pad=7.0)
    
    xlocations = xlocrange
    bars = []
    for yitems, clr, barlabel in zip(yitemslist, colors, barlabels):
        barbar = ax.bar(xlocations, yitems, barwidth, color=clr)
        bars.append(barbar)
        
        # add bar labels
        
        for onebar in barbar:
            xpos = onebar.get_x() + onebar.get_width()/2.
            ypos = 0.9 + onebar.get_height() #1.1 * onebar.get_height()
            ax.text(xpos, ypos, barlabel, rotation=90, va='bottom', ha='center', color='purple')
        
        # update xlocs for the next bar in one xtick
        xlocations = xlocations + barwidth
        
    
    #ax.legend(bars, barlabels)
    
    # add axis labels
    ax.set_title(figurelabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xlocrange+(barwidth*numofbars/2))
    ax.set_xticklabels(xlabels)
    
    if imgpath:
        path = imgpath + os.sep + "-".join(map(lambda x : x.lower().strip(), figurelabel.split())) + ".png"
        plt.savefig(path,dpi=100) 
    #plt.show()
    
    
    



if __name__ == "__main__":
    
    barlabels = ["radikal", "cumhuriyet", "vakit"]
    yitemslist = [[4,5,8,2,6],
                  [5,3,9,3,23],
                  [45,10,8,6,12]]
    colors = ['g', 'r', 'b']
    figurelabel = "newspaper monthly news text counts"
    xlabels = ["Feb,2012",
               "Mar,2012",
               "Apr,2012",
               "May,2012",
               "Jun,2012"]
    ylabel = "count"
    
    multiplebargraphs(barlabels, yitemslist, colors, figurelabel, xlabels, ylabel)
    
    
    
    
    
    
    
    
    
    
    