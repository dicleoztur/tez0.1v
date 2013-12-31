'''
Created on May 5, 2013

@author: dicle
'''



import matplotlib.pyplot as plt
import numpy as np
import colorsys
from mpl_toolkits.mplot3d import Axes3D


def get_n_colors(numofcolors):
    colors=[]
    for i in np.arange(0., 360., 360. / numofcolors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

def set_plotframe(figname, xLabel, yLabel):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5) 
    #fig.set_size_inches(14.5,14.5)      
    plt.autoscale(True, axis='both')
    plt.tight_layout(pad=7.0)
    plt.title(figname)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    return fig
    

def plotline():
    xs = np.arange(100)
    yrange = np.linspace(0, 1.0, 100)
    plotframe(xs, yrange, "random ys")
    
    numofcolors = 5
    linelabels = map(lambda i:str(i), range(numofcolors))
    
    cm = plt.get_cmap('gist_rainbow')     #'gist_rainbow')
    plt.gca().set_color_cycle([cm(float(j)/numofcolors) for j in range(numofcolors)])
    
    for i,lbl in enumerate(linelabels):
        ys = np.random.random_sample(100)
        plt.plot(xs, ys, linestyle='-', marker=2, color=cm(1.*i/numofcolors), label=lbl) 
    plt.legend()


def plot_line(xitems, yitems, linelabel, clr):
    plt.xticks(np.arange(len(xitems)), xitems, rotation=90)
    plt.plot(np.arange(len(xitems)), yitems, color=clr, label=linelabel)



def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin


def scatter3D():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zl, zh)
        cs = randrange(n, 0, 100)
        print xs
        print ys
        print zs
        p = ax.scatter(xs, ys, zs, c=cs, marker=m)
        fig.colorbar(p)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()


def scatter3D2():
    n = 10
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, 5, 10)
    cs = randrange(n, 0, 100)
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = get_n_colors(3)
    
    ax.scatter(xs, ys, zs, c = cs)
    fig.add_axes(ax)
    plt.show()
    

if __name__ == "__main__":
    scatter3D2()
    '''
    plot_line()
    plt.show()
    '''
    
    
    