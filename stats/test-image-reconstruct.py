'''
Created on Jan 21, 2013

@author: dicle
'''


#resource: http://glowingpython.blogspot.it/2011/07/pca-and-image-compression-with-numpy.html

from numpy import mean,cov,cumsum,dot,linalg,size,flipud
from pylab import *

def princomp(A,numpc=0):
 # computing eigenvalues and eigenvectors of covariance matrix
 M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
 [latent,coeff] = linalg.eig(cov(M))
 p = size(coeff,axis=1)
 idx = argsort(latent) # sorting the eigenvalues
 idx = idx[::-1]       # in ascending order
 # sorting eigenvectors according to the sorted eigenvalues
 coeff = coeff[:,idx]
 latent = latent[idx] # sorting eigenvalues
 if numpc < p or numpc >= 0:
  coeff = coeff[:,range(numpc)] # cutting some PCs
 score = dot(coeff.T,M) # projection of the data in the new space
 return coeff,score,latent
 
 

A = imread('/home/dicle/Pictures/rojbin.jpeg') # load an image
A = mean(A,2) # to get a 2-D array
full_pc = size(A,axis=1) # numbers of all the principal components
i = 1
dist = []


print "Taking PCA"
for numpc in range(0,full_pc+10,10): # 0 10 20 ... full_pc
 coeff, score, latent = princomp(A,numpc)
 Ar = dot(coeff,score).T+mean(A,axis=0) # image reconstruction
 # difference in Frobenius norm
 dist.append(linalg.norm(A-Ar,'fro'))
 # showing the pics reconstructed with less than 50 PCs
 if numpc <= 50:
  ax = subplot(2,3,i,frame_on=False)
  ax.xaxis.set_major_locator(NullLocator()) # remove ticks
  ax.yaxis.set_major_locator(NullLocator())
  i += 1 
  imshow(flipud(Ar))
  title('PCs # '+str(numpc))
  gray()

print "Finished PCA"

figure()
imshow(flipud(A))
title('numpc FULL')
gray()
show()
print flipud(A)
