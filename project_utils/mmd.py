import numpy as np
from numpy import sqrt

def grbf(x1, x2, sigma):
    '''
    TAKEN FROM https://github.com/yy1lab/Lyrics-Conditioned-Neural-Melody-Generation/blob/a966efb468673bd251f94d5715f13d0e1d0b02d5/mmd.py
    Calculates the Gaussian radial base function kernel
    '''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape
    
    k1 = np.sum((x1*x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1
    
    k2 = np.sum((x2*x2), 1)
    r = np.tile(k2.T, (n, 1))
    del k2
    
    h = q + r
    del q,r
    
    # The norm
    h = h - 2*np.dot(x1,x2.transpose())
    h = np.array(h, dtype=float)
    
    return np.exp(-1.*h/(2.*pow(sigma,2)))

def Compute_MMD(X,Y):
    '''
    TAKEN FROM https://github.com/yy1lab/Lyrics-Conditioned-Neural-Melody-Generation/blob/a966efb468673bd251f94d5715f13d0e1d0b02d5/mmd.py
    Compute MMD estimate
    '''
    siz = np.min((1000, X.shape[0]))
    sigma = kernelwidthPair(X[0:siz], Y[0:siz])
    Kyy = grbf(Y,Y,sigma)
    Kxy = grbf(X,Y,sigma)
    Kyynd = Kyy-np.diag(np.diagonal(Kyy))
    m = Kxy.shape[0]
    n = Kyy.shape[0]
    u_yy=np.sum(Kyynd)*( 1./(n*(n-1)) )
    u_xy=np.sum(Kxy)/(m*n)
    Kxx = grbf(X, X, sigma)
    Kxxnd = Kxx - np.diag(np.diagonal(Kxx))
    u_xx = np.sum(Kxxnd) * (1. / (m * (m - 1)))
    MMDXY = u_xx + u_yy - 2. * u_xy

    return MMDXY

def kernelwidthPair(x1, x2):
    '''
    TAKEN FROM https://github.com/yy1lab/Lyrics-Conditioned-Neural-Melody-Generation/blob/a966efb468673bd251f94d5715f13d0e1d0b02d5/mmd.py    
    Implementation of the median heuristic. See Gretton 2012
       Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
       in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
       and y of all distances between points from both data sets X and Y.
    '''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape
    
    k1 = np.sum((x1*x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1
    
    k2 = np.sum((x2*x2), 1)
    r = np.tile(k2, (n, 1))
    del k2
    
    h= q + r
    del q,r
    
    # The norm
    h = h - 2*np.dot(x1,x2.transpose())
    h = np.array(h, dtype=float)
    
    mdist = np.median([i for i in h.flat if i])
    
    sigma = sqrt(mdist/2.0)
    if not sigma: sigma = 1
    
    return sigma

