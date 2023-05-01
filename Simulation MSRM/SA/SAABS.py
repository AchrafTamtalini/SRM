# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:27:52 2022

@author: Achraf
"""
import numpy as np
import scipy.stats as sts 

class SA(object):
    def __init__(self, X, c, gamma, K, t, init, epsilon):
        #parameter t for the PR
        self.__t = t
        #compact set
        self.__K = K  
        #realizations of the random vector X
        self.__X = X     
        self.__dim = X.shape[1]
        self.__maxIter = X.shape[0]
        #parameter c in step sequence gamma_n = c / n ** gamma
        self.__c = c
        #paramater gamma in step sequence gamma_n = c / n ** gamma
        self.__gamma = gamma
        #initial value for our SA algorithm
        self.__init = init
        #for the estimation of covariance and jacobian matrix
        self.__epsilon = epsilon
        self.__sigmaEst, self.__jacEst = np.zeros((self.dim, self.dim)), np.zeros((self.dim, self.dim))
        #sequence of the algorithm
        self.__z = np.zeros((self.maxIter, self.dim + 1))

        
        
    @property
    def X(self):
        return self.__X
    
    @property
    def c(self):
        return self.__c
    
    @property
    def gamma(self):
        return self.__gamma
    
    @property
    def K(self):
        return self.__K
    
    @property 
    def t(self):
        return self.__t
    
    @property
    def init(self):
        return self.__init
    
    @property
    def epsilon(self):
        return self.__epsilon
    
    @property
    def maxIter(self):
        return self.__maxIter
    
    @property
    def dim(self):
        return self.__dim
    
    @property
    def z(self):
        return self.__z
    
    
    @property
    def jacEst(self):
        return self.__jacEst
    
    @property
    def sigmaEst(self):
        return self.__sigmaEst
    
    @sigmaEst.setter
    def sigmaEst(self, sigmaEst):
        self.__sigmaEst = sigmaEst
        
    @jacEst.setter
    def jacEst(self, jacEst):
        self.__jacEst = jacEst
        
    
    
    
    def check_gamma(self):
        if self.gamma > 1 or self.gamma <= 0.5:
            raise ValueError('The value of gamma is not accepted')
        return self.gamma
    
    def projection(self, m):
        for i in range(self.dim + 1):
            if m[i] < self.K[i][0]:
                m[i] = self.K[i][0]
            if m[i] > self.K[i][1]:
                m[i] = self.K[i][1]
        return m
    
    def setRM(self):
        gamma = self.check_gamma()
        z = np.zeros((self.dim + 1, self.maxIter))
        z[:, 0] = self.init
        for i in range(1, self.maxIter):
            z[:,i] = self.projection(z[:,i - 1] + (self.c / (i ** gamma)) * self.H(self.X[i], z[:,i - 1]))
        self.__z = z
        return z
            
    def setEst(self):
        gamma = self.check_gamma()
        z = self.__z
        sigmaEst = np.zeros((self.dim + 1, self.dim + 1))
        jacEst = np.zeros((self.dim + 1, self.dim + 1))
        I = np.identity(self.dim + 1)
        for i in range(1, self.maxIter):
            sigmaEst += np.outer(self.H(self.X[i], z[:,i - 1]), self.H(self.X[i], z[:,i - 1]))
            for j in range(self.dim + 1):
                jacEst[:,j] += (self.H(self.X[i], z[:,i - 1] + self.epsilon * I[:,j]) - self.H(self.X[i], z[:,i - 1])) / self.epsilon
        self.sigmaEst = sigmaEst / self.maxIter 
        self.jacEst = jacEst / self.maxIter

    
    #Before calling getPR, we should call setEst first
    def getPR(self):
        if self.gamma == 1:
            raise ValueError('Value of gamma is not less than 1')
        else:
            gamma = self.check_gamma()
            initIndex = int(self.maxIter - (self.t / self.c) * (self.maxIter ** gamma))
            if initIndex < 0:
                raise ValueError('The initial index of the averaging window is negative')
            invA = np.linalg.inv(self.jacEst)
            V = np.dot(invA, np.dot(self.sigmaEst, np.transpose(invA)))
            CI = np.zeros((self.dim, 2))
            zBar = np.mean(self.z[:,initIndex:], axis=1)
            for j in range(self.dim):
                lengthCI = np.sqrt(V[j, j] / (self.t * self.c * self.maxIter ** gamma)) * sts.norm.ppf(0.975)
                CI[j,:] = np.array([zBar[j] - lengthCI, zBar[j] + lengthCI])
            return zBar, CI
          
    
    def H(self, x, z):
        m = z[0: self.dim]
        lam = z[-1]
        res = [0 for i in range(self.dim + 1)]
        res[-1] = self.l(x - m)
        res[0:self.dim] = lam * self.grad(x - m) - 1
        return np.array(res)
    
    def H2(self, x, rho, m):
        return rho - (np.sum(m) + self.l(x - m))
    
    def l(self, m):
        raise NotImplementedError()
        
    def grad(self, m):
        raise NotImplementedError()


