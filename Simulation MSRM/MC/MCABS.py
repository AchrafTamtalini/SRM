# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:48:05 2023

@author: Achraf
"""

        

import numpy as np

#Our base abstract class
class MCLossAbs(object):
    def __init__(self, X):
        self.__X = X
        self.__dim = X.shape[1]
        
    
    @property
    def X(self):
        return self.__X
    
    @property
    def dim(self):
        return self.__dim 
    
        
    def _check_argument(self, m):
        if m is None:
            m = np.zeros(self.__dim)
        else:
            if m.shape != (self.__dim,):
                raise ValueError("""m must be of shape (%i). Given: %s.""" % (self.__dim, m.shape))
        return m
    
    def objective(self, m):
        return np.sum(m)
    
    def objective_jac(self, m):
        return np.ones(self.__dim)
    
    def ineq_constraint(self, m):
        return - self.shortfall_risk(m)
    
    def ineq_constraint_jac(self, m):
        return self.shortfall_risk_jac(m)
    
    def shortfall_risk(self, m=None):
        raise NotImplementedError()
    
    def shortfall_risk_jac(self, m):
        raise NotImplementedError()