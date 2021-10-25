# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:48:49 2021

@author: Achraf
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import math
import mpmath
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D

mpmath.mp.pretty = True
mpmath.dps = 20
class CompoundPoisson2(object):
    
    #T final time, n number of t_i's
    #The law of the jump process is assumed to be exponential
    #cov is the correlation matrix used in the simulation of the poisson distribution
    def __init__(self, T = None, lam = None,  N = 10, lam_exp = None, cov = None):
        self.__T = T
        self.__N = N
        self.__lam = lam
        self.__lam_exp = lam_exp
        self.__cov = cov
        self.__dim = np.size(self.__lam) 
        self.__distr_pois = self.poisson2()
        self.__distr_comp_pois = self.compound_poisson2()
        
    def get_compound_distr(self):
        return self.__distr_comp_pois

    @property
    def T(self):
        if self.__T is None:
            raise AttributeError("The value of T is unset")
        return self.__T
    
    @T.setter
    def T(self, T):
        self.__T = T
    
    @property
    def lam(self):
        if self.__lam is None:
            raise AttributeError("The value of lambda is unset")
        return self.__lam
    @lam.setter
    def lam(self, lam):
        self.__lam = lam
     
    @property
    def mu(self):
        return self.__m
    @mu.setter
    def mu(self, mu):
        self.__mu = mu
    
    @property
    def sigma(self):
        return self.__sigma
    @sigma.setter
    def sigma(self, sigma):
        self.__sigma = sigma
    
        
    
    def poisson2(self):
        N = self.__N
        d = self.__dim
        pois = np.zeros((N, d))
        mean = np.zeros(d)
        cov = self.__cov
        X = st.multivariate_normal.rvs(mean, cov, N)
        U = st.norm.cdf(X)
        for j in range(d):
            pois[:,j] = st.poisson.ppf(U[:, j], self.__lam[j] * self.__T)
        return pois
    
    def rho_pois(self):
        sample = self.__distr_pois
        corr_pois = np.corrcoef(sample.transpose())
        return corr_pois
    
    
    def compound_poisson2(self):
        pois = self.__distr_pois
        N = self.__N
        d = self.__dim       
        comp_pois = np.zeros((N, d))
        for i in range(N):
            for j in range(d):
                Z = st.expon.rvs(scale =  1/self.__lam_exp[j], size = int(pois[i, j]))
                #U = st.uniform.rvs(0, T, int(pois[i, j]))
                comp_pois[i, j] = np.sum(Z)
        return comp_pois
    
    def cov_comp_pois(self):
        sample = self.__distr_comp_pois
        return np.cov(sample.transpose())
    
        
    
    
# obj = CompoundPoisson2(1, [2, 2], 1000000, [3, 3],[[1, 0], [0, 1]])
# sample = obj.get_compound_distr()
# sample1 = st.multivariate_normal.rvs([0, 0], [[1, 0], [0, 1]], 1000000)
# expo = np.exp(sample)
# term1 = expo[:,0]
# term2 = expo[:,1]
# cross = np.multiply(expo[:, 0], expo[:, 1])
# res = 0.5 * (term1 + term2 + cross) - 1
# np.std(res)


#compound second loss alpha = 0
# rho = [-0.7991, -0.4507, -9.1572 * 10**(-4), 0.4706, 0.8602]
# res = [[1.00197667, 0.99902395], [0.99868857, 0.99829183],[1.00120656, 0.99996268],[0.99892493, 0.99724686],[1.00203881, 1.00111636]]
# m1 = [res[i][0] for i in range(len(res)) ]
# m2 = [res[i][1] for i in range(len(res))]
# R = [m1[i] + m2[i] for i in range(len(res))]
# plt.plot(rho, R, 'o-', label = 'R(X)')
# plt.plot(rho, m1, 'o-', label = 'm1')
# plt.plot(rho, m2, 'o-', label = 'm2')
# plt.legend()
# plt.xlabel('rho')


#compound second loss alpha = 1
rho = [-0.7991, -0.4507,-0.1830, -9.1572 * 10**(-4), 0.1842, 0.4706, 0.8602]
res = [[0.70873746, 0.70859876],[0.73830375, 0.73766829], [0.76551428, 0.7626554 ], [0.7900076 , 0.78983028], [0.80934992, 0.80998881], [0.85189357, 0.85152918], [0.91084126, 0.91137544]]
m1 = [res[i][0] for i in range(len(res))]
m2 = [res[i][1] for i in range(len(res))]
R = [m1[i] + m2[i] for i in range(len(res))]
plt.plot(rho, R, 'o-', label = 'R(X)')
plt.plot(rho, m1, 'o-', label = 'm1')
plt.plot(rho, m2, 'o-', label = 'm2')
plt.legend()
plt.xlabel('rho')

#gaussien 1st loss function alpha = 0
# rho = [-0.9, -0.5, -0.2, 0, 0.2, 0.5, 0.9]
# res = [[-0.17647944, -0.17816784], [-0.1732183 , -0.17321948], [-0.17380798, -0.17301348], [-0.17513184, -0.17298256], [-0.17305016, -0.17237054], [-0.17409378, -0.17471387], [-0.17311502, -0.17374435]]
# m1 = [res[i][0] for i in range(len(res))]
# m2 = [res[i][1] for i in range(len(res))]
# R = [m1[i] + m2[i] for i in range(len(res))]
# plt.plot(rho, R, 'o-', label = 'R(X)')
# plt.plot(rho, m1, 'o-', label = 'm1')
# plt.plot(rho, m2, 'o-', label = 'm2')
# plt.legend()
# plt.xlabel('rho')

#gaussien 1st loss function alpha = 1
# rho = [-0.9, -0.5, -0.2, 0, 0.2, 0.5, 0.9]
# res = [[-0.17041321, -0.16631524], [-0.13970434, -0.14270277], [-0.12101297, -0.12146844], [-0.09872361, -0.10135718],[-0.08695495, -0.08517875], [-0.05717697, -0.05695781], [-0.01240347, -0.01308478]]
# m1 = [res[i][0] for i in range(len(res)) ]
# m2 = [res[i][1] for i in range(len(res))]
# R = [m1[i] + m2[i] for i in range(len(res))]
# plt.plot(rho, R, 'o-', label = 'R(X)')
# plt.plot(rho, m1, 'o-', label = 'm1')
# plt.plot(rho, m2, 'o-', label = 'm2')
# plt.legend()
# plt.xlabel('rho')

#compound 1st loss alpha = 0
# rho = [-0.7991, -0.4507, -9.1572 * 10**(-4), 0.4706, 0.8602]
# res = [[0.40556635, 0.40238662], [0.40377374, 0.40309884], [0.40349995, 0.40415358], [0.40490401, 0.40417669], [0.40400765, 0.40351005]]
# m1 = [res[i][0] for i in range(len(res))]
# m2 = [res[i][1] for i in range(len(res))]
# R = [m1[i] + m2[i] for i in range(len(res))]
# plt.plot(rho, R, 'o-', label = 'R(X)')
# plt.plot(rho, m1, 'o-', label = 'm1')
# plt.plot(rho, m2, 'o-', label = 'm2')
# plt.legend()
# plt.xlabel('rho')

#compount 1st loss alpha = 1
# rho = [-0.7991, -0.4507, -9.1572 * 10**(-4), 0.4706, 0.8602]
# res = [[0.42208394, 0.42274955], [0.43450523, 0.4354956 ], [0.44945303, 0.45040965], [0.46966394, 0.46825392], [0.48375373, 0.48292808]]
# m1 = [res[i][0] for i in range(len(res)) ]
# m2 = [res[i][1] for i in range(len(res))]
# R = [m1[i] + m2[i] for i in range(len(res))]
# plt.plot(rho, R, 'o-', label = 'R(X)')
# plt.plot(rho, m1, 'o-', label = 'm1')
# plt.plot(rho, m2, 'o-', label = 'm2')
# plt.legend()
# plt.xlabel('rho')



