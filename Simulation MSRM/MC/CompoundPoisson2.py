# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:24:01 2021

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

def m_n(lam, i, j):
    m, n = 0, 0
    lam_i, lam_j = lam[i], lam[j]
    F_0, G_0 = st.poisson.cdf(0, lam_i), st.poisson.cdf(0, lam_j)
    target_m = F_0 + G_0 - 1
    target_n = G_0 + F_0 - 1
    while target_m < 0:
        m += 1
        target_m = st.poisson.cdf(m, lam_i) + G_0 - 1
    while target_n < 0:
        n += 1
        target_n = st.poisson.cdf(n, lam_j) + F_0 - 1
    return m, n
        

def rho_min(lam, i, j):
    lam_i, lam_j = lam[i], lam[j]
    m, n = m_n(lam, i, j)
    rho_min = -lam_i * lam_j
    for k in range(m):
        for l in range(n):
            rho_min -= min(st.poisson.cdf(k, lam_i) + st.poisson.cdf(l, lam_j) - 1, 0)
    rho_min /= math.sqrt(lam_i * lam_j)
    return rho_min

def rho_max(lam, i, j):
    def min_sf(m, n):
        return min(st.poisson.sf(int(m), lam[i]), st.poisson.sf(int(n), lam[j]))
    double_sum = mpmath.nsum(min_sf, [0, math.inf], [0, math.inf])
    return (double_sum - lam[i] * lam[j]) / math.sqrt(lam[i] * lam[j])


#In this class, we give the matrix of the gausian vector as an input 
class CompoundPoisson2(object):
    
    #T final time, n number of t_i's
    #The law of the jump process is assumed to be gaussian
    def __init__(self, T = None, lam = None, rho_gauss = None,  N = 10, mu = None, sigma = None):
        self.__T = T
        self.__N = N
        self.__lam = lam
        self.__dim = np.size(self.__lam) 
        self.__mu = mu
        self.__sigma = sigma
        #this the covariance matrix of the normal distribution
        self.__rho_gauss = rho_gauss
        self.__dim = np.size(self.__lam) 
        self.__rho_max = np.ones((self.__dim, self.__dim))
        self.__rho_min = np.ones((self.__dim, self.__dim))
        for i in range(self.__dim):
            for j in range(i + 1, self.__dim):
                    self.__rho_min[i][j] = rho_min(self.__T * self.__lam, i, j)
                    self.__rho_min[j][i] = self.__rho_min[i][j]
                    self.__rho_max[i][j] = rho_max(self.__T * self.__lam, i, j)
                    self.__rho_max[j][i] = self.__rho_max[i][j]
        self.__rho_pois = np.ones((self.__dim, self.__dim))
        for i in range(self.__dim):
            for j in range(i + 1, self.__dim):
                    self.__rho_pois[i][j] = self.rho_k_l(i, j)
                    self.__rho_pois[j][i] = self.__rho_pois[i][j]
                    if self.__rho_pois[i][j] > self.__rho_max[i][j] or self.__rho_pois[i][j] < self.__rho_min[i][j]:
                        raise AttributeError("The value of rho is not allowed")
    
        
    def get_rho_pois(self):
        if self.__rho_pois is None:
            raise AttributeError("The value of rho is unset")
        return self.__rho_pois

    def get_T(self):
        if self.__T is None:
            raise AttributeError("The value of T is unset")
        return self.__T
     
    def get_lam(self):
        if self.__lam is None:
            raise AttributeError("The value of lambda is unset")
        return self.__lam
    
    def get_mu(self):
        return self.__mu 
    
    def get_sigma(self):
        return self.__sigma
    
        
    def Z_m_n(self, m, n, k, l, rho_k_l):
        lam_k = self.__lam[k] * self.__T
        lam_l = self.__lam[l] * self.__T
        covar = [[1, rho_k_l], [rho_k_l, 1]]
        if rho_k_l > 1:
            print('rho_k_l', rho_k_l)
        A_m = st.norm.ppf(st.poisson.cdf(int(m - 1), lam_k))
        B_m = st.norm.ppf(st.poisson.cdf(int(m), lam_k))
        C_n = st.norm.ppf(st.poisson.cdf(int(n - 1), lam_l))
        D_n = st.norm.ppf(st.poisson.cdf(int(n), lam_l))
        A = st.multivariate_normal.cdf([B_m, D_n],mean = [0, 0], cov = covar)
        B = st.multivariate_normal.cdf([A_m, D_n],mean = [0, 0], cov = covar)
        C = st.multivariate_normal.cdf([B_m, C_n],mean = [0, 0], cov = covar)
        D =  st.multivariate_normal.cdf([A_m, C_n],mean = [0, 0], cov = covar)
        return A - B - C + D
    
    #This function returns the correlation of the multivariate normal distribution   
    def rho_k_l(self, k, l):
        rho_k_l_gauss = self.__rho_gauss[k][l]
        lam_k = self.__lam[k] * self.__T
        lam_l = self.__lam[l] * self.__T
        sum_term = mpmath.nsum(lambda m, n : m * n * self.Z_m_n(m, n, k, l, rho_k_l_gauss), [1, math.inf], [1, math.inf])
        return (sum_term - lam_k * lam_l) / math.sqrt(lam_k * lam_l)
        
        

    def poisson2(self):
        N = self.__N
        d = self.__dim
        cov = self.__rho_gauss
        pois = np.zeros((N, d))
        mean = np.zeros(d)
        X = st.multivariate_normal.rvs(mean, cov, N)
        U = st.norm.cdf(X)
        for j in range(d):
            pois[:,j] = st.poisson.ppf(U[:, j], self.__lam[j] * self.__T)
        return pois
    
    def compound_poisson2_gauss(self):
        pois = self.poisson2()
        #T = self.__T
        mu = self.__mu
        sigma = self.__sigma
        N = self.__N
        d = self.__dim       
        comp_pois = np.zeros((N, d))
        for i in range(N):
            for j in range(d):
                Z = st.norm.rvs(loc = mu[j], scale = sigma[j], size = int(pois[i, j]))
                #U = st.uniform.rvs(0, T, int(pois[i, j]))
                comp_pois[i, j] = sum(Z)
        return comp_pois

    def compound_poisson2_exp(self, lam_exp):
        pois = self.poisson2()
        #T = self.__T
        N = self.__N
        d = self.__dim       
        comp_pois = np.zeros((N, d))
        for i in range(N):
            for j in range(d):
                Z = st.expon.rvs(loc = 0, scale = 1 / lam_exp[j], size = int(pois[i, j]))
                #U = st.uniform.rvs(0, T, int(pois[i, j]))
                comp_pois[i, j] = sum(Z)
        return comp_pois
    
    def cov_final_X_gauss(self):
        d = self.__dim
        lam = self.__lam
        T = self.__T
        mu = self.__mu
        sigma = self.__sigma
        rho = self.__rho_pois
        res = np.ones((d, d))
        for i in range(d):
            for j in range(i, d):
                if j == i:
                    res[i, j] = lam[i] * T * (sigma[i] ** 2 + mu[i] ** 2)
                else:
                    res[i, j] = mu[i] * mu[j] * rho[i, j] * np.sqrt(lam[i] * lam[j]) * T
                    res[j, i] = res[i, j]
        return res
     
    def cov_final_X_exp(self, lam_exp):
        d = self.__dim
        lam = self.__lam
        T = self.__T
        rho = self.__rho_pois
        res = np.ones((d, d))
        for i in range(d):
            for j in range(i, d):
                if j == i:
                    res[i, j] = lam[i] * T * (2 / lam_exp[i] ** 2)
                else:
                    res[i, j] = (1 / lam_exp[i]) * (1 / lam_exp[j]) * rho[i, j] * np.sqrt(lam[i] * lam[j]) * T
                    res[j, i] = res[i, j]
        return res
        
    #Code to have one complete trajectory        
    # def get_trajectory(self):
    #     n = self.__n
    #     T = self.__T
    #     lam = self.__lam
    #     mu = self.__mu
    #     sigma = self.__sigma
    #     d = self.__d
    #     time = np.array([(i * T) / n for i in range(n + 1)])
    #     traj = [0 for i in range(n + 1)]
    #     N = np.random.poisson(lam * T)
    #     jump_times = np.random.uniform(0, T, N)
    #     Z = np.random.normal(mu, sigma, N)
    #     jump_times.sort()
    #     j, s = 0, 0
    #     for i in range(n + 1):
    #         while j < len(jump_times) and jump_times[j] <= time[i]:
    #             s += Z[j]
    #             j += 1
    #         traj[i] = s
    #     return np.array(traj)
    
    # def Z_m(self, m, k, sigma_k):
    #     lam_k = self.__lam[k] * self.__T
    #     A_m = st.norm.ppf(st.poisson.cdf(int(m), lam_k))
    #     B_m = st.norm.ppf(st.poisson.cdf(int(m - 1), lam_k))
    #     if sigma_k > 0:
    #         A = st.norm.cdf(1 / sigma_k * A_m)
    #         B = st.norm.cdf(1 / sigma_k * B_m)
    #         return A - B
    #     raise AttributeError("value of sigma negative or equal to 0")
    
    # def sig(self, k):
    #     lam_k = self.__lam[k] * self.__T
    #     def helper(sig_k):
    #         if isinstance(sig_k, np.ndarray):
    #             sig_k = sig_k[0]
    #             return mpmath.nsum(lambda m: m ** 2 * Z_m(m, k, sig_k), [1, math.inf]) - lam_k ** 2 - lam_k
    #     return scipy.optimize.fsolve(helper, 1)[0]




def Z_m_n(lam, m, n, k, l, rho_k_l):
    lam_k = lam[k]
    lam_l = lam[l] 
    covar = [[1, rho_k_l], [rho_k_l, 1]]
    A_m = st.norm.ppf(st.poisson.cdf(int(m - 1), lam_k))
    B_m = st.norm.ppf(st.poisson.cdf(int(m), lam_k))
    C_n = st.norm.ppf(st.poisson.cdf(int(n - 1), lam_l))
    D_n = st.norm.ppf(st.poisson.cdf(int(n), lam_l))
    A = st.multivariate_normal.cdf([B_m, D_n],mean = [0, 0], cov = covar)
    B = st.multivariate_normal.cdf([A_m, D_n],mean = [0, 0], cov = covar)
    C = st.multivariate_normal.cdf([B_m, C_n],mean = [0, 0], cov = covar)
    D =  st.multivariate_normal.cdf([A_m, C_n],mean = [0, 0], cov = covar)
    return A - B - C + D

#This function returns the correlation of the multivariate normal distribution   
def rho_k_l(lam, k, l, rho):
    lam_k = lam[k]
    lam_l = lam[l] 
    rho_k_l_pois = rho
    def helper(rho_k_l):
        sum_term = mpmath.nsum(lambda m, n : m * n * Z_m_n(lam, m, n, k, l, rho_k_l), [1, math.inf], [1, math.inf])
        return sum_term - lam_k * lam_l - rho_k_l_pois * math.sqrt(lam_k * lam_l)
    return scipy.optimize.fsolve(helper, 0)[0]
def helper1(lam, k, l, rho_k_l_pois, rho_k_l):
    lam_k = lam[k]
    lam_l = lam[l] 
    sum_term = mpmath.nsum(lambda m, n : m * n * Z_m_n(lam, m, n, k, l, rho_k_l), [1, math.inf], [1, math.inf])
    return sum_term - lam_k * lam_l - rho_k_l_pois * math.sqrt(lam_k * lam_l)

# lam, k, l, rho_pois = [10, 10], 0, 1, 0
# x = np.linspace(-0.99, 0.99, 100)
# y = [0 for i in range(100)]
# for i in range(100):
#     y[i] = helper1(lam, k, l, rho_pois, x[i])
# plt.plot(x, y)
# plt.title("f with lam1=lam2=10 and rho_pois = 0")
# plt.xlabel("normal correlation")
# plt.ylabel("f(rho)")

# m = np.arange(0, 100)
# n = np.arange(0, 100)
# v_x, v_y = np.meshgrid(m, n)
# v_z = np.zeros((100, 100))
# v_z[0, 0] = min(st.poisson.sf(0, 1), st.poisson.sf(0, 1))
# for i in range(100):
#     if i > 0:
#         v_z[i, 0] = v_z[i - 1][99]
#     for j in range(1, 100):
#         v_z[i, j] = v_z[i][j - 1] + min(st.poisson.sf(i, 1), st.poisson.sf(j, 1))
            
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_surface(v_x, v_y, v_z, cmap='Blues')
# for i in range(3):
#     for j in range(i + 1, 3):
#         rho[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
#         rho[j, i] = rho[i, j]
