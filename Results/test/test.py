# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:22:36 2021

@author: Achraf
"""


from fourierABS import AnalyticalLossFunctionAbs
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad, dblquad, IntegrationWarning
import warnings

warnings.filterwarnings('error')

def fourier_integral1d(cplx_integrand, j, x, eta):
    def real_integrand(u):
        return np.real(cplx_integrand(u, j, x, eta))

    real_quad = quad(real_integrand, -np.inf, np.inf, epsrel=1e-4)[0]
    return real_quad

def fourier_integral2d(cplx_integrand, j, k, x, y, eta):
    def real_integrand(u, v):
        return np.real(cplx_integrand(u, v, j, k, x, y, eta))

    real_quad = dblquad(real_integrand, 
                        -np.inf, np.inf,
                        lambda xx: -np.inf, lambda xx: np.inf, 
                        epsrel=1e-4)[0]
    print(real_quad)
    return real_quad

class FourierGaussLossFunction(AnalyticalLossFunctionAbs):
    def __init__(self, mu, sigma, alpha, c=None):
        self.__mu = np.array(mu)
        self.__sigma = np.array(sigma).reshape((len(mu), len(mu)))
        self.__alpha = alpha
        super(FourierGaussLossFunction, self).__init__(len(mu), c)

    def moment_generating(self, t, i):
        mu = self.__mu[i]
        sigma2 = self.__sigma[i, i]
        log_part = mu * t + 0.5 * sigma2 * t**2
        return np.exp(log_part)
    
    def moment_generating2D(self, vec_t, i, j):
        mu = self.__mu[[i, j]]
        sigma2 = [[self.__sigma[i, i], self.__sigma[i, j]],
                  [self.__sigma[j, i], self.__sigma[j, j]]]
        log_part = np.dot(mu, vec_t)
        log_part += 0.5 * np.dot(vec_t, np.dot(sigma2, vec_t))
        return np.exp(log_part)
    
    def e(self, m):
        return (self.__mu - m).sum()
    
    #this corresponds to the integrand when transforming the expectation corresponding to (x^+)**2
    def g_fourier_integrand(self, u, j, m_j, eta):
        i = 1j
        eta_m_iu = eta - i * u
        res = np.exp(-eta_m_iu * m_j)
        res *= self.moment_generating(eta_m_iu, j)
        res /= i * (u + i * eta) ** 3
        return res
    

    def g(self, i, m_i):
        continue_bool = True
        eta = 1.5 * np.random.rand()
        while continue_bool:
            try:
                integral = fourier_integral1d(self.g_fourier_integrand, i, m_i, eta)
                continue_bool = False
                return (1 / np.pi) * integral
            except IntegrationWarning:
                #print "g not converging for x = %s, eta = %s" % (m_i, eta)
                eta = 1.5 * np.random.rand()
    
    #This corresponds to the integrand of the cross terms
    def h_fourier_integrand(self, u, v, j, k, x, y, eta):
        i = 1j
        eta_m_iu = eta - i * np.array([u, v])
        res = np.exp(np.dot(-eta_m_iu, [x, y]))
        res *= self.moment_generating2D(eta_m_iu, j, k)
        res /= (u + i*eta[0])**2 * (v + i*eta[1])**2
        return res 
    
    def h(self, i, j, m_i, m_j):
        continue_bool = True
        eta = 1.5 * np.random.rand(2)
        while continue_bool:
            try:            
                integral = fourier_integral2d(self.h_fourier_integrand, i, j, m_i, m_j, eta)
                continue_bool = False
                return (1 / (2 * np.pi) ** 2) * integral
            except IntegrationWarning:
                eta = 1.5 * np.random.rand(2)
    def jac_g_fourier_integrand(self, u, j, m_j, eta):
        i = 1j
        eta_m_iu = eta - i * u
        res = np.exp(-eta_m_iu * m_j)
        res *= self.moment_generating(eta_m_iu, j)
        res /= (u + i * eta)**2
        return res
    def jac_g(self, i, m_i):
        continue_bool = True
        eta = 1.5 * np.random.rand()
        while continue_bool:
            try:
                integral = fourier_integral1d(self.jac_g_fourier_integrand, i, m_i, eta)
                continue_bool = False
                return (-1 / 2 * np.pi) * integral
            except IntegrationWarning:
                #print "f not converging for x = %s, alpha = %s" % (m_i, eta)
                eta = 1.5 * np.random.rand()
    def jac_h_fourier_integrand(self, u, v, j, k, x, y, eta):
        i = 1j
        eta_m_iu = eta - i * np.array([u, v])
        res = np.exp(np.dot(-eta_m_iu, [x, y]))
        res *= self.moment_generating2D(eta_m_iu, j, k)
        res /= (u + i * eta[0]) ** 2 * (- eta_m_iu[1])
        return res
        
    def jac_h(self, i, j, m_i, m_j):
        continue_bool = True
        eta = 1.5 * np.random.rand(2)
        while continue_bool:
            try:            
                integral = fourier_integral2d(self.jac_h_fourier_integrand, i, j, m_i, m_j, eta)
                continue_bool = False
                return (1 / (2 * np.pi) ** 2) * integral
            except IntegrationWarning:
                eta = 1.5 * np.random.rand(2)
                
    def shortfall_risk(self, m=None):
        m = self._check_argument(m)
        sum_e = self.e(m)
        sum_g, sum_h = 0., 0.
        for i, m_i in enumerate(m):
            sum_g += self.g(i, m_i)
            if self.__alpha != 0:
                for j, m_j in enumerate(m):
                    if j > i:
                        sum_h += self.h(i, j, m_i, m_j)  
        return sum_e + 0.5 * sum_g + self.__alpha * sum_h
    
    def shortfall_risk_jac(self, m):
        m = self._check_argument(m)
        res = []        
        for i, m_i in enumerate(m):
            partial_der = 1 + self.jac_g(i, m_i)
            if self.__alpha != 0:
                for j, m_j in enumerate(m):
                    if i != j:
                        partial_der += self.__alpha * self.jac_h(j, i, m_j, m_i)
            res.append(partial_der)
        return np.array(res)





rho = -0.9
mu = [0., 0.]
sigma = [[1., rho], [rho, 1.]]
c = 1.
alpha  = 1.
loss = FourierGaussLossFunction(mu, sigma, alpha, c)
maxiter = 3500
init = np.ones(loss.dim)

cons = ({'type': 'ineq',
         'fun' : lambda x: loss.ineq_constraint(x),
         'jac' : lambda x: loss.ineq_constraint_jac(x)})
res = minimize(loss.objective, init, 
              jac = loss.objective_jac, 
              constraints = cons, 
               method='SLSQP',
               options={'maxiter': maxiter, 'disp': True})