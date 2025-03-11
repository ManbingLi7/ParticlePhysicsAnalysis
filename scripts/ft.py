#!/usr/bin/env python3

from time import time

import numpy as np
from scipy.integrate import quad, nquad
from scipy.special import erf, log_ndtr

# helper function for erf(z1) - erf(z0) (more precise than intuitive implementation)
def ndtr_erf(z0,z1):
    return 2*(np.exp(log_ndtr(z1*np.sqrt(2)))-np.exp(log_ndtr(z0*np.sqrt(2))))

def gaussian(x, mu, sigma):                                                   
    return 1 / np.sqrt(2* np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))   

#analytical integral of gaus
def gaus_int(x, mu, sigma):
    return 1/2*(1 + erf((x - mu) / (np.sqrt(2)*sigma)))        

# the flux model, can be replaced by anything
def flux(E, phi0=1000, E0=1, gamma=2.7):
    return phi0 * (E / E0 + 1)**(-gamma)


def agauss_int_delta(x0, x1, mean, sigma, asy_factor):                                                                                                                
    norm = 1 / (asy_factor + 1)                                                                                                                                                                                
    left = ndtr_erf((x0 - mean) /(np.sqrt(2)*sigma), (x1 - mean) /(np.sqrt(2)* sigma))                                                                         
    middle = asy_factor * erf((x1 - mean) / (np.sqrt(2)* asy_factor * sigma)) - erf((x0 - mean) / (np.sqrt(2)*sigma))   
    right = asy_factor * ndtr_erf((x0 - mean) / (np.sqrt(2)*asy_factor * sigma), (x1 - mean) / (np.sqrt(2)*asy_factor * sigma))                                           
    return norm * ((x1 <= mean) * left + (x0 >= mean) * right + (x0 < mean) * (x1 > 0) * middle)     

def gaus_int_delta(x0, x1, mean, sigma):                                                                                              
    return 0.5 * ndtr_erf((x0 - mean) / (np.sqrt(2) * sigma), (x1 - mean) / (np.sqrt(2)*sigma))     

# asymmetric gaussian, used as resolution function here
def agauss(x, mu, sigma, alpha):
    norm = 2 / (sigma * np.sqrt(2 * np.pi) * (1 + alpha))
    left = np.exp(-((x-mu)/(np.sqrt(2)*sigma))**2)
    right = np.exp(-((x-mu)/(alpha* np.sqrt(2)* sigma))**2)
    return norm * ((x < mu) * left + (x >= mu) * right)


# analytical integral of asymmetric gaussian, using erf
def agauss_int(x, mu, sigma, alpha):
    norm = 1 / (alpha + 1)
    left = 1 + erf((x - mu) / (np.sqrt(2)*sigma))
    right = 1 + alpha * erf((x - mu) / (alpha * np.sqrt(2) *sigma))
    return norm * ((x < mu) * left + (x >= mu) * right)


def resofunc(x, mu, sigma, alpha, sigma_ratio, fraccore):
    coregaus = gaussian(x, mu, sigma)                                                                               
    asygaus = agauss(x, mu,  sigma_ratio * sigma, alpha) 
    pdf = fraccore * coregaus + (1 - fraccore) * asygaus
    return pdf                                                                                                        

def resofunc_int_delta(x0, x1, mu, sigma, alpha, sigma_ratio, fraccore):
    inter_gaus = gaus_int_delta(x0, x1, mu, sigma)
    inter_asygaus = agauss_int_delta(x0, x1, mu, sigma * sigma_ratio, alpha)   
    sum_int = fraccore * inter_gaus + (1 - fraccore) * inter_asygaus
    return sum_int

# 2d function folding flux with resolution function (agauss)
def folded_flux(Etrue, Erec, phi0, E0, gamma, mu, sigma, alpha):
    return flux(Etrue, phi0=phi0, E0=E0, gamma=gamma) * agauss(Erec - Etrue, mu=mu, sigma=sigma, alpha=alpha)

def folded_flux_resofunc(Etrue, Erec, phi0, E0, gamma, mu, sigma, alpha, sigma_ratio, fraccore):
    return flux(Etrue, phi0=phi0, E0=E0, gamma=gamma) * resofunc(Erec - Etrue, mu=mu, sigma=sigma, alpha=alpha, sigma_ratio=sigma_ratio, fraccore=fraccore)

# 2d integral of folded_flux over Etrue and an Erec-bin (Emin to Emax)
def integrate_2d(Emin, Emax, *args):
    return nquad(folded_flux, ((0, np.inf), (Emin, Emax)), args, opts=dict(limit=200))

def integrate_2d_resofunc(Emin, Emax, *args):
    return nquad(folded_flux_resofunc, ((0, np.inf), (Emin, Emax)), args, opts=dict(limit=200))

# 1d function of folded flux, integrated over a Erec-bin, using agauss_int_delta
def folded_int_flux(Etrue, Erec_min, Erec_max, phi0, E0, gamma, mu, sigma, alpha):
    #return flux(Etrue, phi0=phi0, E0=E0, gamma=gamma) * (agauss_int(Erec_max - Etrue, mu, sigma, alpha) - agauss_int(Erec_min - Etrue, mu, sigma, alpha))
    return flux(Etrue, phi0=phi0, E0=E0, gamma=gamma) * agauss_int_delta(Erec_min - Etrue, Erec_max - Etrue, mu, sigma, alpha)

def folded_int_flux_resofunc(Etrue, Erec_min, Erec_max, phi0, E0, gamma, mu, sigma, alpha, sigma_ratio, fraccore):
    return flux(Etrue, phi0=phi0, E0=E0, gamma=gamma) * resofunc_int_delta(Erec_min - Etrue, Erec_max - Etrue, mu, sigma, alpha, sigma_ratio, fraccore)

# 2d integral over folded flux, using the analytical integral in folded_int_flux
def integrate_1d(Emin, Emax, phi0, E0, gamma, mu, sigma, alpha, min_value=1e-5):
    # try different integral limits. With too wide limits, the integral might just return 0.
    for limits in ((0, np.inf), (Emin / 100, Emax * 100), (Emin / 10, Emax * 10), (Emin / 3, Emax * 3)):
        value, error = quad(folded_int_flux, limits[0], limits[1], (Emin, Emax, phi0, E0, gamma, mu, sigma, alpha), limit=200)
        if value > min_value:
            return value
    return value


def integrate_1d_resofunc(Emin, Emax, phi0, E0, gamma, mu, sigma, alpha, sigma_ratio, fraccore, min_value=1e-5):
    # try different integral limits. With too wide limits, the integral might just return 0.
    for limits in ((0, np.inf), (Emin / 100, Emax * 100), (Emin / 10, Emax * 10), (Emin / 3, Emax * 3)):
        value, error = quad(folded_int_flux_resofunc, limits[0], limits[1], (Emin, Emax, phi0, E0, gamma, mu, sigma, alpha, sigma_ratio, fraccore), limit=200)
        if value > min_value:
            return value
    return value

# sampled 2d integral over folded flux, using the analytical integral in folded_int_flux
def integrate_1d_num(Emin, Emax, phi0, E0, gamma, mu, sigma, alpha, sigma_ratio, fraccore, xmin=1e-2, xmax=1e4, npoints=50000):
    x = np.logspace(np.log10(xmin), np.log10(xmax), npoints, endpoint=False)
    w = x / x.sum() * (xmax - xmin)
    values = folded_int_flux_resofunc(x, Emin, Emax, phi0, E0, gamma, mu, sigma, alpha, sigma_ratio, fraccore)
    return (values * w).sum()


phi0, E0, gamma = 1e4, 1, 2.7
mu, sigma, alpha, sigma_ratio, fraccore = 0, 1, 1.5, 1.8, 0.9
Ebins = np.logspace(0, 2, 100)
t2d = 0
t1d = 0
t1dn = 0
for Emin, Emax in zip(Ebins[:-1], Ebins[1:]):
    t2d -= time()
    int2d, int2d_err = integrate_2d_resofunc(Emin, Emax, phi0, E0, gamma, mu, sigma, alpha, sigma_ratio, fraccore)
    t2d += time()
    t1d -= time()
    int1d = integrate_1d_resofunc(Emin, Emax, phi0, E0, gamma, mu, sigma, alpha, sigma_ratio, fraccore)
    t1d += time()
    t1dn -= time()
    int1dn = integrate_1d_num(Emin, Emax, phi0, E0, gamma, mu, sigma, alpha, sigma_ratio, fraccore)
    t1dn += time()
    print(f"{Emin:>5.1f}-{Emax:>5.1f}: {int2d:>7.4g} = {int1d:>7.4g} = {int1dn:>7.4g}")
print(f"T2d = {t2d:>5.2f}, T1d = {t1d:>5.2f}, T1dn = {t1dn:>5.2f}")
