import numpy as np


def gaussian(x, mu, sigma):
    return np.exp(-(x - mu)**2 /(2 * sigma**2))

def exponential_distribution(x, tau):
    return np.exp(-x / tau) / tau

def gaus_exponential_distribution(x, mu, sigma, ncut, tau):
    return gaussian(x, mu, sigma) + (x - (mu +ncut * sigma)) * np.exp(-(x - mu) / tau) / tau

def exp_tailed_gaussian(x, mu, sigma, ncut, tau, n_r):
    factor = tau / np.sqrt(2 * np.pi * sigma**2) * np.exp(ncut * sigma / tau - ncut**2 / 2)
    y = gaussian(x, mu, sigma)
    tail_high = x > mu + ncut * sigma
    tail_low = x < mu - ncut * sigma
    y[tail_high] =  factor * gaus_exponential_distribution(x[tail_high], mu, sigma, ncut, 1 / tau)
#    y[tail_low] = factor * exponential_distribution(mu - x[tail_low], 1 / tau)                                                                              
    return n_r * y

def asy_gaussian(x, mu, sigma, tau, counts):
    gaus_left = x < mu
    gaus_right = x >= mu
    y = gaussian(x, mu, sigma)
    sigma_right = sigma * tau
    y[gaus_left] = counts * gaussian(x[gaus_left], mu, sigma)
    y[gaus_right] = counts * gaussian(x[gaus_right], mu, sigma_right)
    return y

def cumulative_asy_gaussian(edges, mu, sigma, tau, counts):
    x = (edges[1:] + edges[:-1])/2
    p = asy_gaussian(x,  mu, sigma, tau, counts)
    cp = np.cumsum(p)
    return np.concatenate(([0],cp))

def mass_asy_gaussian(x, mu, sigma, tau, counts,  ratio):
    counts_li6 = counts * (ratio/(1.0+ratio))
    y_li_six = asy_gaussian(x, mu, sigma, tau, counts)
    isotope_factor = 6.0/7.0
    counts_li7 = counts * isotope_factor/(1.0+ratio)
    y_li_seven = asy_gaussian(x, mu/isotope_factor, sigma/isotope_factor, tau, counts_li7)
    y = y_li_six + y_li_seven
    return y

def cumulative_mass_asy_gaus(edges, mu, sigma, tau, counts, ratio):
     x = (edges[1:] + edges[:-1])/2
     p = mass_asy_gaussian(x,  mu, sigma, tau, counts, ratio)
     cp = np.cumsum(p)
     return np.concatenate(([0],cp))

def cumulative_mass(edges, mu, sigma, ncut, tau, counts, ratio):
    x = (edges[1:] + edges[:-1])/2
    p = mass_pdf_exp_tailed_gaussian(x, mu, sigma, ncut, tau, counts, ratio)
    cp = np.cumsum(p)
    return np.concatenate(([0],cp))

def cumulative_exp_tailed_gaussian(edges, mu, sigma, ncut, tau, n_r):
    x = (edges[1:] + edges[:-1])/2
    p = exp_tailed_gaussian(x, mu, sigma, ncut, tau, n_r)
    cp = np.cumsum(p)
    return np.concatenate(([0],cp))
