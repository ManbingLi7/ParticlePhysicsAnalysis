import numpy as np
from iminute import make_func_code
from binnings_collection import kinetic_energy_neculeon_binning

def gaussian(x, mu, sigma):                                                                        
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def ploy_3(x, a, b ,c):
    return a*x**2 + b*x +c

def glob_gaussian(x, y, ua, ub, uc, sa, sb, sc):
    mean = poly3(y, ua, ub, uc)
    sigma = poly3(y, sa, sb, sc)
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mean)**2 /(2* sigma**2))

def glob_GAMG(x_y, ua, ub, uc, sa, sb, sc, ssa, ssb, ssc, fa, fb, fc, ta, tb, tc, counta,countb, countc):
    x_m, y_eng = x_y
    mean = poly3(y_eng, ua, ub, uc)
    sigma_p = poly3(y_eng, sa, sb, sc)
    sigma_factor = poly3(y_eng, ssa, ssb, ssc)
    fraction_sec = poly3(y_eng, fa, fb, fc)
    tau = poly3(y_eng, ta, tb, tc)
    counts = poly3(y_eng, counta, countb, countc)
    x_left = x_m < mean
    x_right = x_m > mean
    X, Y = np.meshgrid(x_m, y_eng)
    z = np.zeros_like(X)
    sigma_sec_left  = sigma_p * sigma_factor
    sigma_sec_right = sigma_p * sigma_factor * tau
    fraction_p = 1.0 - fraction_sec
    z[x_left] = counts * (fraction_p * gaussian(x[x_left], mu, sigma_p) + fraction_sec * guassian(x[x_left], mu, sigma_sec_left))
    z[x_right] = counts * (fraction_p * gaussian(x[x_right], mu, sigma_p) + fraction_sec * guassian(x[x_right], mu, sigma_sec_right))
    return z

def make_limassfun_biGausAsy(binning):
    def limassfunc_biGausAsy(x,  ua, ub, uc, sa, sb, sc, ssa, ssb, ssc, fa, fb, fc, ta, tb, tc, counta,countb, countc, *ratio):
        eny = x[1, :]
        mass = x[0, :]
        mean = poly3(eny, ua, ub, uc)
        sigma_p = poly3(eny, sa, sb, sc)
        sigma_factor = poly3(eny, ssa, ssb, ssc)
        fraction_sec = poly3(eny, fa, fb, fc)
        tau = poly3(eny, ta, tb, tc)
        counts = poly3(eny, counta, countb, countc)
        x_left =  mass < mean
        x_right = mass > mean
        z = np.zeros_like(eny)
        sigma_sec_left  = sigma_p * sigma_factor
        sigma_sec_right = sigma_p * sigma_factor * tau
        fraction_p = 1.0 - fraction_sec
        counts_li6 = counts * (ratio/(1.0+ratio))
        z_li_six = gaus_add_asenygaus(x, mu, sigma_p, sigma_factor, fraction_sec, tau, counts_li6)
        isotope_factor = 6.0/7.0
        index = get_index(binning, eny)
        counts_li7 = counts * isotope_factor/(1.0+ratio[index])
        z_li_seven = gaus_add_asenygaus(x, mu/isotope_factor, sigma_p/isotope_factor, sigma_factor, fraction_sec, tau, counts_li7)
        z = z_li_six + z_li_seven
        return z
    par_name = ["ua", "ub", "uc", "sa", "sb", "sc", "ssa", "ssb", "ssc", "fa", "fb", "fc", "ta", "tb", "tc", "counta", "countb", "countc"]
    ratio_names = [f"ratio_{index}" for index in range(len(binning) - 1)]
    limassfunc_biGausAsy.func_code = make_func_code(par_name+ratio_names)
    return limassfunc_biGausAsy

def aseny_gaussian(x, mu, sigma, tau, counts):
    gaus_left = x < mu
    gaus_right = x >= mu
    eny = gaussian(x, mu, sigma)
    sigma_right = sigma * tau
    eny[gaus_left] = counts * gaussian(x[gaus_left], mu, sigma)
    eny[gaus_right] = counts * gaussian(x[gaus_right], mu, sigma_right)
    return eny

def cumulative_aseny_gaussian(edges, mu, sigma, tau, counts):
    x = (edges[1:] + edges[:-1])/2
    p = aseny_gaussian(x,  mu, sigma, tau, counts)
    cp = np.cumsum(p)
    return np.concatenate(([0],cp))

def gaus_add_asenygaus(x, mu, sigma_p, sigma_factor, fraction_sec, tau, counts):
    #sigma_factor = sigma_sec/sigma_p
    x_left = x < mu
    x_right = x > mu
    eny = np.zeros_like(x) 
    sigma_sec_left  = sigma_p * sigma_factor
    sigma_sec_right = sigma_p * sigma_factor * tau
    fraction_p = 1.0 - fraction_sec
    eny[x_left] = counts * (fraction_p * gaussian(x[x_left], mu, sigma_p) + fraction_sec * guassian(x[x_left], mu, sigma_sec_left))
    eny[x_right] = counts * (fraction_p * gaussian(x[x_right], mu, sigma_p) + fraction_sec * guassian(x[x_right], mu, sigma_sec_right))
    return eny

def cumulative_bigausaseny(edges, sigma_p, sigma_factor, fraction_sec, tau, counts):
    x = (edges[1:] + edges[:-1])/2
    pdf = gaus_plus_asenygaus(x, mu, counts, sigma_p, sigma_factor, fraction_sec, tau)
    cpdf = np.cumsum(pdf)
    return np.contatenate(([0], c))

def limassfunc_biGausAseny(x, mu, sigma_p, sigma_factor, fraction_sec, tau, ratio, counts): 
    counts_li6 = counts * (ratio/(1.0+ratio))
    eny_li_six = gaus_add_asenygaus(x, mu, sigma_p, sigma_factor, fraction_sec, tau, counts_li6)
    isotope_factor = 6.0/7.0
    counts_li7 = counts * isotope_factor/(1.0+ratio)
    eny_li_seven = gaus_add_asenygaus(x, mu/isotope_factor, sigma_p/isotope_factor, sigma_factor, fraction_sec, tau, counts_li7)
    eny = eny_li_six + eny_li_seven
    return eny

def cumulative_limassfunc_biGausAseny(edges, mu, sigma_p, sigma_factor, fraction_sec, tau, ratio, counts):
    x = (edges[1:] + edges[:-1])/2
    cpdf = np.cumsum(massfunc_biGausAseny(x, mu, sigma_p, sigma_factor, fraction_sec, tau, ratio, counts))
    return np.concatenate(([0], cp))

