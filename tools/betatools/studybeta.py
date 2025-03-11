import awkward as ak
import numpy as np
from iminuit.cost import ExtendedBinnedNLL, LeastSquares
from iminuit import Minuit
import uproot
from scipy.optimize import curve_fit
from scipy import interpolate  
from collections import Counter

def gaussian(x, counts, mu, sigma):
    return counts * np.exp(-(x - mu)**2 /(2 * sigma**2))

def cdf_gaussian(edges, counts, mu, sigma):
    x = get_bin_center(edges)
    pdf = gaussian(x, counts, mu, sigma)
    cdf = np.cumsum(pdf)
    return np.concatenate(([0], cdf))

def minuitfit_LL(data, binning, cdffunc, guess):
    loss = ExtendedBinnedNLL(data, binning, cdffunc)
    minuit = Minuit(loss, **guess)
    minuit.errordef = 1
    minuit.migrad()
    minuit.hesse()
    par = minuit.values
    parerr = minuit.errors
    return par, parerr

def gaus(x, mu, sigma):                                                                                                                                                                                
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2)) 

def double_gaus(x, counts, mu, sigma, sigma_ratio, fraction_sec):                                                                                                                                     
    return counts * ((1 - fraction_sec) * gaus(x, mu, sigma) + fraction_sec * gaus(x, mu, sigma_ratio * sigma))

def cdf_double_gaus(edges, counts, mu,  sigma,  sigma_ratio, fraction_sec):
    x = get_bin_center(edges)
    pdf = double_gaus(x, counts, mu,  sigma,  sigma_ratio, fraction_sec)
    cdf = np.cumsum(pdf)
    return np.concatenate(([0], cdf))

def getbeta(events, betaIsLIP):
    if(betaIsLIP):
        return events.richp_beta
    else:
        return events.rich_beta2[:, 0] 

with np.load("lipbetabias_Agl.npz") as bias_file_agl: 
    biaslip_agl = bias_file_agl["mu_all"]     
    var_xcenter_agl= bias_file_agl["var_xcenter"]                                              
    var_ycenter_agl= bias_file_agl["var_ycenter"]                                            
    func_correction_agl = interpolate.interp2d(var_xcenter_agl, var_ycenter_agl, biaslip_agl, kind='cubic')

with np.load("lipbetabias_NaF.npz") as bias_file_naf: 
    biaslip_naf = bias_file_naf["mu_all"]     
    var_xcenter_naf = bias_file_naf["var_xcenter"]                                              
    var_ycenter_naf = bias_file_naf["var_ycenter"]                                            
    func_correction_naf = interpolate.interp2d(var_xcenter_naf, var_ycenter_naf, biaslip_naf, kind='cubic')

def get_index_correction_naf(y_input, x_input):
    return func_correction_naf(y_input, x_input)

def get_index_correction_agl(y_input, x_input):
    return func_correction_agl(y_input, x_input) 

def get_corrected_lipbeta_naf(beta, richx, richy):
    lip_correction = np.zeros(beta.shape)
    for i, x in enumerate(lip_correction):               
        lip_correction[i] = 1.0 + get_index_correction_naf(richy[i], richx[i])
    betalip_corrected = beta/lip_correction
    return betalip_corrected                       

def get_corrected_lipbeta_agl(beta, richx, richy):
    lip_correction = np.zeros(beta.shape)
    for i, x in enumerate(lip_correction):               
        lip_correction[i] = 1.0 + get_index_correction_agl(richy[i], richx[i])
    betalip_corrected = beta/lip_correction
    return betalip_corrected                       

def get_refhits_fraction(events):                                                                                                                                                                         
    hitsStatus = ak.to_numpy(events["richp_hitsStatus"])                                                                                                                                                 
    ndirhits = np.zeros(hitsStatus.shape[0])                                                                                                                                                              
    nrefhits = np.zeros(hitsStatus.shape[0])                                                                                                                                                              
    nusedhits = np.zeros(hitsStatus.shape[0])                                                                                                                                                              
    for i, elem in enumerate(hitsStatus):                                                                                                                                                                  
        hits_counter = Counter(elem)                                                                                                                                                                       
        nrefhits[i] = hits_counter[1] + hits_counter[2]                                                                                                                                                    
        nusedhits[i] = np.size(elem[np.where(elem >= 0)])                                                                                                                                                  
    nref_div_used = nrefhits/nusedhits                                                                                                                                                                     
    return nref_div_used       

def get_usedhits_fraction(events):                                                                                                                                                                         
    hitsStatus = ak.to_numpy(events["richp_hitsStatus"])                                                                                                                                                 
    nusedhits = np.zeros(hitsStatus.shape[0])                                                                                                                                                              
    for i, elem in enumerate(hitsStatus):                                                                                                                                                                  
        hits_counter = Counter(elem)                                                                                                                                                                       
        nusedhits[i] = np.size(elem[np.where(elem >= 0)])                                                                                                                                                  
    return nusedhits     

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    
    return average, variance



