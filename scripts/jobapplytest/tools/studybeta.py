import awkward as ak
import numpy as np
from iminuit.cost import ExtendedBinnedNLL, LeastSquares
from iminuit import Minuit
import uproot
from .binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center, get_binindices
from .selections import *
from scipy.optimize import curve_fit
from scipy import interpolate  
from collections import Counter
from .constants import ISOTOPES_CHARGE, ISOTOPES_MASS, DETECTORS, ISOTOPES
from .graphs import MGraph, slice_graph, plot_graph, slice_graph_by_value, concatenate_graphs
from .calculator import calc_beta
import pickle

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

def minuitfit_Chi(xvalues, yvalues, yvalueserr, func, guess):
    loss = LeastSquares(xvalues, yvalues, yvalueserr, func)
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

def calc_signal_fraction(data, binning, sigma):
    signal_lim = [ -3.0 * sigma, 3.0 * sigma]
    index = np.digitize(signal_lim, binning)
    signal_fraction = np.sum(data[index[0]:index[1]])/np.sum(data)
    return signal_fraction

def hist_number(events, energy_estimator, binning):
    if energy_estimator == "tk_rigidity1":
        energy = ak.to_numpy(events["tk_rigidity1"][:, 0, 0, 1])
    else:
        energy = ak.to_numpy(events["richp_beta"])
 #       energy = ak.to_numpy(events["rich_beta2"][:, 0])
    hist, hist_edges = np.histogram(energy, bins= binning)
    return hist

def hist1d(values, binning):
    hist, hist_edges = np.histogram(values, bins= binning)
    return hist

def hist1d_weighted(values, binning, weight):
    hist, hist_edges = np.histogram(values, bins= binning, weights=weight)
    return hist

def hist2d(values_x, values_y,  binning_x, binning_y):
    hist, hist_edges = np.histogramdd((values_x, values_y), bins= (binning_x, binning_y))
    return hist


def getbeta(events, betaIsLIP):
    if(betaIsLIP):
        return events.richp_beta
    else:
        return events.rich_beta2[:, 0] 


def hist_betabias(events, riglim, binning, isLip=1):
    events_highR = cut_high_rig(events, riglim)
    betahighR = ak.to_numpy(getbeta(events_highR, isLip))
    betabias = betahighR - 1.0
    hist_bias, hist_edges = np.histogram(betabias, bins= binning)
    return hist_bias

def hist_beta(events, binning, isLip=1):
    beta = ak.to_numpy(getbeta(events, isLip))
    hist, hist_edges = np.histogram(beta, bins= binning)
    return hist

def compute3d_avg_and_std(hist3d, binning_x, binning_y, binning_z):
    var_xcenter = get_bin_center(binning_x)
    var_ycenter = get_bin_center(binning_y)
    var_zcenter = get_bin_center(binning_z)
    avg = np.zeros((len(var_xcenter), len(var_ycenter)))
    std = np.zeros((len(var_xcenter), len(var_ycenter)))
    for binx in range(0, len(binning_x) - 1):
            for biny in range(0, len(binning_y) - 1):
                ibin_fitdata = hist3d[binx, biny, :]
                if sum(ibin_fitdata) != 0:
                    avg[binx, biny], std[binx, biny] = weighted_avg_and_std(var_zcenter, ibin_fitdata)
    return avg, std


def compute2d_avg_and_std(hist2d, binning_x, binning_y):
    var_xcenter = get_bin_center(binning_x)
    var_ycenter = get_bin_center(binning_y)
    avg = np.zeros(len(var_xcenter))
    std = np.zeros(len(var_xcenter))
    for binx in range(0, len(binning_x) - 1):
         ibin_fitdata = hist2d[binx, :]
         if sum(ibin_fitdata) != 0:
            avg[binx], std[binx] = weighted_avg_and_std(var_ycenter, ibin_fitdata)
    return avg, std

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    
    return average, variance


def compute_moment(hist2d_data, binning_x, binning_y, guess, fit=0):
    mu = np.zeros(len(binning_x)-1)
    sigma = np.zeros(len(binning_x)-1)
    mu_errs = np.zeros(len(binning_x)-1)
    sigma_errs = np.zeros(len(binning_x)-1)

    for binn in range(0, len(binning_x)-1):
        ibin_fitdata = hist2d_data[binn, :]
        var_ycenter = get_bin_center(binning_y)
        var_xcenter = get_bin_center(binning_x)

      #  norm[binn] = popt[0]                                                                                                                                                                               
        if fit:
            popt, pcov = curve_fit(gaussian, var_ycenter, ibin_fitdata, p0 = guess)
            mu[binn] = popt[1]
            mu_errs[binn] = np.sqrt(np.diag(pcov))[1]
            sigma[binn] = popt[2]
            sigma_errs[binn] = np.sqrt(np.diag(pcov))[2]

        else:
            if (np.sum(ibin_fitdata) != 0):
                mu[binn] = np.average(var_ycenter, weights = ibin_fitdata)
                #mu_errs[binn] = np.sqrt(np.diag(pcov))[1]
                mu_errs[binn] = 0
            else:
                mu[binn] = 0
                mu_errs[binn] = 0
            sigma[binn] = 0.0
            sigma_errs[binn] = 0.0

    return mu, mu_errs, sigma, sigma_errs
 


def getbeta_corr(events):
     cmat_index = ak.to_numpy(events["rich_index"])
     cmat_rawindex = ak.to_numpy(events["rich_rawindex"])
     correction_index = cmat_rawindex/cmat_index
     betalip_raw = ak.to_numpy(events["rich_betap"])
     betalip_corr = ak.to_numpy(events["rich_betap"]) * correction_index
     beta_lip_morecorr = 1.0 + 0.0002029502124347422
     betalip_corr_new = betalip_corr/beta_lip_morecorr
     return betalip_corr_new


with np.load("/home/manbing/Documents/lithiumanalysis/scripts/tools/corrections/heiss_mu_biaslip_raw.npz") as mufile: 
    mubiaslip_raw = mufile["mu_all"]     
    var_xcenter= mufile["var_xcenter"]                                              
    var_ycenter= mufile["var_ycenter"]                                            
    func_correction_agl = interpolate.interp2d(var_xcenter, var_ycenter, mubiaslip_raw, kind='cubic')


with np.load("/home/manbing/Documents/lithiumanalysis/scripts/tools/corrections/lipbetabias_NaF.npz") as mufile_naf: 
    mubiaslip_raw_naf = mufile_naf["mu_all"]     
    var_xcenter_naf = mufile_naf["var_xcenter"]                                              
    var_ycenter_naf = mufile_naf["var_ycenter"]                                            
    func_correction_naf = interpolate.interp2d(var_xcenter_naf, var_ycenter_naf, mubiaslip_raw_naf, kind='cubic')

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

def compute_mean(hist3d, binning_x, binning_y, binning_z):
    var_xcenter = get_bin_center(binning_x)
    var_ycenter = get_bin_center(binning_y)
    var_zcenter = get_bin_center(binning_z)
    mu = np.zeros((len(var_xcenter), len(var_ycenter)))
    for binx in range(0, len(binning_x) - 1):
        for biny in range(0, len(binning_y) - 1):
            ibin_fitdata = hist3d[binx, biny, :]
            if sum(ibin_fitdata) == 0:
                mu[binx, biny] = 0.0
            else:
                mu[binx, biny] = np.average(var_zcenter, weights = ibin_fitdata)
    return mu

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


def correct_charge(qraw, betaraw, betacorr, index, factor_pe):
    qcorr = qraw * np.sqrt(((betaraw * index)**2 - 1) *betacorr**2)/np.sqrt(betaraw**2 * (betacorr**2 * index**2 - 1)) * np.sqrt(factor_pe)
    return qcorr  


def row_mean_and_std(bin_centers, weights, axis=1):           
    mean = (np.expand_dims(bin_centers, axis=1-axis) * weights).sum(axis=axis) / weights.sum(axis=axis)    
    std = np.sqrt(((np.expand_dims(bin_centers, axis=1-axis) - np.expand_dims(mean, axis=axis))**2 * weights).sum(axis=axis) / weights.sum(axis=axis))  
    return mean, std, std / np.sqrt(weights.sum(axis=axis))

def compute_rawmeanstd_2dhist(hist2d_data, xbinning, ybinning):
    xcenter = get_bin_center(xbinning)
    mean = np.zeros(len(xbinning)-1)    
    std = np.zeros(len(xbinning)-1)   
    stdw = np.zeros(len(xbinning)-1)    
    for binn in range(0, len(xbinning)-1):  
        ibin_fitdata = hist2d_data[binn, :] 
        var_ycenter = get_bin_center(ybinning)        
        var_xcenter = get_bin_center(xbinning)
        mean[binn], std[binn], stdw[binn] = row_mean_and_std(var_ycenter, ibin_fitdata)
    return mean, std, stdw

def compute_max_2dhist(hist2d_data, xbinning, ybinning):
    xcenter = get_bin_center(xbinning)
    maxvaluey = np.zeros(len(xbinning) -1)
    for binn in range(0, len(xbinning)-1):  
        ibin_fitdata = hist2d_data[binn, :] 
        var_ycenter = get_bin_center(ybinning)        
        var_xcenter = get_bin_center(xbinning)
        max_index = np.argmax(ibin_fitdata)
        maxvaluey[binn] = var_ycenter[max_index] 
    return maxvaluey



############################################################
#correct CIEMAT beta rich_beta_cor with high rigidity events
############################################################
df_highRPars = np.load('/home/manbing/Documents/lithiumanalysis/scripts/tools/corrections/graph_ParsRICH_HighR.npz')
graph_diff_mean = {}
graphfit_ratio_sigma = {}
graph_ratio_sigma = {}
for dec in ['NaF', 'Agl']:
    graph_diff_mean[dec] = MGraph.from_file(df_highRPars, f'graph_diff_mean_{dec}')
    graph_ratio_sigma[dec] = MGraph.from_file(df_highRPars, f'graph_ratio_sigma_{dec}')
    graphfit_ratio_sigma[dec] = MGraph.from_file(df_highRPars, f'graphfit_ratio_sigma_{dec}')

              
def GetMCRICHTunedBeta_WithHighR(events, beta, dec, iso, exterr='0', given_korr_sigma =None):
    ipoint = graph_diff_mean[dec].get_index(ISOTOPES_CHARGE[iso])
    mean_shift = graph_diff_mean[dec].yvalues[ipoint]
    if given_korr_sigma is None:
        if exterr == '0':
            sigma_scale = graphfit_ratio_sigma[dec].yvalues[ipoint]
        elif exterr == 'up':
            sigma_scale = graphfit_ratio_sigma[dec].yvalues[ipoint] + graphfit_ratio_sigma[dec].yerrs[ipoint] + graph_ratio_sigma[dec].yerrs[ipoint]
        else:
            sigma_scale = graphfit_ratio_sigma[dec].yvalues[ipoint] - graphfit_ratio_sigma[dec].yerrs[ipoint] - graph_ratio_sigma[dec].yerrs[ipoint]   
    
    else:
        sigma_scale = given_korr_sigma
    print('mean_shift:', mean_shift, '  sigma_scale:', sigma_scale)    
    true_rigidity_atRICH = ak.to_numpy(events["mevmom1"][:, 15]/ISOTOPES_CHARGE[iso])      
    true_beta_atRICH = calc_beta(true_rigidity_atRICH, ISOTOPES_MASS[iso], ISOTOPES_CHARGE[iso]) 
    beta_tuned = (beta - true_beta_atRICH) * sigma_scale + true_beta_atRICH - mean_shift
    return beta_tuned
############################################################

df_spline_betakor = {}
with open('/home/manbing/Documents/lithiumanalysis/scripts/tools/corrections/Be_spline_tofbeta_sigmakorr_NaF.pickle', 'rb') as f:
    df_spline_betakor['Be'] = pickle.load(f)

def GetMCTofTunedBeta_WithSpline(events, nuclei, iso):
    SplineKorr  = df_spline_betakor[nuclei]
    tof_beta = events.tof_betahmc
    ekin_tof = calc_ekin_from_beta(tof_beta)
    sigma_scale = SplineKorr(np.log(ekin_tof))
    true_rigidity_atTof = ak.to_numpy(events["mevmom1"][:, 12]/ISOTOPES_CHARGE[iso])
    true_beta_atTof = calc_beta(true_rigidity_atTof, ISOTOPES_MASS[iso], ISOTOPES_CHARGE[iso])
    tof_betatuned = (tof_beta - true_beta_atTof) * sigma_scale + true_beta_atTof
    return tof_betatuned
