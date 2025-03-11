from tools.massfunction_TofGBLV3 import TofInverseMassFunctionFit
from tools.massfunction_NaFGBLV2 import NaFInverseMassFunctionFit
from tools.massfunction_AglGBLV2 import AglInverseMassFunctionFit

import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import tools.roottree as read_tree
from tools.calculator import calc_mass, calc_ekin_from_beta, calc_betafrommomentom, calc_ratio_err

import uproot
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, LeastSquares, NormalConstraint
from iminuit.util import describe, make_func_code
from tools.constants import ISOTOPES, NUCLEI_NUMBER, ISOTOPES_COLOR, ISO_LABELS, ANALYSIS_RANGE_EKIN
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d

from scipy import interpolate
from tools.graphs import MGraph, plot_graph
import uncertainties
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle
from tools.plottools import FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, plot1d_step, FONTSIZE, set_plot_defaultstyle, FONTSIZE_BIG
import uncertainties    
from uncertainties import unumpy 
from tools.constants import DETECTOR_LABEL, DETECTOR_COLOR, ISOTOPES_MASS
from scipy.optimize import curve_fit
from tools.functions import gaussian, asy_gaussian, poly, asy_gaussian_1d, upoly
from tools.constants import expofunc_rig_sigmafactor_7to9, expofunc_rig_sigmafactor_7to10
from tools.massfit_tools import get_fitpdf_witherr, get_fitpdferrorband, get_unp_pars_binbybin, plot_fitmc_pars, plot_fitdata_pars, plot_fitmc_compare_isopars
import random
from typing import Annotated
import pickle

def funcbe10(total_counts, be7, be9):
    ybe10 = total_counts - be7 - be9
    return ybe10


detectors = ["Tof", "NaF", "Agl"]
iso_ratio_guess = {"Be7": 0.6, "Be9": 0.3, "Be10": 0.1}
par_names = {"mean": 0.151, "sigma": 0.016, "fraccore":0.9, "sigma_ratio":1.27, "asy_factor":1.1}

scale_factor = {"Be7": 1, "Be9": 7./9., "Be10": 7.0/10.0}
mean_scale_factor_fromMC = {"Be7": 1, "Be9": 0.77949789, "Be10": 0.70255866}

mcpars_initlims = {'Tof': {'mean': (0.149, 0.158), 'sigma': (0.008, 0.03), 'fraccore': (0.7, 0.98), 'sigma_ratio':(1.2, 1.6), 'asy_factor':(1.1, 1.6)},
                   'NaF': {'mean': (0.13, 0.16), 'sigma': (0.008, 0.03), 'fraccore': (0.89, 0.98), 'sigma_ratio':(1.7, 1.8), 'asy_factor':(1.1, 1.25)},
                   'Agl': {'mean': (0.13, 0.16), 'sigma': (0.008, 0.03), 'fraccore': (0.8, 0.98), 'sigma_ratio':(1.4, 1.8), 'asy_factor':(0.9, 1.2)}}

Nuclei = 'Be'
muscale_factor_name = {'Tof': {iso: [f'muscale_{iso}'] for iso in ISOTOPES[Nuclei][1:]},
                       'NaF': {iso: [f'muscale_{iso}_{a}' for a in ['a', 'b', 'c']] for iso in ISOTOPES[Nuclei][1:]},
                       'Agl': {iso: [f'muscale_{iso}_{a}' for a in ['a', 'b', 'c']] for iso in ISOTOPES[Nuclei][1:]}}

sigscale_factor_name = {'Tof': {iso: [f'sigscale_{iso}_{a}' for a in ['a', 'b', 'c']] for iso in ISOTOPES[Nuclei][1:]},
                        'NaF': {iso: [f'sigscale_{iso}_{a}' for a in ['a', 'b', 'c']] for iso in ISOTOPES[Nuclei][1:]},
                        'Agl': {iso: [f'sigscale_{iso}_{a}' for a in ['a', 'b', 'c']] for iso in ISOTOPES[Nuclei][1:]}}


#'NaF': {'mean': (0.13, 0.16), 'sigma': (0.008, 0.03), 'fraccore': (0.89, 0.91), 'sigma_ratio':(1.7, 1.8), 'asy_factor':(1.2, 1.25)},
#sigma_scale_factor_fromMC = {"Be7": 1, "Be9": 7./9., "Be10": 7./10.}

par_names_axes = {'mean': '$\mathrm{\mu}$',
                  'sigma': '$\mathrm{\sigma_{p}}$',
                  "sigma_ratio": '$\mathrm{ \epsilon(\sigma ratio)}$',
                  "asy_factor":'alpha',
                  "fraccore":'$\mathrm{f_{core}}$',
                  'norm':'Norm'}

poly_deg = {'mean':3, 'sigma':3, 'sigma_ratio':1, 'asy_factor':2, 'fraccore': 1, 'norm':6}

ylim_range = {'mean':        [0.08, 0.16],
              'sigma':       [0.008, 0.024],
              'sigma_ratio': [0.7, 2.5],
              'asy_factor' : [0.8, 2.0],
              'fraccore'   : [0.6, 1.0],
              "norm"       : [0, 40]}

ylim_range_compare = {'mean': [0.08, 0.16],
                      'sigma':       [0.008, 0.024],
                      'sigma_ratio': [0.7, 2.5],
                      'asy_factor' : [0.8, 2.0],
                      'fraccore'   : [0.6, 1.0],
                      "norm"       : [0, 40]}

ylim_range_be7 = {'mean'  :     [0.145, 0.16], 'sigma' : [0.01, 0.03], 'sigma_ratio':[1.0, 2.2], 'asy_factor': [0.8, 1.8], 'fraccore':   [0.6, 1.1], "norm":       [0, 40]}

ylim_range_be = {'Be7':  {'mean': [0.145, 0.16], 'sigma' : [0.01, 0.03],   'sigma_ratio':[1.0, 2.2], 'asy_factor': [0.8, 1.6], 'fraccore':   [0.6, 1.1], "norm":       [0, 40]},
                 'Be9':  {'mean': [0.1, 0.12],   'sigma' : [0.008, 0.018], 'sigma_ratio':[1.0, 2.2], 'asy_factor': [0.8, 1.6], 'fraccore':   [0.6, 1.1], "norm":       [0, 40]},
                 'Be10': {'mean': [0.09, 0.11],  'sigma' : [0.008, 0.018], 'sigma_ratio':[1.0, 2.2], 'asy_factor': [0.8, 1.6], 'fraccore':   [0.6, 1.1], "norm":       [0, 40]}}

detectors_energyrange = {"Tof": [0.45, 1.2], "NaF": [1.1, 4.0], "Agl": [4.0, 12]} #Agl: 4.0, 10

fit_range_mass_nuclei = [0.06, 0.225]

def fill_guess_binbybin(guess, par_names, massfit_mc, isotopes=None):
    mc_counts = massfit_mc.get_data_infitrange()
    num_bins = massfit_mc.num_energybin
    for name, value in par_names.items():
        for ibin in range(num_bins):
            guess[f'{name}_{ibin}'] = value
            for iso in isotopes:
                isonum = NUCLEI_NUMBER[iso]
                if len(isotopes) == 1:
                    guess[f'n{isonum}_{ibin}'] = mc_counts[ibin].sum()

                else:
                    guess[f'n{isonum}_{ibin}'] = mc_counts[ibin].sum() * iso_ratio_guess[iso]

    #mean set specifically for difference isotope
    for ibin in range(num_bins):
         for iso in isotopes:
              isonum = NUCLEI_NUMBER[iso]
              guess[f'mean_{ibin}'] = 1/isonum * 0.93
              
    return guess 

def update_guess(guess, fitpars):
    for i, key in enumerate(guess.keys()):
        guess[key] = fitpars[key]['value']
    return guess

def update_guess_with_polyfit(guess, graph_poly_par, num_bins):
    for name, value in par_names.items():
        for ibin in range(num_bins):
            guess[f'{name}_{ibin}']  = graph_poly_par[name].yvalues[ibin]
    return guess

def get_limit_pars_binbybinfit(lim_pars, par_name, num_bins, limrange):
    for ibin in range(num_bins): 
        lim_pars[f'{par_name}_{ibin}'] = limrange[f'{par_name}'] 
    return lim_pars

polyfitp0 = {dec: {} for dec in detectors}

polyfitp0["Agl"] = {"mean": [0.15, 0.0, 0.0], "sigma":[0.016, 0.0, 0.0, 0.001], 'sigma_ratio': [1.5, 0.1, 0.1], 'asy_factor':[1.1, 0.1, 0.001], 'fraccore': [0.8, 0.1, 0.1]}
polyfitp0["NaF"] = {"mean": [0.15, 0.0, 0.0], "sigma":[0.016, 0.0, 0.0], 'sigma_ratio': [1.5], 'asy_factor':[1.1], 'fraccore': [0.75]}
polyfitp0["Tof"] = {"mean": [0.15, 0.0, 0.0], "sigma":[0.016, 0.001, 0.001], 'sigma_ratio': [1.2, 0.001, 0.001], 'asy_factor':[1.4, 0.01, 0.001], 'fraccore': [0.823, 0.1, 0.1]}

polyfit_pars = {dec: {} for dec in detectors}

def get_polyfit_pars(massfit, fit_parameters, graph_par):
    for dec in detectors:
        par = unumpy.nominal_values(fit_parameters[dec])
        par_err =  unumpy.std_devs(fit_parameters[dec])
        for i, parname in enumerate(par_names.keys()):
            n_x = massfit[dec].num_energybin
            graph_par[dec][parname] = MGraph(massfit[dec].x_fit_energy, par[i* n_x: (i+1)*n_x], yerrs=par_err[i* n_x: (i+1)*n_x])
            
    for i, par_name in enumerate(par_names.keys()):
        for dec in detectors:
            popt, pcov = curve_fit(poly, np.log(graph_par[dec][par_name].xvalues), graph_par[dec][par_name].yvalues, p0 = polyfitp0[dec][par_name])
            polyfit_pars[dec][par_name] = popt           
    return polyfit_pars

#def update_guess_simultaneousfit_pvalues(polyfit_pars, guess):
#    for dec in detectors:
#        combined_initial_par_array = np.concatenate(list(polyfit_pars[dec].values()))
#        keys = list(guess[dec].keys())
#        for i in range(len(combined_initial_par_array)):
#            guess[dec][keys[i]] = combined_initial_par_array[i]

def update_guess_simultaneousfit_pvalues(polyfit_pars, guess):
        combined_initial_par_array = np.concatenate(list(polyfit_pars.values()))
        keys = list(guess.keys())
        for i in range(len(combined_initial_par_array)):
            guess[keys[i]] = combined_initial_par_array[i]

def update_guess_simultaneousfit_pvalues_withpvalues(init_pvalues, guess_pvalues):
    for dec in detectors:
        for i, key in enumerate(guess_pvalues[dec].keys()):
            #random_number = random.uniform(0.995, 1.005)
            guess_pvalues[dec][key] = init_pvalues[dec][key]['value'] 


def update_guess_binbybinfit_initial(graph_template_pars_from_poly_par, isotopes, counts):
    guess_binbybin = {dec: {} for dec in detectors}
    for dec in detectors:
        for name, value in par_names.items():
            for ibin in range(massfit[dec].num_energybin):
                guess_binbybin[dec][f'{name}_{ibin}'] = graph_template_pars_from_poly_par[dec][name].yvalues[ibin]
        for iso in isotopes:
            isonum = NUCLEI_NUMBER[iso]
            for ibin in range(massfit[dec].num_energybin):
                guess_binbybin[dec][f'n{isonum}_{ibin}'] = counts[ibin].sum() * iso_ratio_guess[iso]
            
    return guess_binbybin

def initial_guess(guess_initial, guess_pvalues, massfit, nuclei, detector, mufactorfile, sigfactorfile, isConstraint):
    for key, value in guess_pvalues.items():
        guess_initial[key] = value

    for iso in ISOTOPES[Nuclei][1:]:
        for inum, ikey in enumerate(muscale_factor_name[detector][iso]):
            print(detector, ikey, inum)
            guess_initial[ikey] = mufactorfile[iso][inum]

    for iso in ISOTOPES[Nuclei][1:]:
        for inum, ikey in enumerate(sigscale_factor_name[detector][iso]):
            guess_initial[ikey] = sigfactorfile[iso][inum]
 

    counts = massfit.get_data_infitrange()
    for iso in ISOTOPES[nuclei]:
        if isConstraint and  iso == "Be10":
            continue
        else:
            isonum = NUCLEI_NUMBER[iso]
            for ibin in range(massfit.num_energybin):            
                guess_initial[f'n{isonum}_{ibin}'] = counts[ibin].sum() * iso_ratio_guess[iso]

                

def initial_guess_mciso_simufit(guess_initial, guess_pvalues, massfit,  nuclei, detector, isotopes, isConstraint=True):
    for key, value in guess_pvalues.items():
        guess_initial[key] = value
    counts = massfit.get_data_infitrange()
    if isConstraint:
        for iso in isotopes[:-1]:
            isonum = NUCLEI_NUMBER[iso]
            for ibin in range(massfit.num_energybin):            
                guess_initial[f'n{isonum}_{ibin}'] = counts[ibin].sum() * iso_ratio_guess[iso]
    else:
        for iso in isotopes:
            isonum = NUCLEI_NUMBER[iso]
            for ibin in range(massfit.num_energybin):            
                guess_initial[f'n{isonum}_{ibin}'] = counts[ibin].sum() * iso_ratio_guess[iso]

def draw_parameters_binbybin(massfit, fit_parameters, plotdir, plotname, detectors):
    graph_template_pars_from_poly_par = {dec: {} for dec in detectors}
    graph_par = {dec: {} for dec in detectors} 
    for dec in detectors:
        par = unumpy.nominal_values(fit_parameters[dec])
        par_err =  unumpy.std_devs(fit_parameters[dec])
        for i, parname in enumerate(par_names.keys()):
            n_x = massfit[dec].num_energybin
            graph_par[dec][parname] = MGraph(massfit[dec].x_fit_energy, par[i* n_x: (i+1)*n_x], yerrs=par_err[i* n_x: (i+1)*n_x])

    for i, par_name in enumerate(par_names.keys()):
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))
        fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)
        for dec in detectors:
            #plot_graph(fig, ax1, graph_par[dec][par_name], color=DETECTOR_COLOR[dec], label=DETECTOR_LABEL[dec], style="EP", xlog=False, ylog=False, scale=None, markersize=22)
            print(dec, par_name, graph_par[dec][par_name])
            
            popt, pcov = curve_fit(poly, np.log(graph_par[dec][par_name].xvalues), graph_par[dec][par_name].yvalues, p0 = polyfitp0[dec][par_name])
            polyfit_pars[dec][par_name] = popt
            graph_template_pars_from_poly_par[dec][par_name] = MGraph(graph_par[dec][par_name].xvalues, poly(np.log(graph_par[dec][par_name].xvalues), *popt), np.zeros_like(graph_par[dec][par_name].xvalues))
            #ax1.plot(graph_par[dec][par_name].xvalues, poly(np.log(graph_par[dec][par_name].xvalues), *popt), '-', color='black')
            #ax2.plot(graph_par[dec][par_name].xvalues, poly(np.log(graph_par[dec][par_name].xvalues), *popt)/graph_par[dec][par_name].yvalues, '.', color=DETECTOR_COLOR[dec], markersize=20)
        #ax1.set_ylabel(f'{par_names_axes[par_name]}')        
        #ax1.legend(loc='upper right', fontsize=FONTSIZE-1)
        ax2.set_ylabel("ratio")
        ax2.set_ylim([0.9, 1.1])
        ax2.grid()
        set_plot_defaultstyle(ax1)
        set_plot_defaultstyle(ax2)
        ax1.get_yticklabels()[0].set_visible(False)
        plt.subplots_adjust(hspace=.0)
        ax1.set_xticklabels([])
        ax2.set_xlabel("Ekin/n (GeV/n)")
        ax1.set_ylim(ylim_range_be7[par_name])
        savefig_tofile(fig, plotdir, f"fit_be7_{par_name}_{plotname}", show=False)
    return graph_template_pars_from_poly_par


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    #parser.add_argument("--filedata", default="/home/manbing/Documents/Data/data_BeP7/", help="file of data (mass vs energy 2d histogram) to be analysis")
    #parser.add_argument("--filedata", default="/home/manbing/Documents/Data/data_BeP8/BeISS_masshist_Ekinp8rebin_rmbadrun.npz", help="file of data (mass vs energy 2d histogram) to be analysis")
    #parser.add_argument("--filedata", default="/home/manbing/Documents/Data/data_BeP8/BeISS_masshist_EkinP8.npz", help="file of data (mass vs energy 2d histogram) to be analysis")
    #parser.add_argument("--filedata", default="/home/manbing/Documents/Data/data_BeP8/BeMassHist/BeISS_masshist_EkinP8GBL_finebin10yr_notofedge.npz", help="file of data (mass vs energy 2d histogram) to be analysis")
    #parser.add_argument("--filedata", default="/home/manbing/Documents/Data/data_BeP8/BeMassHist/BeMC_hist_mass_tuned1p1Agl_rebin.npz", help="file of data (mass vs energy 2d histogram) to be analysis")
    #parser.add_argument("--filedata", default="/home/manbing/Documents/Data/data_BeP8/BeMassHist/BeISS_masshist_EkinP8GBL_Rebin.npz", help="file of data (mass vs energy 2d histogram) to be analysis")
    #parser.add_argument("--filedata", default="/home/manbing/Documents/Data/data_BeP8/BeMassHist/BeISS_masshist_EkinP8GBL_fine10yr_nottofedge.npz", help="file of data (mass vs energy 2d histogram) to be analysis")
    #parser.add_argument("--filedata", default="/home/manbing/Documents/Data/data_BeP8/BeMassHist/BeISS_masshist_EkinP8GBL_rebin10yr_nottofedge.npz", help="file of data (mass vs energy 2d histogram) to be analysis")
    parser.add_argument("--filedata", default="/home/manbing/Documents/Data/data_BeP8/BeMassHist/BeISS_masshist_EkinP8GBL_rebin11yr_nottofedge_withP.npz", help="file of data (mass vs energy 2d histogram) to be analysis")
    parser.add_argument("--filename_mc", default="/home/manbing/Documents/Data/data_BeP8/BeMassHist/BeMC_hist_mass_tuned1p1Agl_rebin.npz", help="file of data (mass vs energy 2d histogram) to be analysis")
    parser.add_argument("--plotdir", default="plots/BeP8/fitData_P811yr_rebin_ThesisMay", help="Directory to store plots and result files in.")
    parser.add_argument("--nuclei", default="Be", help="Directory to store plots and result files in.")
    parser.add_argument("--isconstraint", default=True, help="choose if constraint the total N in fitting data")
    parser.add_argument("--isrebin", default=True, type=bool, help="choose if analysis is done in rebin version(can be seen from the name of the data files")
    parser.add_argument("--isFreeP", default=False, type=bool, help="choose if is_nonlinearconstraint")
    parser.add_argument("--isGBL", default=True, type=bool, help="choose if is_nonlinearconstraint")

    #parser.add_argument("--detectors", nargs="+", default=["NaF", "Agl"], help="Directory to store plots and result files in.")
    args = parser.parse_args()
    os.makedirs(args.plotdir, exist_ok=True)

    nuclei = args.nuclei
    #get the TH2 histogram
    isotopes_atom_num = [NUCLEI_NUMBER[iso] for iso in ISOTOPES[args.nuclei]]
    isotopes = ISOTOPES[args.nuclei]

    fit_range_mass = dict()
    for i, iso in enumerate(isotopes):
        mass_ingev = ISOTOPES_MASS[iso]
        fit_range_mass[iso] = [1/(mass_ingev*1.83), 1/(mass_ingev * 0.68)]   
        #fit_range_mass[iso] = [1/(mass_ingev*1.9), 1/(mass_ingev * 0.6)]
        #fit_range_mass[iso] = [1/(isotopes_atom_num[i]*1.85), 1/(isotopes_atom_num[i]* 0.65)]
        

    #########################################################
    #Read ErrorBand limits
    df_parlimsGBL = np.load('/home/manbing/Documents/Data/data_BeP8/FitParsRange/graph_parslim_TunedUncertaintymax.npz')
    df_parlimsChoutko = np.load('/home/manbing/Documents/Data/data_BeP8/FitParsRange/graph_parslim_TunedUncertainty_GBL.npz')

    df_parlims = {'GBL': df_parlimsGBL, 'choutko': df_parlimsChoutko}
    
    #df_parlims = np.load('/home/manbing/Documents/Data/data_BeP8/graph_parslim_UnTunedToTuned.npz')
    
    ErrorBandLim = {dec: {} for dec in detectors}
    graph_low = {dec: {} for dec in detectors}
    graph_up = {dec: {} for dec in detectors}
    rangelow = {dec: {} for dec in detectors}
    rangeup  = {dec: {} for dec in detectors}

    '''
    for dec in detectors:
        for parname in par_names.keys():
            graph_low[dec][parname] = MGraph.from_file(df_parlims, f'graph_{parname}low_{dec}')
            graph_up[dec][parname] = MGraph.from_file(df_parlims, f'graph_{parname}up_{dec}')
            rangelow[dec][parname] = np.array(graph_low[dec][parname].yvalues) 
            rangeup[dec][parname] = np.array(graph_up[dec][parname].yvalues) 
            print(dec, parname, 'low:', rangelow[dec][parname])
            print(dec, parname, 'up:', rangeup[dec][parname])
            ErrorBandLim[dec][parname] = np.vstack((rangelow[dec][parname], rangeup[dec][parname]))
    '''    
    #########################################################

    ################################################################################
    #Step5:  using the initial values from MC. fit the data with the simultaneous fit
    ################################################################################
    #entry objects for the fit 
    graph_counts_iso = {dec: {} for dec in detectors}
    graph_ratio_counts = {dec: {} for dec in detectors}
    hist2d_mass_energy = {dec: {} for dec in detectors}

    #this can be change according to which datasample(s) you want to use
    #OptimizedDataSample = [iso for iso in ISOTOPES[args.nuclei]]
    OptimizedDataSample = ["Be7", 'Be9', 'Be10']

    #read the file of 2d data histogram
    filenames = args.filedata 
    with np.load(filenames) as massfile:
        for dec in detectors:
            for OpIso in OptimizedDataSample:
                hist2d_mass_energy[dec][OpIso] = Histogram.from_file(massfile, f"Be_{dec}Opt{OpIso}_mass_ciemat")
                #hist2d_mass_energy[dec][OpIso] = Histogram.from_file(massfile, f"BeMCMix_{dec}_mass_test")

                fig = plt.figure(figsize=(20, 15))
                plot = fig.subplots(1, 1)
                plot2dhist(fig, plot, xbinning=hist2d_mass_energy[dec][OpIso].binnings[0].edges[1:-1],
                           ybinning=hist2d_mass_energy[dec][OpIso].binnings[1].edges[1:-1],
                           counts=hist2d_mass_energy[dec][OpIso].values[1:-1, 1:-1],
                           xlabel=None, ylabel=None, zlabel="counts", zmin=None, zmax=None,
                           setlogx=False, setlogy=False, setscilabelx=True, setscilabely=True,  setlogz=False)
                plot.text(0.05, 0.98, f"{dec}-Op{OpIso}_ISS", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", fontweight="bold")
                plot.set_ylabel("1/mass(1/GeV)", fontsize=30)
                plot.set_xlabel("Ekin/n(GeV/n)", fontsize=30)
                plot.set_xlim(detectors_energyrange[dec])
                
                savefig_tofile(fig, args.plotdir, f"hist_ISS_mass_Op{OpIso}_{dec}", show=False)

    with np.load(args.filename_mc) as mc_file:
        hist2d_mass_energy_mc = {dec: {iso: WeightedHistogram.from_file(mc_file, f"{iso}MC_{dec}_mass")  for iso in ISOTOPES[nuclei]} for dec in detectors}
        
    MassFunctionFit = {"Tof": TofInverseMassFunctionFit, "NaF": NaFInverseMassFunctionFit, "Agl": AglInverseMassFunctionFit}  
    massfit = {Opiso: {} for Opiso in OptimizedDataSample}
    fit_parameters = {Opiso: {} for Opiso in OptimizedDataSample}
    dict_fit_parameters = {Opiso: {dec: {} for dec in detectors} for  Opiso in OptimizedDataSample}
    
    ##############################################################################################
    #these values are not necessary because it would be in the end updated with the results of MC from the steps before
    ##############################################################################################
    guess_pvalues = {"Tof": {'mua':0.15200487, 'mub':-0.00191607, 'muc':-0.00085857,
                             'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949,   
                             'fraccore_a':0.79175096, 'fraccore_b': 0.1, 'fraccore_c': 0.1,
                             'sigma_ratio_a':1.2, 'sigma_ratio_b':0.1,  'sigma_ratio_c': 0.01,
                             'asy_factor_a':1.1, 'asy_factor_b':0.1, 'asy_factor_c':0.01},
                     "NaF": {'mua':0.15200487, 'mub':-0.00191607, 'muc':-0.00085857,
                             'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949,
                             'fraccore':0.79175096, 'sigma_ratio': 1.2, 'asy_factor': 1.3},
                     "Agl": {'mua':0.15200487, 'mub':-0.00191607, 'muc':-0.00085857,
                             'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949, 'sigd': 0.0001,
                             'fraccore_a': 0.8, 'fraccore_b': 0.1, 'fraccore_c': 0.1, 
                             'sigma_ratio_a': 1.1, 'sigma_ratio_b': 0.1, 'sigma_ratio_c': 0.01,
                             'asy_factor_a':1.1, 'asy_factor_b':0.1, 'asy_factor_c':0.01}}

    guess_initial = {dec: {} for dec in detectors}
    #p_pars_mc7 = np.load('/home/manbing/Documents/lithiumanalysis/scripts/plots/BeP8/fitMCUnTuned_Rebin/pvalues_mc7_iter0.npz', allow_pickle=True)
    if args.isrebin:
        #p_pars_mc7 = np.load('/home/manbing/Documents/lithiumanalysis/scripts/plots/BeP8/fitMCTunedBetaGBL_Rebin/pvalues_mc7_iter0.npz', allow_pickle=True)
        p_pars_mc7 = np.load('/home/manbing/Documents/lithiumanalysis/scripts/plots/BeP8/fitMCTunedBetaGBL_Rebin24Apr/pvalues_mc7_iter0.npz', allow_pickle=True)
        #p_pars_mc7 = np.load('/home/manbing/Documents/Data/data_BeP8/BeMCPars/pvalues_mc7_iter0.npz', allow_pickle=True)
    else:
        p_pars_mc7 = np.load('/home/manbing/Documents/Data/data_BeP8/BeMCPars/pvalues_mc7_iter0_1p1Agl.npz', allow_pickle=True)
        
    dictpar = {}    
    dictpar['Tof'] = p_pars_mc7['Tof'].item()
    dictpar['NaF'] = p_pars_mc7['NaF'].item()
    dictpar['Agl'] = p_pars_mc7['Agl'].item()

    print('dictpar:', dictpar)

    ############################################################################################
    #read the scale factor and write them to the initial guess
    ############################################################################################
    with open('/home/manbing/Documents/Data/data_BeP8/FitParsRange/polypars_mufactor.pickle', 'rb') as file:
        mufactorfile = pickle.load(file)

    with open('/home/manbing/Documents/Data/data_BeP8/FitParsRange/polypars_sigfactor.pickle', 'rb') as file:
        sigfactorfile = pickle.load(file)
        

   
    ###############################################################################################
    ####these values are fixed for Be, again this should be changed accordingly for other nuclei###
    ###############################################################################################
    
    
    #fixed_pvalues = {'Tof': ['fraccore', 'siga', 'sigb', 'sigc', 'sigma_ratio_a', 'sigma_ratio_b', 'asy_factor_a', 'asy_factor_b'],
    #                 'NaF': ['fraccore', 'siga', 'sigb', 'sigc', 'sigma_ratio', 'asy_factor'],
    #                 'Agl': ['siga', 'sigb', 'sigc', 'sigd', 'sigma_ratio_a', 'sigma_ratio_b', 'sigma_ratio_c', 'asy_factor_a', 'asy_factor_b', 'asy_factor_c']}
    lim_pars_beiss = {dec: {} for dec in detectors}
    fixed_pars_beiss = {dec: {} for dec in detectors}

    fixed_pvalues = {'Tof': [],
                     'NaF': [],
                     'Agl': []}  
                     #+ [f'muscale_{iso}_{a}' for a in ['a', 'b', 'c'] for iso in ISOTOPES[Nuclei][1:]] +  [f'sigscale_{iso}_{a}' for a in ['a', 'b', 'c'] for iso in ISOTOPES[Nuclei][1:]]} #'mua', 'mub', 'muc', 'siga', 'sigb', 'sigc', 'sigd', 'fraccore', 'sigma_ratio_a', 'sigma_ratio_b', 'sigma_ratio_c', 'asy_factor_a', 'asy_factor_b', 'asy_factor_c']}

    #p_pars_fitdatafree = np.load('/home/manbing/Documents/lithiumanalysis/scripts/plots/BeP8/fitDataTunedMC_Rebin_Free/pvalues_fitDataFree.npz', allow_pickle=True)
    #p_pars_fitdatafree = np.load('/home/manbing/Documents/lithiumanalysis/scripts/plots/BeP8/fitDataGBL_Rebin_FreeP/pvalues_fitDataFree.npz', allow_pickle=True)
    #dictpar_fitdatafree = {}
    #dictpar_fitdatafree['Be7'] = p_pars_fitdatafree['Be7'].item()
    #update_guess_simultaneousfit_pvalues_withpvalues(dictpar_fitdatafree['Be7'], guess_pvalues)
    
    #setup fit for data
    isscovariance = {Opiso: {} for Opiso in OptimizedDataSample}
    guesserr_dec = {'Tof': None,
                    'NaF': '/home/manbing/Documents/Data/data_BeP8/FitParsRange/splines_pars_uncertaintymax_BetaRig.pkl',
                    'Agl': '/home/manbing/Documents/Data/data_BeP8/FitParsRange/splines_pars_uncertaintymax_BetaRig.pkl'}

    for dec in detectors:
        for OpIso in OptimizedDataSample:
            massfit[OpIso][dec] = MassFunctionFit[dec](nuclei, isotopes, hist2d_mass_energy[dec][OpIso], detectors_energyrange[dec], fit_range_mass_nuclei, dec, args.isconstraint)
            
            initial_guess(guess_initial[dec], guess_pvalues[dec],  massfit[OpIso][dec], nuclei, dec, mufactorfile[dec], sigfactorfile[dec], args.isconstraint)
                        
            update_guess_simultaneousfit_pvalues(dictpar[dec], guess_initial[dec])
            print(dec, guess_initial[dec])


            fixed_pars_beiss[dec] = fixed_pvalues[dec]
            #if dec == 'Tof':
            #    fixed_pars_beiss[dec] = fixed_pvalues[dec]  + [f'muscale_{iso}' for iso in ISOTOPES[nuclei][1:]] + [f'sigscale_{iso}_{a}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]]
            #else:
            #    fixed_pars_beiss[dec] = fixed_pvalues[dec]  + [f'muscale_{iso}_{a}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]] + [f'sigscale_{iso}_{a}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]]
           
            
            fit_parameters[OpIso][dec], dict_fit_parameters[OpIso][dec], isscovariance[OpIso][dec] = massfit[OpIso][dec].perform_fit(guess_initial[dec], guesserr=guesserr_dec[dec],  fit_simultaneous=True,  fixed_pars = fixed_pars_beiss[dec], verbose=True, parlim=ErrorBandLim[dec], fitFreeP=args.isFreeP)
            #fit_parameters[OpIso][dec], dict_fit_parameters[OpIso][dec], isscovariance[OpIso][dec] = massfit[OpIso][dec].perform_fit(guess_initial[dec], fit_simultaneous=True, fixed_pars = fixed_pars_beiss[dec], verbose=True, parlim=ErrorBandLim[dec])
            
            for name in guess_initial[dec].keys():
                print(name, 'guess:', guess_initial[dec][name], ' ', 'fitResult:', dict_fit_parameters[OpIso][dec][name]['value'])
                
            massfit[OpIso][dec].draw_fit_results(fit_parameters[OpIso][dec], dict_fit_parameters[OpIso][dec], args.plotdir, fit_simultaneous=True)
            massfit[OpIso][dec].draw_fit_results_compare_datamc(fit_parameters[OpIso][dec], hist2d_mass_energy_mc[dec], dict_fit_parameters[OpIso][dec], args.plotdir, fit_simultaneous=True) 
    with open(os.path.join(args.plotdir, 'FitData_ResultsPars.pickle'), 'wb') as file:
        pickle.dump(fit_parameters, file)
        
    with open(os.path.join(args.plotdir, 'FitData_ResultsPars.pickle'), 'rb') as file:
        dict_results = pickle.load(file)

    print('save results in file:')
    print(dict_results['Be7']['NaF'])
    
    for OpIso in OptimizedDataSample:
        for parname in par_names.keys():
            plot_fitdata_pars(parname, detectors, massfit[OpIso], fit_parameters[OpIso], polyfitp0, par_names_axes, ylim_range_be7, nuclei, args.plotdir, guess_initial, plot_mc=True, figname=f'ISSOpt{OpIso}', guesserr='/home/manbing/Documents/Data/data_BeP8/FitParsRange/splines_pars_uncertaintymax_BetaRig.pkl', OptIso=OpIso)

    
    #np.savez(os.path.join('/home/manbing/Documents/lithiumanalysis/scripts/plots/BeP8/fitDataGBL_Rebin_FreeP', 'pvalues_fitDataFree.npz'), **dict_fit_parameters)   
    graph_counts_iso = {dec: {OpIso: {} for OpIso in OptimizedDataSample} for dec in detectors}
    dict_graph_counts = dict()

    ##################################################################################################
    #Here is also to be changed accroding to different nuclei, think about how to make it more general
    #################################################################################################
    if args.isconstraint:
        isotopes_atom_num = [NUCLEI_NUMBER[iso] for iso in ISOTOPES[args.nuclei][:-1]]
    else:
        isotopes_atom_num = [NUCLEI_NUMBER[iso] for iso in ISOTOPES[args.nuclei]]
        
    fit_norm =  {dec: {} for dec in detectors}
    fit_norm_err =  {dec: {} for dec in detectors}
        
    for isonum, iso in zip(isotopes_atom_num, ISOTOPES[args.nuclei]):
        for dec in detectors:
            for OpIso in OptimizedDataSample:
                fit_norm[dec][OpIso] = {isonum: np.array([dict_fit_parameters[OpIso][dec][f"n{isonum}_{ibin}"]['value'] for ibin in range(massfit[OpIso][dec].num_energybin)]) for isonum in isotopes_atom_num}
                fit_norm_err[dec][OpIso] = {isonum: np.array([dict_fit_parameters[OpIso][dec][f"n{isonum}_{ibin}"]['error'] for ibin in range(massfit[OpIso][dec].num_energybin)]) for isonum in isotopes_atom_num}
                graph_counts_iso[dec][OpIso][iso] = MGraph(massfit[OpIso][dec].x_fit_energy, fit_norm[dec][OpIso][isonum], fit_norm_err[dec][OpIso][isonum])
                graph_counts_iso[dec][OpIso][iso].add_to_file(dict_graph_counts, f'graph_counts_{dec}Opt{OpIso}_{iso}')

                
    counts_be10 = {dec: {} for dec in detectors}
    if args.isconstraint: 
        for dec in detectors:
            for OpIso in OptimizedDataSample:
                counts_be10[dec][OpIso] = massfit[OpIso][dec].get_data_infitrange().sum(axis=1) - fit_norm[dec][OpIso][7] - fit_norm[dec][OpIso][9]
                yerrbe10 = np.sqrt(graph_counts_iso[dec][OpIso]["Be7"].yerrs ** 2 + graph_counts_iso[dec][OpIso]["Be9"].yerrs ** 2 - 2 * 0.8*abs(graph_counts_iso[dec][OpIso]["Be7"].yerrs * graph_counts_iso[dec][OpIso]["Be9"].yerrs))
                graph_counts_iso[dec][OpIso]["Be10"] =  MGraph(massfit[OpIso][dec].x_fit_energy, counts_be10[dec][OpIso], yerrbe10)     
                graph_counts_iso[dec][OpIso]["Be10"].add_to_file(dict_graph_counts, f'graph_counts_{dec}Opt{OpIso}_Be10')

    np.savez(os.path.join(args.plotdir, f"graph_massfit_counts.npz"), **dict_graph_counts)


if __name__ == "__main__":
    main()


###########################################################################################


