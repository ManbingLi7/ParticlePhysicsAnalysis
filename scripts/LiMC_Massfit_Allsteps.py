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
from tools.lithium_massfittools import get_fitpdf_witherr, get_fitpdferrorband, get_unp_pars_binbybin, plot_fitmc_pars, plot_fitdata_pars, plot_fitmc_compare_isopars


detectors = ["Tof", "NaF", "Agl"]
iso_ratio_guess = {"Li6": 0.5, "Li7": 0.5}
par_names = {"mean": 0.18, "sigma": 0.016, "fraccore":0.9, "sigma_ratio":1.27, "asy_factor":1.1}


mcpars_initlims = {'Tof': {'mean': (0.145, 0.19), 'sigma': (0.008, 0.03), 'fraccore': (0.75, 0.9), 'sigma_ratio':(1.3, 3.0), 'asy_factor':(1.1, 1.7)},
                   'NaF': {'mean': (0.14, 0.19), 'sigma': (0.008, 0.03), 'fraccore': (0.85, 0.98), 'sigma_ratio':(1.4, 1.8), 'asy_factor':(1.0, 1.25)},
                   'Agl': {'mean': (0.14, 0.19), 'sigma': (0.008, 0.03), 'fraccore': (0.8, 0.95), 'sigma_ratio':(1.7, 2.0), 'asy_factor':(0.98, 1.1)}}

                   #'NaF': {'mean': (0.13, 0.16), 'sigma': (0.008, 0.03), 'fraccore': (0.89, 0.91), 'sigma_ratio':(1.7, 1.8), 'asy_factor':(1.2, 1.25)},
                   #sigma_scale_factor_fromMC = {"Be7": 1, "Be9": 7./9., "Be10": 7./10.}

par_names_axes = {'mean': '$\mathrm{\mu}$',
                  'sigma': '$\mathrm{\sigma_{p}}$',
                  "sigma_ratio": '$\mathrm{ \epsilon(\sigma ratio)}$',
                  "asy_factor":'alpha',
                  "fraccore":'$\mathrm{f_{core}}$',
                  'norm':'Norm'}

poly_deg = {'mean':3, 'sigma':3, 'sigma_ratio':1, 'asy_factor':2, 'fraccore': 1, 'norm':6}

ylim_range = {'mean':        [0.08, 0.19],
              'sigma':       [0.008, 0.024],
              'sigma_ratio': [0.7, 2.5],
              'asy_factor' : [0.8, 2.0],
              'fraccore'   : [0.6, 1.0],
              "norm"       : [0, 40]}

ylim_range_compare = {'mean': [0.14, 0.19],
                      'sigma':       [0.008, 0.024],
                      'sigma_ratio': [0.7, 2.5],
                      'asy_factor' : [0.8, 1.8],
                      'fraccore'   : [0.6, 1.0],
                      "norm"       : [0, 40]}

ylim_range_be7 = {'mean'  :     [0.16, 0.185],
                  'sigma' :     [0.01, 0.03],
                  'sigma_ratio':[1.0, 2.2],
                  'asy_factor': [0.8, 1.8],
                  'fraccore':   [0.6, 1.1],
                  "norm":       [0, 40]}

detectors_energyrange = {"Tof": [0.33, 1.3], "NaF": [0.9, 6.0], "Agl": [3.5, 12]} #Agl: 4.0, 10

fit_range_mass_nuclei = [0.05, 0.23]

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


polyfitp0["Agl"] = {"mean": [0.18, 0.0, 0.0], "sigma":[0.016, 0.0, 0.0, 0.001], 'sigma_ratio': [1.5, 0.1, 0.1], 'asy_factor':[1.1, 0.1, 0.001], 'fraccore': [0.8, 0.1, 0.1]}  
polyfitp0["NaF"] = {"mean": [0.18, 0.0, 0.0], "sigma":[0.016, 0.0, 0.0], 'sigma_ratio': [1.5], 'asy_factor':[1.1], 'fraccore': [0.75]}
polyfitp0["Tof"] = {"mean": [0.18, 0.0, 0.0], "sigma":[0.016, 0.001, 0.001], 'sigma_ratio': [1.2, 0.001, 0.001], 'asy_factor':[1.4, 0.01, 0.001], 'fraccore': [0.823, 0.1, 0.1]} 



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

def update_guess_simultaneousfit_pvalues(polyfit_pars, guess):
    for dec in detectors:
        combined_initial_par_array = np.concatenate(list(polyfit_pars[dec].values()))
        keys = list(guess[dec].keys())
        for i in range(len(list(keys))):
            guess[dec][keys[i]] = combined_initial_par_array[i]

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

def initial_guess(guess_initial, guess_pvalues, fixed_rigsigma_factor, massfit,  nuclei, detector, isConstraint=True):
    for key, value in guess_pvalues.items():
        guess_initial[key] = value
    for key, value in fixed_rigsigma_factor.items():
        guess_initial[key] = value 
    counts = massfit.get_data_infitrange()
    for iso in ISOTOPES[nuclei]:
        if isConstraint and  iso == "Li7":
            continue
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
            print(dec, parname, graph_par[dec][par_name])
            plot_graph(fig, ax1, graph_par[dec][par_name], color=DETECTOR_COLOR[dec], label=DETECTOR_LABEL[dec], style="EP", xlog=False, ylog=False, scale=None, markersize=22)
            
            popt, pcov = curve_fit(poly, np.log(graph_par[dec][par_name].xvalues), graph_par[dec][par_name].yvalues, p0 = polyfitp0[dec][par_name])
            polyfit_pars[dec][par_name] = popt
            graph_template_pars_from_poly_par[dec][par_name] = MGraph(graph_par[dec][par_name].xvalues, poly(np.log(graph_par[dec][par_name].xvalues), *popt), np.zeros_like(graph_par[dec][par_name].xvalues))
            ax1.plot(graph_par[dec][par_name].xvalues, poly(np.log(graph_par[dec][par_name].xvalues), *popt), '-', color='black')
            ax2.plot(graph_par[dec][par_name].xvalues, poly(np.log(graph_par[dec][par_name].xvalues), *popt)/graph_par[dec][par_name].yvalues, '.', color=DETECTOR_COLOR[dec], markersize=20)
        ax1.set_ylabel(f'{par_names_axes[par_name]}')        
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

def update_guess_scaling_mean(guess, fromiso, toiso, massfit):
    num = massfit.num_energybin
    mass_scale = ISOTOPES_MASS[fromiso]/ISOTOPES_MASS[toiso]
    for ibin in range(num):
         guess[f'mean_{ibin}'] = guess[f'mean_{ibin}'] * mass_scale
    return guess

def update_guess_scaling_sigma(guess, fromiso, toiso, rig_scale_pars, massfit):
    num = massfit.num_energybin
    mass_scale = ISOTOPES_MASS[fromiso]/ISOTOPES_MASS[toiso]
    xval = massfit.x_fit_energy
    rig_scale = expo_func(xval, *rig_scale_pars)
    for ibin in range(num):
        guess[f'sigma_{ibin}'] = guess[f'sigma_{ibin}'] * mass_scale / rig_scale[ibin]
    return guess

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    #parser.add_argument("--filename", default="trees/massfit/Be9iss_Agl_masshist.root",   help="Path to root file to read tree from")
    #parser.add_argument("--filename_mc", default="/home/manbing/Documents/Data/data_LiP8/MassHist/LiMC_hist_mass_beta0_fluxweight.npz",  help="Path to root file to read tree from")
    parser.add_argument("--filename_mc", default="/home/manbing/Documents/Data/data_LiP8/Hist2D/LiMC_hist_MassHist_B1308_FineNewBin_ToFShiftCutoff_CutToFEdge.npz",  help="Path to root file to read tree from")
    parser.add_argument("--plotdir", default="plots/LiP8/LiMC_hist_MassHist_B1308_FineNewBin_ToFShiftCutoff_CutToFEdge", help="Directory to store plots and result files in.")
    parser.add_argument("--filedata", default="/home/manbing/Documents/Data/data_Li_flux/Li_dict_hist_mass_Ekin.npz", help="file of data (mass vs energy 2d histogram) to be analysis")    
    parser.add_argument("--nuclei", default="Li", help="Directory to store plots and result files in.")
    parser.add_argument("--isconstraint", default=True, help="choose if constraint the total N in fitting data")
    
    #parser.add_argument("--detectors", nargs="+", default=["NaF", "Agl"], help="Directory to store plots and result files in.")
    args = parser.parse_args()
    os.makedirs(args.plotdir, exist_ok=True)

    nuclei = args.nuclei
    #get the TH2 histogram
    isotopes_atom_num = [NUCLEI_NUMBER[iso] for iso in ISOTOPES[args.nuclei]]
    isotopes = ISOTOPES[args.nuclei]

    fit_range_mass = dict()
    for iso in isotopes:
        mass_ingev = ISOTOPES_MASS[iso]
        #fit_range_mass[iso] = [1/(mass_ingev*1.85), 1/(mass_ingev * 0.65)]   
        fit_range_mass[iso] = [1/(mass_ingev*1.9), 1/(mass_ingev * 0.7)]
        print(iso, fit_range_mass[iso])
        
    # get mc histogram
    hist2d_mass_energy_mc = {dec: {} for dec in detectors}
    with np.load(args.filename_mc) as mc_file:
        drawhline1 = {"Tof": [0.61, 0.73], "NaF": [2.57, 2.88], "Agl": [6.56, 7.18]}
        drawhline2 = {"Tof": [0.61, 0.73], "NaF": [2.57, 2.88], "Agl": [6.56, 7.18]}  
        for dec in detectors:
            for iso in ISOTOPES[nuclei]:
                hist2d_mass_energy_mc[dec][iso] = WeightedHistogram.from_file(mc_file, f"{iso}MC_{dec}_mass")
                fig = plt.figure(figsize=(20, 15))
                plot = fig.subplots(1, 1)
                plot2dhist(fig, plot, xbinning=hist2d_mass_energy_mc[dec][iso].binnings[0].edges[1:-1],
                           ybinning=hist2d_mass_energy_mc[dec][iso].binnings[1].edges[1:-1],
                           counts=hist2d_mass_energy_mc[dec][iso].values[1:-1, 1:-1],
                           xlabel=None, ylabel=None, zlabel="counts", zmin=None, zmax=None,
                           setlogx=False, setlogy=False, setscilabelx=True, setscilabely=True,  setlogz=True)
                plot.text(0.05, 0.98, f"{dec}-{iso}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", fontweight="bold")
                plot.set_ylabel("1/mass(1/GeV)", fontsize=30)
                plot.set_xlabel("Ekin/n(GeV/n)", fontsize=30)
                plot.set_xlim(ANALYSIS_RANGE_EKIN[dec])
                plot.axvline(x=drawhline1[dec][0], color='red', linestyle='-', linewidth=3)
                plot.axvline(x=drawhline1[dec][1], color='red', linestyle='-', linewidth=3)
                savefig_tofile(fig, args.plotdir, f"hist2d_mass_{iso}_{dec}", show=False)

                
    ###########################################################################
    # Step1:fit Li6mc
    ###########################################################################
    MCISO = isotopes[0]
    guess_mciso = {dec: {} for dec in detectors}
    massfit_mciso = dict()
    resultpars_mciso = dict()
    dict_results_mciso = {dec: {} for dec in detectors}
    limpars_mciso =  {dec: {} for dec in detectors}
    graphpars_pfit_iter0 = {dec: {} for dec in detectors}
    MassFunctionFit = {"Tof": TofInverseMassFunctionFit, "NaF": NaFInverseMassFunctionFit, "Agl": AglInverseMassFunctionFit}
    for dec in detectors:
        massfit_mciso[dec] = MassFunctionFit[dec](nuclei, [MCISO], hist2d_mass_energy_mc[dec][MCISO], detectors_energyrange[dec], fit_range_mass[MCISO], dec, False)
        guess_mciso[dec] = fill_guess_binbybin(guess_mciso[dec], par_names, massfit_mciso[dec], isotopes=[MCISO])
        #fixed_pars_be7 = ['fraccore', 'sigma_ratio']
        limpars_mciso[dec] = get_limit_pars_binbybinfit(limpars_mciso[dec], 'fraccore', massfit_mciso[dec].num_energybin, mcpars_initlims[dec])
        limpars_mciso[dec] = get_limit_pars_binbybinfit(limpars_mciso[dec], 'sigma_ratio', massfit_mciso[dec].num_energybin, mcpars_initlims[dec])
        limpars_mciso[dec] = get_limit_pars_binbybinfit(limpars_mciso[dec], 'asy_factor', massfit_mciso[dec].num_energybin, mcpars_initlims[dec])
        resultpars_mciso[dec], dict_results_mciso[dec], m_covariance = massfit_mciso[dec].perform_fit(guess_mciso[dec], lim_pars=limpars_mciso[dec], fit_simultaneous=False, verbose=True)

        massfit_mciso[dec].draw_fit_results_mc(resultpars_mciso[dec], dict_results_mciso[dec], args.plotdir, fit_simultaneous=False, guess=None, figname='iter0')


    graphpars_pfit_iter0 = draw_parameters_binbybin(massfit_mciso, resultpars_mciso, args.plotdir, f'{MCISO}_iter0', detectors)
    for name in par_names.keys():
        plot_fitmc_pars(name, detectors, dict_results_mciso, massfit_mciso, polyfitp0, args.plotdir, par_names_axes, ylim_range_be7, nuclei, iso, legend='iter0')

        
    pvalues_mc7_iter0 = get_polyfit_pars(massfit_mciso, resultpars_mciso, graphpars_pfit_iter0)    
    np.savez(os.path.join(args.plotdir, 'pvalues_mc7_iter0.npz'), **pvalues_mc7_iter0) 
    #####################################################################
    #Step2: mc be7 iteration 1 binbybin fit
    #####################################################################
    guess_mciso_iter1 = {dec: {} for dec in detectors}
    massfit_mciso_iter1 = dict()
    resultpars_mciso_iter1 = dict()
    dict_results_mciso_iter1 = { dec: {} for dec in detectors}
    limpars_mciso_iter1 = {dec: {} for dec in detectors}
    fixpars_mciso_iter1 = {dec: {} for dec in detectors}
    graphpars_pfit_iter1 = {dec: {} for dec in detectors}
    
    for dec in detectors:
        massfit_mciso_iter1[dec] = MassFunctionFit[dec](nuclei, [MCISO], hist2d_mass_energy_mc[dec][MCISO], detectors_energyrange[dec], fit_range_mass[MCISO], dec, False)
        guess_mciso_iter1[dec] = update_guess(guess_mciso[dec], dict_results_mciso[dec])
        guess_mciso_iter1[dec] = update_guess_with_polyfit(guess_mciso_iter1[dec], graphpars_pfit_iter0[dec], massfit_mciso_iter1[dec].num_energybin)
        limpars_mciso_iter1[dec] = get_limit_pars_binbybinfit(limpars_mciso_iter1[dec], 'sigma_ratio', massfit_mciso_iter1[dec].num_energybin,  mcpars_initlims[dec])
        fixpars_mciso_iter1 = ['sigma_ratio', 'fraccore']

        resultpars_mciso_iter1[dec], dict_results_mciso_iter1[dec], m_covariance = massfit_mciso_iter1[dec].perform_fit(guess_mciso_iter1[dec], lim_pars=limpars_mciso_iter1[dec], fit_simultaneous=False)  
        massfit_mciso_iter1[dec].draw_fit_results_mc(resultpars_mciso_iter1[dec], dict_results_mciso_iter1[dec], args.plotdir, fit_simultaneous=False, guess=None, figname='iter1')
        
    graphpars_pfit_iter1 = draw_parameters_binbybin(massfit_mciso_iter1, resultpars_mciso_iter1, args.plotdir, f'{MCISO}_iter1', detectors)

    for name in par_names.keys():
        plot_fitmc_pars(name, detectors, dict_results_mciso_iter1, massfit_mciso_iter1, polyfitp0, args.plotdir, par_names_axes, ylim_range_be7, nuclei, MCISO, legend='iter1')

    p_pars_mc7 = get_polyfit_pars(massfit_mciso_iter1, resultpars_mciso_iter1, graphpars_pfit_iter1)    
    np.savez(os.path.join(args.plotdir, 'pvalues_mc7_iter1.npz'), **p_pars_mc7) 

    ##################################################################################
    #Step3 :fit Be7MC simultaneously
    ##################################################################################
    '''
    massfit_mciso_simu = dict()
    resultpars_mciso_simu = dict()
    dict_results_mciso_simu = {dec: {} for dec in detectors}
    guess_pvalues_mciso_simu = {"Tof": {'mua':0.18, 'mub':-0.00191607, 'muc':-0.00085857,
                                        'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949,
                                        'fraccore':0.79175096, 
                                        'sigma_ratio_a':1.2, 'sigma_ratio_b':0.001, 'sigma_ratio_c':0.001, 
                                        'asy_factor_a':1.1, 'asy_factor_b':0.1, 'asy_factor_c':0.01},
                                "NaF": {'mua':0.15200487, 'mub':-0.00191607, 'muc':-0.00085857,
                                        'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949,
                                        'fraccore':0.79175096, 'sigma_ratio': 1.2, 'asy_factor': 1.3},
                                "Agl": {'mua':0.15200487, 'mub':-0.00191607, 'muc':-0.00085857,
                                        'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949, 'sigd': 0.0001,
                                        'fraccore_a':0.79175096, 'fraccore_b': 0.001, 'fraccore_c': 0.001,
                                        'sigma_ratio_a':1.2, 'sigma_ratio_b':0.001, 'sigma_ratio_c':0.001, 
                                        'asy_factor_a':1.1, 'asy_factor_b':0.1, 'asy_factor_c':0.01}}

    limpars_mciso_simu= {dec: {} for dec in detectors}
    fixpars_mciso_simu = {dec: {} for dec in detectors}
    update_guess_simultaneousfit_pvalues(p_pars_mc7, guess_pvalues_mciso_simu)
    
    #setup fit for data
    guess_initial_mciso_simu = {dec: {} for dec in detectors}    
    for dec in detectors:
        massfit_mciso_simu[dec] = MassFunctionFit[dec]([MCISO], hist2d_mass_energy_mc[dec][MCISO], detectors_energyrange[dec], fit_range_mass[MCISO], dec, True)
        initial_guess_mciso_simufit(guess_initial_mciso_simu[dec], guess_pvalues_mciso_simu[dec],  massfit_mciso_simu[dec], nuclei, dec, [MCISO])
        limpars_mciso_simu[dec] = {'fraccore': [0.7, 0.9], 'asy_factor':[1.0, 1.3], 'sigma_ratio':[1.4, 2.0]}
        #fixed_pars_beiss[dec] = ['siga', 'sigb', 'sigc', 'fraccore', 'asy_factor_a', 'asy_factor_b', 'asy_factor_c', 'sigma_ratio_a', 'sigma_ratio_b', 'sigma_ratio_c'] + [f'ex{a}_{iso}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]]
        resultpars_mciso_simu[dec], dict_results_mciso_simu[dec], m_covariance= massfit_mciso_simu[dec].perform_fit(guess_initial_mciso_simu[dec], fit_simultaneous=True, verbose=True)
        massfit_mciso_simu[dec].draw_fit_results(resultpars_mciso_simu[dec], dict_results_mciso_simu[dec], args.plotdir, fit_simultaneous=True, figname="simu_log", setylog=True)
    
    for parname in par_names.keys():
        plot_fitdata_pars(parname, detectors, massfit_mciso_simu, resultpars_mciso_simu, resultpars_mciso_simu, polyfitp0, par_names_axes, ylim_range_be7, nuclei, args.plotdir, plot_mc=True, figname=f'Li6MC')
    ''' 
     
    
    ##################################################################################
    #the scaling of sigma is understood and using the script plot_template_mass.py
    #Step4: fixed the parameters and scaling to Be9 and Be10, with only the \alpha free. compare the scaling of alpha
    ##################################################################################    
    guess_mciso_iter2 = {iso: {dec: {} for dec in detectors} for iso in isotopes}
    massfit_mciso_iter2 = {iso: {} for iso in isotopes}
    resultpars_mciso_iter2 = {iso: {} for iso in isotopes}
    dict_results_mciso_iter2 = {iso: {dec: {} for dec in detectors} for iso in isotopes}
    limpars_mciso_iter2 = {dec: {} for dec in detectors}
    fixpars_mciso_iter2 = {dec: {} for dec in detectors}
    graphpars_pfit_iter2 = {iso: {dec: {} for dec in detectors} for iso in isotopes}

    for dec in detectors:
        for iso in isotopes:
            massfit_mciso_iter2[iso][dec] = MassFunctionFit[dec](nuclei, [iso], hist2d_mass_energy_mc[dec][iso], detectors_energyrange[dec], fit_range_mass[iso], dec, False)
            guess_mciso_iter2[iso][dec] = fill_guess_binbybin(guess_mciso_iter2[iso][dec], par_names, massfit_mciso_iter2[iso][dec], isotopes=[iso])
            guess_mciso_iter2[iso][dec] = update_guess_with_polyfit(guess_mciso_iter2[iso][dec], graphpars_pfit_iter1[dec], massfit_mciso_iter2[iso][dec].num_energybin)
            if iso != MCISO:
                guess_mciso_iter2[iso][dec] = update_guess_scaling_mean(guess_mciso_iter2[iso][dec], MCISO,  iso, massfit_mciso_iter2[iso][dec])
                #guess_mciso_iter2[iso][dec] = update_guess_scaling_sigma(guess_mciso_iter2[iso][dec], MCISO, iso, rig_pars[iso], massfit_mciso_iter2[iso][dec])
                
            limpars_mciso_iter2[dec] = get_limit_pars_binbybinfit(limpars_mciso_iter2[dec], 'asy_factor', massfit_mciso_iter2[iso][dec].num_energybin,  mcpars_initlims[dec])
            limpars_mciso_iter2[dec] = get_limit_pars_binbybinfit(limpars_mciso_iter2[dec], 'sigma_ratio', massfit_mciso_iter2[iso][dec].num_energybin,  mcpars_initlims[dec])
            #fixpars_mciso_iter2[dec] = ['fraccore']
            fixpars_mciso_iter2[dec] = ['fraccore']

            resultpars_mciso_iter2[iso][dec], dict_results_mciso_iter2[iso][dec], m_covariance = massfit_mciso_iter2[iso][dec].perform_fit(guess_mciso_iter2[iso][dec], lim_pars=limpars_mciso_iter2[dec], fixed_pars=fixpars_mciso_iter2[dec], fit_simultaneous=False)
            massfit_mciso_iter2[iso][dec].draw_fit_results_mc(resultpars_mciso_iter2[iso][dec], dict_results_mciso_iter2[iso][dec], args.plotdir, fit_simultaneous=False, guess=None, figname=f'iter2_{iso}')

    
    #    plot_fitmc_pars("fraccore",    detectors, dict_results_mciso_iter2[iso], massfit_mciso_iter2[iso], 1, args.plotdir, par_names_axes, ylim_range_be7, nuclei, iso, legend=f'{iso}iter2')
    #    plot_fitmc_pars("asy_factor",  detectors, dict_results_mciso_iter2[iso], massfit_mciso_iter2[iso], 3, args.plotdir, par_names_axes, ylim_range_be7, nuclei, iso, legend=f'{iso}iter2')
    plt.close()
    for name in par_names.keys():
        plot_fitmc_compare_isopars(name, detectors, dict_results_mciso_iter2, massfit_mciso_iter2, args.plotdir, polyfitp0, par_names_axes, ylim_range_compare, nuclei, isotopes, legend='iter2')


    

if __name__ == "__main__":
    main()





