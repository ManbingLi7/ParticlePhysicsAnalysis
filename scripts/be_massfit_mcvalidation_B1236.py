from tools.MassFunction import expo_func
from tools.massfunction_Tof import TofInverseMassFunctionFit
from tools.massfunction_NaF import NaFInverseMassFunctionFit
from tools.massfunction_Agl import AglInverseMassFunctionFit

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

def funcbe10(total_counts, be7, be9):
    ybe10 = total_counts - be7 - be9
    return ybe10


detectors = ["Tof", "NaF", "Agl"]
iso_ratio_guess = {"Be7": 0.6, "Be9": 0.3, "Be10": 0.1}
par_names = {"mean": 0.151, "sigma": 0.016, "fraccore":0.9, "sigma_ratio":1.27, "asy_factor":1.1}

scale_factor = {"Be7": 1, "Be9": 7./9., "Be10": 7.0/10.0}
mean_scale_factor_fromMC = {"Be7": 1, "Be9": 0.77949789, "Be10": 0.70255866}

mcpars_initlims = {'Tof': {'mean': (0.13, 0.16), 'sigma': (0.008, 0.03), 'fraccore': (0.8, 0.95), 'sigma_ratio':(1.2, 1.6), 'asy_factor':(1.1, 1.5)},
                   'NaF': {'mean': (0.13, 0.16), 'sigma': (0.008, 0.03), 'fraccore': (0.89, 0.95), 'sigma_ratio':(1.7, 1.8), 'asy_factor':(1.1, 1.25)},
                   'Agl': {'mean': (0.13, 0.16), 'sigma': (0.008, 0.03), 'fraccore': (0.8, 0.95), 'sigma_ratio':(1.4, 1.7), 'asy_factor':(1.0, 1.2)}}

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

ylim_range_be7 = {'mean'  :     [0.14, 0.16], 'sigma' : [0.01, 0.03], 'sigma_ratio':[1.0, 2.2], 'asy_factor': [0.8, 1.8], 'fraccore':   [0.6, 1.1], "norm":       [0, 40]}

ylim_range_be = {'Be7':  {'mean': [0.14, 0.16], 'sigma' : [0.01, 0.03],   'sigma_ratio':[1.0, 2.2], 'asy_factor': [0.8, 1.6], 'fraccore':   [0.6, 1.1], "norm":       [0, 40]},
                 'Be9':  {'mean': [0.1, 0.12],   'sigma' : [0.008, 0.018], 'sigma_ratio':[1.0, 2.2], 'asy_factor': [0.8, 1.6], 'fraccore':   [0.6, 1.1], "norm":       [0, 40]},
                 'Be10': {'mean': [0.09, 0.11],  'sigma' : [0.008, 0.018], 'sigma_ratio':[1.0, 2.2], 'asy_factor': [0.8, 1.6], 'fraccore':   [0.6, 1.1], "norm":       [0, 40]}}

detectors_energyrange = {"Tof": [0.42, 1.1], "NaF": [1.1, 4.2], "Agl": [4.0, 12]} #Agl: 4.0, 10

fit_range_mass_nuclei = [0.055, 0.22]

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

polyfitp0["Agl"] = {"mean": [0.15, 0.0, 0.0], "sigma":[0.016, 0.0, 0.0, 0.001], 'sigma_ratio': [1.5, 0.1, 0.1], 'asy_factor':[1.1, 0.1, 0.001], 'fraccore': [0.75]}
polyfitp0["NaF"] = {"mean": [0.15, 0.0, 0.0], "sigma":[0.016, 0.0, 0.0], 'sigma_ratio': [1.5], 'asy_factor':[1.1], 'fraccore': [0.75]}
polyfitp0["Tof"] = {"mean": [0.15, 0.0, 0.0], "sigma":[0.016, 0.0, 0.0, 0.001], 'sigma_ratio': [1.2, 0.001, 0.001], 'asy_factor':[1.4, 0.01, 0.001], 'fraccore': [0.823]}

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

def update_guess_simultaneousfit_pvalues_withpvalues(init_pvalues, guess_pvalues):
    for dec in detectors:
        for i, key in enumerate(guess_pvalues[dec].keys()):
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

def initial_guess(guess_initial, guess_pvalues, fixed_rigsigma_factor, massfit,  nuclei, detector, isConstraint):
    for key, value in guess_pvalues.items():
        guess_initial[key] = value
    for key, value in fixed_rigsigma_factor.items():
        guess_initial[key] = value 
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
    #parser.add_argument("--filename_mc", default="/home/manbing/Documents/Data/data_mc/dfile/BeMC_dict_hist_mass_ekin.npz",  help="Path to root file to read tree from")
    parser.add_argument("--filename_mc", default="/home/manbing/Documents/Data/data_mc/dfile/BeMC_histmass_Ekin_B1236dst_rawmc.npz",  help="Path to root file to read tree from")
    parser.add_argument("--plotdir", default="plots/massfit/mcB1236_v2", help="Directory to store plots and result files in.")
    parser.add_argument("--filedata", default="/home/manbing/Documents/Data/jiahui/MC_Events/p7_common_sel/Be_histmass_ekin_jiahui.npz", help="file of data (mass vs energy 2d histogram) to be analysis")    
    parser.add_argument("--outputdir", default="plots/massfit/mcB1236_v2", help="Directory to store plots and result files in.")
    parser.add_argument("--nuclei", default="Be", help="Directory to store plots and result files in.")
    parser.add_argument("--isconstraint", default=True, help="choose if constraint the total N in fitting data")
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
        fit_range_mass[iso] = [1/(mass_ingev*1.85), 1/(mass_ingev * 0.65)]   
        #fit_range_mass[iso] = [1/(mass_ingev*1.9), 1/(mass_ingev * 0.6)]
        #fit_range_mass[iso] = [1/(isotopes_atom_num[i]*1.85), 1/(isotopes_atom_num[i]* 0.65)]
        
    # get mc histogram
    hist2d_mass_energy_mc = {dec: {} for dec in detectors}
    with np.load(args.filename_mc) as mc_file:
        drawhline1 = {"Tof": [0.61, 0.73], "NaF": [2.57, 2.88], "Agl": [6.56, 7.18]}
        drawhline2 = {"Tof": [0.61, 0.73], "NaF": [2.57, 2.88], "Agl": [6.56, 7.18]}  
        for dec in detectors:
            for iso in ISOTOPES[nuclei]:
                hist2d_mass_energy_mc[dec][iso] = WeightedHistogram.from_file(mc_file, f"{iso}MC_{dec}_mass_build")
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
    # Step1:fit be7mc
    ###########################################################################
    MCISO = isotopes[0]
    MCISO_PLOTDIR = os.path.join(args.plotdir, f"fit{MCISO}")
    guess_mciso = {dec: {} for dec in detectors}
    massfit_mciso = dict()
    resultpars_mciso = dict()
    dict_results_mciso = {dec: {} for dec in detectors}
    limpars_mciso =  {dec: {} for dec in detectors}
    graphpars_pfit_iter0 = {dec: {} for dec in detectors}
    MassFunctionFit = {"Tof": TofInverseMassFunctionFit, "NaF": NaFInverseMassFunctionFit, "Agl": AglInverseMassFunctionFit}
    for dec in detectors:
        massfit_mciso[dec] = MassFunctionFit[dec]([MCISO], hist2d_mass_energy_mc[dec][MCISO], detectors_energyrange[dec], fit_range_mass[MCISO], dec, False)
        guess_mciso[dec] = fill_guess_binbybin(guess_mciso[dec], par_names, massfit_mciso[dec], isotopes=[MCISO])
        #fixed_pars_be7 = ['fraccore', 'sigma_ratio']
        limpars_mciso[dec] = get_limit_pars_binbybinfit(limpars_mciso[dec], 'mean', massfit_mciso[dec].num_energybin, mcpars_initlims[dec])
        limpars_mciso[dec] = get_limit_pars_binbybinfit(limpars_mciso[dec], 'sigma', massfit_mciso[dec].num_energybin, mcpars_initlims[dec])
        limpars_mciso[dec] = get_limit_pars_binbybinfit(limpars_mciso[dec], 'fraccore', massfit_mciso[dec].num_energybin, mcpars_initlims[dec])
        limpars_mciso[dec] = get_limit_pars_binbybinfit(limpars_mciso[dec], 'sigma_ratio', massfit_mciso[dec].num_energybin, mcpars_initlims[dec])
        limpars_mciso[dec] = get_limit_pars_binbybinfit(limpars_mciso[dec], 'asy_factor', massfit_mciso[dec].num_energybin, mcpars_initlims[dec])
        resultpars_mciso[dec], dict_results_mciso[dec] = massfit_mciso[dec].perform_fit(guess_mciso[dec], lim_pars=limpars_mciso[dec], fit_simultaneous=False, verbose=True)
        massfit_mciso[dec].draw_fit_results_mc(resultpars_mciso[dec], dict_results_mciso[dec], MCISO_PLOTDIR, fit_simultaneous=False, guess=None, figname=f"{MCISO}iter0")


    graphpars_pfit_iter0 = draw_parameters_binbybin(massfit_mciso, resultpars_mciso, MCISO_PLOTDIR, f'{MCISO}_iter0', detectors)
    for name in par_names.keys():
        plot_fitmc_pars(name, detectors, dict_results_mciso, massfit_mciso, polyfitp0, MCISO_PLOTDIR, par_names_axes, ylim_range_be7, nuclei, iso, legend='iter0', figname='iter0')

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
        massfit_mciso_iter1[dec] = MassFunctionFit[dec]([MCISO], hist2d_mass_energy_mc[dec][MCISO], detectors_energyrange[dec], fit_range_mass[MCISO], dec, False)
        guess_mciso_iter1[dec] = update_guess(guess_mciso[dec], dict_results_mciso[dec])
        guess_mciso_iter1[dec] = update_guess_with_polyfit(guess_mciso_iter1[dec], graphpars_pfit_iter0[dec], massfit_mciso_iter1[dec].num_energybin)
        limpars_mciso_iter1[dec] = get_limit_pars_binbybinfit(limpars_mciso_iter1[dec], 'asy_factor', massfit_mciso_iter1[dec].num_energybin,  mcpars_initlims[dec])
        fixpars_mciso_iter1 = ['fraccore']
        resultpars_mciso_iter1[dec], dict_results_mciso_iter1[dec] = massfit_mciso_iter1[dec].perform_fit(guess_mciso_iter1[dec], lim_pars=limpars_mciso_iter1[dec], fixed_pars=fixpars_mciso_iter1, fit_simultaneous=False)
        massfit_mciso_iter1[dec].draw_fit_results_mc(resultpars_mciso_iter1[dec], dict_results_mciso_iter1[dec], MCISO_PLOTDIR, fit_simultaneous=False, guess=None, figname=f"{MCISO}iter1", drawlog=True)
        
    graphpars_pfit_iter1 = draw_parameters_binbybin(massfit_mciso_iter1, resultpars_mciso_iter1, MCISO_PLOTDIR, f'{MCISO}_iter1', detectors)

    for name in par_names.keys():
        plot_fitmc_pars(name, detectors, dict_results_mciso_iter1, massfit_mciso_iter1, polyfitp0, MCISO_PLOTDIR, par_names_axes, ylim_range_be7, nuclei, MCISO, legend='iter1')

    p_pars_mc7 = get_polyfit_pars(massfit_mciso_iter1, resultpars_mciso_iter1, graphpars_pfit_iter1)    

    #####################################################################
    #Step2.2: mc be7 iteration 1 binbybin fit
    #####################################################################
    guess_mciso_iter12 = {dec: {} for dec in detectors}
    massfit_mciso_iter12 = dict()
    resultpars_mciso_iter12 = dict()
    dict_results_mciso_iter12 = { dec: {} for dec in detectors}
    limpars_mciso_iter12 = {dec: {} for dec in detectors}
    graphpars_pfit_iter12 = {dec: {} for dec in detectors}
    
    for dec in detectors:
        massfit_mciso_iter12[dec] = MassFunctionFit[dec]([MCISO], hist2d_mass_energy_mc[dec][MCISO], detectors_energyrange[dec], fit_range_mass[MCISO], dec, False)
        guess_mciso_iter12[dec] = update_guess(guess_mciso[dec], dict_results_mciso_iter1[dec])
        guess_mciso_iter12[dec] = update_guess_with_polyfit(guess_mciso_iter12[dec], graphpars_pfit_iter1[dec], massfit_mciso_iter12[dec].num_energybin)
        limpars_mciso_iter12[dec] = get_limit_pars_binbybinfit(limpars_mciso_iter12[dec], 'asy_factor', massfit_mciso_iter12[dec].num_energybin,  mcpars_initlims[dec])
        fixpars_mciso_iter12 = ['fraccore']
        resultpars_mciso_iter12[dec], dict_results_mciso_iter12[dec] = massfit_mciso_iter12[dec].perform_fit(guess_mciso_iter12[dec], lim_pars=limpars_mciso_iter12[dec], fixed_pars=fixpars_mciso_iter12, fit_simultaneous=False)
        massfit_mciso_iter12[dec].draw_fit_results_mc(resultpars_mciso_iter12[dec], dict_results_mciso_iter12[dec], MCISO_PLOTDIR, fit_simultaneous=False, guess=None, figname=f"{MCISO}iter12")
        
    graphpars_pfit_iter12 = draw_parameters_binbybin(massfit_mciso_iter12, resultpars_mciso_iter12, MCISO_PLOTDIR, f'{MCISO}_iter12', detectors)

    for name in par_names.keys():
        plot_fitmc_pars(name, detectors, dict_results_mciso_iter12, massfit_mciso_iter12, polyfitp0, MCISO_PLOTDIR, par_names_axes, ylim_range_be7, nuclei, MCISO, legend='iter12')

    #p_pars_mc7 = get_polyfit_pars(massfit_mciso_iter12, resultpars_mciso_iter12, graphpars_pfit_iter12)    

    ##################################################################################
    #Step3 :fit Be7MC simultaneously
    ##################################################################################

    massfit_mciso_simu = dict()
    resultpars_mciso_simu = dict()
    dict_results_mciso_simu = {dec: {} for dec in detectors}
    guess_pvalues_mciso_simu = {"Tof": {'mua':0.15200487, 'mub':-0.00191607, 'muc':-0.00085857, 
                                        'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949, 'sigd':0.001,
                                        'fraccore':0.79175096, 
                                        'sigma_ratio_a':1.2, 'sigma_ratio_b':0.001,   'sigma_ratio_c':0.001,
                                        'asy_factor_a':1.1, 'asy_factor_b':0.1, 'asy_factor_c':0.001},
                                "NaF": {'mua':0.15200487, 'mub':-0.00191607, 'muc':-0.00085857,
                                        'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949,
                                        'fraccore':0.79175096, 'sigma_ratio': 1.2, 'asy_factor': 1.3},
                                "Agl": {'mua':0.15200487, 'mub':-0.00191607, 'muc':-0.00085857,
                                        'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949, 'sigd': 0.0001,
                                        'fraccore':0.79175096,
                                        'sigma_ratio_a':1.2, 'sigma_ratio_b':0.001,   'sigma_ratio_c':0.001,
                                        'asy_factor_a':1.1, 'asy_factor_b':0.1, 'asy_factor_c':0.01}}

    limpars_mciso_simu= {dec: {} for dec in detectors}
    fixpars_mciso_simu = {dec: {} for dec in detectors}
    update_guess_simultaneousfit_pvalues(p_pars_mc7, guess_pvalues_mciso_simu)
    
    #setup fit for data
    guess_initial_mciso_simu = {dec: {} for dec in detectors}    
    for dec in detectors:
        massfit_mciso_simu[dec] = MassFunctionFit[dec]([MCISO], hist2d_mass_energy_mc[dec][MCISO], detectors_energyrange[dec], fit_range_mass[MCISO], dec, is_constraint=True)
        initial_guess_mciso_simufit(guess_initial_mciso_simu[dec], guess_pvalues_mciso_simu[dec],  massfit_mciso_simu[dec], nuclei, dec, [MCISO])
        limpars_mciso_simu[dec] = {'sigma_ratio':[1.1, 1.8]}
        #fixed_pars_beiss[dec] = ['siga', 'sigb', 'sigc', 'fraccore', 'asy_factor_a', 'asy_factor_b', 'asy_factor_c', 'sigma_ratio_a', 'sigma_ratio_b', 'sigma_ratio_c'] + [f'ex{a}_{iso}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]]
        fixpars_mciso_simu[dec] = ['fraccore']
        resultpars_mciso_simu[dec], dict_results_mciso_simu[dec] = massfit_mciso_simu[dec].perform_fit(guess_initial_mciso_simu[dec], fixed_pars=fixpars_mciso_simu[dec], fit_simultaneous=True, verbose=True)
        massfit_mciso_simu[dec].draw_fit_results(resultpars_mciso_simu[dec], dict_results_mciso_simu[dec], args.plotdir, fit_simultaneous=True, figname="Be7MC")
        
    for name in par_names.keys():
        plot_fitdata_pars(name, detectors, massfit_mciso_simu, resultpars_mciso_simu, resultpars_mciso_iter1, polyfitp0, par_names_axes, ylim_range_be7, nuclei, args.plotdir,  guess_initial_mciso_simu, plot_mc=False, plot_guess=True,  figname='Be7MC')
     
    
    ##################################################################################
    #the scaling of sigma is understood and using the script plot_template_mass.py
    #Step4: fixed the parameters and scaling to Be9 and Be10, with only the \alpha free. compare the scaling of alpha
    ##################################################################################
    # read the scale factor from rigidity
    # here is only for Be rigidity, need to be modified for other nuclei
    filename_rig = "/home/manbing/Documents/Data/data_rig/graph_rigsigma_scale_factor.npz"
    with np.load(filename_rig) as file_rig:
        graph_rig_sigmafactor_7to9 = MGraph.from_file(file_rig, "graph_rig_reso_sigma_7to9")
        graph_rig_sigmafactor_7to10 = MGraph.from_file(file_rig, "graph_rig_reso_sigma_7to10")
        
    popt9, pcov9 = curve_fit(expo_func, graph_rig_sigmafactor_7to9.xvalues[1:], graph_rig_sigmafactor_7to9.yvalues[1:], p0 = [1.0, 4.0, 10])
    popt10, pcov10 = curve_fit(expo_func, graph_rig_sigmafactor_7to10.xvalues[1:], graph_rig_sigmafactor_7to10.yvalues[1:], p0 = [1.0, 4.0, 10])
    y_fit_expo_factor9 = expo_func(graph_rig_sigmafactor_7to9.xvalues[:], *popt9)
    y_fit_expo_factor10 = expo_func(graph_rig_sigmafactor_7to10.xvalues[:], *popt10)

    graph_9to7_sigmafactor = MGraph(graph_rig_sigmafactor_7to9.xvalues, 7/(9 * y_fit_expo_factor9), yerrs=graph_rig_sigmafactor_7to9.yerrs)
    graph_10to7_sigmafactor = MGraph(graph_rig_sigmafactor_7to9.xvalues, 7/(10 * y_fit_expo_factor10), yerrs=graph_rig_sigmafactor_7to10.yerrs)
    #end reading the rigidity scaling, modify this so that the script is independent of nuclei

    guess_mciso_iter2 = {iso: {dec: {} for dec in detectors} for iso in isotopes}
    massfit_mciso_iter2 = {iso: {} for iso in isotopes}
    resultpars_mciso_iter2 = {iso: {} for iso in isotopes}
    dict_results_mciso_iter2 = {iso: {dec: {} for dec in detectors} for iso in isotopes}
    limpars_mciso_iter2 = {dec: {} for dec in detectors}
    fixpars_mciso_iter2 = {'Tof': ['fraccore', 'sigma_ratio'],
                           'NaF': ['fraccore', 'sigma_ratio'],
                           'Agl': ['fraccore', 'sigma_ratio']}
                           
    graphpars_pfit_iter2 = {iso: {dec: {} for dec in detectors} for iso in isotopes}

    rig_pars = dict()
    rig_pars = {"Be9": popt9, "Be10": popt10}
    
    for dec in detectors:
        for iso in isotopes:
            massfit_mciso_iter2[iso][dec] = MassFunctionFit[dec]([iso], hist2d_mass_energy_mc[dec][iso], detectors_energyrange[dec], fit_range_mass[iso], dec, False)
            guess_mciso_iter2[iso][dec] = fill_guess_binbybin(guess_mciso_iter2[iso][dec], par_names, massfit_mciso_iter2[iso][dec], isotopes=[iso])
            guess_mciso_iter2[iso][dec] = update_guess_with_polyfit(guess_mciso_iter2[iso][dec], graphpars_pfit_iter12[dec], massfit_mciso_iter2[iso][dec].num_energybin)
            if iso != MCISO:
                guess_mciso_iter2[iso][dec] = update_guess_scaling_mean(guess_mciso_iter2[iso][dec], MCISO,  iso, massfit_mciso_iter2[iso][dec])
                guess_mciso_iter2[iso][dec] = update_guess_scaling_sigma(guess_mciso_iter2[iso][dec], MCISO, iso, rig_pars[iso], massfit_mciso_iter2[iso][dec])

            #limpars_mciso_iter2[dec] = get_limit_pars_binbybinfit(limpars_mciso_iter2[dec], 'fraccore', massfit_mciso_iter2[iso][dec].num_energybin,  mcpars_initlims[dec])
            limpars_mciso_iter2[dec] = get_limit_pars_binbybinfit(limpars_mciso_iter2[dec], 'asy_factor', massfit_mciso_iter2[iso][dec].num_energybin,  mcpars_initlims[dec])
            #limpars_mciso_iter2[dec] = get_limit_pars_binbybinfit(limpars_mciso_iter2[dec], 'sigma_ratio', massfit_mciso_iter2[iso][dec].num_energybin,  mcpars_initlims[dec])
            #resultpars_mciso_iter2[iso][dec], dict_results_mciso_iter2[iso][dec] = massfit_mciso_iter2[iso][dec].perform_fit(guess_mciso_iter2[iso][dec], fit_simultaneous=False)
            resultpars_mciso_iter2[iso][dec], dict_results_mciso_iter2[iso][dec] = massfit_mciso_iter2[iso][dec].perform_fit(guess_mciso_iter2[iso][dec], fixed_pars=fixpars_mciso_iter2[dec], fit_simultaneous=False)
        
            massfit_mciso_iter2[iso][dec].draw_fit_results_mc(resultpars_mciso_iter2[iso][dec], dict_results_mciso_iter2[iso][dec], MCISO_PLOTDIR, fit_simultaneous=False, guess=None, figname=f'{iso}iter2', drawlog=True)
        
    for iso in isotopes:
        graphpars_pfit_iter2[iso] = draw_parameters_binbybin(massfit_mciso_iter2[iso], resultpars_mciso_iter2[iso], MCISO_PLOTDIR, f'{MCISO}_iter2', detectors)
        for name in par_names.keys():
            plot_fitmc_pars(name, detectors, dict_results_mciso_iter2[iso], massfit_mciso_iter2[iso], polyfitp0, MCISO_PLOTDIR, par_names_axes, ylim_range_be[iso], nuclei, iso, figname='iter2', legend='iter2')

    pvalues_mc7_iter2 = get_polyfit_pars(massfit_mciso_iter2[MCISO], resultpars_mciso_iter2[MCISO], graphpars_pfit_iter2[MCISO])
    print('########################################')
    print('iter2:')
    print('pvalues_mc7_iter2', pvalues_mc7_iter2)
    print('########################################')
    
    for name in par_names.keys():    
        plot_fitmc_compare_isopars(name, detectors, dict_results_mciso_iter2, massfit_mciso_iter2, args.plotdir, polyfitp0, par_names_axes, ylim_range_compare, nuclei, isotopes, legend='iter2')


    ##############################################################################################
    #parameters intialize
    #these values are not necessary because it would be in the end updated with the results of MC from the steps before
    ##############################################################################################
    guess_pvalues = {"Tof": {'mua':0.15200487, 'mub':-0.00191607, 'muc':-0.00085857, 
                             'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949, 'sigd':0.001,
                             'fraccore':0.79175096,
                             'sigma_ratio_a':1.2, 'sigma_ratio_b':0.001, 'sigma_ratio_c':0.001, 
                             'asy_factor_a':1.1, 'asy_factor_b':0.1, 'asy_factor_c':0.01},
                     "NaF": {'mua':0.15200487, 'mub':-0.00191607, 'muc':-0.00085857,
                             'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949,
                             'fraccore':0.79175096, 'sigma_ratio': 1.2, 'asy_factor': 1.3},
                     "Agl": {'mua':0.15200487, 'mub':-0.00191607, 'muc':-0.00085857,
                             'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949, 'sigd': 0.0001,
                             'fraccore':0.79175096, 
                             'sigma_ratio_a':1.2, 'sigma_ratio_b':0.001, 'sigma_ratio_c':0.001, 
                             'asy_factor_a':1.1, 'asy_factor_b':0.1, 'asy_factor_c':0.01}}

    #update_guess_simultaneousfit_pvalues(pvalues_mc7_iter2, guess_pvalues)
    update_guess_simultaneousfit_pvalues_withpvalues(dict_results_mciso_simu, guess_pvalues)
    fixed_rigsigma_factor = {"exa_Be9":  1.18459342, 'exb_Be9': 127.19756028, 'exc_Be9': 67.10310689,
                             'exa_Be10':  2.73491775, 'exb_Be10': 181.3605604, 'exc_Be10': 392.48417433}
                            
    guess_initial_mcmix = {dec: {} for dec in detectors}    

    ################################################################################
    #Step5 Fit MC mixture
    ################################################################################
    #read 2d histogram mcmix
    hist2d_mass_energy_mcmix = dict()
    #with np.load("/home/manbing/Documents/Data/data_mc/dfile/BeMC_dict_hist_mass_ekin.npz") as mc_file:
    #
    #with np.load("/home/manbing/Documents/Data/jiahui/MC_Events/BeMC_histmass_jiahuicommsel_reweight_test.npz") as mc_file:
    with np.load(args.filename_mc) as mc_file:
        for dec in detectors:
            hist2d_mass_energy_mcmix[dec] = WeightedHistogram.from_file(mc_file, f'{nuclei}MCMix_{dec}_mass_build')

    #prepare objects for the fit:
    graph_counts_iso_mcmix = {dec: {} for dec in detectors}
    dict_graph_counts_mcmix = dict() 
    massfit_mcmix = dict()
    resultpars_mcmix = dict()
    dict_results_mcmix = {dec: {} for dec in detectors}

    #fit simultaneously
    lim_pars_mcmix = {dec: {} for dec in detectors}
    #fixed_pars_mcmix = {"Tof": ['siga', 'sigb', 'sigc', 'sigd', 'sigma_ratio_a', 'sigma_ratio_b', 'sigma_ratio_c','fraccore', 'asy_factor_a', 'asy_factor_b', 'asy_factor_c'] + [f'ex{a}_{iso}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]],
    #                    "NaF": ['siga', 'sigb', 'sigc', 'sigma_ratio', 'fraccore', 'asy_factor'] + [f'ex{a}_{iso}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]],
    #                    

    #fixed_pars_mcmix = {"Tof": [f'ex{a}_{iso}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]],
    #fixed_pars_mcmix = {"Tof": ['fraccore'] + [f'ex{a}_{iso}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]],
    fixed_pars_mcmix =  {"Tof":  ['fraccore'] + [f'ex{a}_{iso}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]],
                         "NaF":  ['fraccore', 'sigma_ratio'] + [f'ex{a}_{iso}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]],
                         "Agl": ['mua', 'mub', 'muc', 'siga', 'sigb', 'sigc', 'sigd', 'sigma_ratio_a', 'sigma_ratio_b',  'fraccore', 'asy_factor_a', 'asy_factor_b', 'asy_factor_c'] + [f'ex{a}_{iso}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]]}
                         #"Agl":  ['fraccore', 'sigma_ratio_a', 'sigma_ratio_b', 'sigma_ratio_c', 'asy_factor_a', 'asy_factor_b', 'asy_factor_c'] + [f'ex{a}_{iso}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]]}
    
    for dec in detectors:
        massfit_mcmix[dec] = MassFunctionFit[dec](isotopes, hist2d_mass_energy_mcmix[dec], detectors_energyrange[dec], fit_range_mass_nuclei, dec, True)
        #initial guess with the fit range
        initial_guess(guess_initial_mcmix[dec], guess_pvalues[dec], fixed_rigsigma_factor, massfit_mcmix[dec], nuclei, dec, True)
        #lim_pars_mcmix[dec] = {}
        resultpars_mcmix[dec], dict_results_mcmix[dec] = massfit_mcmix[dec].perform_fit(guess_initial_mcmix[dec], fit_simultaneous=True, fixed_pars = fixed_pars_mcmix[dec], verbose=True)
        massfit_mcmix[dec].draw_fit_results(resultpars_mcmix[dec], dict_results_mcmix[dec], args.plotdir, fit_simultaneous=True, figname="mcmix")
    
    for name in par_names.keys():   
        plot_fitdata_pars(name, detectors, massfit_mcmix, resultpars_mcmix, resultpars_mciso_simu, polyfitp0, par_names_axes, ylim_range_be7, nuclei, args.plotdir,  guess_initial_mcmix, plot_mc=True, figname="mcmix")
    

    ##################################################################################################
    #Here is also to be changed accroding to different nuclei, think about how to make it more general
    ##################################################################################################
    iso_nums = [7, 9, 10]
    fit_norm =  dict() 
    fit_norm_err =  dict()
    if massfit_mcmix["Tof"].is_constraint:
        iso_nums = [7, 9]
    else:
        iso_nums = [7, 9, 10]
        
    for isonum, iso in zip(iso_nums, ISOTOPES[args.nuclei]):
        fig, ax1 = plt.subplots(1, figsize=(20, 12))
        fig.subplots_adjust(left= 0.11, right=0.96, bottom=0.1, top=0.96)
        set_plot_defaultstyle(ax1)
        for dec in detectors:
            fit_norm[dec] = {isonum: np.array([dict_results_mcmix[dec][f"n{isonum}_{ibin}"]['value'] for ibin in range(massfit_mcmix[dec].num_energybin)]) for isonum in iso_nums}
            fit_norm_err[dec] = {isonum: np.array([dict_results_mcmix[dec][f"n{isonum}_{ibin}"]['error'] for ibin in range(massfit_mcmix[dec].num_energybin)]) for isonum in iso_nums}
            
            graph_counts_iso_mcmix[dec][iso] = MGraph(massfit_mcmix[dec].x_fit_energy, fit_norm[dec][isonum], fit_norm_err[dec][isonum])
            graph_counts_iso_mcmix[dec][iso].add_to_file(dict_graph_counts_mcmix, f'graph_counts_{dec}_{iso}')
            ax1.errorbar(massfit_mcmix[dec].x_fit_energy, fit_norm[dec][isonum], yerr=fit_norm_err[dec][isonum], fmt='.', markersize=18, color=ISOTOPES_COLOR[iso], label=f'{ISO_LABELS[iso]}')
        ax1.set_xlabel('Ekin/n (GeV/n)', fontsize=FONTSIZE+3)
        ax1.set_ylabel('N (counts)', fontsize=FONTSIZE+3)
        ax1.text(0.03, 0.98, f"{DETECTOR_LABEL[dec]}", fontsize=FONTSIZE+2, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black")
        savefig_tofile(fig, args.plotdir, f"counts_fromfit_{iso}_mcmix", show=True)
    
    counts_be10 = dict()
    if massfit_mcmix["Tof"].is_constraint:
        for dec in detectors:
            counts_be10[dec] = massfit_mcmix[dec].get_data_infitrange().sum(axis=1) - fit_norm[dec][7] - fit_norm[dec][9]
            yerrbe10 = np.sqrt(graph_counts_iso_mcmix[dec]["Be7"].yerrs ** 2 + graph_counts_iso_mcmix[dec]["Be9"].yerrs ** 2 - 2 * 0.8*abs(graph_counts_iso_mcmix[dec]["Be7"].yerrs * graph_counts_iso_mcmix[dec]["Be9"].yerrs))
            graph_counts_iso_mcmix[dec]["Be10"] =  MGraph(massfit_mcmix[dec].x_fit_energy, counts_be10[dec], yerrbe10)     
            graph_counts_iso_mcmix[dec]["Be10"].add_to_file(dict_graph_counts_mcmix, f'graph_counts_{dec}_Be10')
        
    np.savez(os.path.join(args.outputdir, f"graph_counts_mcmix_rawmc.npz"), **dict_graph_counts_mcmix)

    ################################################################################
    #Step5:  using the initial values from MC. fit the data with the simultaneous fit
    ################################################################################
    #entry objects for the fit 
    graph_counts_iso = {dec: {} for dec in detectors}
    graph_ratio_counts = {dec: {} for dec in detectors}
    hist2d_mass_energy = {dec: {} for dec in detectors}

    #this can be change according to which datasample(s) you want to use
    OptimizedDataSample = [iso for iso in ISOTOPES[args.nuclei]]
    #OptimizedDataSample = ["Be7"]

    #read the file of 2d data histogram
    filenames = args.filedata 
    with np.load(filenames) as massfile:
        for dec in detectors:
            for OpIso in OptimizedDataSample:
                hist2d_mass_energy[dec][OpIso] = Histogram.from_file(massfile, f"Be_{dec}Opt{OpIso}_mass")
    
    massfit = {Opiso: {} for Opiso in OptimizedDataSample}
    fit_parameters = {Opiso: {} for Opiso in OptimizedDataSample}
    dict_fit_parameters = {Opiso: {dec: {} for dec in detectors} for  Opiso in OptimizedDataSample}

    ##############################################################################################
    #these values are not necessary because it would be in the end updated with the results of MC from the steps before
    ##############################################################################################
    guess_pvalues = {"Tof": {'mua':0.15200487, 'mub':-0.00191607, 'muc':-0.00085857,
                             'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949, 'sigd':0.001,
                             'fraccore':0.79175096, 'sigma_ratio_a':1.2, 'sigma_ratio_b':0.1,  'sigma_ratio_c': 0.01,
                             'asy_factor_a':1.1, 'asy_factor_b':0.1, 'asy_factor_c':0.01},
                     "NaF": {'mua':0.15200487, 'mub':-0.00191607, 'muc':-0.00085857,
                             'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949,
                             'fraccore':0.79175096, 'sigma_ratio': 1.2, 'asy_factor': 1.3},
                     "Agl": {'mua':0.15200487, 'mub':-0.00191607, 'muc':-0.00085857,
                             'siga':0.0180672, 'sigb': -0.00123484 , 'sigc': 0.00175949, 'sigd': 0.0001,
                             'fraccore':0.79175096, 'sigma_ratio_a': 1.1, 'sigma_ratio_b': 0.1, 'sigma_ratio_c': 0.01,
                             'asy_factor_a':1.1, 'asy_factor_b':0.1, 'asy_factor_c':0.01}}

    lim_pars_beiss = {dec: {} for dec in detectors}
    fixed_pars_beiss = {dec: {} for dec in detectors}
    update_guess_simultaneousfit_pvalues(p_pars_mc7, guess_pvalues)
    
    ###############################################################################################
    ####these values are fixed for Be, again this should be changed accordingly for other nuclei###
    ###############################################################################################
    #fixed_rigsigma_factor = {"exa_Be9":  1.00733236, 'exb_Be9': 28.14031131, 'exc_Be9': 5.03277699,
    #                         'exa_Be10':  1.01313499, 'exb_Be10': 25.85808338, 'exc_Be10': 5.18650221}

    fixed_rigsigma_factor = {"exa_Be9":  1.18459342, 'exb_Be9': 127.19756028, 'exc_Be9': 67.10310689,
                             'exa_Be10':  2.73491775, 'exb_Be10': 181.3605604, 'exc_Be10': 392.48417433}
    
    fixed_pvalues = {'Tof': ['fraccore', 'siga', 'sigb', 'sigc', 'sigma_ratio_a', 'sigma_ratio_b', 'asy_factor_a', 'asy_factor_b'],
                     'NaF': ['siga', 'sigb', 'sigc', 'fraccore', 'sigma_ratio', 'asy_factor'],
                     'Agl': ['fraccore', 'siga', 'sigb', 'sigc', 'sigd', 'sigma_ratio_a', 'sigma_ratio_b', 'sigma_ratio_c', 'asy_factor_a', 'asy_factor_b', 'asy_factor_c']}

    #fixed_pvalues = {'Tof': ['fraccore'],
    #                 'NaF': ['fraccore'],
    #                 'Agl': ['fraccore']}
    
    #setup fit for data
    guess_initial = {dec: {} for dec in detectors}    
    for dec in detectors:
        for OpIso in OptimizedDataSample:
            massfit[OpIso][dec] = MassFunctionFit[dec](isotopes, hist2d_mass_energy[dec][OpIso], detectors_energyrange[dec], fit_range_mass_nuclei, dec, args.isconstraint)
            initial_guess(guess_initial[dec], guess_pvalues[dec], fixed_rigsigma_factor, massfit[OpIso][dec], nuclei, dec, args.isconstraint)
            lim_pars_beiss[dec] = {'fraccore': [0.7, 0.9], 'asy_factor':[1.0, 1.23], 'sigma_ratio':[1.4, 2.0]}
            #fixed_pars_beiss[dec] = ['siga', 'sigb', 'sigc', 'fraccore', 'asy_factor_a', 'asy_factor_b', 'asy_factor_c', 'sigma_ratio_a', 'sigma_ratio_b', 'sigma_ratio_c'] + [f'ex{a}_{iso}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]]
            fixed_pars_beiss[dec] = fixed_pvalues[dec] + [f'ex{a}_{iso}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]]
            fit_parameters[OpIso][dec], dict_fit_parameters[OpIso][dec] = massfit[OpIso][dec].perform_fit(guess_initial[dec], fit_simultaneous=True, fixed_pars = fixed_pars_beiss[dec], verbose=True)
            massfit[OpIso][dec].draw_fit_results(fit_parameters[OpIso][dec], dict_fit_parameters[OpIso][dec], args.plotdir, fit_simultaneous=True)
    
    for OpIso in OptimizedDataSample:
        for parname in par_names.keys():
            plot_fitdata_pars(parname, detectors, massfit[OpIso], fit_parameters[OpIso], resultpars_mciso_simu, polyfitp0, par_names_axes, ylim_range_be7, nuclei, args.plotdir, guess_initial, plot_mc=True, figname=f'ISSOpt{OpIso}')


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



'''
   ################################################################################
    #Step5:  using the initial values from MC. fit the data with the simultaneous fit
    ################################################################################
    #entry objects for the fit 
    dict_graph_counts_T7 = dict()
    graph_counts_iso = {dec: {} for dec in detectors}
    graph_ratio_counts = {dec: {} for dec in detectors}
    hist2d_mass_energy = dict()
 
    #read the file of 2d data histogram
    filenames = args.filedata 
    with np.load(filenames) as massfile:
        for dec in detectors:
            hist2d_mass_energy[dec] = Histogram.from_file(massfile, f"Be_{dec}_mass_ciemat")
            
    graph_counts_before_fit = dict()
    graph_histentries_before_fit = dict()
    
    massfit = dict()
    fit_parameters = dict()
    dict_fit_parameters = {dec: {} for dec in detectors}
    
    lim_pars_beiss = {dec: {} for dec in detectors}
    fixed_pars_beiss = {dec: {} for dec in detectors}
                            
    #setup fit for data
    guess_initial = {dec: {} for dec in detectors}    
    for dec in detectors:
        print("dec:", dec)
        massfit[dec] = MassFunctionFit[dec](isotopes, hist2d_mass_energy[dec], detectors_energyrange[dec], fit_range_mass_nuclei, dec, is_constraint=True)
        counts = massfit[dec].get_data_infitrange()                  
        #initial guess with the fit range
        initial_guess(guess_initial[dec], guess_pvalues[dec], fixed_rigsigma_factor, massfit[dec], nuclei, dec)
        print(f"{dec} guess_initial:")
        for i, item in enumerate(guess_initial[dec].items()):
            if i < 13:
                print(item, '\t')

        lim_pars_beiss[dec] = {'fraccore': [0.7, 0.9], 'asy_factor':[1.0, 1.23], 'sigma_ratio':[1.4, 2.0]}
        #fixed_pars_beiss[dec] = ['siga', 'sigb', 'sigc', 'fraccore', 'asy_factor_a', 'asy_factor_b', 'asy_factor_c', 'sigma_ratio_a', 'sigma_ratio_b', 'sigma_ratio_c'] + [f'ex{a}_{iso}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]]
        fixed_pars_beiss[dec] = [f'ex{a}_{iso}' for iso in ISOTOPES[nuclei][1:] for a in ["a", "b", "c"]]
        fit_parameters[dec], dict_fit_parameters[dec] = massfit[dec].perform_fit(guess_initial[dec], fit_simultaneous=True, fixed_pars = fixed_pars_beiss[dec], verbose=True)
        massfit[dec].draw_fit_results(fit_parameters[dec], args.plotdir, fit_simultaneous=True)
    
    
    plot_fitdata_pars('mean', detectors, massfit, fit_parameters, guess_initial, dict_results_mciso_iter1, polyfitp0, par_names_axes, ylim_range_be7, nuclei, args.plotdir,  plot_mc=True)
    plot_fitdata_pars('sigma', detectors, massfit, fit_parameters, guess_initial, dict_results_mciso_iter1, polyfitp0, par_names_axes, ylim_range_be7, nuclei, args.plotdir,  plot_mc=True)
    plot_fitdata_pars('fraccore', detectors, massfit, fit_parameters, guess_initial, dict_results_mciso_iter1, polyfitp0, par_names_axes, ylim_range_be7, nuclei, args.plotdir,  plot_mc=True)
    plot_fitdata_pars('asy_factor', detectors, massfit, fit_parameters, guess_initial, dict_results_mciso_iter1, polyfitp0, par_names_axes, ylim_range_be7, nuclei, args.plotdir,  plot_mc=True)
    plot_fitdata_pars('sigma_ratio', detectors, massfit, fit_parameters, guess_initial, dict_results_mciso_iter1, polyfitp0, par_names_axes, ylim_range_be7, nuclei, args.plotdir,  plot_mc=True)
    
    graph_counts_iso = {dec: {} for dec in detectors}
    dict_graph_counts_T7 = dict()
'''



