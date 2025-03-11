
from tools.massfunction_Tof import TofInverseMassFunctionFit
import os
import numpy as np
import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import tools.roottree as read_tree
from tools.calculator import calc_mass, calc_ekin_from_beta, calc_betafrommomentom, calc_ratio_err

import uproot
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, LeastSquares, NormalConstraint
from iminuit.util import describe, make_func_code
from tools.constants import ISOTOPES, NUCLEI_NUMBER, ISOTOPES_COLOR, ISO_LABELS
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

detectors = ["Tof", "NaF", "Agl"]
iso_ratio_guess = {"Be7": 0.6, "Be9": 0.3, "Be10": 0.1, 'B10': 1.0, 'B11': 1.0}
par_names = {"mean": 0.15, "sigma": 0.016, "fraccore":1.0, "sigma_ratio":1.0, "asy_factor":1.0}

scale_factor = {"Be7": 1, "Be9": 7./9., "Be10": 7.0/10.0}
mean_scale_factor_fromMC = {"Be7": 1, "Be9": 0.77949789, "Be10": 0.70255866}

sigma_scale_factor_fromMC = {"Tof": {"Be7": 1, "Be9": 7./9, "Be10": 7./10},
                             "NaF": {"Be7": 1, "Be9": 0.792093, "Be10": 0.7218902},
                             "Agl": {"Be7": 1, "Be9": 0.792093, "Be10": 0.7218902}}
#sigma_scale_factor_fromMC = {"Be7": 1, "Be9": 7./9., "Be10": 7./10.}

par_names_axes = {'mean': '$\mathrm{\mu}$',
                  'sigma': '$\mathrm{\sigma_{p}}$',
                  "sigma_ratio": '$\mathrm{ \epsilon(\sigma ratio)}$',
                  "asy_factor":'alpha',
                  "fraccore":'$\mathrm{f_{core}}$',
                  'norm':'Norm'}

poly_deg = {'mean':3, 'sigma':3, 'sigma_ratio':1, 'asy_factor':1, 'fraccore': 1, 'norm':6}

ylim_range = {'mean':        [0.08, 0.16],
              'sigma':       [0.008, 0.024],
              'sigma_ratio': [0.7, 2.5],
              'asy_factor' : [0.8, 2.0],
              'fraccore'   : [0.6, 1.0],
              "norm"       : [0, 40]}

ylim_range_be7 = {'mean'  :     [0.145, 0.16],
                  'sigma' :     [0.012, 0.024],
                  'sigma_ratio':[0.7, 2.5],
                  'asy_factor': [0.8, 2.0],
                  'fraccore':   [0.6, 1.0],
                  "norm":       [0, 40]}

detectors_energyrange = {"Tof": [0.4, 2.5], "NaF": [1.1, 7.0], "Agl": [4.0, 12]} #Agl: 4.0, 10




def get_unp_pars_binbybin(par_name, dict_pars, num_energybin):
    yvals = [dict_pars[f'{par_name}_{ibin}']['value'] for ibin in range(num_energybin)]  
    yvals_err = [dict_pars[f'{par_name}_{ibin}']['error'] for ibin in range(num_energybin)]  
    return yvals, yvals_err

def get_fitpdf_witherr(xvalue, fit_parameters,  func):
    fit_values_with_errors = func(xvalue, *fit_parameters)
    fit_values = unumpy.nominal_values(fit_values_with_errors)
    fit_value_errors = unumpy.std_devs(fit_values_with_errors)
    return fit_values, fit_value_errors

def get_fitpdferrorband(xvalue, fit_parameters, func):
    fit_values_with_errors = func(xvalue, *fit_parameters)
    fit_values = unumpy.nominal_values(fit_values_with_errors)
    fit_value_errors = unumpy.std_devs(fit_values_with_errors)
    return fit_values - fit_value_errors, fit_values + fit_value_errors



def fill_guess_binbybin(guess, par_names, data_counts, num_bins, isotopes=None):
    for name, value in par_names.items():
        for ibin in range(num_bins):
            guess[f'{name}_{ibin}'] = value
            for iso in isotopes:
                isonum = NUCLEI_NUMBER[iso]
                if len(isotopes) == 1:
                    guess[f'n{isonum}_{ibin}'] = data_counts[ibin].sum()

                else:
                    guess[f'n{isonum}_{ibin}'] = data_counts[ibin].sum() * iso_ratio_guess[iso]
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
        lim_pars[f'{par_name}_{ibin}'] = limrange 
    return lim_pars

polyfitp0 = {dec: {} for dec in detectors}

polyfitp0["Agl"] = {"mean": [0.1, 0.0, 0.0], "sigma":[0.016, 0.0, 0.0, 0.0], 'sigma_ratio': [1.1], 'asy_factor':[1.1], 'fraccore': [0.75]}
polyfitp0["NaF"] = {"mean": [0.1, 0.0, 0.0], "sigma":[0.016, 0.0, 0.0], 'sigma_ratio': [1.1], 'asy_factor':[1.1], 'fraccore': [0.75]}
polyfitp0["Tof"] = {"mean": [0.1, 0.0, 0.0], "sigma":[0.016, 0.0, 0.0], 'sigma_ratio': [1.1], 'asy_factor':[1.2], 'fraccore': [0.75]}

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
        if isConstraint and  iso == "Be10":
            continue
        isonum = NUCLEI_NUMBER[iso]
        for ibin in range(massfit.num_energybin):            
            guess_initial[f'n{isonum}_{ibin}'] = counts[ibin].sum() * iso_ratio_guess[iso]

        
def draw_parameters_binbybin(massfit, fit_parameters, graph_par, plotdir, plotname):
    graph_template_pars_from_poly_par = {dec: {} for dec in detectors}
    
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
        #print("mean_fit:", popt, " _err:", pcov)
        #pars_mean = np.polyfit(np.log(sub_graph_mean[iso].getx()), sub_graph_mean[iso].gety(), 0)
        #yplot = np.poly1d(pars_mean)
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
    print("polyfit_pars:", polyfit_pars)
    return graph_template_pars_from_poly_par

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    #parser.add_argument("--filename", default="trees/massfit/Be9iss_Agl_masshist.root",   help="Path to root file to read tree from")
    parser.add_argument("--filename_mc", default="/home/manbing/Documents/Data/data_mc/dfile/BeMC1220Mit_dict_hist_mass_ekin.npz",  help="Path to root file to read tree from")
    parser.add_argument("--plotdir", default="plots/mass_reso", help="Directory to store plots and result files in.")
    parser.add_argument("--filedata", default="/home/manbing/Documents/Data/data_be_flux/Be_dict_hist_mass_Ekin.npz", help="file of data (mass vs energy 2d histogram) to be analysis")    
    parser.add_argument("--datadir", default="/home/manbing/Documents/Data/data_be_flux", help="Directory to store plots and result files in.")
    parser.add_argument("--nuclei", default="Be", help="Directory to store plots and result files in.")
    parser.add_argument("--mciso", default="Be7", help="Directory to store plots and result files in.")
    #parser.add_argument("--detectors", nargs="+", default=["NaF", "Agl"], help="Directory to store plots and result files in.")
    args = parser.parse_args()
    os.makedirs(args.plotdir, exist_ok=True)

    nuclei = args.nuclei
    #get the TH2 histogram
    isotopes_atom_num = [NUCLEI_NUMBER[iso] for iso in ISOTOPES[args.nuclei]]
    isotopes = ISOTOPES[args.nuclei]
    MCISO = args.mciso
    mass = ISOTOPES_MASS[MCISO]
    fit_range_mass = [1/(mass*1.5), 1/(mass * 0.7)]
    
    # Step1:fit mc iso bin by bin
    # get mc histogram
    hist2d_mass_energy_mc = {dec: {} for dec in detectors}
    with np.load(args.filename_mc) as mc_file:
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
                savefig_tofile(fig, args.plotdir, f"hist2d_mass_{iso}_{dec}", show=False)
 
    #initial guess for mc:
    guess_mc = {dec: {} for dec in detectors}
    massfit_mc = dict()
    fit_parameters_mc_be7 = dict()
    dict_fit_results_be7 = {dec: {} for dec in detectors}
    lim_fraccore = {dec: {} for dec in detectors}
    for dec in detectors:
        massfit_mc[dec] = TofInverseMassFunctionFit([MCISO], hist2d_mass_energy_mc[dec][MCISO], detectors_energyrange[dec], fit_range_mass, dec, False)          
        mc_counts = massfit_mc[dec].get_data_infitrange()
        guess_mc[dec] = fill_guess_binbybin(guess_mc[dec], par_names, mc_counts, massfit_mc[dec].num_energybin, isotopes=[MCISO])
        #lim_fraccore[dec] = get_limit_pars_binbybinfit(lim_fraccore[dec], 'fraccore', massfit_mc[dec].num_energybin, (0.9, 1.0))
        lim_fraccore[dec] = get_limit_pars_binbybinfit(lim_fraccore[dec], 'sigma', massfit_mc[dec].num_energybin, (0.1, 0.3))
        fix_pars_mc = ["fraccore", 'sigma_ratio', 'asy_factor']
        fit_parameters_mc_be7[dec], dict_fit_results_be7[dec] = massfit_mc[dec].perform_fit(guess_mc[dec], fit_simultaneous=False, fixed_pars=fix_pars_mc, verbose=False)
        
    graph_par_be7 = {dec: {} for dec in detectors}
    graph_poly_par_iter1 = draw_parameters_binbybin(massfit_mc, fit_parameters_mc_be7, graph_par_be7, os.path.join(args.plotdir, "fitbe7/iter1"), f'{MCISO}_iter1')

    for dec in detectors:
        massfit_mc[dec].draw_fit_results_mc(fit_parameters_mc_be7[dec], dict_fit_results_be7[dec], os.path.join(args.plotdir, "fitbe7/iter1"), fit_simultaneous=False, guess=None)
        
    #mc be7 iteration 2 
    guess_mc_iter2 = {dec: {} for dec in detectors}
    fit_parameters_mc_be7_iter2 = dict()
    dict_fit_results_be7_iter2 = {dec: {} for dec in detectors}


    
    for dec in detectors:
        #guess_mc_iter2[dec] = update_guess(guess_mc[dec], dict_fit_results_be7[dec])
        #guess_mc_iter2[dec] = update_guess_with_polyfit(guess_mc[dec], graph_poly_par_iter1[dec], massfit_mc[dec].num_energybin)
        mc_counts = massfit_mc[dec].get_data_infitrange()  
        guess_mc_iter2[dec] = fill_guess_binbybin(guess_mc_iter2[dec], par_names, mc_counts, massfit_mc[dec].num_energybin, isotopes=[MCISO])
        #lim_fraccore[dec] = get_limit_pars_binbybinfit(lim_fraccore[dec], 'fraccore', massfit_mc[dec].num_energybin, (0.9, 0.95))
        fix_pars_mc = ["fraccore", 'sigma_ratio', 'asy_factor']
        lim_fraccore[dec] = get_limit_pars_binbybinfit(lim_fraccore[dec], 'sigma', massfit_mc[dec].num_energybin, (0.01, 0.03))
        fit_parameters_mc_be7_iter2[dec], dict_fit_results_be7_iter2[dec] = massfit_mc[dec].perform_fit(guess_mc_iter2[dec], fixed_pars=fix_pars_mc, lim_pars= lim_fraccore[dec],  fit_simultaneous=False, verbose=False)

    graph_par_be7_iter2 = {dec: {} for dec in detectors}
    graph_poly_par_iter2 = draw_parameters_binbybin(massfit_mc, fit_parameters_mc_be7_iter2, graph_par_be7_iter2, os.path.join(args.plotdir, "fitbe7/iter2"), f'{MCISO}_iter2')
    p_pars_mc7 = get_polyfit_pars(massfit_mc, fit_parameters_mc_be7_iter2, graph_poly_par_iter2)
    
    for dec in detectors:
        massfit_mc[dec].draw_fit_results_mc(fit_parameters_mc_be7_iter2[dec], dict_fit_results_be7_iter2[dec], os.path.join(args.plotdir, "fitbe7/iter2"), fit_simultaneous=False, guess=None, figname='iter2')
    plt.show()

    #fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(24, 16)
    fig, ax1 = plt.subplots(1, 1, figsize=(24, 16)) 
    fig.subplots_adjust(left= 0.12, right=0.97, bottom=0.08, top=0.95)
    for dec in detectors:
        ysigma_init1, ysigma_init_err1 = get_unp_pars_binbybin("sigma", dict_fit_results_be7[dec], massfit_mc[dec].num_energybin)
        graph_sigma_init1 = MGraph(massfit_mc[dec].x_fit_energy, ysigma_init1, ysigma_init_err1)
        ysigma_init2, ysigma_init_err2 = get_unp_pars_binbybin("sigma", dict_fit_results_be7_iter2[dec], massfit_mc[dec].num_energybin)
        graph_sigma_init2 = MGraph(massfit_mc[dec].x_fit_energy, np.array(ysigma_init2) * ISOTOPES_MASS[MCISO], np.array(ysigma_init_err2) *ISOTOPES_MASS[MCISO])
        #plot_graph(fig, ax1, graph_sigma_init1, color=DETECTOR_COLOR[dec], style="EP", xlog=True, ylog=False, scale=None, markersize=22, label="iter 1", markerfacecolor="none")
        xval = np.log(graph_sigma_init1.xvalues)
        popt_iter1, pcov_iter1 = curve_fit(poly, xval, graph_sigma_init1.yvalues, p0 = [0.1, 0.001, 0.0001, 0.00005])
        polypars_iter1 = uncertainties.correlated_values(popt_iter1, pcov_iter1)
        popt_iter2, pcov_iter2 = curve_fit(poly, xval, graph_sigma_init2.yvalues, p0 = [0.1, 0.001, 0.0001, 0.00005])
        polypars_iter2 = uncertainties.correlated_values(popt_iter2, pcov_iter2)
        yfit_iter1, yfit_err_iter1 = get_fitpdf_witherr(xval, polypars_iter1, upoly)
        yfit_iter2, yfit_err_iter2 = get_fitpdf_witherr(xval, polypars_iter2, upoly)
        yfit_lower_iter1, yfit_upper_iter1 = get_fitpdferrorband(xval, polypars_iter1, upoly)
        yfit_lower_iter2, yfit_upper_iter2 = get_fitpdferrorband(xval, polypars_iter2, upoly)
        
        #ax1.plot(massfit_mc[dec].x_fit_energy, yfit_iter1,  "--", color='lightblue',  label='fit iter1')
        ax1.plot(massfit_mc[dec].x_fit_energy, yfit_iter2, "--",  color=DETECTOR_COLOR[dec], label= f'fit {dec}')
        plot_graph(fig, ax1, graph_sigma_init2, color=DETECTOR_COLOR[dec],  style="EP", xlog=True, ylog=False, scale=None, markersize=22, label=f'{dec}')
        #ax1.fill_between(massfit_mc[dec].x_fit_energy, yfit_lower_iter1, yfit_upper_iter1, color='lightblue', alpha=0.4, label='Error band iter1')
        ax1.fill_between(massfit_mc[dec].x_fit_energy, yfit_lower_iter2, yfit_upper_iter2, color=DETECTOR_COLOR[dec], alpha=0.4)
        
        ax1.set_ylabel(r'$\mathrm{\sigma_{m}/m}$')
        ax1.set_xlabel('Ekin/N(GeV/n)')
        ax1.set_ylim([0.05, 0.3])
        #ax1.legend(loc="upper right", fontsize=28)
        ax1.text(0.07, 0.98, f"{MCISO} MC", fontsize=FONTSIZE+2, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
        ax1.text(0.2, 0.5, f"Tof", fontsize=FONTSIZE+2, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color=DETECTOR_COLOR["Tof"], fontweight="bold")
        ax1.text(0.6, 0.5, f"NaF", fontsize=FONTSIZE+2, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color=DETECTOR_COLOR["NaF"], fontweight="bold")
        ax1.text(0.85, 0.5, f"Aerogel", fontsize=FONTSIZE+2, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color=DETECTOR_COLOR["Agl"], fontweight="bold")
        
        # Calculate the pull
        delta = yfit_iter2 - graph_sigma_init2.yvalues
        pull_sigma = delta / graph_sigma_init2.yerrs
        graph_pull_sigma = MGraph(massfit_mc[dec].x_fit_energy,  pull_sigma, np.zeros_like(pull_sigma))
        #plot_graph(fig, ax2, graph_pull_sigma, color="black", label=r'$\mathrm{\frac{y_{fit} - y_{d}}{\sqrt{e^{2}_{fit} + e^{2}_{d}}}}$', style="EP", xlog=True, ylog=False, scale=None, markersize=20) 
        
        # Calculate the ratio
        ratio_sigma = yfit_iter2 / np.array(ysigma_init2)
        ratio_sigma_error = calc_ratio_err(yfit_iter2, np.array(ysigma_init2), yfit_err_iter2, np.array(ysigma_init_err2))
        graph_ratio_sigma = MGraph(massfit_mc[dec].x_fit_energy,  ratio_sigma, ratio_sigma_error)
        
        #plot_graph(fig, ax3, graph_ratio_sigma, color="black", label='fit/data', style="EP", xlog=True, ylog=False, scale=None, markersize=20) 

        plt.subplots_adjust(hspace=.0)
        set_plot_defaultstyle(ax1)
        #ax1.set_xscale('log')
        #ax1.get_yticklabels()[0].set_visible(False)
        ax1.set_xticks([0.4, 0.6, 1, 2, 5, 10])
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())  
        #ax2.set_ylabel("pull")
        #ax2.set_ylim([-5, 5])
        #ax2.grid()

        ax1.set_xlabel("Ekin/n (GeV/n)")
        #set_plot_defaultstyle(ax2)
        #ax2.get_yticklabels()[0].set_visible(False)
        #ax2.set_xticklabels([])
        #ax3.set_ylabel("ratio")
        #ax3.set_xscale('log')
        #ax3.grid()
       
    savefig_tofile(fig, args.plotdir, f"{nuclei}_mass_resolution_1", show=True)

    fig, ax1 = plt.subplots(1, 1, figsize=(24, 16)) 
    fig.subplots_adjust(left= 0.12, right=0.97, bottom=0.08, top=0.95)
    for dec in detectors:
        ysigma_init1, ysigma_init_err1 = get_unp_pars_binbybin("sigma", dict_fit_results_be7[dec], massfit_mc[dec].num_energybin)
        graph_sigma_init1 = MGraph(massfit_mc[dec].x_fit_energy, ysigma_init1, ysigma_init_err1)
        ysigma_init2, ysigma_init_err2 = get_unp_pars_binbybin("sigma", dict_fit_results_be7_iter2[dec], massfit_mc[dec].num_energybin)
        graph_sigma_init2 = MGraph(massfit_mc[dec].x_fit_energy, np.array(ysigma_init2) * ISOTOPES_MASS[MCISO], np.array(ysigma_init_err2) *ISOTOPES_MASS[MCISO])
        #plot_graph(fig, ax1, graph_sigma_init1, color=DETECTOR_COLOR[dec], style="EP", xlog=True, ylog=False, scale=None, markersize=22, label="iter 1", markerfacecolor="none")
        xval = np.log(graph_sigma_init1.xvalues)
        popt_iter1, pcov_iter1 = curve_fit(poly, xval, graph_sigma_init1.yvalues, p0 = [0.1, 0.001, 0.0001, 0.00005])
        polypars_iter1 = uncertainties.correlated_values(popt_iter1, pcov_iter1)
        popt_iter2, pcov_iter2 = curve_fit(poly, xval, graph_sigma_init2.yvalues, p0 = [0.1, 0.001, 0.0001, 0.00005])
        polypars_iter2 = uncertainties.correlated_values(popt_iter2, pcov_iter2)
        yfit_iter1, yfit_err_iter1 = get_fitpdf_witherr(xval, polypars_iter1, upoly)
        yfit_iter2, yfit_err_iter2 = get_fitpdf_witherr(xval, polypars_iter2, upoly)
        yfit_lower_iter1, yfit_upper_iter1 = get_fitpdferrorband(xval, polypars_iter1, upoly)
        yfit_lower_iter2, yfit_upper_iter2 = get_fitpdferrorband(xval, polypars_iter2, upoly)
        
        ax1.plot(massfit_mc[dec].x_fit_energy, yfit_iter2, "--",  color=DETECTOR_COLOR[dec], label= f'fit {dec}')
        plot_graph(fig, ax1, graph_sigma_init2, color=DETECTOR_COLOR[dec],  style="EP", xlog=True, ylog=False, scale=None, markersize=22, label=f'{dec}')
        ax1.fill_between(massfit_mc[dec].x_fit_energy, yfit_lower_iter2, yfit_upper_iter2, color=DETECTOR_COLOR[dec], alpha=0.3)

        
        ax1.set_ylabel(r'$\mathrm{\sigma_{m}/m}$')
        ax1.set_xlabel('Ekin/N(GeV/n)')
        ax1.set_ylim([0.07, 0.2])
        #ax1.legend(loc="upper right", fontsize=28)
        ax1.text(0.07, 0.98, f"{MCISO} MC", fontsize=FONTSIZE+2, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
        ax1.text(0.07, 0.9, f"Tof", fontsize=FONTSIZE+2, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color=DETECTOR_COLOR["Tof"], fontweight="bold")
        ax1.text(0.6, 0.9, f"NaF", fontsize=FONTSIZE+2, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color=DETECTOR_COLOR["NaF"], fontweight="bold")
        ax1.text(0.85, 0.9, f"Aerogel", fontsize=FONTSIZE+2, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color=DETECTOR_COLOR["Agl"], fontweight="bold")
        
        
        plt.subplots_adjust(hspace=.0)
        set_plot_defaultstyle(ax1)
        
        ax1.set_xticks([0.4, 0.6, 1, 2, 5, 10])
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())  
        ax1.set_xlabel("Ekin/n (GeV/n)")
        
    ax1.fill_betweenx(np.linspace(0.05, 0.2, 100), 1.0, 4.0, alpha=0.1, color=DETECTOR_COLOR['NaF'])         
    savefig_tofile(fig, args.plotdir, f"{nuclei}_mass_resolution_2", show=True)



   
    plt.show()

if __name__ == "__main__":
    main()


    '''

    #plot the resolution
    #fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))
    fig, ax1 = plt.subplots(1, figsize=(20, 12))
    fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)
    for dec in detectors:            
        poly_mean = poly(np.log(massfit_mc[dec].x_fit_energy), guess_initial[dec]['mua'], guess_initial[dec]['mub'], guess_initial[dec]['muc'])
        poly_mean_fitresult = poly(np.log(massfit_mc[dec].x_fit_energy), dict_fit_parameters[dec]['mua']['value'],  dict_fit_parameters[dec]['mub']['value'],  dict_fit_parameters[dec]['muc']['value'])
        ax1.plot(massfit_mc[dec].x_fit_energy, poly_mean, '--', color=DETECTOR_COLOR[dec], label=f'{dec} init mc')
        ax1.plot(massfit_mc[dec].x_fit_energy, poly_mean_fitresult, '-', color=DETECTOR_COLOR[dec], label=f'{dec} fit')        
        ax1.set_ylabel(r'$\mathrm{\mu}$')
        ax1.set_xlabel('Ekin/N(GeV/n)')
        #ax1.set_ylim([0.1, 0.16])
        ax1.legend(fontsize=35)

    fig, ax1 = plt.subplots(1, figsize=(20, 12))
    fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)
    for dec in detectors:            
        poly_sigma = poly(np.log(massfit_mc[dec].x_fit_energy), guess_initial[dec]['siga'], guess_initial[dec]['sigb'], guess_initial[dec]['sigc'], guess_initial[dec]['sigd'])
        poly_sigma_fitresult = poly(np.log(massfit_mc[dec].x_fit_energy), dict_fit_parameters[dec]['siga']['value'],
                                    dict_fit_parameters[dec]['sigb']['value'],  dict_fit_parameters[dec]['sigc']['value'], dict_fit_parameters[dec]['sigd']['value'])

        print(dec, "sigma poly fit:")
        print(guess_initial[dec]['siga'],"  ", dict_fit_parameters[dec]['siga'])
        print(guess_initial[dec]['sigb'], "  ", dict_fit_parameters[dec]['sigb'])
        print(guess_initial[dec]['sigc'], "  ", dict_fit_parameters[dec]['sigc'])
        print(guess_initial[dec]['sigd'], "  ", dict_fit_parameters[dec]['sigd'])
        #print(dict_fit_parameters[dec]['sigb']['value'], dict_fit_parameters[dec]['sigb']['error'])
        #print(dict_fit_parameters[dec]['sigc']['value'], dict_fit_parameters[dec]['sigc']['error'])
        #print(dict_fit_parameters[dec]['sigd']['value'], dict_fit_parameters[dec]['sigd']['error'])
        p0_text = r'$\mathrm{{p_{{0}} = {:.4f} \pm {:.4f} }}$'.format(dict_fit_parameters[dec]['siga']['value'], dict_fit_parameters[dec]['siga']['error'])
        p1_text = r'$\mathrm{{p_{{0}} = {:.4f} \pm {:.4f} }}$'.format(dict_fit_parameters[dec]['sigb']['value'], dict_fit_parameters[dec]['sigb']['error'])
        p2_text = r'$\mathrm{{p_{{0}} = {:.4f} \pm {:.4f} }}$'.format(dict_fit_parameters[dec]['sigc']['value'], dict_fit_parameters[dec]['sigc']['error'])
        p3_text = r'$\mathrm{{p_{{0}} = {:.4f} \pm {:.4f} }}$'.format(dict_fit_parameters[dec]['sigd']['value'], dict_fit_parameters[dec]['sigd']['error'])
        ax1.text(0.05, 0.4, p0_text, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", fontweight="bold")
        ax1.text(0.05, 0.4, p1_text, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", fontweight="bold")
        ax1.text(0.05, 0.4, p2_text, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", fontweight="bold")
        ax1.text(0.05, 0.4, p3_text, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", fontweight="bold")

        ax1.plot(massfit_mc[dec].x_fit_energy, poly_sigma, '--', color=DETECTOR_COLOR[dec], label=f'{dec} guess')
        
        ax1.plot(massfit_mc[dec].x_fit_energy, poly_sigma_fitresult, '-', color=DETECTOR_COLOR[dec], label=f'{dec} fit')        
        ax1.set_ylabel(r'$\mathrm{\sigma}$')
        ax1.set_xlabel('Ekin/N(GeV/n)')
        #ax1.set_ylim([0.1, 0.16])
        ax1.legend()
        sig_pars = [dict_fit_parameters[dec]['siga'], dict_fit_parameters[dec]['sigb'], dict_fit_parameters[dec]['sigc'], dict_fit_parameters[dec]['sigd']]                                
        np.savez(os.path.join(args.plotdir, 'poly_parameters_sigma_3deg.npz'), parameters=sig_pars)
        savefig_tofile(fig, args.plotdir, f"fit_sigma_{dec}", show=True)



    #plot sigma ratio
    fig, ax1 = plt.subplots(1, figsize=(20, 12))
    fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)
    for dec in detectors:            
        poly_sigratio = poly(np.log(massfit_mc[dec].x_fit_energy), guess_initial[dec]['sigma_ratio'])
        poly_sigratio_fitresult = poly(np.log(massfit_mc[dec].x_fit_energy), dict_fit_parameters[dec]['sigma_ratio']['value'])
        ax1.plot(massfit_mc[dec].x_fit_energy, poly_sigratio, '--', color=DETECTOR_COLOR[dec], label=f'{dec} init mc')
        ax1.plot(massfit_mc[dec].x_fit_energy, poly_sigratio_fitresult, '-', color=DETECTOR_COLOR[dec], label=f'{dec} fit')        
        ax1.set_ylabel(r'$\mathrm{\sigma_{s}/\sigma_{p} ~ \epsilon}$')
        
        ax1.set_xlabel('Ekin/N(GeV/n)')
        savefig_tofile(fig, args.plotdir, f"fit_sigmaratio_{dec}", show=True)
        #ax1.set_ylim([0.1, 0.16])
        ax1.legend()
'''
