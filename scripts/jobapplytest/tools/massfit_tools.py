import os
import numpy as np
import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines 
import tools.roottree as read_tree
from tools.calculator import calc_mass, calc_ekin_from_beta, calc_betafrommomentom, calc_ratio_err

import uproot
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, LeastSquares, NormalConstraint
from iminuit.util import describe, make_func_code
from tools.constants import ISOTOPES, NUCLEI_NUMBER, ISOTOPES_COLOR, ISO_LABELS, DETECTOR_COLOR, ISOTOPES_MASS, ANALYSIS_RANGE_RIG
from tools.histograms import Histogram
from tools.functions import gaussian, asy_gaussian, poly, upoly

from scipy import interpolate
from tools.graphs import MGraph, plot_graph
from scipy.optimize import curve_fit
import uncertainties
from uncertainties import unumpy
from uncertainties import ufloat
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, plot1d_step, FONTSIZE, set_plot_defaultstyle
from tools.mass_function import InverseMassFunctionFit, expo_func


################################
####tools for binbybin fit######
################################

def get_unp_pars_binbybin(par_name, dict_pars, num_energybin):
    yvals = [dict_pars[f'{par_name}_{ibin}']['value'] for ibin in range(num_energybin)]  
    yvals_err = [dict_pars[f'{par_name}_{ibin}']['error'] for ibin in range(num_energybin)]  
    return yvals, yvals_err

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

def get_limit_pars_binbybinfit(lim_pars, par_name, num_bins, limrange):
    for ibin in range(num_bins): 
        lim_pars[f'{par_name}_{ibin}'] = limrange 
    return lim_pars

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


def plot_fitmc_pars(parname, detectors, dict_fit_results_be7, massfit, poly_initp0, plotdir, ylabel, yrange, nuclei, isotope, figname=None, legend=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios':[0.4, 0.3, 0.3]}, figsize=(26, 16)) 
    fig.subplots_adjust(left= 0.12, right=0.97, bottom=0.08, top=0.95)
    xaxistext = {"Tof": 0.03, "NaF": 0.33, "Agl": 0.75}
    df_pars_save = dict()
    for dec in detectors:
        print(dec, parname, poly_initp0[dec][parname])
        polydeg = len(poly_initp0[dec][parname])
        y_pars, y_pars_err = get_unp_pars_binbybin(parname, dict_fit_results_be7[dec], massfit[dec].num_energybin)
        graph_pars = MGraph(massfit[dec].x_fit_energy, y_pars, y_pars_err)
        plot_graph(fig, ax1, graph_pars, color=DETECTOR_COLOR[dec],  style="EP", xlog=True, ylog=False, scale=None, markersize=22)
        xval = np.log(graph_pars.xvalues)
        #popt, pcov = curve_fit(poly, xval, graph_init.yvalues, p0 = np.ones(polydeg)*0.1, sigma=graph_init.yerrs, absolute_sigma=True)
        popt, pcov = curve_fit(poly, xval, graph_pars.yvalues, p0 = np.ones(polydeg)*0.1)
        polypars = uncertainties.correlated_values(popt, pcov)
        yfit, yfit_err = get_fitpdf_witherr(xval, polypars, upoly)  
        yfit_lower, yfit_upper = get_fitpdferrorband(xval, polypars, upoly)
        if dec == "Tof":
            ax1.plot(massfit[dec].x_fit_energy, yfit, "--",  color='black', label='poly fit')
        else:
            ax1.plot(massfit[dec].x_fit_energy, yfit, "--",  color='black')
            
        ax1.fill_between(massfit[dec].x_fit_energy, yfit_lower, yfit_upper, color='green', alpha=0.3)
        ax1.set_ylabel(ylabel[parname])
        ax1.set_xlabel('Ekin/N(GeV/n)')
        ax1.set_ylim(yrange[parname])
        ax1.legend(loc="lower right", fontsize=28)
        if  dec == "Tof":  
            ax1.text(xaxistext[dec], 0.98, f"{isotope} MC {dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
        else:
            ax1.text(xaxistext[dec], 0.98, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")

        #save the graph
        graph_pars.add_to_file(df_pars_save, f'graph_{parname}_{dec}')
        
            
        # Calculate the pull
        delta = yfit - y_pars
        pull = delta / y_pars_err
        graph_pull = MGraph(massfit[dec].x_fit_energy,  pull, np.zeros_like(pull))
        plot_graph(fig, ax2, graph_pull, color="black", label=r'$\mathrm{\frac{y_{fit} - y_{d}}{\sqrt{e^{2}_{fit} + e^{2}_{d}}}}$', style="EP", xlog=True, ylog=False, scale=None, markersize=20)
        if parname == 'mean':
            ax1.grid(axis='y')
            ax1.axhline(y=1/ISOTOPES_MASS[isotope], color='red', linestyle='--', linewidth=3)
        # Calculate the ratio
        ratio_par = yfit / np.array(y_pars)
        ratio_par_error = calc_ratio_err(yfit, np.array(y_pars), yfit_err, np.array(y_pars_err))
        graph_ratio = MGraph(massfit[dec].x_fit_energy,  ratio_par, ratio_par_error)
        
        plot_graph(fig, ax3, graph_ratio, color="black", label='fit/data', style="EP", xlog=True, ylog=False, scale=None, markersize=20) 

        equation_text_3deg = r'$\mathrm{\sigma_{p} = p_{0} + p_{1} \cdot x + p_{2} \cdot x^{2} + p_{3} \cdot x^{3}}$'
        equation_text_2deg = r'$\mathrm{\sigma_{p} = p_{0} + p_{1} \cdot x + p_{2} \cdot x^{2}}$'
        equation_text_1deg = r'$\mathrm{\sigma_{p} = p_{0} + p_{1} \cdot x }$'
        #equation_text = 
        #ax1.text(0.03, 0.9, equation_text_3deg, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='black')            
        for deg in range(polydeg):
            p_text = r'$\mathrm{{p_{{{}}} = {:.4f} \pm {:.4f}}}$'.format(deg, popt[deg], unumpy.std_devs(polypars)[deg])  
            ax1.text(xaxistext[dec], 0.88-0.07*deg, p_text, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='black')            

        plt.subplots_adjust(hspace=.0)
        #ax1.get_yticklabels()[0].set_visible(False)
        ax1.set_xticklabels([])
        set_plot_defaultstyle(ax1)
        ax2.set_ylabel("pull")
        ax2.set_ylim([-5, 5])
        ax3.set_ylim([0.95, 1.05])
        ax2.grid(axis='y')
        ax2.set_xscale('log')        
        set_plot_defaultstyle(ax2)
        #ax2.get_yticklabels()[0].set_visible(False)
        ax2.set_xticklabels([])
        ax3.set_xlabel("Ekin/n (GeV/n)")
        ax3.set_ylabel("ratio")
        ax3.set_xscale('log')
        ax3.grid(axis='y')
        set_plot_defaultstyle(ax3)

    if figname is not None:
        savefig_tofile(fig, plotdir, f"{nuclei}fit_{parname}_mc_{polydeg}deg_{figname}", show=True)
        np.savez(os.path.join(plotdir, f'df_{parname}_{figname}'), **df_pars_save)
    else:
        savefig_tofile(fig, plotdir, f"{nuclei}fit_{parname}_mc_{polydeg}deg_{legend}", show=True)



def plot_fitmc_compare_isopars(parname, detectors, dict_fit_results, massfit, plotdir, poly_initp0, ylabel, yrange, nuclei, isotopes, legend=None):
    xaxistext = {"Tof": 0.03, "NaF": 0.33, "Agl": 0.75}
    ylim_compare = {"mean":[0.6, 0.9], "sigma": [0.6, 0.9], "fraccore": [0.9, 1.1], "sigma_ratio": [0.9, 1.1], "asy_factor": [0.9, 1.1]}
    ratio_be9be7 = dict()
    ratio_be10be7 = dict()
    df_sigma_ratio = dict()
    graph_init = {dec: {} for dec in detectors}
    xtick_dec = {"Tof": [0.4, 0.6, 1.0], "NaF": [1.0, 2.0, 4.0], "Agl":[4.0, 6.0, 10.0]}
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(25, 15)) 
    fig.subplots_adjust(left= 0.12, right=0.97, bottom=0.08, top=0.95)
    for dec in detectors:
        polydeg = len(poly_initp0[dec][parname])
        for iso in isotopes: 
            y_init, y_init_err = get_unp_pars_binbybin(parname, dict_fit_results[iso][dec], massfit[iso][dec].num_energybin)
            graph_init[dec][iso] = MGraph(massfit[iso][dec].x_fit_energy, y_init, y_init_err)
            if dec == "Tof":
                plot_graph(fig, ax1, graph_init[dec][iso], color=ISOTOPES_COLOR[iso],  style="EP", xlog=True, ylog=False, scale=None, markersize=22, label=f'{ISO_LABELS[iso]}')
            else:
                plot_graph(fig, ax1, graph_init[dec][iso], color=ISOTOPES_COLOR[iso],  style="EP", xlog=True, ylog=False, scale=None, markersize=22)
                
            xval = graph_init[dec][iso].xvalues
            popt, pcov = curve_fit(poly, np.log(xval), graph_init[dec][iso].yvalues, p0 = 1 / (10 ** np.arange(1, polydeg+1)), sigma=graph_init[dec][iso].yerrs, absolute_sigma=True)
            polypars = uncertainties.correlated_values(popt, pcov)
            yfit, yfit_err = get_fitpdf_witherr(np.log(xval), polypars, upoly)  
            yfit_lower, yfit_upper = get_fitpdferrorband(np.log(xval), polypars, upoly)
            ax1.plot(xval, yfit, "--",  color=ISOTOPES_COLOR[iso])
            ax1.fill_between(xval, yfit_lower, yfit_upper, color=ISOTOPES_COLOR[iso], alpha=0.3)

        ratio_be9be7[dec] = graph_init[dec]["Be9"]/graph_init[dec]["Be7"]
        ratio_be10be7[dec] = graph_init[dec]["Be10"]/graph_init[dec]["Be7"]
        ratio_be9be7[dec].add_to_file(df_sigma_ratio, f'graph_mass_sigma_9to7_{dec}')
        ratio_be10be7[dec].add_to_file(df_sigma_ratio, f'graph_mass_sigma_10to7_{dec}')
        
        plot_graph(fig, ax2, ratio_be9be7[dec], color=ISOTOPES_COLOR['Be9'], style="EP", xlog=True, ylog=False, scale=None, markersize=22, label='Be9/Be7')
        plot_graph(fig, ax2, ratio_be10be7[dec], color=ISOTOPES_COLOR['Be10'], style="EP", xlog=True, ylog=False, scale=None, markersize=22, label='Be10/Be7')
        if dec =="Tof":
            ax1.text(xaxistext[dec], 0.98, f"{nuclei} MC {dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
        else:
            ax1.text(xaxistext[dec], 0.98, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
        #ax2.axhline(y=1.0, color='red', linestyle='--')
        
        #label_iso = []
        #for iso in isotopes:
        #    label_iso.append(mlines.Line2D([], [], marker='o', markersize=22, color=ISOTOPES_COLOR[iso], label=f'{ISO_LABELS[iso]}'))
        #    legend1 = ax1.legend(handles=[label_ for label_ in label_iso], loc='upper left', bbox_to_anchor=(0.0, 1.02), fontsize=20)
    np.savez(os.path.join(plotdir, f"df_ratio_{parname}_{legend}.npz"), **df_sigma_ratio)
    ax1.legend(loc='center right', fontsize=FONTSIZE-2) 
    if parname == "mean" or parname == "sigma":
        ax2.axhline(y=7.0/9.0, color='orange', linestyle='--')
        #ax2.axhline(y=ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be10'], color='green', linestyle='--')
        ax2.axhline(y=7.0/10.0, color='green', linestyle='--')
        label_ref9 = mlines.Line2D([], [], linestyle='--', color=ISOTOPES_COLOR['Be9'], label='ref m7/m9')
        label_ref10 = mlines.Line2D([], [], linestyle='--', color=ISOTOPES_COLOR['Be10'], label='ref m7/m10')
        legend2 = ax2.legend(handles=[label_ref9, label_ref10], loc='upper left', bbox_to_anchor=(0.05, 1.02), fontsize=20)
        ax2.add_artist(legend2)

    if parname == "mean":
        label_com9 = mlines.Line2D([], [], linestyle='--', marker = "o",  markersize= 14, color=ISOTOPES_COLOR['Be9'], label=r"$\mathrm{\mu_{9}}$/$\mathrm{\mu_{7}}$")
        label_com10 = mlines.Line2D([], [], linestyle='--', marker = 'o', markersize= 14, color=ISOTOPES_COLOR['Be10'], label=r"$\mathrm{\mu_{10}}$/$\mathrm{\mu_{7}}$")   
        legend3 = ax2.legend(handles=[label_com9, label_com10], loc='upper right', bbox_to_anchor=(0.9, 1.02), fontsize=20)   
        ax2.add_artist(legend3)

    if parname == "sigma":
        label_com9 = mlines.Line2D([], [],  marker = "o",  markersize= 12, color=ISOTOPES_COLOR['Be9'], label=r"$\mathrm{\sigma_{9}}$/$\mathrm{\sigma_{7}}$")
        label_com10 = mlines.Line2D([], [],  marker = 'o', markersize= 12, color=ISOTOPES_COLOR['Be10'], label=r"$\mathrm{\sigma_{10}}$/$\mathrm{\sigma_{7}}$")   
        legend3 = ax2.legend(handles=[label_com9, label_com10], loc='upper right', bbox_to_anchor=(0.9, 1.02), fontsize=20)   
        ax2.add_artist(legend3)

    ax1.fill_betweenx(np.linspace(0.05, 0.3, 100), 1.0, 4.0, alpha=0.1, color="tab:blue")  
    ax1.set_ylabel(ylabel[parname])
    ax1.set_ylim(yrange[parname])
    ax1.legend(loc="lower right", fontsize=28)
    plt.subplots_adjust(hspace=.0)
    ax1.set_xticklabels([])
    set_plot_defaultstyle(ax1)
    ax2.set_xlabel('Ekin/N(GeV/n)')
    ax2.set_ylabel("ratio")
    ax2.grid(axis='y')
    ax1.set_xticks(xtick_dec[dec]) 
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()) 
    ax2.set_ylim(ylim_compare[parname])
    set_plot_defaultstyle(ax2)
    savefig_tofile(fig, plotdir, f'{nuclei}fit_{parname}_compare_mc_{legend}', show=True)



def plot_fitmc_compare_isopars_rigreso(parname, detectors, dict_fit_results, massfit, plotdir, poly_initp0, ylabel, yrange, nuclei, isotopes, legend=None):
    xaxistext = {"Tof": 0.03, "NaF": 0.33, "Agl": 0.75}
    ylim_compare = {"mean":[0.9, 1.1], "sigma": [0.9, 1.1], "fraccore": [0.9, 1.1], "sigma_ratio": [0.9, 1.1], "asy_factor": [0.9, 1.1]}
    ratio_be9be7 = dict()
    ratio_be10be7 = dict()
    graph_init = {dec: {} for dec in detectors}
    xtick_dec = {"Tof": [0.4, 0.6, 1.0], "NaF": [1.0, 2.0, 4.0], "Agl":[4.0, 6.0, 10.0]}
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(24, 16)) 
    fig.subplots_adjust(left= 0.12, right=0.97, bottom=0.08, top=0.95)
    df_sigma_ratio = dict()
    df_sigma = dict()
    for dec in detectors:
        polydeg = len(poly_initp0[dec][parname])
        for iso in isotopes: 
            y_init, y_init_err = get_unp_pars_binbybin(parname, dict_fit_results[iso][dec], massfit[iso][dec].num_energybin)
            graph_init[dec][iso] = MGraph(massfit[iso][dec].x_fit_energy, y_init, y_init_err)
            if dec == "Tof":
                plot_graph(fig, ax1, graph_init[dec][iso], color=ISOTOPES_COLOR[iso],  style="EP", xlog=True, ylog=False, scale=None, markersize=22, label=f'{ISO_LABELS[iso]}')
            else:
                plot_graph(fig, ax1, graph_init[dec][iso], color=ISOTOPES_COLOR[iso],  style="EP", xlog=True, ylog=False, scale=None, markersize=22)


            graph_init[dec][iso].add_to_file(df_sigma, f'graph_sigma_{dec}_{iso}')
            
            xval = graph_init[dec][iso].xvalues
            popt, pcov = curve_fit(poly, np.log(xval), graph_init[dec][iso].yvalues, p0 = 1 / (10 ** np.arange(1, polydeg+1)), sigma=graph_init[dec][iso].yerrs, absolute_sigma=True)
            polypars = uncertainties.correlated_values(popt, pcov)
            yfit, yfit_err = get_fitpdf_witherr(np.log(xval), polypars, upoly)  
            yfit_lower, yfit_upper = get_fitpdferrorband(np.log(xval), polypars, upoly)
            ax1.plot(xval, yfit, "--",  color=ISOTOPES_COLOR[iso])
            ax1.fill_between(xval, yfit_lower, yfit_upper, color=ISOTOPES_COLOR[iso], alpha=0.3)

        
        ratio_be9be7[dec] = graph_init[dec]["Be9"]/graph_init[dec]["Be7"]
        ratio_be10be7[dec] = graph_init[dec]["Be10"]/graph_init[dec]["Be7"]
        ratio_be9be7[dec].add_to_file(df_sigma_ratio, f'graph_mass_sigma_9to7_{dec}')
        ratio_be10be7[dec].add_to_file(df_sigma_ratio, f'graph_mass_sigma_10to7_{dec}')
    
        plot_graph(fig, ax2, ratio_be9be7[dec], color=ISOTOPES_COLOR['Be9'], style="EP", xlog=True, ylog=False, scale=None, markersize=22, label='Be9/Be7')
        plot_graph(fig, ax2, ratio_be10be7[dec], color=ISOTOPES_COLOR['Be10'], style="EP", xlog=True, ylog=False, scale=None, markersize=22, label='Be10/Be7')
        #ax1.text(xaxistext[dec], 0.88, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")        
        #ax2.axhline(y=1.0, color='red', linestyle='--')
        
        #label_iso = []
        #for iso in isotopes:
        #    label_iso.append(mlines.Line2D([], [], marker='o', markersize=22, color=ISOTOPES_COLOR[iso], label=f'{ISO_LABELS[iso]}'))
        #    legend1 = ax1.legend(handles=[label_ for label_ in label_iso], loc='upper left', bbox_to_anchor=(0.0, 1.02), fontsize=20)
    np.savez(os.path.join(plotdir, f"df_rigreso_{parname}_ratio.npz"), **df_sigma_ratio)
    np.savez(os.path.join(plotdir, f"df_reso_{parname}.npz"), **df_sigma)
    ax1.legend(loc='center right', fontsize=FONTSIZE-2)  
    if parname == "mean":
        label_com9 = mlines.Line2D([], [], linestyle='--', marker = "o",  markersize= 12, color=ISOTOPES_COLOR['Be9'], label=r"$\mathrm{\mu_{7}}$/$\mathrm{\mu_{9}}$")
        label_com10 = mlines.Line2D([], [], linestyle='--', marker = 'o', markersize= 12, color=ISOTOPES_COLOR['Be10'], label=r"$\mathrm{\mu_{7}}$/$\mathrm{\mu_{10}}$")   
        legend3 = ax2.legend(handles=[label_com9, label_com10], loc='upper right', bbox_to_anchor=(0.9, 1.02), fontsize=20)   
        ax2.add_artist(legend3)
        
    if parname == "sigma":
        label_com9 = mlines.Line2D([], [],  marker = "o",  markersize= 11, color=ISOTOPES_COLOR['Be9'], label=r"$\mathrm{\sigma_{9}}$/$\mathrm{\sigma_{7}}$")
        label_com10 = mlines.Line2D([], [],  marker = 'o', markersize= 11, color=ISOTOPES_COLOR['Be10'], label=r"$\mathrm{\sigma_{10}}$/$\mathrm{\sigma_{7}}$")   
        legend3 = ax2.legend(handles=[label_com9, label_com10], loc='upper right', bbox_to_anchor=(0.9, 1.02), fontsize=20)   
        ax2.add_artist(legend3)

    ax1.set_ylabel(ylabel[parname])
    

    ax1.legend(loc="lower right", fontsize=28)
    ax1.text(0.07, 0.98, f"Be MC", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold") 
    ax1.set_ylim(yrange[parname])
    ax1.fill_betweenx(np.linspace(yrange[parname][0], yrange[parname][1], 100), ANALYSIS_RANGE_RIG['Be']['NaF'][0], ANALYSIS_RANGE_RIG['Be']['NaF'][1], alpha=0.1, color=DETECTOR_COLOR['NaF'])  
    plt.subplots_adjust(hspace=.0)
    #ax1.get_yticklabels()[0].set_visible(False)
    ax1.set_xticklabels([])
    set_plot_defaultstyle(ax1)
    ax2.set_xlabel('Rigidity(GeV)')
    ax2.set_ylabel("ratio")
    ax2.grid(axis='y')
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()) 
    ax2.set_ylim(ylim_compare[parname])
    set_plot_defaultstyle(ax2)
    savefig_tofile(fig, plotdir, f'{nuclei}fit_{parname}_compare_mc_{legend}', show=True)


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



def plot_fitdata_pars(parname, detectors, massfit, fit_parameters, polypars_initp0, ylabel, yrange, nuclei, plotdir, guess_initial, plot_mc=False, plot_guess=True, figname=None, fitparslim=None):
    parname_s = {'Tof': {'mean': 'mua', 'sigma':'siga' , 'fraccore': 'fraccore', 'asy_factor': 'asy_factor_a', 'sigma_ratio': 'sigma_ratio_a'},
                 'NaF': {'mean': 'mua', 'sigma':'siga' , 'fraccore': 'fraccore', 'asy_factor': 'asy_factor', 'sigma_ratio': 'sigma_ratio'},
                 'Agl': {'mean': 'mua', 'sigma':'siga' , 'fraccore': 'fraccore_a', 'asy_factor': 'asy_factor_a', 'sigma_ratio': 'sigma_ratio_a'}}
    xaxistext = {"Tof": 0.03, "NaF": 0.33, "Agl": 0.64}
    fig, ax1 = plt.subplots(1, figsize=(20, 12))
    fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)
    df_fitdata = {}
    
    
    for dec in detectors:
        par_index = massfit[dec].get_polypars_index()
        pindex_s = par_index[parname_s[dec][parname]]
        nparpoly = len(polypars_initp0[dec][parname])
        pindex_e = pindex_s + nparpoly 
        #poly_par = poly(np.log(massfit[dec].x_fit_energy), guess_initial[dec]['siga'], guess_initial[dec]['sigb'], guess_initial[dec]['sigc'])    
        #y_init, y_init_err = get_unp_pars_binbybin(parname, dict_fit_results_be7[dec], massfit[dec].num_energybin)
        #graph_parinit = MGraph(massfit[dec].x_fit_energy, y_init, y_init_err)
        #plot_graph(fig, ax1, graph_parinit, color=DETECTOR_COLOR[dec],  style="EP", xlog=True, ylog=False, scale=None, markersize=20, label="MC")

        guess_initial_values = np.array(list(guess_initial[dec].values()))
        xval = massfit[dec].x_fit_energy
        yguess = poly(np.log(xval), *guess_initial_values[pindex_s:pindex_e])
        if plot_guess:
            ax1.plot(xval, yguess, '--', color='black', label=f'{dec} guess')

        if fitparslim is not None:
            graph_low = MGraph.from_file(fitparslim, f'graph_{parname}low_{dec}') 
            graph_up = MGraph.from_file(fitparslim, f'graph_{parname}up_{dec}')
            ax1.fill_between(graph_low.xvalues, graph_low.yvalues, graph_up.yvalues, color='grey', alpha=0.2)
        
        #yinit, yinit_err = get_fitpdf_witherr(np.log(xval), init_parameters[dec][pindex_s:pindex_e], upoly)
        #yinit_lower, yinit_upper = get_fitpdferrorband(np.log(xval), init_parameters[dec][pindex_s:pindex_e], upoly)
        #if plot_mc:
        #    ax1.plot(xval, yinit, '--', color='black', label=f'{dec} init')
        #    ax1.fill_between(xval, yinit_lower, yinit_upper, color='grey', alpha=0.1)
        yfit, yfit_err = get_fitpdf_witherr(np.log(xval), fit_parameters[dec][pindex_s:pindex_e], upoly)      
        ax1.plot(xval, yfit, '-', color=DETECTOR_COLOR[dec], label=f'{dec} fit (Data)')
        yfit_lower, yfit_upper = get_fitpdferrorband(np.log(xval), fit_parameters[dec][pindex_s:pindex_e], upoly)
        ax1.fill_between(xval, yfit_lower, yfit_upper, color=DETECTOR_COLOR[dec], alpha=0.3)
        ax1.set_ylabel(ylabel[parname])
        ax1.set_xlabel('Ekin/N(GeV/n)')
        ax1.set_ylim(yrange[parname])
        ax1.legend(loc="lower right", fontsize=30)
        ax1.text(xaxistext[dec], 0.98, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
        for ideg in range(nparpoly):
            p_text = r'$\mathrm{{p_{{{}}} = {:.4f} \pm {:.4f}}}$'.format(ideg, unumpy.nominal_values(fit_parameters[dec])[pindex_s+ ideg], unumpy.std_devs(fit_parameters[dec])[pindex_s + ideg])  
            ax1.text(xaxistext[dec], 0.9-0.06* ideg, p_text, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='black')


    if figname is not None:
        savefig_tofile(fig, plotdir, f"{nuclei}_{parname}_{figname}", show=True)
    else:
        savefig_tofile(fig, plotdir, f"{nuclei}_fitData_{parname}", show=True)


def update_guess_with_polyfit(guess, graph_poly_par, num_bins):
    for name, value in par_names.items():
        for ibin in range(num_bins):
            guess[f'{name}_{ibin}']  = graph_poly_par[name].yvalues[ibin]
            return guess

def get_limit_pars_binbybinfit(lim_pars, par_name, num_bins, limrange):
    for ibin in range(num_bins): 
        lim_pars[f'{par_name}_{ibin}'] = limrange 
    return lim_pars



def update_guess_simultaneousfit_pvalues(polyfit_pars, guess):
    for dec in detectors:
        combined_initial_par_array = np.concatenate(list(polyfit_pars[dec].values()))
        keys = list(guess[dec].keys())
        for i in range(len(list(keys))):
            guess[dec][keys[i]] = combined_initial_par_array[i]

def update_guess(guess, fitpars):
    for i, key in enumerate(guess.keys()):
        guess[key] = fitpars[key]['value']
    return guess

def update_guess_with_polyfit(guess, graph_poly_par, num_bins):
    for name, value in par_names.items():
        for ibin in range(num_bins):
            guess[f'{name}_{ibin}']  = graph_poly_par[name].yvalues[ibin]
    return guess

def update_guess_simultaneousfit_pvalues(polyfit_pars, guess):
    for dec in detectors:
        combined_initial_par_array = np.concatenate(list(polyfit_pars[dec].values()))
        keys = list(guess[dec].keys())
        for i in range(len(list(keys))):
            guess[dec][keys[i]] = combined_initial_par_array[i]

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



def plot_fitMCISO_pars(parname, detectors, massfit, fit_parameters, guess_initial, dict_fit_results_be7, polypars_initp0, ylabel, yrange, nuclei, isotopes, plotdir, plot_mc=True):
    parname_s = {'Tof': {'mean': 'mua', 'sigma':'siga' , 'fraccore': 'fraccore', 'asy_factor': 'asy_factor_a', 'sigma_ratio': 'sigma_ratio_a'},
                 'NaF': {'mean': 'mua', 'sigma':'siga' , 'fraccore': 'fraccore', 'asy_factor': 'asy_factor_a', 'sigma_ratio': 'sigma_ratio'},
                 'Agl': {'mean': 'mua', 'sigma':'siga' , 'fraccore': 'fraccore_a', 'asy_factor': 'asy_factor_a', 'sigma_ratio': 'sigma_ratio_a'}}
    fig, ax1 = plt.subplots(1, figsize=(20, 12))
    fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)
    for iso in isotopes:
        for dec in detectors:
            par_index = massfit[iso][dec].get_polypars_index()
            pindex_s = par_index[parname_s[dec][parname]]
            nparpoly = len(polypars_initp0[dec][parname])
            pindex_e = pindex_s + nparpoly 
            poly_par = poly(np.log(massfit[iso][dec].x_fit_energy), guess_initial[dec]['siga'], guess_initial[dec]['sigb'], guess_initial[dec]['sigc'])
            
            y_init, y_init_err = get_unp_pars_binbybin(parname, dict_fit_results_be7[dec], massfit[iso][dec].num_energybin)
            graph_parinit = MGraph(massfit[iso][dec].x_fit_energy, y_init, y_init_err)
            if plot_mc:
                plot_graph(fig, ax1, graph_parinit, color=DETECTOR_COLOR[dec],  style="EP", xlog=True, ylog=False, scale=None, markersize=22, label="MC")

            xval = massfit[iso][dec].x_fit_energy
            yfit, yfit_err = get_fitpdf_witherr(np.log(xval), fit_parameters[iso][dec][pindex_s:pindex_e], upoly)      
            ax1.plot(xval, yfit, '-', color="black", label=f'{dec} fit(Data)')
            yfit_lower, yfit_upper = get_fitpdferrorband(np.log(xval), fit_parameters[iso][dec][pindex_s:pindex_e], upoly)
            ax1.fill_between(xval, yfit_lower, yfit_upper, color='blue', alpha=0.3, label='Error band')
            ax1.set_ylabel(ylabel[parname])
            ax1.set_xlabel('Ekin/N(GeV/n)')
            ax1.set_ylim(yrange[parname])
            ax1.legend(loc="lower right", fontsize=30)
            
            for ideg in range(nparpoly):
                p_text = r'$\mathrm{{p_{{{}}} = {:.4f} \pm {:.4f}}}$'.format(ideg, unumpy.nominal_values(fit_parameters[iso][dec])[pindex_s+ ideg], unumpy.std_devs(fit_parameters[iso][dec])[pindex_s + ideg])  
                ax1.text(0.03, 0.9-0.08* ideg, p_text, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='black')            
    savefig_tofile(fig, plotdir, f"{nuclei}_fitdata_{parname}", show=True)



def plot_fitmc_compare_isopars_betareso(parname, detectors, dict_fit_results, massfit, plotdir, poly_initp0, ylabel, yrange, nuclei, isotopes, legend=None):
    xaxistext = {"Tof": 0.03, "NaF": 0.33, "Agl": 0.75}
    ylim_compare = {"mean":[0.9, 1.1], "sigma": [0.9, 1.1], "fraccore": [0.9, 1.1], "sigma_ratio": [0.9, 1.1], "asy_factor": [0.9, 1.1]}
    ratio_be9be7 = dict()
    ratio_be10be7 = dict()
    graph_init = {dec: {} for dec in detectors}
    xtick_dec = {"Tof": [0.4, 0.6, 1.0], "NaF": [1.0, 2.0, 4.0], "Agl":[4.0, 6.0, 10.0]}
    df_sigma_ratio = dict()
    df_sigma = dict()
    for dec in detectors:
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(24, 16)) 
        fig.subplots_adjust(left= 0.12, right=0.97, bottom=0.08, top=0.95)
        polydeg = len(poly_initp0[dec][parname])
        for iso in isotopes: 
            y_init, y_init_err = get_unp_pars_binbybin(parname, dict_fit_results[iso][dec], massfit[iso][dec].num_energybin)
            graph_init[dec][iso] = MGraph(massfit[iso][dec].x_fit_energy, y_init, y_init_err)
            if dec == "Tof":
                plot_graph(fig, ax1, graph_init[dec][iso], color=ISOTOPES_COLOR[iso],  style="EP", xlog=True, ylog=False, scale=None, markersize=22, label=f'{ISO_LABELS[iso]}')
            else:
                plot_graph(fig, ax1, graph_init[dec][iso], color=ISOTOPES_COLOR[iso],  style="EP", xlog=True, ylog=False, scale=None, markersize=22)


            graph_init[dec][iso].add_to_file(df_sigma, f'graph_sigma_{dec}_{iso}')
            
            xval = graph_init[dec][iso].xvalues
            popt, pcov = curve_fit(poly, np.log(xval), graph_init[dec][iso].yvalues, p0 = 1 / (10 ** np.arange(1, polydeg+1)), sigma=graph_init[dec][iso].yerrs, absolute_sigma=True)
            polypars = uncertainties.correlated_values(popt, pcov)
            yfit, yfit_err = get_fitpdf_witherr(np.log(xval), polypars, upoly)  
            yfit_lower, yfit_upper = get_fitpdferrorband(np.log(xval), polypars, upoly)
            ax1.plot(xval, yfit, "--",  color=ISOTOPES_COLOR[iso])
            ax1.fill_between(xval, yfit_lower, yfit_upper, color=ISOTOPES_COLOR[iso], alpha=0.3)

        
        ratio_be9be7[dec] = graph_init[dec]["Be9"]/graph_init[dec]["Be7"]
        ratio_be10be7[dec] = graph_init[dec]["Be10"]/graph_init[dec]["Be7"]
        ratio_be9be7[dec].add_to_file(df_sigma_ratio, f'graph_mass_sigma_9to7_{dec}')
        ratio_be10be7[dec].add_to_file(df_sigma_ratio, f'graph_mass_sigma_10to7_{dec}')
    
        plot_graph(fig, ax2, ratio_be9be7[dec], color=ISOTOPES_COLOR['Be9'], style="EP", xlog=True, ylog=False, scale=None, markersize=22, label='Be9/Be7')
        plot_graph(fig, ax2, ratio_be10be7[dec], color=ISOTOPES_COLOR['Be10'], style="EP", xlog=True, ylog=False, scale=None, markersize=22, label='Be10/Be7')
        #ax1.text(xaxistext[dec], 0.88, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")        
        #ax2.axhline(y=1.0, color='red', linestyle='--')
        
        #label_iso = []
        #for iso in isotopes:
        #    label_iso.append(mlines.Line2D([], [], marker='o', markersize=22, color=ISOTOPES_COLOR[iso], label=f'{ISO_LABELS[iso]}'))
        #    legend1 = ax1.legend(handles=[label_ for label_ in label_iso], loc='upper left', bbox_to_anchor=(0.0, 1.02), fontsize=20)
        ax1.legend(loc='center right', fontsize=FONTSIZE-2)  
        if parname == "mean":
            label_com9 = mlines.Line2D([], [], linestyle='--', marker = "o",  markersize= 12, color=ISOTOPES_COLOR['Be9'], label=r"$\mathrm{\mu_{7}}$/$\mathrm{\mu_{9}}$")
            label_com10 = mlines.Line2D([], [], linestyle='--', marker = 'o', markersize= 12, color=ISOTOPES_COLOR['Be10'], label=r"$\mathrm{\mu_{7}}$/$\mathrm{\mu_{10}}$")   
            legend3 = ax2.legend(handles=[label_com9, label_com10], loc='upper right', bbox_to_anchor=(0.9, 1.02), fontsize=20)   
            ax2.add_artist(legend3)

        if parname == "sigma":
            label_com9 = mlines.Line2D([], [],  marker = "o",  markersize= 11, color=ISOTOPES_COLOR['Be9'], label=r"$\mathrm{\sigma_{9}}$/$\mathrm{\sigma_{7}}$")
            label_com10 = mlines.Line2D([], [],  marker = 'o', markersize= 11, color=ISOTOPES_COLOR['Be10'], label=r"$\mathrm{\sigma_{10}}$/$\mathrm{\sigma_{7}}$")   
            legend3 = ax2.legend(handles=[label_com9, label_com10], loc='upper right', bbox_to_anchor=(0.9, 1.02), fontsize=20)   
            ax2.add_artist(legend3)
            
        ax1.set_ylabel(ylabel[parname])
        ax1.set_ylim(yrange[dec][parname])      

        ax1.legend(loc="lower right", fontsize=28)
        ax1.text(0.07, 0.98, f"Be MC", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold") 

        plt.subplots_adjust(hspace=.0)
        #ax1.get_yticklabels()[0].set_visible(False)
        ax1.set_xticklabels([])
        set_plot_defaultstyle(ax1)
        ax2.set_xlabel('Rigidity(GeV)')
        ax2.set_ylabel("ratio")
        ax2.grid(axis='y')
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()) 
        ax2.set_ylim(ylim_compare[parname])
        set_plot_defaultstyle(ax2)
        savefig_tofile(fig, plotdir, f'{nuclei}fit_{parname}_compare_mc_{legend}_{dec}', show=True)
    np.savez(os.path.join(plotdir, f"df_betareso_{parname}_isoratio.npz"), **df_sigma_ratio)
    np.savez(os.path.join(plotdir, f"df_betareso_{parname}.npz"), **df_sigma)


def plot_fitmc_pars_betareso(parname, detectors, dict_fit_results_be7, massfit, poly_initp0, plotdir, ylabel, yrange, nuclei, isotope, figname=None, legend=None):
    xaxistext = {"Tof": 0.03, "NaF": 0.33, "Agl": 0.75}
    for dec in detectors:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios':[0.4, 0.3, 0.3]}, figsize=(26, 16)) 
        fig.subplots_adjust(left= 0.12, right=0.97, bottom=0.08, top=0.95)
        print(dec, parname, poly_initp0[dec][parname])
        polydeg = len(poly_initp0[dec][parname])
        y_init, y_init_err = get_unp_pars_binbybin(parname, dict_fit_results_be7[dec], massfit[dec].num_energybin)
        graph_init = MGraph(massfit[dec].x_fit_energy, y_init, y_init_err)
        plot_graph(fig, ax1, graph_init, color=DETECTOR_COLOR[dec],  style="EP", xlog=True, ylog=False, scale=None, markersize=22)
        xval = np.log(graph_init.xvalues)
        #popt, pcov = curve_fit(poly, xval, graph_init.yvalues, p0 = np.ones(polydeg)*0.1, sigma=graph_init.yerrs, absolute_sigma=True)
        popt, pcov = curve_fit(poly, xval, graph_init.yvalues, p0 = np.ones(polydeg)*0.1)
        polypars = uncertainties.correlated_values(popt, pcov)
        yfit, yfit_err = get_fitpdf_witherr(xval, polypars, upoly)  
        yfit_lower, yfit_upper = get_fitpdferrorband(xval, polypars, upoly)
        if dec == "Tof":
            ax1.plot(massfit[dec].x_fit_energy, yfit, "--",  color='black', label='poly fit')
        else:
            ax1.plot(massfit[dec].x_fit_energy, yfit, "--",  color='black')
        ax1.fill_between(massfit[dec].x_fit_energy, yfit_lower, yfit_upper, color='green', alpha=0.3)
        ax1.set_ylabel(ylabel[parname])
        ax1.set_xlabel(r'$\mathrm{\gamma ~ (1/\sqrt{(1-\beta^{2})})}$')
        ax1.set_ylim(yrange[dec][parname])
        ax1.legend(loc="lower right", fontsize=28)
        if  dec == "Tof":  
            ax1.text(xaxistext[dec], 0.98, f"{isotope} MC {dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
        else:
            ax1.text(xaxistext[dec], 0.98, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
        # Calculate the pull
        delta = yfit - y_init
        pull = delta / y_init_err
        graph_pull = MGraph(massfit[dec].x_fit_energy,  pull, np.zeros_like(pull))
        plot_graph(fig, ax2, graph_pull, color="black", label=r'$\mathrm{\frac{y_{fit} - y_{d}}{\sqrt{e^{2}_{fit} + e^{2}_{d}}}}$', style="EP", xlog=True, ylog=False, scale=None, markersize=20)
        if parname == 'mean':
            ax1.grid(axis='y')
            ax1.axhline(y=1/ISOTOPES_MASS[isotope], color='red', linestyle='--', linewidth=3)
        # Calculate the ratio
        ratio_par = yfit / np.array(y_init)
        ratio_par_error = calc_ratio_err(yfit, np.array(y_init), yfit_err, np.array(y_init_err))
        graph_ratio = MGraph(massfit[dec].x_fit_energy,  ratio_par, ratio_par_error)
        
        plot_graph(fig, ax3, graph_ratio, color="black", label='fit/data', style="EP", xlog=True, ylog=False, scale=None, markersize=20) 

        equation_text_3deg = r'$\mathrm{\sigma_{p} = p_{0} + p_{1} \cdot x + p_{2} \cdot x^{2} + p_{3} \cdot x^{3}}$'
        equation_text_2deg = r'$\mathrm{\sigma_{p} = p_{0} + p_{1} \cdot x + p_{2} \cdot x^{2}}$'
        equation_text_1deg = r'$\mathrm{\sigma_{p} = p_{0} + p_{1} \cdot x }$'
        #equation_text = 
        #ax1.text(0.03, 0.9, equation_text_3deg, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='black')            
        for deg in range(polydeg):
            p_text = r'$\mathrm{{p_{{{}}} = {:.4f} \pm {:.4f}}}$'.format(deg, popt[deg], unumpy.std_devs(polypars)[deg])  
            ax1.text(xaxistext[dec], 0.88-0.07*deg, p_text, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='black')            

        plt.subplots_adjust(hspace=.0)
        #ax1.get_yticklabels()[0].set_visible(False)
        ax1.set_xticklabels([])
        set_plot_defaultstyle(ax1)
        ax2.set_ylabel("pull")
        ax2.set_ylim([-5, 5])
        ax3.set_ylim([0.95, 1.05])
        ax2.grid(axis='y')
        ax2.set_xscale('log')        
        set_plot_defaultstyle(ax2)
        #ax2.get_yticklabels()[0].set_visible(False)
        ax2.set_xticklabels([])
        ax3.set_xlabel(r'$\mathrm{\gamma ~ (1/\sqrt{(1-\beta^{2})})}$')
        ax3.set_ylabel("ratio")
        ax3.set_xscale('log')
        ax3.grid(axis='y')
        set_plot_defaultstyle(ax3)
        if figname is not None:
            savefig_tofile(fig, plotdir, f"{nuclei}fit_{parname}_mc_{polydeg}deg_{legend}_{figname}_{dec}", show=True)
        else:
            savefig_tofile(fig, plotdir, f"{nuclei}fit_{parname}_mc_{polydeg}deg_{legend}_{dec}", show=True)
