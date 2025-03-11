import multiprocessing as mp
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import uproot
import uproot3 
from tools.roottree import read_tree
from tools.selections import *
import scipy.stats
from scipy.optimize import curve_fit
from tools.studybeta import hist1d, hist2d, hist_beta, getbeta, hist_betabias, compute_moment   
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, FIGSIZE_MID, FIGSIZE_BIG, setplot_defaultstyle, format_order_of_magnitude, FONTSIZE, savefig_tofile, FONTSIZE_BIG, plot1d_errorbar_v2, plot1d_step, FONTSIZE_MID, set_plot_defaultstyle
from tools.studybeta import calc_signal_fraction, hist1d, hist1d_weighted
from tools.binnings_collection import  fbinning_energy
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, cdf_double_gaus, double_gaus
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.binnings import Binning
from tools.constants import NUCLEI_CHARGE, ANALYSIS_RANGE_EKIN, ISOTOPES_MASS, ISOTOPES_COLOR, ISO_LABELS
from tools.calculator import calc_ekin_from_beta
from tools.calculator import calculate_efficiency_and_error, calculate_efficiency_and_error_weighted, calculate_efficiency_weighted
from tools.statistics import poly_func
from tools.graphs import MGraph, slice_graph, concatenate_graphs, plot_graph, slice_graph_by_value, compute_pull_graphs
import ROOT
from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F
from iminuit import Minuit     
from iminuit.cost import ExtendedBinnedNLL, LeastSquares, NormalConstraint, ExtendedUnbinnedNLL
from iminuit.util import describe, make_func_code 
from scipy.interpolate import UnivariateSpline
import pickle
from tools.utilities import save_spline_to_file
from tools.functions import normalized_gaussian, cumulative_norm_gaus, cumulative_normalize_exp_modified_gaussian, normalize_exp_modified_gaussian
from tools.functions import gaussian, asy_gaussian, poly, asy_gaussian_1d
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.interpolate import make_interp_spline
import matplotlib.lines as mlines

#xbinning = fbinning_energy_agl()
setplot_defaultstyle()

kNucleiBinsRebin = np.array([0.8,1.00,1.16,1.33,1.51,1.71,1.92,2.15,2.40,2.67,2.97,3.29,3.64,4.02,4.43,4.88, 5.37,5.90,6.47,7.09,7.76,8.48,9.26, 10.1,11.0,12.0,13.0,14.1,15.3,16.6,18.0,19.5,21.1,22.8,24.7,26.7,28.8,31.1,33.5,36.1, 38.9, 41.9,45.1,48.5,52.2,60.3,69.7,80.5,93.0,108., 116.,147.,192.,259.,379.,660., 1300, 3300.])
kNucleiBinsRebin_center = get_bin_center(kNucleiBinsRebin)
kNucleiNbinRebin = 57

def unorm_gaussian(x, counts, mu, sigma):   
    return counts * np.exp(-(x - mu)**2 /(2 * sigma**2))  

def expo_func(x, pa, pb, pc):
    pdf = pa* (1 - np.exp((x-pb)/pc))
    return pdf


def make_charge_template_fit(nT, templates_par):
    def template_fit(x, *pars):
        #assert(len(pars) == len(templates))
        #assert(len(x) == len(templates[0]))
        pdf = np.zeros(x.shape)
        norm = pars[0:3]
        mu = pars[3:6]
        for i, ipar in enumerate(norm):
            pdf += norm[i] * (gaus_asygaus(x, mu[i], *(templates_par[i][1:]))) 
        return pdf
    parnames = ['x'] +  [f"T_{i}" for i in range(nT)] + [f"mu_{i}" for i in range(nT)]
    template_fit.func_code = make_func_code(parnames)
    return template_fit


def draw_gaussian(x, mean, sigma, norm):
    pdf = norm * gaussian(x, mean, sigma)
    return pdf


def gaus_asygaus(x, mean, sigma, sigma_ratio, asy_factor, fraccore, norm):
    coregaus = gaussian(x, mean, sigma)
    asygaus = asy_gaussian_1d(x, mean,  sigma_ratio * sigma, asy_factor)
    pdf = norm * (fraccore * coregaus + (1 - fraccore) * asygaus)
    return pdf


def cumulative_gaus(edges, mean, sigma, norm):
    x = (edges[1:] + edges[:-1])/2
    pdf = norm * gaussian(x, mean, sigma)
    cpdf = np.cumsum(pdf)
    return np.concatenate(([0], cpdf))


def cumulative_gaus_asygaus(edges, mean, sigma, sigma_ratio, asy_factor, fraccore, norm):
    x = (edges[1:] + edges[:-1])/2
    pdf = gaus_asygaus(x, mean, sigma, sigma_ratio, asy_factor, fraccore, norm)
    cpdf = np.cumsum(pdf)
    return np.concatenate(([0], cpdf))

def gaus_asygaus_part1(x, mean, sigma, sigma_ratio, asy_factor, fraccore, norm):
    coregaus = gaussian(x, mean, sigma)
    asygaus = asy_gaussian_1d(x, mean,  sigma_ratio * sigma, asy_factor)
    pdf = norm * (1 - fraccore) * asygaus
    return pdf

def gaus_asygaus_part2(x, mean, sigma, sigma_ratio, asy_factor, fraccore, norm):
    coregaus = gaussian(x, mean, sigma)
    asygaus = asy_gaussian_1d(x, mean,  sigma_ratio * sigma, asy_factor)
    pdf = norm * fraccore * coregaus
    return pdf


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    #parser.add_argument("--filename", default="/home/manbing/Documents/Data/data_mc/dfile/BeMC_histmass_Ekin_B1220rwth_rawmc_v3.npz",   help="Path to root file to read tree from")
    parser.add_argument("--filename", default="/home/manbing/Documents/Data/data_BeP8/BeMC_masshist_Ekin_B1236dst_rawmc_finebin_v6.npz",   help="Path to root file to read tree from")
    parser.add_argument("--resultdir", default="/home/manbing/Documents/Data/data_mc/dfile", help="Directory to store plots and result files in.")
    parser.add_argument("--plotdir", default="/home/manbing/Documents/lithiumanalysis/scripts/plots/massfit/fitmassgaus", help="Directory to store plots and result files in.")    
    parser.add_argument("--nuclei", default="Be", help="Directory to store plots and result files in.")
    parser.add_argument("--detectors", nargs="+", default=["RichAgl"], help="Directory to store plots and result files in.")
    parser.add_argument("--iteration", default=3, help="the iteration times for the fit")
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    
    
    nuclei = args.nuclei
    iteration = args.iteration
    templates_name = ["Be7", "Be9", "Be10"]
    tem_color = ["tab:blue", "tab:red", "tab:green"]
    xbinning = Binning(fbinning_energy())

    #detectors = ["Tof"]
    detectors = ["NaF"]

    #fitminbin = fit_mass_range[0]
    #fitmaxbin = fit_mass_range[1]

    #read mc data:
    hist2d_mass_energy = {dec: {} for dec in detectors}
    with np.load(args.filename) as massfile:
        for dec in detectors:
            for iso in ISOTOPES[nuclei]:
                hist2d_mass_energy[dec][iso] = WeightedHistogram.from_file(massfile, f"{iso}MC_{dec}_mass")
                fig = plt.figure(figsize=(20, 15))
                plot = fig.subplots(1, 1)
                #plot_histogram_2d(plot, hist2d_mass_energy[dec][iso], scale=None, transpose=False, show_overflow=True, show_overflow_x=None, show_overflow_y=None, label=None)
                plot2dhist(fig, plot, xbinning=hist2d_mass_energy[dec][iso].binnings[0].edges[1:-1], ybinning=hist2d_mass_energy[dec][iso].binnings[1].edges[1:-1], counts=hist2d_mass_energy[dec][iso].values[1:-1, 1:-1], xlabel=None, ylabel=None, zlabel="counts", zmin=None, zmax=None, setlogx=False, setlogy=False, setscilabelx=True, setscilabely=True,  setlogz=True)
                plot.text(0.05, 0.98, f"{dec}-{iso}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", fontweight="bold")
                plot.set_ylabel("1/mass(1/GeV)", fontsize=30)
                plot.set_xlabel("Ekin/n(GeV/n)", fontsize=30)
                savefig_tofile(fig, args.plotdir, f"hist2d_mass_{iso}_{dec}", show=False)
    #definations 
    par_names = ['mean', 'sigma', 'norm']
    par_names_axes = ['$\mathrm{\mu}$', '$\mathrm{\sigma_{p}}$', '$\mathrm{ \epsilon(\sigma ratio)}$', 'alpha', '$\mathrm{f_{core}}$', 'Norm']  
    poly_deg = {'mean': 3, 'sigma':3, 'sigma_ratio':3, 'asy_factor':0, 'fraccore': 1, 'norm':6}
    dict_pars = {dec: {iso: {par: MGraph(binning=xbinning, labels=["Ekin(GeV/n)", f"{par}"]) for par in par_names} for iso in ISOTOPES[nuclei]} for dec in detectors}
    chisquare = {iso: np.zeros(kNucleiNbinRebin)  for iso in ISOTOPES[nuclei]}

    fit_mass_range = dict()
    for iso in ISOTOPES[nuclei]:
        mass_ingev = ISOTOPES_MASS[iso]
        fit_mass_range[iso] = [1/(mass_ingev*1.2), 1/(mass_ingev * 0.87)]   
    
    for dec in detectors:
        energy_range = ANALYSIS_RANGE_EKIN[dec]
        mass_range = {"Be7": [0.06, 0.23], "Be9": [0.05, 0.18], "Be10": [0.05, 0.17]}  
        mass_binning = hist2d_mass_energy[dec][iso].binnings[1]
        xvalues = mass_binning.bin_centers[1:-1]
        
        iso_mass_ratio = {"Be7": 1.0, "Be9": 9/7, "Be10": 10/7}
        for iso in ISOTOPES[nuclei]:
            fit_range = mass_binning.get_indices(fit_mass_range[iso])
            bin_range = xbinning.get_indices(energy_range, with_overflow=False)
            for ibin in range(bin_range[0], bin_range[1]+1):
                i_binnum = ibin - bin_range[0]
                hist_ibin = hist2d_mass_energy[dec][iso].project(ibin)
                counts = hist_ibin.values[1:-1][fit_range[0]: fit_range[1]]
                counts_err = np.sqrt(hist_ibin.squared_values[1:-1][fit_range[0]: fit_range[1]])

                guess = dict(mean=1/ISOTOPES_MASS[iso], sigma=0.016/iso_mass_ratio[iso], norm=1000)
                init_ms = [1000, 1/ISOTOPES_MASS[iso], 0.013]
                popt, pcov = curve_fit(unorm_gaussian, mass_binning.bin_centers[1: -1][fit_range[0]:fit_range[1]], counts, p0 = init_ms)   
                guess['mean'] = popt[1]   
                guess['sigma'] = abs(popt[2]) 
                loss = ExtendedBinnedNLL(counts, mass_binning.edges[1: -1][fit_range[0]:fit_range[1]+1], cumulative_gaus)
                #pars_guess = dict()
                #for par in dict_pars[temp].keys():
                #    pars_guess[par] = par_polynomials[par](np.log(kNucleiBinsRebin_center[ibin]))
                m = Minuit(loss, **guess)

                inverse_mass = 1/ISOTOPES_MASS[iso]
                m.limits['mean'] = (inverse_mass*0.95, inverse_mass*1.05)
                m.limits['sigma'] = (0.008, 0.03)
                m.migrad()
                
                for par in dict_pars[dec][iso].keys():  
                    dict_pars[dec][iso][par].yvalues[ibin-1] = m.values[par]
                    dict_pars[dec][iso][par].yerrs[ibin-1] = m.errors[par]
                    
                fit_templates = draw_gaussian(mass_binning.bin_centers[1:-1], *m.values)

                
                fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14)) 
                plot1d_errorbar_v2(fig, ax1, xvalues[fit_range[0]: fit_range[1]], counts=hist_ibin.values[1:-1][fit_range[0]: fit_range[1]], err=np.sqrt(hist_ibin.squared_values[1:-1][fit_range[0]: fit_range[1]]), label_x="1/mass", label_y="counts",  style=".", color="black", setlogy=0, markersize=20)
                plot1d_errorbar_v2(fig, ax1, xvalues, counts=hist_ibin.values[1:-1], err=np.sqrt(hist_ibin.squared_values[1:-1]), label_x="1/mass", label_y="counts",  style=".", color="black", setlogy=0, markersize=20)
                ax1.plot(xvalues[fit_range[0]: fit_range[1]], fit_templates[fit_range[0]: fit_range[1]], "-", color="red", label='fit', linewidth=3)
           
                ax1.text(0.03, 0.98, f"{iso}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')
                ax1.text(0.03, 0.88, f"{iso}: [{xbinning.edges[ibin+1]:.2f}, {xbinning.edges[ibin+2]:.2f}] GeV/n", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')

                # display legend with some fit info
                #fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fmin.reduced_chi2:.1f}",]
                #ax2.plot(xvalues, templates[iT]/fit_templates[iT], "-", color="black", label="ratio")

                #ax1.legend(fontsize=FONTSIZE_BIG)
                set_plot_defaultstyle(ax1)
                set_plot_defaultstyle(ax2)
                ax1.set_ylabel("Counts")

                ax1.set_xticklabels([])
                #ax1.set_yscale('log')
                #ax1.set_ylim([0.1, 8*np.max(fit_templates)])
                ax1.get_yticklabels()[0].set_visible(False)

                pull = (counts - fit_templates[fit_range[0]: fit_range[1]])/np.sqrt(hist_ibin.squared_values[1:-1][fit_range[0]: fit_range[1]])
                pull[np.isinf(pull)] = 0 
                
                chisquare = np.sum(pull**2)/(len(pull) - 3)
                
                fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {chisquare:.1f}", f"$\\mu$ ={m.values['mean']:.4f}$\\pm$ {m.errors['mean']:.4f}",  f"$\\sigma$ ={m.values['sigma']:.4f}$\\pm$ {m.errors['sigma']:.4f}"]
                ax1.legend(title="\n".join(fit_info), frameon=False, fontsize=24, loc='upper right')
                #plot1d_errorbar_v2(fig, ax2, xvalues, counts=(hist_ibin.values[1:-1]-fit_templates)/np.sqrt(hist_ibin.squared_values[1:-1]), err=np.zeros(len(xvalues)), label_x="Charge", label_y=r"(fit-data)/$\mathrm{\sigma_{data}}$",  style=".", color="black", setlogy=0, markersize=20)
                plot1d_errorbar_v2(fig, ax2, xvalues[fit_range[0]: fit_range[1]], counts=pull, err=np.zeros(len(pull)), label_x="1/mass", label_y=r"(fit-data)/$\mathrm{\sigma_{data}}$",  style=".", color="black", setlogy=0, markersize=20)
                ax2.plot(xvalues, np.zeros_like(xvalues), '--', color='black')
                ax2.set_ylim([-3.9, 3.9])
                ax2.grid()
                plt.subplots_adjust(hspace=.0)                             
                savefig_tofile(fig, args.plotdir, f"fit_mass_{iso}_{dec}_{ibin}_log", show=False)

        #####################################################
        #plot_mean be9
        #####################################################
        ylim_range = {'mean':[0.08, 0.17], 'sigma':[0.008, 0.024], 'sigma_ratio':[1.3, 2.0], 'asy_factor':[0.9, 1.8], 'fraccore':[0.8, 1.0], "norm": [0, 40]}
        df_dict = {}
        for dec in detectors:
            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.55, 0.45]}, figsize=(16, 14))
            fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)   
            sub_graph_mean = dict()
            for iso in ISOTOPES[nuclei]:
                sub_graph_mean[iso] = slice_graph_by_value(dict_pars[dec][iso]["mean"], energy_range)
                if iso != "Be10":
                    plot_graph(fig, ax1, sub_graph_mean[iso], color=ISOTOPES_COLOR[iso], label=ISO_LABELS[iso], style="EP", xlog=False, ylog=False, scale=None, markersize=22)
                    popt, pcov = curve_fit(poly, np.log(sub_graph_mean[iso].xvalues), sub_graph_mean[iso].yvalues, p0 = [1/ISOTOPES_MASS[iso], 0.0001])
                    ax1.plot(sub_graph_mean[iso].xvalues, poly(np.log(sub_graph_mean[iso].xvalues), *popt), '-', color=ISOTOPES_COLOR[iso])
                    print(iso, " mean_fit:", popt, " _err:", pcov)
                    pars_mean = np.polyfit(np.log(sub_graph_mean[iso].getx()), sub_graph_mean[iso].gety(), 0)
                    yplot = np.poly1d(pars_mean)
                ax1.set_ylabel(r'$\mathrm{\mu}$')
                    #axes[ipar].plot(sub_graph.getx(), yplot(np.log(sub_graph.getx())) , "-", color=ISOTOPES_COLOR[iso])
                ax1.set_ylim([0.1, 0.16])
            
            ratio_mean_be9be7 = sub_graph_mean['Be9']/sub_graph_mean['Be7']
            ratio_mean_be10be7 = sub_graph_mean['Be10']/sub_graph_mean['Be7']
            ratio_mean_be9be7.add_to_file(df_dict, f'graph_ratio_mean_be9be7_{dec}')
            ratio_mean_be10be7.add_to_file(df_dict, f'graph_ratio_mean_be10be7_{dec}')
            
            popt, pcov = curve_fit(poly, np.log(ratio_mean_be9be7.xvalues), ratio_mean_be9be7.yvalues, p0 = [0.779])
            print("ratio_mean_be9be7: ",  "popt:", popt , "pcov:", pcov)
            p_text = r'$\mathrm{{p = {:.4f} \pm {:.4f} }}$'.format(popt[0], np.sqrt(pcov[0, 0]))
            ref_text = r'$\mathrm{{ref = {:.4f}}}$'.format(ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be9']) 
            ax2.text(0.4, 0.98, p_text, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes, color='black')
            ax2.text(0.4, 0.9, ref_text, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes, color='black')
            popt10, pcov10 = curve_fit(poly, np.log(ratio_mean_be10be7.xvalues), ratio_mean_be10be7.yvalues, p0 = [0.7])
            print("ratio_mean_be10be7:", "popt:", popt10 , "pcov:", pcov10)
            ax2.plot(ratio_mean_be9be7.xvalues, poly(np.log(ratio_mean_be9be7.xvalues), popt), '-', color=ISOTOPES_COLOR['Be9'],  label=r"$\mathrm{\mu_{9}}$/$\mathrm{\mu_{7}}$")
            #ax2.plot(ratio_mean_be10be7.xvalues, poly(np.log(ratio_mean_be10be7.xvalues), popt10), '-', color=ISOTOPES_COLOR['Be10'],  label=r"$\mathrm{\mu_{10}}$/$\mathrm{\mu_{7}}$")

            ax2.axhline(y=ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be9'], color='red', linestyle='--', label='ref m7/m9')
            #ax2.axhline(y=ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be10'], color='green', linestyle='--')
            ax1.text(0.8, 0.9, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", fontweight="bold")
            plot_graph(fig, ax2, sub_graph_mean['Be9']/sub_graph_mean['Be7'], color=ISOTOPES_COLOR['Be9'], style="EP", xlog=False, ylog=False, scale=None, markersize=22)
            plot_graph(fig, ax2, sub_graph_mean['Be10']/sub_graph_mean['Be7'], color=ISOTOPES_COLOR['Be10'], style="EP", xlog=False, ylog=False, scale=None, markersize=22)
            ax1.legend(loc='center right', fontsize=FONTSIZE-2)
            ax2.legend(loc='lower right', fontsize=FONTSIZE-2)
            ax2.set_ylabel("ratio")
            ax2.set_ylim([0.77, 0.79])
            ax2.grid()
            set_plot_defaultstyle(ax1)
            set_plot_defaultstyle(ax2)
            ax1.get_yticklabels()[0].set_visible(False)
            plt.subplots_adjust(hspace=.0)
            ax1.set_xticklabels([])
            ax2.set_xlabel("Ekin/n (GeV/n)")

            savefig_tofile(fig, args.plotdir, f"mean_compare_be9_{dec}P8", show=True)

        np.savez(os.path.join(args.plotdir, f'df_mean_ratio_{dec}P8.npz'), **df_dict)
        ####################################################################
        #plot mean be10
        ####################################################################
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))
        fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)   
        for dec in detectors:
            sub_graph_mean = dict()
            for iso in ISOTOPES[nuclei]:
                sub_graph_mean[iso] = slice_graph_by_value(dict_pars[dec][iso]["mean"], energy_range)
                if iso != "Be9":
                    plot_graph(fig, ax1, sub_graph_mean[iso], color=ISOTOPES_COLOR[iso], label=ISO_LABELS[iso], style="EP", xlog=False, ylog=False, scale=None, markersize=22)
                    popt, pcov = curve_fit(poly, np.log(sub_graph_mean[iso].xvalues), sub_graph_mean[iso].yvalues, p0 = [1/ISOTOPES_MASS[iso], 0.0001])
                    ax1.plot(sub_graph_mean[iso].xvalues, poly(np.log(sub_graph_mean[iso].xvalues), *popt), '-', color=ISOTOPES_COLOR[iso])
                    print(iso, " mean_fit:", popt, " _err:", pcov)
                    pars_mean = np.polyfit(np.log(sub_graph_mean[iso].getx()), sub_graph_mean[iso].gety(), 0)
                    yplot = np.poly1d(pars_mean)
                    ax1.set_ylabel(r'$\mathrm{\mu}$')
                    #axes[ipar].plot(sub_graph.getx(), yplot(np.log(sub_graph.getx())) , "-", color=ISOTOPES_COLOR[iso])
                    ax1.set_ylim([0.1, 0.16])
            
            ratio_mean_be10be7 = sub_graph_mean['Be10']/sub_graph_mean['Be7']
            popt10, pcov10 = curve_fit(poly, np.log(ratio_mean_be10be7.xvalues), ratio_mean_be10be7.yvalues, p0 = [0.7])
            print("ratio_mean_be10be7:", "popt:", popt10 , "pcov:", pcov10)
            ax2.plot(ratio_mean_be10be7.xvalues, poly(np.log(ratio_mean_be10be7.xvalues), popt10), '-', color=ISOTOPES_COLOR['Be10'],  label=r"$\mathrm{\mu_{10}}$/$\mathrm{\mu_{7}}$")
            p_text = r'$\mathrm{{p = {:.4f} \pm {:.4f}}}$'.format(popt10[0], np.sqrt(pcov10[0, 0]))
            ref_text = r'$\mathrm{{ref = {:.4f}}}$'.format(ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be10']) 
            ax2.text(0.4, 0.9, p_text, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes, color='black')
            ax2.text(0.4, 0.98, ref_text, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes, color='black')

            ax2.axhline(y=ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be10'], color='green', linestyle='--')
            ax1.text(0.8, 0.9, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", fontweight="bold")
            plot_graph(fig, ax2, sub_graph_mean['Be10']/sub_graph_mean['Be7'], color=ISOTOPES_COLOR['Be10'], style="EP", xlog=False, ylog=False, scale=None, markersize=22)
            ax1.legend(loc='center right', fontsize=FONTSIZE-2)
            ax2.set_ylabel("ratio")
            ax2.set_ylim([0.69, 0.71])
            ax2.grid()
            set_plot_defaultstyle(ax1)
            set_plot_defaultstyle(ax2)
            ax1.get_yticklabels()[0].set_visible(False)
            plt.subplots_adjust(hspace=.0)
            ax1.set_xticklabels([])
            ax2.set_xlabel("Ekin/n (GeV/n)")

        savefig_tofile(fig, args.plotdir, f"mean_compare_be10_{dec}P8", show=True)

        ### sigma ######################################################
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))
        fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)
        ratio_sigma_be9be7 = dict()
        ratio_sigma_be10be7 = dict()
        for dec in detectors:
            sub_graph_sigma = dict()
            for iso in ISOTOPES[nuclei]:
                sub_graph_sigma[iso] = slice_graph_by_value(dict_pars[dec][iso]["sigma"], energy_range)
                print("sub_graph_sigma[iso].xvalues:", sub_graph_sigma[iso].xvalues)
                plot_graph(fig, ax1, sub_graph_sigma[iso], color=ISOTOPES_COLOR[iso], label=f'{ISO_LABELS[iso]}', style="EP", xlog=False, ylog=False, scale=None, markersize=22)
                popt, pcov = curve_fit(poly, np.log(sub_graph_sigma[iso].xvalues), sub_graph_sigma[iso].yvalues, p0 = [0.016/iso_mass_ratio[iso], 0.001, 0.0001])
                ax1.plot(sub_graph_sigma[iso].xvalues, poly(np.log(sub_graph_sigma[iso].xvalues), *popt), '-', color=ISOTOPES_COLOR[iso])
                print(iso, " sigma_fit_pars:", popt, " _err:", pcov)
                pars_sigma = np.polyfit(np.log(sub_graph_sigma[iso].getx()), sub_graph_sigma[iso].gety(), 0)
                yplot = np.poly1d(pars_sigma)
                ax1.set_ylabel(r'$\mathrm{\sigma_{p}}$')
                #axes[ipar].plot(sub_graph.getx(), yplot(np.log(sub_graph.getx())) , "-", color=ISOTOPES_COLOR[iso])
                ax1.set_ylim(ylim_range['sigma'])
                ax1.legend(loc='upper left', fontsize=FONTSIZE)
            ratio_sigma_be9be7[dec] = sub_graph_sigma['Be9']/sub_graph_sigma['Be7']
            ratio_sigma_be10be7[dec] = sub_graph_sigma['Be10']/sub_graph_sigma['Be7']
            popt, pcov = curve_fit(poly, np.log(ratio_sigma_be9be7[dec].xvalues), ratio_sigma_be9be7[dec].yvalues, p0 = [0.779])
            ax2.plot(ratio_sigma_be9be7[dec].xvalues, poly(np.log(ratio_sigma_be9be7[dec].xvalues), popt), '-', color=ISOTOPES_COLOR['Be9'], label=r"$\mathrm{\sigma_{9}}$/$\mathrm{\sigma_{7}}$")
            popt10, pcov10 = curve_fit(poly, np.log(ratio_sigma_be10be7[dec].xvalues), ratio_sigma_be10be7[dec].yvalues, p0 = [0.7])
            ax2.plot(ratio_sigma_be10be7[dec].xvalues, poly(np.log(ratio_sigma_be10be7[dec].xvalues), popt10), '-', color=ISOTOPES_COLOR['Be10'], label=r"$\mathrm{\sigma_{9}}$/$\mathrm{\sigma_{7}}$")
            ax2.axhline(y=ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be9'], color='red', linestyle='--')
            ax2.axhline(y=ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be10'], color='red', linestyle='--')
            ax2.legend(fontsize=20)
            #ax2.axhline(y=7/10, color='red', linestyle='--', label='y = m7/m9')
            plot_graph(fig, ax2, sub_graph_sigma['Be9']/sub_graph_sigma['Be7'], color=ISOTOPES_COLOR['Be9'], style="EP", xlog=False, ylog=False, scale=None, markersize=22)

            equation_text = r'$\mathrm{\sigma_{p} = p_{0} + p_{1} \cdot x + p_{2} \cdot x^{2}}$'
            ax1.text(0.5, 0.1, equation_text, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='black')
            ax1.text(0.8, 0.9, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", fontweight="bold")
        label_ref9 = mlines.Line2D([], [], linestyle='--', color=ISOTOPES_COLOR['Be9'], label='ref m7/m9')
        legend1 = ax2.legend(handles=[label_ref9], loc='upper left', bbox_to_anchor=(0.0, 1.02), fontsize=20)
        ax2.add_artist(legend1)
        ax2.set_ylabel("ratio")
        ax2.legend(fontsize=20)
        ax2.set_ylim([0.6, 0.9])
        ax2.grid()
        set_plot_defaultstyle(ax1)
        set_plot_defaultstyle(ax2)
        ax1.get_yticklabels()[0].set_visible(False)
        plt.subplots_adjust(hspace=.0)
        ax1.set_xticklabels([])
        ax2.set_xlabel("Ekin/n (GeV/n)")
        savefig_tofile(fig, args.plotdir, f"sigma_compare", show=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))
        fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)
        ratio_sigma_be9be7 = dict()
        ratio_sigma_be10be7 = dict()
        for dec in detectors:
            sub_graph_sigma = dict()
            for iso in ISOTOPES[nuclei]:
                sub_graph_sigma[iso] = slice_graph_by_value(dict_pars[dec][iso]["sigma"], energy_range)
                print("sub_graph_sigma[iso].xvalues:", sub_graph_sigma[iso].xvalues)
                plot_graph(fig, ax1, sub_graph_sigma[iso], color=ISOTOPES_COLOR[iso], label=f'{ISO_LABELS[iso]}', style="EP", xlog=False, ylog=False, scale=None, markersize=22)
                popt, pcov = curve_fit(poly, np.log(sub_graph_sigma[iso].xvalues), sub_graph_sigma[iso].yvalues, p0 = [0.016/iso_mass_ratio[iso], 0.001, 0.0001, 0.0001])
                ax1.plot(sub_graph_sigma[iso].xvalues, poly(np.log(sub_graph_sigma[iso].xvalues), *popt), '-', color=ISOTOPES_COLOR[iso])
                print(iso, " sigma_fit_pars:", popt, " _err:", np.sqrt(np.diag(pcov)))
                pars_sigma = np.polyfit(np.log(sub_graph_sigma[iso].getx()), sub_graph_sigma[iso].gety(), 0)
                yplot = np.poly1d(pars_sigma)
                ax1.set_ylabel(r'$\mathrm{\sigma_{p}}$')
                #axes[ipar].plot(sub_graph.getx(), yplot(np.log(sub_graph.getx())) , "-", color=ISOTOPES_COLOR[iso])
                ax1.set_ylim(ylim_range['sigma'])
            ax1.legend(loc='upper left', fontsize=FONTSIZE)
            equation_text_3deg = r'$\mathrm{\sigma_{p} = p_{0} + p_{1} \cdot x + p_{2} \cdot x^{2} + p_{3} \cdot x^{3}}$'
            ax1.text(0.5, 0.1, equation_text_3deg, fontsize=25, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='black')
            ratio_sigma_be9be7[dec] = sub_graph_sigma['Be9']/sub_graph_sigma['Be7']
            ratio_sigma_be10be7[dec] = sub_graph_sigma['Be10']/sub_graph_sigma['Be7']
            popt, pcov = curve_fit(poly, np.log(ratio_sigma_be9be7[dec].xvalues), ratio_sigma_be9be7[dec].yvalues, p0 = [0.779])
            popt10, pcov10 = curve_fit(poly, np.log(ratio_sigma_be10be7[dec].xvalues), ratio_sigma_be10be7[dec].yvalues, p0 = [0.7])
            ax2.plot(ratio_sigma_be9be7[dec].xvalues, poly(np.log(ratio_sigma_be9be7[dec].xvalues), popt), '-', color=ISOTOPES_COLOR['Be9'], label=r"$\mathrm{\sigma_{9}}$/$\mathrm{\sigma_{7}}$")
            ax2.plot(ratio_sigma_be10be7[dec].xvalues, poly(np.log(ratio_sigma_be10be7[dec].xvalues), popt10), '-', color=ISOTOPES_COLOR['Be10'], label=r"$\mathrm{\sigma_{10}}$/$\mathrm{\sigma_{7}}$")
            ax2.axhline(y=ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be9'], color='red', linestyle='--')
            #ax2.axhline(y=7/9, color='red', linestyle='--', label='ref m7/m9')
            ax2.axhline(y=ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be10'], color='green', linestyle='--')
            ax2.legend(fontsize=20)
            #ax2.axhline(y=7/10, color='red', linestyle='--', label='y = m7/m9')
            plot_graph(fig, ax2, sub_graph_sigma['Be9']/sub_graph_sigma['Be7'], color=ISOTOPES_COLOR['Be9'], style="EP", xlog=False, ylog=False, scale=None, markersize=22)
            plot_graph(fig, ax2, sub_graph_sigma['Be10']/sub_graph_sigma['Be7'], color=ISOTOPES_COLOR['Be10'], style="EP", xlog=False, ylog=False, scale=None, markersize=22)
        ax1.text(0.8, 0.9, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", fontweight="bold")
        label_ref9 = mlines.Line2D([], [], linestyle='--', color=ISOTOPES_COLOR['Be9'], label='ref m7/m9')
        label_ref10 = mlines.Line2D([], [], linestyle='--', color=ISOTOPES_COLOR['Be10'], label='ref m7/m10')
        legend1 = ax2.legend(handles=[label_ref10], loc='upper left', bbox_to_anchor=(0.0, 1.02), fontsize=20)
        ax2.add_artist(legend1)
        ax2.set_ylabel("ratio")
        ax2.legend(fontsize=20)
        ax2.set_ylim([0.6, 0.9])
        ax2.grid()
        set_plot_defaultstyle(ax1)
        set_plot_defaultstyle(ax2)
        ax1.get_yticklabels()[0].set_visible(False)
        plt.subplots_adjust(hspace=.0)
        ax1.set_xticklabels([])
        ax2.set_xlabel("Ekin/n (GeV/n)")
        savefig_tofile(fig, args.plotdir, f"sigma_compare_3deg", show=True)

        # read the scale factor from rigidity
        filename_rig = "/home/manbing/Documents/Data/data_rig/graph_rigsigma_scale_factor.npz"
        with np.load(filename_rig) as file_rig:
             graph_rig_sigmafactor_7to9 = MGraph.from_file(file_rig, "graph_rig_reso_sigma_7to9")
             graph_rig_sigmafactor_7to10 = MGraph.from_file(file_rig, "graph_rig_reso_sigma_7to10")

        fig, ax1 = plt.subplots(figsize=(20, 14))
        fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95) 
        plot_graph(fig, ax1, graph_rig_sigmafactor_7to9, color=ISOTOPES_COLOR['Be9'], style="EP", xlog=False, ylog=False, scale=None, markersize=22, label=r'$\mathrm{\sigma_{(1/R, Be7)}/\sigma_{(1/R, Be9)}}$')
        plot_graph(fig, ax1, graph_rig_sigmafactor_7to10, color=ISOTOPES_COLOR['Be10'], style="EP", xlog=False, ylog=False, scale=None, markersize=22, label=r'$\mathrm{\sigma_{(1/R, Be7)}/\sigma_{(1/R, Be10)}}$')
        popt, pcov = curve_fit(expo_func, graph_rig_sigmafactor_7to9.xvalues[1:], graph_rig_sigmafactor_7to9.yvalues[1:], p0 = [1.0, 4.0, 10])
        ax1.plot(graph_rig_sigmafactor_7to9.xvalues[1:], expo_func(graph_rig_sigmafactor_7to9.xvalues[1:], *popt), '-', color=ISOTOPES_COLOR['Be9'])
        print("graph_rig_sigmafactor_7to9:", popt)
        popt, pcov = curve_fit(expo_func, graph_rig_sigmafactor_7to10.xvalues[1:], graph_rig_sigmafactor_7to10.yvalues[1:], p0 = [1.0, 4.0, 10])
        print("graph_rig_sigmafactor_7to10:", popt)
        ax1.plot(graph_rig_sigmafactor_7to10.xvalues[1:], expo_func(graph_rig_sigmafactor_7to10.xvalues[1:], *popt), '-', color=ISOTOPES_COLOR['Be10'])

        y_fit_expo_factor9 = expo_func(graph_rig_sigmafactor_7to9.xvalues[:], *popt)
        y_fit_expo_factor10 = expo_func(graph_rig_sigmafactor_7to10.xvalues[:], *popt)
        
        ax1.set_ylim([0.92, 1.02])
        ax1.set_xlabel("Ekin/N(GeV/n)")
        ax1.set_ylabel(r"$\mathrm{\sigma_{1/R} ~ ratio}$")
        ax1.legend(fontsize=30)
        savefig_tofile(fig, args.plotdir, f"rigidity_sigma_scale_factor", show=True)
        
        fig, ax1 = plt.subplots(figsize=(20, 14))
        fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95) 
        for dec in detectors:
            graph_9to7_sigmafactor = MGraph(graph_rig_sigmafactor_7to9.xvalues, 7/(9 * y_fit_expo_factor9), yerrs=np.zeros_like(graph_rig_sigmafactor_7to9.xvalues))
            graph_10to7_sigmafactor = MGraph(graph_rig_sigmafactor_7to10.xvalues, 7/(10 * y_fit_expo_factor10), yerrs=np.zeros_like(graph_rig_sigmafactor_7to10.xvalues))
            graph_9to7_sigmafactor = slice_graph_by_value(graph_9to7_sigmafactor, ANALYSIS_RANGE_EKIN[dec])
            graph_10to7_sigmafactor = slice_graph_by_value(graph_10to7_sigmafactor, ANALYSIS_RANGE_EKIN[dec])
            plot_graph(fig, ax1, ratio_sigma_be9be7[dec], color=ISOTOPES_COLOR['Be9'], style="EP", xlog=False, ylog=False, scale=None, markersize=22, label=r"$\mathrm{\sigma_{(m,9)}/\sigma_{(m,7)}}$")
            plot_graph(fig, ax1, ratio_sigma_be10be7[dec], color=ISOTOPES_COLOR['Be10'], style="EP", xlog=False, ylog=False, scale=None, markersize=22, label=r"$\mathrm{\sigma_{(m,10)}/\sigma_{(m,7)}}$")
            #plot_graph(fig, ax1, graph_9to7_sigmafactor, color=ISOTOPES_COLOR['Be9'], style="EP", xlog=False, ylog=False, scale=None, markersize=22, markerfacecolor='none', label=r"$\mathrm{(m7/m9)/k}$")
            ax1.plot(graph_9to7_sigmafactor.xvalues, graph_9to7_sigmafactor.yvalues, '-', color="red")
            ax1.plot(graph_10to7_sigmafactor.xvalues, graph_10to7_sigmafactor.yvalues, '-', color="green")
            plot_graph(fig, ax1, graph_10to7_sigmafactor, color=ISOTOPES_COLOR['Be10'], style="EP", xlog=False, ylog=False, scale=None, markersize=22, markerfacecolor='none', label=r"$\mathrm{(m10/m9)/k}$")
            #popt, pcov = curve_fit(expo_func, ratio_sigma_be9be7[dec].xvalues, ratio_sigma_be9be7[dec].yvalues, p0 = [0.7, 4.0, 10])
            #print("expo_func_be9:", popt)
            

        #ax1.text(0.8, 0.98, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", fontweight="bold")    
        ax1.axhline(y=ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be9'], color=ISOTOPES_COLOR['Be9'], linestyle='--')
        ax1.axhline(y=ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be10'], color=ISOTOPES_COLOR['Be10'], linestyle='--')
        label_ref9 = mlines.Line2D([], [], linestyle='--', color=ISOTOPES_COLOR['Be9'], label='ref m7/m9')
        label_ref10 = mlines.Line2D([], [], linestyle='--', color=ISOTOPES_COLOR['Be10'], label='ref m7/m10')
        legend1 = ax1.legend(handles=[label_ref9, label_ref10], loc='lower left', bbox_to_anchor=(0.0, 0.1), fontsize=FONTSIZE-2)
        ax1.add_artist(legend1)

        ax1.set_ylim([0.6, 0.85])
        ax1.set_xlabel("Ekin/N(GeV/n)")
        ax1.set_ylabel(r"$\mathrm{\sigma_{c} ~ ratio}$")
        ax1.legend(fontsize=FONTSIZE-2, loc="lower right")
        savefig_tofile(fig, args.plotdir, f"sigma_scale_factor_{dec}", show=True)
        
        

    plt.show()
    
if __name__ == "__main__":
    main()
    


