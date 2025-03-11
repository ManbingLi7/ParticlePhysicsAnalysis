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
from tools.binnings_collection import  fbinning_energy_agl
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, cdf_double_gaus, double_gaus
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.binnings import Binning
from tools.constants import NUCLEI_CHARGE
from tools.calculator import calc_ekin_from_beta
from tools.calculator import calculate_efficiency_and_error, calculate_efficiency_and_error_weighted, calculate_efficiency_weighted
from tools.statistics import poly_func
from tools.graphs import MGraph, slice_graph, concatenate_graphs, plot_graph
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

#xbinning = fbinning_energy_agl()
setplot_defaultstyle()

kNucleiBinsRebin = np.array([0.8,1.00,1.16,1.33,1.51,1.71,1.92,2.15,2.40,2.67,2.97,3.29,3.64,4.02,4.43,4.88, 5.37,5.90,6.47,7.09,7.76,8.48,9.26, 10.1,11.0,12.0,13.0,14.1,15.3,16.6,18.0,19.5,21.1,22.8,24.7,26.7,28.8,31.1,33.5,36.1, 38.9, 41.9,45.1,48.5,52.2,60.3,69.7,80.5,93.0,108., 116.,147.,192.,259.,379.,660., 1300, 3300.])
kNucleiBinsRebin_center = get_bin_center(kNucleiBinsRebin)
kNucleiNbinRebin = 57

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


def gaus_asygaus(x, mean, sigma, sigma_ratio, asy_factor, fraccore, norm):
    coregaus = gaussian(x, mean, sigma)
    asygaus = asy_gaussian_1d(x, mean,  sigma_ratio * sigma, asy_factor)
    pdf = norm * (fraccore * coregaus + (1 - fraccore) * asygaus)
    return pdf

def cumulative_gaus_asygaus(edges, mean, sigma, sigma_ratio, asy_factor, fraccore, norm):
    x = (edges[1:] + edges[:-1])/2
    pdf = gaus_asygaus(x, mean, sigma, sigma_ratio, asy_factor, fraccore, norm)
    cpdf = np.cumsum(pdf)
    return np.concatenate(([0], cpdf))



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--resultdir", default="plots/l1q", help="Directory to store plots and result files in.")
    parser.add_argument("--nuclei", default="Be", help="Directory to store plots and result files in.")
    parser.add_argument("--detectors", nargs="+", default=["RichAgl"], help="Directory to store plots and result files in.")
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    
    hist_l1q = []
    nT = 3
    templates_name = ["Be", "Boron", "Carbon"]
    tem_color = ["tab:blue", "tab:red", "tab:green"]
    xbinning = Binning(kNucleiBinsRebin)
    ybinning = Binning(np.linspace(2, 14, 500))
    fit_charge_range = ybinning.get_indices([3.3, 6.65])
    fitminbin = fit_charge_range[0]
    fitmaxbin = fit_charge_range[1]
    
    hist2d_qz = dict()
    with open('/home/manbing/Documents/Data/jiahui/efficiency/eff_cor_be_10yr_updated.pkl', 'rb') as f:
        data_jiahui_eff = pickle.load(f)

    with open(os.path.join(args.resultdir, 'polyfits_4.pickle'), 'rb') as file:
        par_polynomials = pickle.load(file)
        
    #with uproot.open("/home/manbing/Documents/Data/data_effacc/lq_template.root") as rootfile:
    with uproot.open("/home/manbing/Documents/Data/data_BeP8/efficiency/l1qtemplate_ISSP8GBL.root") as rootfile:
        for i, tem in enumerate(templates_name):
            qtem = i + 4
            print(templates_name[i], qtem)
            tree = rootfile[f'ql1_z{qtem}']
            qz = tree['q_template2'].array()
            qrig = tree['tk_rig'].array()
            hist2d_qz[tem] = Histogram(xbinning, ybinning, labels=["Rigidity (GV)", f"Z_{tem}"])
            hist2d_qz[tem].fill(qrig, qz)
            fig = plt.figure(figsize=(20, 15))
            plot = fig.subplots(1, 1)
            plot_histogram_2d(plot, hist2d_qz[tem], scale=None, transpose=False, show_overflow=True, show_overflow_x=None, show_overflow_y=None, label=None)


        boron_background = np.zeros(kNucleiNbinRebin)
        eff = np.zeros(kNucleiNbinRebin)
        efferr = np.zeros(kNucleiNbinRebin)
        graph_boron_background = MGraph(xvalues=get_bin_center(kNucleiBinsRebin), yvalues=np.zeros(len(kNucleiBinsRebin)-1), yerrs=np.zeros(len(kNucleiBinsRebin)-1))
        
        par_names = ['mean', 'sigma', 'sigma_ratio', 'asy_factor', 'fraccore', 'norm']
        
        par_names_axes = ['$\mathrm{\mean}$', '$\mathrm{sigma', 'sigma_ratio', 'asy_factor', 'fraccore', 'norm']
        poly_deg = {'mean': 3, 'sigma':3, 'sigma_ratio':3, 'asy_factor':0, 'fraccore': 0, 'norm':6}
        dict_pars = {temp: {par: MGraph(binning=xbinning, labels=["Rigidity(GV)", f"{par}"]) for par in par_names} for temp in templates_name}
        chisquare = {temp: np.zeros(kNucleiNbinRebin)  for temp in templates_name}    
        print(dict_pars.keys())

        minbin = 0
        maxbin = len(kNucleiBinsRebin) - 1
        #minbin = 30
        #maxbin = 32

        for ibin in range(minbin, maxbin):
            i_num = ibin - minbin
            hist_l1q.append(rootfile[f'h_l1q_{ibin}'])
            templates = np.zeros([nT, len(ybinning.bin_centers[1:-1])])
            fit_templates = np.zeros([nT, len(ybinning.bin_centers[1:-1])])

            charge_xbin = ybinning.edges[1:-1][fitminbin:fitmaxbin+1]
            charge_binning_fit = Binning(charge_xbin)            
            
            xvalues_fit = get_bin_center(charge_xbin)
            yvalues_fit = hist_l1q[i_num].values()[fitminbin:fitmaxbin]
            yvalues_fit[yvalues_fit == 0] = 1
            yvalueserr_fit = hist_l1q[i_num].errors()[fitminbin:fitmaxbin]
            yvalueserr_fit[yvalueserr_fit == 0.0] = 1

            templates_par = np.zeros([nT, 6])
            print(ibin, "bin")
            for iT, temp in enumerate(templates_name):
                qsel = iT + 4
                fit_range = ybinning.get_indices([qsel-0.8, qsel+1.0])
                templates[iT] = hist2d_qz[temp].values[ibin+1, 1:-1]
                #templates[iT] = templates[iT]/np.sum(templates[iT])
                
                hist_ibin = hist2d_qz[temp].project(ibin)

                loss = ExtendedBinnedNLL(templates[iT][fit_range[0]: fit_range[1]], ybinning.edges[1: -1][fit_range[0]:fit_range[1]+1], cumulative_gaus_asygaus)
                guess = dict(mean=qsel-0.02, sigma=0.15, sigma_ratio=1.2, asy_factor=1.45, fraccore=0.3, norm= 3000)  
                pars_guess = dict()
                for par in dict_pars[temp].keys():
                     pars_guess[par] = par_polynomials[par](np.log(kNucleiBinsRebin_center[ibin]))


                if pars_guess['norm'] < 0:
                    pars_guess['norm'] = 500
                    
                pars_guess['asy_factor'] = 1.5
                print(pars_guess)
                #guess = dict(mean=qsel-0.1, sigma=0.12, sigma_ratio=1.8, asy_factor=1.3, fraccore=0.15)
                m = Minuit(loss, **pars_guess)
                #m.limits["norm"]=(300, 800)
                m.limits["mean"]=(qsel-0.1, qsel+0.05)
                m.limits["sigma"]=(0.1, 0.2)
                m.limits["asy_factor"]=(1.3, 1.7)
                m.limits['sigma_ratio'] = (1.0, 2.0)
                m.fixed['fraccore'] = True
                
                m.migrad()
                #print(m)
                chisquare[temp][ibin] = m.fval / m.ndof
                for par in dict_pars[temp].keys():
                    
                    dict_pars[temp][par].yvalues[ibin] = m.values[par]
                    dict_pars[temp][par].yerrs[ibin] = m.errors[par]


                templates_par[iT] = np.array(m.values)
                #print(templates_par[iT])
                
                xvalues = hist2d_qz[temp].binnings[1].bin_centers[1:-1]

                fit_templates[iT] = gaus_asygaus(xvalues, *m.values)

                #fit_templates[iT] = fit_templates[iT]/np.sum(fit_templates[iT])
                #print(np.sum(fit_templates[iT]))                
                fit_guess = gaus_asygaus(xvalues, **guess)
                #fit_guess = fit_guess/np.sum(fit_guess)
                #print(np.sum(fit_guess))
                
                fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14)) 
                #plot_histogram_1d(ax1, hist_ibin, style="iss", color="black", label="data", scale=None, gamma=None, xlog=False, ylog=False, shade_errors=False)
                #ax1.plot(xvalues, templates[iT], ".", color="black", label="data", markersize=25)
                plot1d_errorbar_v2(fig, ax1, xvalues, counts=templates[iT], err=np.sqrt(templates[iT]), label_x="Charge", label_y="counts",  style=".", color="black", setlogy=0, markersize=20)
                ax1.plot(xvalues, fit_templates[iT], "-", color="black", label="fit")
                #ax1.plot(xvalues, fit_guess, "-", color="blue", label="guess")            
                ax1.text(0.03, 0.98, f"{templates_name[iT]}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color=tem_color[iT], weight='bold')
                ax1.text(0.03, 0.85, r"$\mathrm{{R = [{}, {}] GV}}$".format(xbinning.edges[ibin+1], xbinning.edges[ibin+2]), fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')
                # display legend with some fit info
                fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fmin.reduced_chi2:.1f}",]
                #ax2.plot(xvalues, templates[iT]/fit_templates[iT], "-", color="black", label="ratio")
                ax1.legend(title="\n".join(fit_info), frameon=False)
                #ax1.legend(fontsize=FONTSIZE_BIG)
                ax1.set_xlim([qsel-1.0, qsel+1.0])
                ax2.set_xlim([qsel-1.0, qsel+1.0])
                set_plot_defaultstyle(ax1)
                set_plot_defaultstyle(ax2)
                ax1.set_ylabel("Counts")
                ax1.set_xticks([])
                ax1.set_xticklabels([])
                ax1.set_yscale('log')
                ax1.set_ylim([1, 10*np.max(templates[iT])])
                ax1.get_yticklabels()[0].set_visible(False)
                plot1d_errorbar_v2(fig, ax2, xvalues, counts=(templates[iT]-fit_templates[iT])/np.sqrt(templates[iT]), err=np.zeros(len(xvalues)), label_x="Charge", label_y=r"(fit-data)/$\mathrm{\sigma_{data}}$",  style=".", color="black", setlogy=0, markersize=20)
                ax2.set_ylim([-3.9, 3.9])
                ax2.grid()
                plt.subplots_adjust(hspace=.0)                             

                savefig_tofile(fig, args.resultdir, f"fit/fit_{qsel}_{ibin}_log", show=False)


        
        pars_poly = {temp: dict() for temp in templates_name}
        splinefit = {temp: dict() for temp in templates_name}
        y_splinefit = {temp: dict() for temp in templates_name}
        polyfit = {temp: dict() for temp in templates_name}
        ylim_range = {'mean':[3.9, 4.1], 'sigma':[0.05, 0.25], 'sigma_ratio':[1.0, 2.0], 'asy_factor':[1.0, 2.0], 'fraccore':[0.1, 0.8], "norm": [0, 2000]}
        for par in dict_pars["Be"].keys():
            fig = plt.figure(figsize=(20, 15))
            plot = fig.subplots()    
            for iT, temp in enumerate(templates_name):
                if (iT == 0):
                    print(dict_pars[temp][par].xvalues)
                    print(dict_pars[temp][par].yvalues)
                    print(dict_pars[temp][par].yerrs)

                    plot_graph(fig, plot, dict_pars[temp][par], color=tem_color[iT], label=None, style="EP", xlog=True, ylog=False, scale=None, markersize=22)
                    pars_poly[temp][par] = np.polyfit(np.log(dict_pars[temp][par].getx()), dict_pars[temp][par].gety(), 3)
                    splinefit[temp][par] = UnivariateSpline(dict_pars[temp][par].getx()[5:-5], dict_pars[temp][par].gety()[5:-5], k=3, s=100)
                    y_splinefit[temp][par] = splinefit[temp][par](np.log(dict_pars[temp][par].getx()))
                    #polyfit[temp][par] = pars_poly[temp][par](np.log(dict_pars[temp][par].getx()))
                    plot.set_ylim(ylim_range[par])
                    #plot.plot(rig_bin_centers, polyfit[par], "-", color="tab:blue", label="fit")
                    #plot.plot(dict_pars[temp][par].getx(), y_splinefit[temp][par], "-", color=tem_color[iT], label="fit")
                    savefig_tofile(fig, args.resultdir, f"fit_temp_pars_{par}_iter3", show=False)


        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(25, 20))
        fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)   
        temp = "Be"
        polynomials = {}
        for ipar, par  in enumerate(dict_pars[temp].keys()):
            if (ipar < 3):
                plot_graph(fig, axes[ipar], dict_pars[temp][par], color=tem_color[iT], label=None, style="EP", xlog=True, ylog=False, scale=None, markersize=22)
                pars_poly[temp][par] = np.polyfit(np.log(dict_pars[temp][par].getx()), dict_pars[temp][par].gety(), 5)
                splinefit[temp][par] = UnivariateSpline(dict_pars[temp][par].getx()[10:-10], dict_pars[temp][par].gety()[10:-10], k=3, s=5)
                y_splinefit[temp][par] = splinefit[temp][par](np.log(dict_pars[temp][par].getx()[10:-10]))
                print(dict_pars[temp][par].getx()[3:-10])
                yplot = np.poly1d(pars_poly[temp][par])
                polynomials[par] = np.poly1d(pars_poly[temp][par]) 
                axes[ipar].plot(dict_pars[temp][par].getx(), yplot(np.log(dict_pars[temp][par].getx())) , "-", color=tem_color[iT])
                axes[ipar].set_ylim(ylim_range[par])
                axes[ipar].set_xlim([1.9, 1300])
                plt.subplots_adjust(hspace=.0)
        axes[0].get_yticklabels()[0].set_visible(False)
        axes[1].get_yticklabels()[0].set_visible(False)
        savefig_tofile(fig, args.resultdir, f"fit_charge_pars_{temp}_iter4", show=True)

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(25, 20))
        fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)   
        temp = "Be"
        
        for ipar, par  in enumerate(dict_pars[temp].keys()):
            if (ipar > 2):
                plot_graph(fig, axes[ipar-3], dict_pars[temp][par], color=tem_color[iT], label=None, style="EP", xlog=True, ylog=False, scale=None, markersize=22)
                pars_poly[temp][par] = np.polyfit(np.log(dict_pars[temp][par].getx()[3:-7]), dict_pars[temp][par].gety()[3:-7], poly_deg[par])
                splinefit[temp][par] = UnivariateSpline(dict_pars[temp][par].getx()[10:-1], dict_pars[temp][par].gety()[10:-1], k=3, s=5)
                y_splinefit[temp][par] = splinefit[temp][par](np.log(dict_pars[temp][par].getx()[10:-10]))
                ypoly = np.poly1d(pars_poly[temp][par])
                polynomials[par] = np.poly1d(pars_poly[temp][par])
                if (ipar < 5):
                    axes[ipar-3].plot(dict_pars[temp][par].getx(), ypoly(np.log(dict_pars[temp][par].getx())), "-", color=tem_color[iT])
                    
                axes[ipar-3].set_ylim(ylim_range[par])
                axes[ipar-3].set_xlim([1.9, 1300])
                plt.subplots_adjust(hspace=.0)
                
        #axes[2].plot(dict_pars[temp][par].getx(), chisquare[temp], '.', color='black', markersize=18)
        #axes[2].set_ylabel(r'$\mathrm{\chi^{2}/d.o.f}$')
        axes[2].set_xlabel('Rigidity(GV)')
        axes[2].set_xscale('log')
        axes[2].set_xlim([1.9, 1300])
        #axes[2].set_ylim([0, 4])
        axes[2].set_xticks([2, 5, 10, 100, 300, 500, 1000])
        axes[0].get_yticklabels()[0].set_visible(False)
        axes[1].get_yticklabels()[0].set_visible(False)
        set_plot_defaultstyle(axes[0])
        set_plot_defaultstyle(axes[1])
        set_plot_defaultstyle(axes[2])
        savefig_tofile(fig, args.resultdir, f"fit_charge_pars_{temp}_2_iter4", show=True)

        pars_poly[temp]['norm'] = np.polyfit(np.log(dict_pars[temp][par].getx()[1:-1]), dict_pars[temp][par].gety()[1:-1], 6)
        polynomials['norm'] = np.poly1d(pars_poly[temp]['norm'])


        fig, ax1 = plt.subplots(figsize=(18, 15))
        ax1.plot(dict_pars[temp][par].getx(), chisquare[temp], '.', color='black', markersize=18)
        ax1.set_ylabel(r'$\mathrm{\chi^{2}/d.o.f}$')
        ax1.set_xlabel('Rigidity(GV)')
        ax1.set_xscale('log')
        ax1.set_xlim([1.9, 1300])
        ax1.set_ylim([0, 5])
        
        with open(os.path.join(args.resultdir, 'polyfits_5.pickle'), 'wb') as file1:
            pickle.dump(polynomials, file1)

        with open(os.path.join(args.resultdir, 'polypars_5.pickle'), 'wb') as file2:
            pickle.dump(pars_poly, file2)

        
    plt.show()
    
if __name__ == "__main__":
    main()
    
