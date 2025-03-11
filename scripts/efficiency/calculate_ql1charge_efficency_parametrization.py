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
from tools.calculator import calc_ekin_from_beta, calc_ratio_err
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


def gaus_asygaus(x, mean, sigma, sigma_ratio, asy_factor, fraccore):
    coregaus = gaussian(x, mean, sigma)
    asygaus = asy_gaussian_1d(x, mean,  sigma_ratio * sigma, asy_factor)
    pdf = fraccore * coregaus + (1 - fraccore) * asygaus
    return pdf

def cumulative_gaus_asygaus(edges, mean, sigma, sigma_ratio, asy_factor, fraccore):
    x = (edges[1:] + edges[:-1])/2
    pdf = gaus_asygaus(x, mean, sigma, sigma_ratio, asy_factor, fraccore)
    cpdf = np.cumsum(pdf)
    return np.concatenate(([0], cpdf))



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--resultdir", default="plots/l1qBe", help="Directory to store plots and result files in.")
    parser.add_argument("--nuclei", default="Be", help="Directory to store plots and result files in.")
    parser.add_argument("--detectors", nargs="+", default=["RichAgl"], help="Directory to store plots and result files in.")
    parser.add_argument("--savedatadir", default="plots/l1qBe", help="dataname for the output file and plots.") 
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    
    hist_l1q = []
    nT = 3
    nuclei = args.nuclei
    templates_name = ["Be", "B", 'C']
    tem_color = ["tab:blue", "tab:red", "tab:green"]
    xbinning = Binning(kNucleiBinsRebin)
    ybinning = Binning(np.linspace(2, 14, 241))
    fit_charge_range = ybinning.get_indices([NUCLEI_CHARGE[nuclei]-0.4, NUCLEI_CHARGE[nuclei] + 2.0 + 0.7])
    print(NUCLEI_CHARGE[nuclei]-0.4, NUCLEI_CHARGE[nuclei] + 2.0 + 0.65)
    
    fitminbin = fit_charge_range[0]
    fitmaxbin = fit_charge_range[1]

    
    hist2d_qz = dict()
    with open('/home/manbing/Documents/Data/jiahui/efficiency/eff_cor_be_10yr_updated.pkl', 'rb') as f:
        data_jiahui_eff = pickle.load(f)

   # with open(os.path.join(args.resultdir, 'polyfits_4.pickle'), 'rb') as file:                                                                                                                                      
   #    par_polynomials = pickle.load(file)
        
    with uproot.open(f"/home/manbing/Documents/Data/data_BeP8/efficiency/{nuclei}_l1qtemplate_ISSP8GBL.root") as rootfile:
        for i, tem in enumerate(templates_name):
            qtem = i + int(NUCLEI_CHARGE[nuclei])
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
        boron_background_err = np.zeros(kNucleiNbinRebin)
        eff = np.zeros(kNucleiNbinRebin)
        efferr = np.zeros(kNucleiNbinRebin)
        
        graph_boron_background = MGraph(xvalues=get_bin_center(kNucleiBinsRebin), yvalues=np.zeros(len(kNucleiBinsRebin)-1), yerrs=np.zeros(len(kNucleiBinsRebin)-1))
        
        par_names = ['mean', 'sigma', 'sigma_ratio', 'asy_factor', 'fraccore']
        dict_pars = {temp: {par: MGraph(binning=xbinning, labels=["Rigidity(GV)", f"{par}"]) for par in par_names} for temp in templates_name}
        chisquare = {temp: np.zeros(kNucleiNbinRebin)  for temp in templates_name} 
        print(dict_pars.keys())

        minbin = 0
        maxbin = len(kNucleiBinsRebin) -1 
        #minbin = 24
        #maxbin = 26

        for ibin in range(minbin, maxbin):
            i_num = ibin - minbin
            hist_l1q.append(rootfile[f'h_l1q_ub_{ibin}'])
            templates = np.zeros([nT, 240])
            templateserrors = np.zeros([nT, 240])
            fit_templates = np.zeros([nT, 240])

            charge_xbin = ybinning.edges[1:-1][fitminbin:fitmaxbin+1]
            charge_binning_fit = Binning(charge_xbin)            
            
            xvalues_fit = get_bin_center(charge_xbin)
            yvalues_fit = hist_l1q[i_num].values()[fitminbin:fitmaxbin]
            yvalues_fit[yvalues_fit == 0] = 1
            yvalueserr_fit = hist_l1q[i_num].errors()[fitminbin:fitmaxbin]
            yvalueserr_fit[yvalueserr_fit == 0.0] = 1

            templates_par = np.zeros([nT, 5])
            
            for iT, temp in enumerate(templates_name):
                qsel = iT + NUCLEI_CHARGE[nuclei]
                fit_range = ybinning.get_indices([iT+2.55, iT+6.6])
                fit_range_tem = ybinning.get_indices([qsel-0.8, qsel+1.0])
                
                templates[iT] = hist2d_qz[temp].values[ibin+1, 1:-1]
                templateserrors[iT] = np.sqrt(templates[iT])
                templateserrors[iT] = templateserrors[iT]/np.sum(templates[iT] * ybinning.bin_widths[1:-1])
                templates[iT] = templates[iT]/np.sum(templates[iT] * ybinning.bin_widths[1:-1])
                
                hist_ibin = hist2d_qz[temp].project(ibin)
                loss = ExtendedBinnedNLL(templates[iT], hist_ibin.binnings[0].edges[1:-1], cumulative_gaus_asygaus)
                #guess = dict(mean=qsel, sigma=0.15, sigma_ratio=1.2, asy_factor=1.5, fraccore=0.5)
                guess = dict(mean=qsel-0.02, sigma=0.15, sigma_ratio=1.2, asy_factor=1.45, fraccore=0.3)
                #for par in guess.keys():                                                                                                                                                                         
                #     guess[par] = par_polynomials[par](np.log(kNucleiBinsRebin_center[ibin]))   

                m = Minuit(loss, **guess)
                guess['asy_factor'] = 1.5    
                m.limits["mean"]=(qsel-0.1, qsel+0.1) 
                m.limits["sigma"]=(0.1, 0.2)     
                m.limits["asy_factor"]=(1.3, 1.7)                                                                                                                                                               
                m.limits['sigma_ratio'] = (1.0, 2.0)
                m.limits['fraccore'] = (0.3, 0.7)
                #m.fixed['fraccore'] = True  
                m.migrad()
                print(m)                
                chisquare[temp][ibin] = m.fval / m.ndof      
                print('chi:', chisquare[temp][ibin])
                
                for par in dict_pars[temp].keys():
                    dict_pars[temp][par].yvalues[i_num] = m.values[par]
                    dict_pars[temp][par].yerrs[i_num] = m.errors[par]

                templates_par[iT] = np.array(m.values)
                print(templates_par[iT])
                
                xvalues = hist2d_qz[temp].binnings[1].bin_centers[1:-1]

                fit_templates[iT] = gaus_asygaus(xvalues, *m.values)
                fit_templates[iT] = fit_templates[iT]/np.sum(fit_templates[iT] * ybinning.bin_widths[1:-1])
                
                fit_guess = gaus_asygaus(xvalues, **guess)

                fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14)) 
                #plot_histogram_1d(ax1, hist_ibin, style="iss", color="black", label="data", scale=None, gamma=None, xlog=False, ylog=False, shade_errors=False)
                #ax1.plot(xvalues, templates[iT], ".", color="black", label="data", markersize=25)
                plot1d_errorbar_v2(fig, ax1, xvalues, counts=templates[iT], err=templateserrors[iT], label_x="Charge", label_y="counts",  style=".", color="black", setlogy=0, markersize=20)
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
                #ax1.set_yscale('log')
                #ax1.set_ylim([1, 10*np.max(templates[iT])])
                ax1.set_ylim([0, 1.3*np.max(templates[iT])])
                ax1.get_yticklabels()[0].set_visible(False)
                plot1d_errorbar_v2(fig, ax2, xvalues, counts=(templates[iT]-fit_templates[iT])/templateserrors[iT], err=np.zeros(len(xvalues)), label_x="Charge", label_y=r"(fit-data)/$\mathrm{\sigma_{data}}$",  style=".", color="black", setlogy=0, markersize=20)
                ax2.set_ylim([-3.9, 3.9])
                ax2.grid()
                plt.subplots_adjust(hspace=.0)                             

                savefig_tofile(fig, args.resultdir, f"fit/fit_{qsel}_{ibin}_linear", show=False)


            
            be_charge_l1_fitfunc = make_charge_template_fit(nT, templates_par)
            cost = LeastSquares(xvalues_fit, yvalues_fit, yvalueserr_fit, be_charge_l1_fitfunc)
            guess = {"T_0": 1000, "T_1": 50, "T_2": 50, "mu_0": NUCLEI_CHARGE[nuclei], "mu_1": NUCLEI_CHARGE[nuclei]+1.0 , "mu_2": NUCLEI_CHARGE[nuclei]+2.0}
            m_be_q = Minuit(cost, **guess)
            m_be_q.limits['mu_1'] = (NUCLEI_CHARGE[nuclei]+1 - 0.1, NUCLEI_CHARGE[nuclei]+1 +0.05)
            m_be_q.migrad()
            print(m_be_q)

            
            fit_bel1q = be_charge_l1_fitfunc(xvalues_fit, *m_be_q.values) 
            fig = plt.figure(figsize=(20, 13))
            ax1 = fig.subplots()
            
            plot1d_errorbar_v2(fig, ax1, get_bin_center(charge_xbin), counts=yvalues_fit, err=yvalueserr_fit, label_x="Charge", label_y="counts",  style=".",  label="L1Q", color="black", setlogy=1, markersize=28)
            ax1.axvline(x=NUCLEI_CHARGE[nuclei]+0.6, color='black', linestyle='--', linewidth=3.0)
            q_templates = dict()
            for iT, temp in enumerate(templates_name):
                q_templates[temp] = m_be_q.values[iT] * gaus_asygaus(get_bin_center(charge_xbin), m_be_q.values[iT+3], *(templates_par[iT][1:]))
                print(np.sum(gaus_asygaus(get_bin_center(charge_xbin), m_be_q.values[iT+3], *(templates_par[iT][1:]))))
                #plot1d_step(fig, ax1, xbinning=ybinning.edges[1:-1][fitminbin:fitmaxbin+1], counts= q_templates,
                #            err=None, label_x="L1 Z", label_y="counts",  legend=f"l2q {temp}", col=tem_color[iT], setlogy=1, alpha=0.8)
                ax1.fill_between(get_bin_center(charge_xbin), y1=q_templates[temp], y2=0, where=None, interpolate=False, step=None, alpha=0.5, color=tem_color[iT])

            ax1.plot(get_bin_center(charge_xbin), fit_bel1q, label="fit", color="tab:orange", linewidth=4.0)
            #plot1d_step(fig, ax1, xbinning=charge_xbin, counts=fit_bel1q, label_x="Charge", label_y="counts",
            #            err=np.zeros(len(fit_bel1q)), legend="fit", col="tab:orange", setlogy=1, linewidth=3.0)

            ax1.set_ylim([1, 10 * max(yvalues_fit)])
            ax1.set_xlim([NUCLEI_CHARGE[nuclei]-0.4, NUCLEI_CHARGE[nuclei]+2.0+0.6])
            ax1.text(0.65, 0.75, r"$\mathrm{{R = [{}, {}] GV}}$".format(xbinning.edges[ibin+1], xbinning.edges[ibin+2]), fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')
            ax1.text(0.15, 0.88, f"{templates_name[0]}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color=tem_color[0], weight='bold')
            ax1.text(0.5, 0.5, f"{templates_name[1]}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color=tem_color[1], weight='bold')
            ax1.text(0.8, 0.5, f"{templates_name[2]}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color=tem_color[2], weight='bold')
            ax1.legend(fontsize=FONTSIZE_BIG)

            #calculate background/purity
            cut_bin_index = charge_binning_fit.get_indices([NUCLEI_CHARGE[nuclei]-0.5, NUCLEI_CHARGE[nuclei]+0.7])
            #boron_background[ibin] = (q_templates[templates_name[1]][:cut_bin_index[1]].sum()/fit_bel1q[:cut_bin_index[1]].sum()) *0.9
            boron_background[ibin] = q_templates[templates_name[1]][:cut_bin_index[1]].sum()/fit_bel1q[:cut_bin_index[1]].sum()
            
            a1 = q_templates[templates_name[1]][:cut_bin_index[1]].sum()
            a2 = fit_bel1q[:cut_bin_index[1]].sum()
            boron_background_err[ibin] = calc_ratio_err(a1, a2, np.sqrt(a1), np.sqrt(a2))
            
            ax1.text(0.01, 0.98, r"$\mathrm{{Background = {:.2f}\% }}$".format(boron_background[ibin]*100), fontsize=FONTSIZE_BIG-2, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')
            savefig_tofile(fig, args.resultdir, f"l1q_templatefit_{ibin}_log", show=False)

            #calculate efficiency based on template Be
            eff[ibin] = q_templates[nuclei][cut_bin_index[0]:cut_bin_index[1]].sum()/q_templates[nuclei][cut_bin_index[0]:].sum()
            
            efferr[ibin] =  eff[ibin] * (1.0 - eff[ibin])/(q_templates[nuclei][:cut_bin_index[1]].sum()/20)

    
        be_signal = (1 - boron_background)
        graph_boron_background.yvalues = boron_background
        graph_boron_background.yerrs = boron_background_err
        print(graph_boron_background)
        
        fig = plt.figure(figsize=(16, 10))
        ax1 = fig.subplots()
        plot1d_errorbar_v2(fig, ax1, get_bin_center(kNucleiBinsRebin), counts=boron_background*100,
                           err=boron_background_err*100,  label_x="Rigidity (GV)", label_y="Background [%]", setlogx=1, label="l1q", color="black", markersize=28)
        ax1.text(0.05, 0.95, r"$\mathrm{L1Q < 4.7}$", fontsize=FONTSIZE_MID, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='normal')
        ax1.set_ylim([0.0, 1])
        ax1.set_xlim([1.9, 1000])
        ax1.set_xticks([2, 5, 10, 30,  100,  300, 1000])                                                                                                                                                   
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())    
        savefig_tofile(fig, args.resultdir, f"l1q_upppercut_background", show=True)

        #read jiahui data and compare
        pd_jiahui_l1purity = pd.read_csv("/home/manbing/Documents/Data/jiahui/efficiencies/l1q_purity.csv",  sep='\s+', header=0)
        graph_l1purity_jiahui = MGraph(xvalues=pd_jiahui_l1purity["bin_center[GV]"], yvalues=pd_jiahui_l1purity["dt_withcut"], yerrs=pd_jiahui_l1purity["dt_withcut_err"])
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 13))    
        fig.subplots_adjust(left= 0.12, right=0.96, bottom=0.1, top=0.95)
        plot1d_errorbar_v2(fig, ax1, get_bin_center(kNucleiBinsRebin), counts=be_signal,
                           err=boron_background_err,  label_x="Rigidity (GV)", label_y="Purity", setlogx=1, label="this", color="tab:orange", markersize=28)
        plot_graph(fig, ax1, graph_l1purity_jiahui, label=f"Jiahui", style="EP", markersize=21, color='black', markerfacecolor='none')
        spline_purity = UnivariateSpline(np.log(get_bin_center(kNucleiBinsRebin)[5:-1]), be_signal[5:-1], k=3, s=100)
        ax1.plot(get_bin_center(kNucleiBinsRebin), spline_purity(np.log(kNucleiBinsRebin_center)), "-", color="tab:orange")
        spline_purity_jiahui = UnivariateSpline(np.log(get_bin_center(kNucleiBinsRebin)[5:-1]), graph_l1purity_jiahui.yvalues[5:-1], k=3, s=100)
        ax1.plot(get_bin_center(kNucleiBinsRebin), spline_purity_jiahui(np.log(kNucleiBinsRebin_center)), "-", color="black")
        ax2.plot(get_bin_center(kNucleiBinsRebin), spline_purity(np.log(kNucleiBinsRebin_center))/spline_purity_jiahui(np.log(kNucleiBinsRebin_center)), "-", color="black")

        ax2.plot(kNucleiBinsRebin_center, be_signal/graph_l1purity_jiahui.yvalues, ".", color="black", markersize=20)
        #ax1.text(0.05, 0.95, r"$\mathrm{L1Q < 4.7}$", fontsize=FONTSIZE_MID, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='normal')
        ax1.set_ylim([0.97, 1.01])
        #ax2.plot(kNucleiBinsRebin_center, /eff_cor_l1q_jw(np.log(kNucleiBinsRebin_center)), "-", color="black")
        #x1.set_ylim([0.96, 1.02])
        ax2.set_ylim([0.995, 1.005])
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        ax2.set_xlim([1.9, 1300])
        ax2.set_xticks([2, 5,  10, 30,  80,  300, 1000])
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.get_yticklabels()[0].set_visible(False)
        ax1.sharex(ax2)
        ax1.legend(loc="lower right")
        ax2.set_xlabel("Rigidity (GV)")
        ax2.set_ylabel("this/J.W")
        ax1.text(0.05, 0.95, "Below L1 background", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
        ax2.grid(linewidth=1)
        set_plot_defaultstyle(ax1)
        set_plot_defaultstyle(ax2)
        plt.subplots_adjust(hspace=.0)
        ax2.text(0.05, 0.95, "Difference around 0.2%", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes, color="black", weight='bold') 
        savefig_tofile(fig, args.resultdir, f"l1q_upppercut_purity", show=True)                        


        #n_pass = hist2d_qz["Be"].values[:,:cutlim_index[1]].sum(axis=1)
        #n_tot = hist2d_qz["Be"].values.sum(axis=1)  
        #eff_c, efferr_c = calculate_efficiency_and_error(n_pass, n_tot, "ISS")
        fig = plt.figure(figsize=(16, 14))
        ax1 = fig.subplots()
        plot1d_errorbar_v2(fig, ax1,  get_bin_center(kNucleiBinsRebin), counts=eff,
                           err=efferr,  label_x="Rigidity (GV)", label_y="Efficiency", color="tab:orange", setlogx=1, markersize=25)
        ax1.set_ylim([0.95, 1.01])
        ax1.text(0.05, 0.95, r"$\mathrm{L1Q < 4.7}$", fontsize=FONTSIZE_MID, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='normal')
        #spline_l1q_eff = UnivariateSpline(np.log(get_bin_center(kNucleiBinsRebin)[1:-3:2]), eff[1:-3:2], w=1/efferr[1:-3:2], k=5, s=1000)
        spline_l1q_effcor = UnivariateSpline(np.log(get_bin_center(kNucleiBinsRebin)[1:-1]), eff[1:-1], k=3, s=100)
        spline_2 = make_interp_spline(np.log(get_bin_center(kNucleiBinsRebin)[1:-3]),  eff[1:-3], k=3)
        save_spline_to_file(spline_l1q_effcor, args.savedatadir, "spline_l1q_effcor.pickle")
        savefig_tofile(fig, args.resultdir, f"l1q_upppercut_efficiency", show=True)

        #open jiahui L1Q efficiency csv file:
        pd_jiahui_lq = pd.read_csv("/home/manbing/Documents/Data/jiahui/efficiencies/l1_upper_limit_eff_data_alone.csv",  sep='\s+', header=0)    
        graph_lq_jiahui = MGraph(xvalues=pd_jiahui_lq["bin_center[GV]"], yvalues=pd_jiahui_lq["dt"], yerrs=pd_jiahui_lq["dt_err"])
        
        initial_guess = np.zeros(3) # Initial guess for the polynomial coefficients
        print(eff,efferr)
        eff_mod = eff
        eff_mod[53:] = 0.994 * np.ones_like(eff_mod[53:])
        fit_coeffs, _ = curve_fit(poly_func, np.log(get_bin_center(kNucleiBinsRebin)[:57]), eff_mod[:57], sigma=1/efferr[:57],  p0=np.zeros(4))
        fit_l1q =  poly_func(np.log(get_bin_center(kNucleiBinsRebin)), *fit_coeffs)
        print(fit_l1q)

        
        #plot the efficiency corrections all together
        #read the L1Q Charge efficiency
        eff_cor_l1q_jw = data_jiahui_eff[2][3]
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 13))    
        figure.subplots_adjust(left= 0.12, right=0.96, bottom=0.1, top=0.95)

        plot1d_errorbar_v2(fig, ax1,  get_bin_center(kNucleiBinsRebin), counts=eff,
                           err=efferr,  label_x="Rigidity (GV)", label_y="Efficiency", color="tab:orange", setlogx=1, markersize=25, label=f"{nuclei} ISS")
        plot1d_errorbar_v2(figure, ax1,  graph_lq_jiahui.getx(), graph_lq_jiahui.gety(), err=graph_lq_jiahui.get_yerrors(), label="Jiahui ISS", color="grey", style=".", markersize=20)
        ax1.plot(kNucleiBinsRebin_center, eff_cor_l1q_jw(np.log(kNucleiBinsRebin_center)), "-", color="black")
        ax1.plot(kNucleiBinsRebin_center, spline_l1q_effcor(np.log(kNucleiBinsRebin_center)), "-", color="tab:orange")
        #ax1.plot(kNucleiBinsRebin_center, spline_2(np.log(kNucleiBinsRebin_center)), "--", color="tab:orange")
        #ax1.plot(kNucleiBinsRebin_center, fit_l1q, "-", color="tab:orange")
        ax2.plot(kNucleiBinsRebin_center, fit_l1q/eff_cor_l1q_jw(np.log(kNucleiBinsRebin_center)), "-", color="black")

        print(graph_lq_jiahui.getx(), len(graph_lq_jiahui.getx()))
        #ax2.plot(kNucleiBinsRebin_center, eff/graph_lq_jiahui.gety(), ".", color="black", markersize=20)
        ax1.set_ylabel("Efficiency")
        ax1.set_ylim([0.93, 1.02])
        ax2.set_ylim([0.995, 1.01])
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        ax2.set_xlim([1.9, 1300])
        ax2.set_xticks([2, 5,  10, 30,  80,  300, 1000])
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.get_yticklabels()[0].set_visible(False)
        ax1.sharex(ax2)
        ax1.legend(loc="lower left")
        ax2.set_xlabel("Rigidity (GV)")
        ax2.set_ylabel("this/J.W")
        ax1.text(0.05, 0.95, "L1Q upper cut", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
        ax2.grid(linewidth=1)
        set_plot_defaultstyle(ax1)
        set_plot_defaultstyle(ax2)
        ax2.text(0.05, 0.95, "Difference around 0.5%", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes, color="black", weight='bold') 
        plt.subplots_adjust(hspace=.0)
        savefig_tofile(figure, args.resultdir, f"hist_compare_jw_l1q_effcor", show=True)

        dict_savegraph = {}
        graph_eff_cor_l1q = MGraph(get_bin_center(kNucleiBinsRebin), eff, efferr)
        graph_eff_cor_l1q.add_to_file(dict_savegraph, 'graph_effcor_l1q')
        graph_boron_background.add_to_file(dict_savegraph, 'graph_boron_background')
        np.savez(os.path.join(args.savedatadir, f'graph_l1q_effcor_{nuclei}.npz'), **dict_savegraph)
        

    plt.show()
                        
if __name__ == "__main__":
    main()
    
