import ROOT
import uproot
import uproot.behaviors.TGraph
import uproot3                                                                              
import os
import csv
import pandas as pd
import numpy as np
import multiprocessing as mp
import awkward as ak   
import matplotlib.pyplot as plt
import seaborn as sns
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center, compute_dayfromtime
from tools.binnings_collection import mass_binning, fbinning_energy, LithiumRigidityBinningFullRange, BeRigidityBinningRICHRange
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar_v2, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, FONTSIZE_BIG, FONTSIZE_MID, plot1d_errorbar, plot1d_step
from tools.calculator import calc_rig_from_ekin, calc_ratio_err, calc_ekin_from_beta
from tools.constants import ISOTOPES_MASSES, NUCLEI_CHARGE, NUCLEIS, ISOTOPES, ISOTOPES_COLOR
from tools.histograms import Histogram
from tools.binnings import Binning
from tools.roottree import read_tree
from tools.selections import *
import pickle
from tools.graphs import MGraph, slice_graph, plot_graph
from scipy.optimize import curve_fit
from tools.statistics import poly_func 

setplot_defaultstyle()
def plot_comparison_histogram(figure=None, ax1=None, ax2=None, comhist=None, refhist=None, xlabel=None, ylabel=None):
    if figure == None:
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14)) 
        
    plot1d_errorbar(figure, ax1, hist_a.binnings[0].bin_centers[1:-1], comhist.values, err=comhist.get_errors(), label_x=xlabel, label_y=ylabel, col='tab:blue')
    plot1d_step(figure, ax1,  hist_a.binnings[0].bin_centers[1:-1], refhist.values, err=refhist.get_errors(), label_x=xlabel, label_y=ylabel, col="tab:orange")
    pull = comhist.values/refhist.values
    pull_err = calc_ratio_err(comhist.values, refhist.values, comhist.get_errors(), refhist.get_errors())    
    plot1d_errorbar(figure, ax2, hist_a.binnings[0].bin_centers[1:-1], counts=pull, err=pull_err,  label_x=xlabel, label_y=ylabel, legend=None,  col="black", setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
    plt.subplots_adjust(hspace=.0)                             
    ax1.legend()                                         
    ax2.sharex(ax1)


def plot_comparison_nphist(figure=None, ax1=None, ax2=None, x_binning=None, com=None, com_err=None, ref=None, ref_err=None, xlabel=None, ylabel=None, legendA=None, legendB=None, colorA="tab:orange", colorB="tab:green", colorpull="black"):
    if figure == None: 
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))

    #plot1d_errorbar(figure, ax1, x_binning, ref, err=ref_err, label_x=xlabel, label_y=ylabel, col=colorB, legend=legendB)
    plot1d_step(figure, ax1,  x_binning, ref, err=ref_err, label_x=xlabel, label_y=ylabel, col=colorB, legend=legendB)  
    plot1d_errorbar(figure, ax1, x_binning, com, err=com_err, label_x=xlabel, label_y=ylabel, col=colorA, legend=legendA)

    #plot1d_step(figure, ax1,  x_binning, ref, err=ref_err, label_x=xlabel, label_y=ylabel, col=colorB, legend=legendB)
    pull = np.array(com/ref)
    
    #pull_err = ratioerr(pull, com, ref, com_err, ref_err)
    pull_err = np.zeros(len(pull))   
    plot1d_errorbar(figure, ax2, x_binning, counts=pull, err=pull_err,  label_x=xlabel, label_y=r"$\mathrm{this/ref}$", legend=None,  col=colorpull, setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
    plt.subplots_adjust(hspace=.0)                             
    ax1.legend()                                         
    ax2.sharex(ax1)

def read_values_from_hist(hist):
    values = hist.values
    errors = hist.get_errors()
    return values, errors

def calculate_flux(hist_counts, hist_acc, hist_effcorr, hist_measuringtime, binning):
    xbinning = histcounts.binnings[0]
    counts, countserr = read_values_from_hist(hist_counts)
    acceptance, accerr = read_values_from_hist(hist_acc)
    effcorr, effcorrerr = read_values_from_hist(hist_effcorr)
    time = hist_measuringtime.values
    deltabin = binning[1:-1] - binning[:-2]
    
    flux = counts/(acceptance * effcor * time * deltabin)
    fluxerr = flux * np.sqrt((countserr/counts)**2 + (accerr/acceptance)**2 + (effcorrerr/effcorr)**2)

rich_selectors = {"LIP": {"Tof": selector_tof, "NaF": selector_naf_lipvar, "Agl": selector_agl_lipvar},    
             "CIEMAT": {"Tof":selector_tof, "NaF": selector_naf_ciematvar, "Agl": selector_agl_ciematvar}}   

def main():                                                                                 
    import argparse                                                                     
    parser = argparse.ArgumentParser()                                                      
    parser.add_argument("--filename", default="/home/manbing/Documents/Data/data_iss/BeISS_NucleiSelection_BetaCor.root",help="Path to fils")
    parser.add_argument("--treename", default="amstreea",help="Path to fils")                                                                       
    parser.add_argument("--resultdir", default="plots/isotopesflux")
    parser.add_argument("--filetime", default="/home/manbing/Documents/Data/expo_time/expo_time_sf_finebin_10yr.root", help="Path to file measuring time")
    parser.add_argument("--fileacc", default="trees/acceptance/acc_nucleiselection.csv", help="Path to file acceptance")
    parser.add_argument("--fileeff", default="$DATA/data_flux/", help="Path to file acceptance")
    parser.add_argument("--variable", default="Ekin", help="analysis in rigidity or in kinetic energy per nucleon")
    parser.add_argument("--nuclei", default="Be", help="the analyzed nuclei")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")  
    args = parser.parse_args()                                                    
    os.makedirs(args.resultdir, exist_ok=True)
    nuclei = args.nuclei
    detectors = ["NaF", "Agl"]
    detectors_alias = {"Tof":"tof", "NaF":"naf", "Agl": "agl"}
    variable = args.variable

    #xbinning = Binning(LithiumRigidityBinningFullRange())
    xbinning = {"Rigidity": Binning(BeRigidityBinningRICHRange()), "Ekin":Binning(fbinning_energy())}
    xlabel = {"Rigidity": "Rigidity(GV)", "Ekin": "Ekin/n (GeV/n)"}


    dict_graph_counts = dict() 
    graph_eff = dict()

    #read the tot efficiency corrections
    with open('plots/effcor/total/spline_Be7_nucleiselection_effcor.pickle', 'rb') as f:
        spline_be_nuclei_effcor = pickle.load(f)
        
        
    be_nuclei_effcor = spline_be_nuclei_effcor(np.log(xbinning[variable].bin_centers[1:-1]))

    with np.load("plots/effcor/rich/graph_rich_effcor.npz") as file_sub_detector_effcor:
        dict_graph_sub_dec_effcor = {dec:  MGraph.from_file(file_sub_detector_effcor, f"{dec}") for dec in detectors}

    total_effcor = dict()
    graph_total_effcor = dict()
    graph_total_effcor_ekin = dict()
    spline_rich_eff = dict()

    fit_range = {"NaF": [5, 100], "Agl": [10, 100]}
    dict_fit_pars = dict()
    graph_total_effcor_ekin = {dec: dict() for dec in detectors}
    xrigidity_from_ekin = dict()
    
    for dec in detectors:
        graph_be_effcor = MGraph(xbinning[variable].bin_centers[1:-1], be_nuclei_effcor, np.zeros(len(be_nuclei_effcor)))
        total_effcor[dec] = graph_be_effcor.yvalues * dict_graph_sub_dec_effcor[dec].yvalues[1:-1]
        
        graph_total_effcor[dec] = MGraph(graph_be_effcor.xvalues, total_effcor[dec], np.zeros(len(total_effcor[dec])))
        

        fit_min = graph_total_effcor[dec].get_index(fit_range[dec][0])                                                                                                                        
        fit_max = graph_total_effcor[dec].get_index(fit_range[dec][1])
        slice_graph_effcor = slice_graph(graph_total_effcor[dec], fit_min, fit_max)
        dict_fit_pars[dec], _ = curve_fit(poly_func, np.log(slice_graph_effcor.xvalues), slice_graph_effcor.yvalues, p0=np.zeros(3))     
        
        for iso in ISOTOPES[nuclei]:
            xrigidity_from_ekin[iso] = calc_rig_from_ekin(xbinning["Ekin"].bin_centers[1:-1], ISOTOPES_MASSES[iso], NUCLEI_CHARGE[nuclei])
            eff_cor_ekin = poly_func(np.log(xrigidity_from_ekin[iso]), *dict_fit_pars[dec])
            graph_total_effcor_ekin[dec][iso] = MGraph(xbinning["Ekin"].bin_centers[1:-1], eff_cor_ekin)
         
    graph_corrected_acc = {dec: dict() for dec in detectors}
    
    with np.load("plots/acceptance/Be_dict_graph_rawacc_Ekin_v0.npz") as file_acc:
        graph_acc = {dec: {iso: MGraph.from_file(file_acc, f"raw_acc_{dec}_{iso}") for iso in ISOTOPES[nuclei]} for dec in detectors}
        
    #plot corrected acceptance
    for dec in detectors:
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(21, 14))    
        for i, iso in enumerate(ISOTOPES[nuclei]):

            acc_corr = graph_acc[dec][iso].yvalues * graph_total_effcor_ekin[dec][iso].yvalues
            graph_corrected_acc[dec][iso] = MGraph(xbinning["Ekin"].bin_centers[1:-1], acc_corr, graph_acc[dec][iso].yerrs)
            
            #plot_graph(figure, ax1, graph_acc[dec][iso], color=ISOTOPES_COLOR[iso], label=f"{iso}", style="EP", xlog=True, ylog=False)
            plot_graph(figure, ax1, graph_corrected_acc[dec][iso], color=ISOTOPES_COLOR[iso], label=f"{iso}", style="EP", xlog=True, ylog=False, markersize=20)
            ax1.legend()
            ax2.set_ylim([0.9, 1.1])
            ax1.set_xscale("log")
            ax1.set_ylabel(r"$\mathrm{Acceptance (m^{2} sr)}$")
            ax1.text(0.05, 0.98, f"{dec}", fontsize=30, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')
            
        savefig_tofile(figure, args.resultdir, f"acc_{nuclei}_{dec}_{variable}", 1)

    
    acc_jiahuifile = {dec: {iso :f'/home/manbing/Documents/Data/jiahui/isotope_fluxes/Acceptance/{dec}_{iso}.txt' for iso in ISOTOPES[nuclei]} for dec in detectors}
    acc_ref = {detector: dict() for detector in detectors}
    for dec in detectors:
        for iso in ISOTOPES[nuclei]:  
            acc_ref[dec][iso] = pd.read_csv(acc_jiahuifile[dec][iso],  sep='\s+', header=0)
            
    for dec in detectors:
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))    
        for i, iso in enumerate(ISOTOPES[nuclei]):
            plot_comparison_nphist(figure, ax1, ax2, x_binning=xbinning["Ekin"].edges[1:-1], com=graph_corrected_acc[dec][iso].yvalues,
                                   com_err=graph_corrected_acc[dec][iso].yerrs, ref=acc_ref[dec][iso]["Eff_Acc_corrected"],
                                   ref_err=np.zeros(len(acc_ref[dec][iso]["Eff_Acc_corrected"])),
                                   xlabel="Ekin/n(GeV/n", ylabel=r"$\mathrm{Acceptance_{corr} (m^{2} sr)}$",
                                   legendA=f"this {iso}", legendB=f"jiahui {iso}", colorA=ISOTOPES_COLOR[iso], colorB=ISOTOPES_COLOR[iso])
        ax2.set_ylim([0.9, 1.1])
        ax1.text(0.05, 0.98, f"{dec}", fontsize=30, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')
        savefig_tofile(figure, args.resultdir, f"Acceptance_corr_{nuclei}_{dec}_{variable}", 1)


    for dec in detectors:
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))    
        for i, iso in enumerate(ISOTOPES[nuclei]):
            plot_comparison_nphist(figure, ax1, ax2, x_binning=xbinning["Ekin"].edges[1:-1], com=graph_corrected_acc[dec][iso].yvalues,
                                   com_err=graph_corrected_acc[dec][iso].yerrs, ref=graph_acc[dec][iso].yvalues, ref_err=graph_acc[dec][iso].yerrs,
                                   xlabel="Ekin/n(GeV/n", ylabel=r"$\mathrm{Acceptance (m^{2} sr)}$",
                                   legendA=f"corrected {iso}", legendB=f"raw {iso}", colorA=ISOTOPES_COLOR[iso], colorB=ISOTOPES_COLOR[iso])
        ax2.set_ylim([0.9, 1.1])
        ax1.text(0.05, 0.98, f"{dec}", fontsize=30, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')
        savefig_tofile(figure, args.resultdir, f"Acceptance_compare_raw_corr_{nuclei}_{dec}_{variable}", 1)



    plt.show()

if __name__ == "__main__":   
    main()

    
