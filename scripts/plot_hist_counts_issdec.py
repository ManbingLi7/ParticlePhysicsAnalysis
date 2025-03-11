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
from tools.binnings_collection import mass_binning, fbinning_energy, LithiumRigidityBinningFullRange, Rigidity_Analysis_Binning_FullRange
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar_v2, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, FONTSIZE_BIG, FONTSIZE_MID, plot1d_errorbar, plot1d_step, FONTSIZE
from tools.calculator import calc_rig_from_ekin, calc_ratio_err, calc_ekin_from_beta, calc_mass
from tools.constants import ISOTOPES_MASS, NUCLEI_CHARGE, NUCLEIS, DETECTOR_COLOR
from tools.constants import * 
from tools.histograms import Histogram, plot_histogram_2d
from tools.binnings import Binning
from tools.roottree import read_tree
from tools.selections import *

import pickle
from tools.graphs import MGraph, slice_graph, slice_graph_by_value, plot_graph

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

    plot1d_errorbar(figure, ax1, x_binning, ref, err=ref_err, label_x=xlabel, label_y=ylabel, col=colorB, legend=legendB)        
    plot1d_errorbar(figure, ax1, x_binning, com, err=com_err, label_x=xlabel, label_y=ylabel, col=colorA, legend=legendA)

    #plot1d_step(figure, ax1,  x_binning, ref, err=ref_err, label_x=xlabel, label_y=ylabel, col=colorB, legend=legendB)
    pull = np.array(com/ref)
    print(pull)
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
    parser.add_argument("--resultdir", default="plots/counts2")
    parser.add_argument("--datadir", default="/home/manbing/Documents/Data/data_be_flux")
    parser.add_argument("--filetime", default="/home/manbing/Documents/Data/expo_time/expo_time_sf_finebin_10yr.root", help="Path to file measuring time")
    parser.add_argument("--fileacc", default="trees/acceptance/acc_nucleiselection.csv", help="Path to file acceptance")
    parser.add_argument("--fileeff", default="$DATA/data_flux/", help="Path to file acceptance")
    parser.add_argument("--variable", default="Ekin", help="analysis in rigidity or in kinetic energy per nucleon")
    parser.add_argument("--nuclei", default="Be", help="the analyzed nuclei")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")  
    args = parser.parse_args()                                                    
    os.makedirs(args.resultdir, exist_ok=True)
    nuclei = args.nuclei
    detectors = ["Tof", "Agl", "NaF"]
    detectors_alias = {"Tof": "tof", "Agl":"agl", "NaF": "naf"}
    variable = args.variable
    massbinning = Binning(mass_binning())
    inverse_mass_binning = Binning(np.linspace(0.05, 0.3, 100))
    
    #xbinning = Binning(LithiumRigidityBinningFullRange())
    xbinning = {"Rigidity": Binning(Rigidity_Analysis_Binning_FullRange()), "Ekin":Binning(fbinning_energy())}
    xlabel = {"Rigidity": "Rigidity(GV)", "Ekin": "Ekin/n (GeV/n)"}
    
    jiahui_filename = {dec: f'/home/manbing/Documents/Data/jiahui/isotope_fluxes/Event_counts_before_fit/evt_count_verR_{dec_alias}_10yr.txt' for dec, dec_alias in detectors_alias.items()}
    jiahui_filename_ekin = {dec: f'/home/manbing/Documents/Data/jiahui/isotope_fluxes/Event_counts_before_fit/evt_count_{dec_alias}_Be7_optimized_10yr.txt' for dec, dec_alias in detectors_alias.items()}

    jiahui_ref = {"Rigidity": jiahui_filename, "Ekin": jiahui_filename_ekin}
    

    dict_graph_counts_betalimit = dict()
    dict_hist_counts_betalimit = dict()
    
    graph_counts = dict()
    sub_graph_counts = dict()
    graph_jiahui_counts = dict()
    sub_graph_jiahui_counts = dict()
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(21, 14))
    with np.load(os.path.join(args.resultdir, f"{nuclei}_dict_graph_counts_{variable}.npz")) as countsfile:
        for dec in detectors:
            graph_counts[dec] = MGraph.from_file(countsfile, f"{dec}_counts")
            print(graph_counts[dec])
            sub_graph_counts[dec] = slice_graph_by_value(graph_counts[dec], ANALYSIS_RANGE_EKIN[dec])
            
            #hist_counts[dec].add_to_file(dict_hist_counts, f"{nuclei}_{dec}_counts")
            pd_jiahuicounts = pd.read_csv(jiahui_ref[variable][dec],  sep='\s+', header=0)    
            jiahui_counts = np.array(pd_jiahuicounts['EvtCount'])
            jiahui_countserr = np.zeros(len(jiahui_counts))


            print(pd_jiahuicounts)

            x = xbinning[variable].bin_centers[1:-1]
            xbinedge = xbinning[variable].edges[1:-1]
            graph_jiahui_counts = MGraph(get_bin_center(xbinedge), jiahui_counts, jiahui_countserr)
            sub_graph_jiahui_counts = slice_graph_by_value(graph_jiahui_counts, ANALYSIS_RANGE_EKIN[dec])
            plot_graph(fig, ax1, sub_graph_jiahui_counts, color='magenta', label=None, style="EP", xlog=1, ylog=1, scale=None, markersize=20, marker='+')
            plot_graph(fig, ax1, sub_graph_counts[dec], color='black', label=None, style="EP", xlog=True, ylog=True, scale=None, markersize=22)

            plot_graph(fig, ax2, sub_graph_jiahui_counts/sub_graph_counts[dec], color='black', label=None, style="EP", xlog=1, ylog=0, scale=None, markersize=20)
            
            


            #plot_comparison_nphist(figure, ax1, ax2, x_binning=xbinedge, com=sub_graph_counts[dec].yvalues, com_err=graph_counts[dec].yerrs, ref=sub_graph_jiahui_counts.,
             #              ref_err=jiahui_countserr, ylabel="counts", xlabel=xlabel[variable],
             #             legendA=f"this {dec}", legendB=f"J.W {dec}")

    #ax1.set_yscale("log")
    ax1.fill_betweenx(np.linspace(500, 200000, 1000), ANALYSIS_RANGE_EKIN["NaF"][0],  ANALYSIS_RANGE_EKIN["NaF"][1], alpha=0.1, color='tab:blue')
    ax2.fill_betweenx(np.linspace(0.98, 1.02, 100), ANALYSIS_RANGE_EKIN["NaF"][0],  ANALYSIS_RANGE_EKIN["NaF"][1], alpha=0.1, color='tab:blue')
    ax2.grid()
    ax2.set_ylim([0.98, 1.02])
    ax1.set_ylim([500, 200000])
    #ax1.text(0.98, 0.75, "Raw Flux", fontsize=30, verticalalignment='top', horizontalalignment='right', transform=ax1.transAxes, color="black", weight='bold')
    #ax1.text(0.98, 0.68, "this: no background subtraction", fontsize=30, verticalalignment='top', horizontalalignment='right', transform=ax1.transAxes, color="black", weight='bold') 
    #ax2.plot(x, [1]*len(x), 'b--')
    plt.subplots_adjust(hspace=.0)                             

    ax1.legend()                                         
    ax2.sharex(ax1)
    ax2.set_xlabel("Ekin/n (GeV/n)")
    ax2.set_ylabel("Ratio")
    ax1.set_ylabel("Counts")
    ax1.text(0.05, 0.98, f"Tof", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')
    ax1.text(0.4, 0.98, f"NaF", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')
    ax1.text(0.7, 0.98, f"Aerogel", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')
    ax2.text(0.05, 0.98, f"J.W/this", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes, color="black", weight='bold')
    ax1.legend()

    savefig_tofile(fig, args.resultdir, f"counts_compare_jiahui_alldec_{variable}", 1)
        
    plt.show()

if __name__ == "__main__":   
    main()

    
