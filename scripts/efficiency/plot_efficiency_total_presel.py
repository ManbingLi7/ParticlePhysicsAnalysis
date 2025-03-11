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
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, FIGSIZE_MID, FIGSIZE_BIG, setplot_defaultstyle, format_order_of_magnitude, FONTSIZE, savefig_tofile, FONTSIZE_BIG, plot1d_errorbar_v2, tick_length, tick_labelsize, tick_width, set_plot_defaultstyle
from tools.studybeta import calc_signal_fraction, hist1d, hist1d_weighted
from tools.binnings_collection import  fbinning_energy
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, cdf_double_gaus, double_gaus
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d
from tools.binnings import Binning
from tools.constants import NUCLEI_CHARGE, ANALYSIS_RANGE_RIG, ISOTOPES_COLOR, ISOTOPES
from tools.calculator import calc_ekin_from_beta
from tools.calculator import calculate_efficiency_and_error, calculate_efficiency_and_error_weighted, calculate_efficiency_weighted
from tools.statistics import poly_func
from tools.graphs import MGraph, slice_graph, concatenate_graphs, plot_graph
from tools.utilities import get_spline_from_graph, save_spline_to_file, get_graph_from_spline, get_spline_from_file
import pickle
from scipy.interpolate import make_interp_spline, BSpline
from scipy.interpolate import UnivariateSpline
import pandas as pd
from tools.jupytertools import *

#xbinning = fbinning_energy_agl()

kNucleiBinsRebin = np.array([0.8,1.00,1.16,1.33,1.51,1.71,1.92,2.15,2.40,2.67,2.97,3.29,3.64,4.02,4.43,4.88,    
                             5.37,5.90,6.47,7.09,7.76,8.48,9.26, 10.1,11.0,12.0,13.0,14.1,15.3,16.6,18.0,19.5,21.1,22.8,24.7,26.7,28.8,31.1,33.5,36.1,    
                             38.9,41.9,45.1,48.5,52.2,60.3,69.7,80.5,93.0,108., 116.,147.,192.,259.,379.,660., 1300, 3300.])
kNucleiBinsRebin_center = get_bin_center(kNucleiBinsRebin)
RIG_XLIM =[1.8, 1000]
RIG_XLABEL = "Rigidity (GV)"

nuclei = 'Be'

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--resultdir", default="plots/total", help="Directory to store plots and result files in.")
    parser.add_argument("--datadir", default=f"/home/manbing/Documents/Data/data_{nuclei}P8/efficiency/spline_effcor", help="dataname for the output file and plots.")
    
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)

    
    
    spline_l1q_effcor = get_spline_from_file(os.path.join(args.datadir,"spline_l1q_effcor.pickle"), "spline_l1q_eff")
    #polyfit_l1q_eff = get_nppolyfit(np.log(kNucleiBinsRebin_center), spline_l1q_effcor(kNucleiBinsRebin_center), 2)
    l1q_effcor = spline_l1q_effcor(np.log(kNucleiBinsRebin_center))
    
    spline_bz_effcor = get_spline_from_file(os.path.join(args.datadir, "spline_bz_effcor.pickle"), "spline_bz_eff")
    bz_effcor = spline_bz_effcor(np.log(kNucleiBinsRebin_center))

    spline_inntrk_effcor = get_spline_from_file(os.path.join(args.datadir, "spline_inntrk_effcor.pickle"), "spline_inntrk_effcor")
    inntrk_effcor = spline_inntrk_effcor(np.log(kNucleiBinsRebin_center))

    spline_tofq_effcor = get_spline_from_file(os.path.join(args.datadir, "spline_tofq_effcor.pickle"), "spline_tofq_effcor")
    tofq_effcor = spline_tofq_effcor(np.log(kNucleiBinsRebin_center))

    spline_trigger_effcor = get_spline_from_file(os.path.join(args.datadir, "spline_trigger_effcor.pickle"), "spline_trigger_effcor")
    trigger_effcor = spline_trigger_effcor(np.log(kNucleiBinsRebin_center))

    xbinning = Binning(kNucleiBinsRebin)
    tran_index = xbinning.get_indices([80])[0]

    spline_innq_effcor = get_spline_from_file(os.path.join(args.datadir, "spline_innq_effcor.pickle"), "spline_innq_effcor")                                                                                    
    innq_effcor = spline_innq_effcor(np.log(kNucleiBinsRebin_center))

    spline_sedtrk_effcor = get_spline_from_file(os.path.join(args.datadir, "spline_sedtrk_effcor_he.pickle"), "spline_sedtrk_effcor")                                                                           
    sedtrk_effcor = spline_sedtrk_effcor(np.log(kNucleiBinsRebin_center))



    with np.load(os.path.join(args.datadir,  "graph_inntrk_effcor.npz")) as file_bz:
        graph_inntrk_effcor = MGraph.from_file(file_bz, "graph_inntrk_effcor")


    dict_total_effcor = dict()
    #Tof
    figure, ax1 = plt.subplots(1, 1, figsize=(23, 13))
    figure.subplots_adjust(left= 0.13, right=0.96, bottom=0.12, top=0.95)
    ax1.plot(kNucleiBinsRebin_center, bz_effcor, "--", color="tab:orange", label="Big-Z", linewidth=3)
    ax1.plot(kNucleiBinsRebin_center, tofq_effcor, "--", color="tab:green", label="UTof Q", linewidth=3)
    ax1.plot(kNucleiBinsRebin_center, trigger_effcor, "--", color="magenta", label="Trigger", linewidth=3)
    ax1.plot(kNucleiBinsRebin_center, l1q_effcor, "--", color="cyan", label="L1Q", linewidth=3)
    #ax1.plot(kNucleiBinsRebin_center, tofvelocity_effcor, "-", color="brown", label="Tof-V")
    ax1.plot(graph_inntrk_effcor.xvalues, graph_inntrk_effcor.yvalues, "--", color="tab:blue", label="Inn Trk", linewidth=3)
    ax1.plot(kNucleiBinsRebin_center, sedtrk_effcor, "--", color="grey", label="Background", linewidth=3)
    ax1.plot(kNucleiBinsRebin_center, innq_effcor, "--", color="red", label="Inn Trk Q", linewidth=3)    

    print(kNucleiBinsRebin_center)
    print(graph_inntrk_effcor.xvalues)

    total_effcor_tof = bz_effcor[5:] * tofq_effcor[5:] * trigger_effcor[5:] * l1q_effcor[5:] *  graph_inntrk_effcor.yvalues * innq_effcor[5:] * sedtrk_effcor[5:]
    graph_total_effcor_tof = MGraph(kNucleiBinsRebin_center[5:], total_effcor_tof, yerrs=np.zeros_like(total_effcor_tof))
    graph_total_effcor_tof.add_to_file(dict_total_effcor, "graph_total_effcor_tof")

    spline_total_effcor_tof =  UnivariateSpline(np.log(graph_total_effcor_tof.getx()[1:70]), total_effcor_tof[1:70], k=3, s=5)   
    ax1.plot(kNucleiBinsRebin_center[5:], spline_total_effcor_tof(np.log(graph_total_effcor_tof.getx())), '-', label='Total', color='black', linewidth=4)

    ax1.grid(axis='y')
    ax1.set_ylim([0.91, 1.048])
    ax1.set_xscale("log")
    ax1.set_xlim([1.8, 1300])
    ax1.set_xticks([2, 5,  10, 20, 50, 100, 1000])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax1.set_xlabel(RIG_XLABEL, fontsize=FONTSIZE, fontweight='bold')
    ax1.set_ylabel("Efficiency Correction", fontsize=FONTSIZE, fontweight='bold')
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FONTSIZE-5)
    plt.subplots_adjust(right=0.75)
    #set_plot_style(ax1)
    #ax1.text(0.1, 0.98, f"Tof", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
    
    plt.rcParams["font.weight"] = "bold"                                                                                                                                     
    plt.rcParams["axes.labelweight"] = "bold"                                                                                                                                  
    plt.rcParams['font.size']= 28                                                                                                                                              
    plt.rcParams['xtick.top'] = True                                                                                                                                           
    plt.rcParams['ytick.right'] = True                                                                                                                                           
    ax1.tick_params(axis='both', which="major",direction='in', length=tick_length, width=tick_width, labelsize=tick_labelsize)                                                
    ax1.tick_params(axis='both', which="minor",direction='in', length=tick_length/2.0, width=tick_width, labelsize=tick_labelsize)        
    for axis in ['top','bottom','left','right']:                                                                                                                                    
        ax1.spines[axis].set_linewidth(3)
        
    plt.minorticks_on()                
    plt.show()
    savefig_tofile(figure, args.resultdir, f"{nuclei}hist_total_effcor_presel", show=True)
    np.savez(os.path.join(args.datadir, f"{nuclei}_graph_total_effcor_presel.npz"), **dict_total_effcor)  
    
        
if __name__ == "__main__":
    main()

    
