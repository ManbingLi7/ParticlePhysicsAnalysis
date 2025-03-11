#!/usr/bin/env python3
import multiprocessing as mp  
import os   
from collections import Counter     
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from tools.calculator import calc_mass, calc_ekin_from_beta         
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center, compute_dayfromtime      
from tools.studybeta import hist1d, hist2d, hist_beta, getbeta, hist_betabias, compute_moment, get_corrected_lipbeta_naf, compute_mean, get_refhits_fraction, correct_charge, compute_rawmeanstd_2dhist, compute_max_2dhist 
from tools.roottree import read_tree
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center, compute_dayfromtime, LithiumRigidityBinningFullRange
from tools.selections import *
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, plot1d_errorbar_v2, FONTSIZE_MID, FONTSIZE
# This program will read all events from the root trees,
# select the events with ToF charge around 2,
# create a histogram of the inner tracker charge,
# then plot that histogram and save the plot.
binning_days = np.linspace(0, 3600, 100)
binning_rigidity = LithiumRigidityBinningFullRange()


def main():
    import argparse    
    parser = argparse.ArgumentParser()    
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)") 
    parser.add_argument("--filenameA", default= "/home/manbing/Documents/Data/jiahui/isotope_fluxes/Event_counts_before_fit/.root",  help="(e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--filenameB", default= "/home/manbing/Documents/Data/data_iss/BeISS_NucleiSelection.root",  help="(e.g. results/ExampleAnalysisTree*.root)")   
    parser.add_argument("--dataname", default="heiss", help="give a name to describe the dataset")
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file.")   
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")   
    parser.add_argument("--resultdir", default="plots/lipbetanaf", help="Directory to store plots and result files in.")  
    parser.add_argument("--qsel",  default=2.0, type=float, help="the charge of the particle to be selected")                
    args = parser.parse_args()

    # create a binning for the charge histogram, with 100 bins (i.e. 101 bin edges)
    hist_rigidity_A = np.zeros(len(binning_rigidity) - 1)
    hist_rigidity_B = np.zeros(len(binning_rigidity) - 1)
    # loop over chunks of events in the root file(s), since we cannot load all events into memory at the same time    
    for events in read_tree(args.filenameA, args.treename, chunk_size=args.chunk_size):
        runnum = events.run
        events = geomagnetic_IGRF_cutoff(events, 1.2)
        rigidityA = ak.to_numpy((events.tk_rigidity1)[:, 0, 2, 1])        
        hist_rigidity_A += hist1d(rigidityA, binning_rigidity)

    # loop over chunks of events in the root file(s), since we cannot load all events into memory at the same time
    for events in read_tree(args.filenameB, args.treename, chunk_size=args.chunk_size):
        events = remove_badrun_indst(events)
        runnum = events.run
        events = geomagnetic_IGRF_cutoff(events, 1.2)
        rigidityB = ak.to_numpy((events.tk_rigidity1)[:, 0, 2, 1])
        hist_rigidity_B += hist1d(rigidityB, binning_rigidity)

        #mostfreq_npe = np.bincount(npe_lip).argmax()
        # create histogram of the charge values of this chunk of events:
    
    # Now the loop over all events is done, plot histogram:
    setplot_defaultstyle()
    figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0.7, 0.3]}, figsize=(17, 12))                                                                                                  
    figure.subplots_adjust(left= 0.18, right=0.96, bottom=0.1, top=0.95)  
    #plot = figure.subplots(1)
    #figure = plt.figure(figsize=FIGSIZE_BIG) 
    #plot = figure.subplots(1)  
    plot1dhist(figure=figure, plot=ax1, xbinning=binning_rigidity, counts=hist_rigidity_A, label_x= "Rigidity (GV)", setlogx=True, setlogy=False, setscilabely=True, col="tab:blue", legend="Jiahui")
    plot1dhist(figure=figure, plot=ax1, xbinning=binning_rigidity, counts=hist_rigidity_B, label_x= "Rigidity (GV)", setlogx=True, setlogy=False, setscilabely=True, col="tab:orange", legend="this")
    #plot1dhist(figure=figure, plot=ax2, xbinning=binning_rigidity, counts=hist_rigidity_A/hist_rigidity_B, label_x= "Rigidity (GV)", setlogx=True, setlogy=False, setscilabely=False)
    ax1.legend()
    ratio = (hist_rigidity_A - hist_rigidity_B)/hist_rigidity_A
    plot1d_errorbar_v2(figure, ax2, get_bin_center(binning_rigidity), ratio, err=np.zeros(len(ratio)), label_x="Rigidity (GV)", label_y="(J-M)/J",  style=".",  legendfontsize=FONTSIZE, setlogx=True, setlogy=False, setscilabelx=False, setscilabely=False, drawlegend=True)
    savefig_tofile(figure, args.resultdir, f"Jiahui_rigidity", 1)
    plt.subplots_adjust(hspace=.0)                                                                                                                                                                        
    ax2.sharex(ax1)
    ax2.grid()          
    plt.show()

if __name__ == "__main__":
    main()
