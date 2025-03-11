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
from tools.binnings_collection import mass_binning, fbinning_energy, LithiumRigidityBinningFullRange, Rigidity_Analysis_Binning_FullRange, fbinning_inversemass, fbinning_energy_rebin
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar_v2, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, FONTSIZE_BIG, FONTSIZE_MID, plot1d_errorbar, plot1d_step
from tools.calculator import calc_rig_from_ekin, calc_ratio_err, calc_ekin_from_beta, calc_mass, calc_inverse_mass
from tools.constants import ISOTOPES_MASS, NUCLEI_CHARGE, NUCLEIS, MASS_NUCLEON_GEV, ISOTOPES
from tools.histograms import Histogram, plot_histogram_2d
from tools.binnings import Binning, make_lin_binning  , make_log_binning
from tools.roottree import read_tree
from tools.selections import *
import pickle
from tools.graphs import MGraph, slice_graph

setplot_defaultstyle()

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


binning_rig_residual = make_lin_binning(-1.0, 1.0, 550)     
binning_rig_resolution = make_lin_binning(-1, 1, 200)

FIGNAME = 'P8GBL_rebin'

def main():                                                                                 
    import argparse                                                                     
    parser = argparse.ArgumentParser()                                                      
    #parser.add_argument("--filename", default="/home/manbing/Documents/Data/data_iss/BeISS_NucleiSelection_BetaCor.root",help="Path to fils")
    #parser.add_argument("--filename", default="/home/manbing/Documents/Data/data_BeP8/rootfile/OxygenIss_P8_CIEBetaCor.root",help="Path to fils")
    parser.add_argument("--nuclei", default="Be", help="the analyzed nuclei")
    parser.add_argument("--filename", default="/home/manbing/Documents/Data/data_BeP8/rootfile/BeISS_P8_CIEBeta.root",help="Path to fils")
    parser.add_argument("--treename", default="amstreea",help="Path to fils")                    
    parser.add_argument("--plotdir", default="plots/counts")
    parser.add_argument("--datadir", default="/home/manbing/Documents/Data/data_BeP8/presel_flux")
    parser.add_argument("--variable", default="Rigidity", help="analysis in rigidity or in kinetic energy per nucleon")
    parser.add_argument("--isrebin", default=True, type=bool,  help="is rebin version")
    parser.add_argument("--isGBL", default=True, type=bool,  help="is rebin version")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")  
    

    args = parser.parse_args()                                                    
    os.makedirs(args.plotdir, exist_ok=True)
    nuclei = args.nuclei
    detectors = ["Tof", "NaF", "Agl"]
    #detectors = ["Tof"]
    detectors_alias = {"Tof": "tof", "Agl":"agl", "NaF": "naf"}
    isotopes = ISOTOPES[nuclei]
    variable = args.variable
    massbinning = Binning(mass_binning())
    inverse_mass_binning = Binning(fbinning_inversemass(nuclei))

    #xbinning = Binning(LithiumRigidityBinningFullRange())
    RigidityBinning = Binning(np.array([1.51, 1.71, 1.92, 2.15, 2.4, 2.67, 2.97, 3.29, 3.64, 4.02, 4.43, 4.88, 5.37, 5.9,
                                6.47, 7.09, 7.76, 8.48, 9.26, 10.1, 11, 12, 13, 14.1, 15.3, 16.6, 18, 19.5, 21.1, 22.8,
                                24.7, 26.7, 28.8, 31.1]))
    binning_cutoff = make_log_binning(1.5, 30, 70) 
    rigidity_binning = make_lin_binning(1.5, 40, 100)  
    
    xbinning = {"Rigidity": rigidity_binning, "Ekin":Binning(fbinning_energy_rebin()) if args.isrebin else Binning(fbinning_energy())} 
    xlabel = {"Rigidity": "Rigidity(GV)", "Ekin": "Ekin/n (GeV/n)"}
    
    dict_hist = dict()
    hist_rig_vs_cutoff = {}
    hist_rig_vs_cutoff = Histogram(binning_cutoff, rigidity_binning, labels=["Geomagnetic cutoff (GV)", f"Rigidity (GV)"])
    
    for events in read_tree(args.filename, args.treename, chunk_size=args.chunk_size):
                #selections
        events = remove_badrun_indst(events)
        events = events[events.is_ub_l1 == 1]
        events = SelectCleanEvent(events)
        #events = geomagnetic_IGRF_cutoff(events, 1.1)
        hist_rig_vs_cutoff.fill(ak.to_numpy((events.mcutoffi)[:, 1, 1]), ak.to_numpy(events.tk_rigidity1[:, 1, 2, 1]))
                
        
        hist_rig_vs_cutoff.add_to_file(dict_hist, f"hist_{nuclei}_rig_cutoff")
                

    np.savez(os.path.join(args.datadir, f"{nuclei}ISS_RigVsCutoff.npz"), **dict_hist)
    plt.show()

if __name__ == "__main__":   
    main()

    
