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
from tools.calculator import calc_rig_from_ekin, calc_ratio_err, calc_ekin_from_beta, calc_mass, calc_ekin_from_rigidity, calc_inverse_mass, calc_beta, calc_ekin_from_rigidity_iso
from tools.constants import ISOTOPES_MASS, NUCLEI_CHARGE, NUCLEIS, ISOTOPES_CHARGE
from tools.histograms import Histogram, plot_histogram_2d, plot_histogram_1d, WeightedHistogram
from tools.binnings import Binning, make_lin_binning
from tools.roottree import read_tree
from tools.selections import *
import pickle
from tools.graphs import MGraph, slice_graph

setplot_defaultstyle()

def read_values_from_hist(hist):
    values = hist.values
    errors = hist.get_errors()
    return values, errors

rich_selectors = {"LIP": {"Tof": selector_tof, "NaF": selector_naf_lipvar, "Agl": selector_agl_lipvar},    
             "CIEMAT": {"Tof":selector_tof, "NaF": selector_naf_ciematvar, "Agl": selector_agl_ciematvar}}   

def get_bin_content(histogram, values):
    result = np.array([histogram.values[index] for index in histogram.binnings[0].get_indices(values)])
    return result

Nuclei = 'Be'

def main():                                                                                 
    import argparse                                                                     
    parser = argparse.ArgumentParser()                                                      
    parser.add_argument("--filename", default="/home/manbing/Documents/Data/data_iss/BeISS_NucleiSelection_BetaCor.root",help="Path to fils")
    parser.add_argument("--treename", default="amstreea",help="Path to fils")                                                                       
    parser.add_argument("--resultdir", default=f'/home/manbing/Documents/Data/data_{Nuclei}P8')
    parser.add_argument("--plotdir", default=f"/home/manbing/Documents/Data/data_{Nuclei}P8")
    parser.add_argument("--datadir", default=f"/home/manbing/Documents/Data/data_{Nuclei}P8/rootfile")
    parser.add_argument("--variable", default="Ekin", help="analysis in rigidity or in kinetic energy per nucleon")
    parser.add_argument("--nuclei", default=Nuclei, help="the analyzed nuclei")
    parser.add_argument("--isrebin", default=True, type=bool,  help="if with rebin")
    parser.add_argument("--isP8", default=True, type=bool,  help="if with Pass8")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")  
    args = parser.parse_args()                                                    
    os.makedirs(args.resultdir, exist_ok=True)
    nuclei = args.nuclei
    detectors = ["Tof", "NaF", "Agl"]
    detectors_alias = {"Tof": "tof", "Agl":"agl", "NaF": "naf"}
    variable = args.variable
    massbinning = Binning(mass_binning())
    inverse_mass_binning = Binning(fbinning_inversemass(nuclei))

    
    fine_inversemassbinning = make_lin_binning(0.05, 0.25, 2500)
    mass_delta_binning = make_lin_binning(-0.001, 0.006, 1200)
    #filenames = {iso: os.path.join(args.datadir, f"{iso}MC_BetaCor.root") for iso in ISOTOPES[nuclei]}
    filenames = {iso: os.path.join(args.datadir, f"{iso}MC_B1236P8_CIEBetaCor.root") for iso in ISOTOPES[nuclei]}
    #filenames = {iso: os.path.join(args.datadir, f"{iso}MC_B1236_dst.root") for iso in ISOTOPES[nuclei]}
    #filenames = {iso: os.path.join(args.datadir, f"{iso}MC_NucleiSelection.root") for iso in ISOTOPES[nuclei]}
    #filenames = {iso: os.path.join(args.datadir, f"{iso}B1236l1_NucleiSelection.root") for iso in ISOTOPES[nuclei]}
                 
    #xbinning = Binning(LithiumRigidityBinningFullRange())
    xbinning = {"Rigidity": Binning(Rigidity_Analysis_Binning_FullRange()), "Ekin":Binning(fbinning_energy_rebin()) if args.isrebin else Binning(fbinning_energy())}
    print(len(xbinning['Ekin'].edges))
    
    xlabel = {"Rigidity": "Rigidity(GV)", "Ekin": "Ekin/n (GeV/n)"}
    

    dict_hist_mass = {}

    hist_true_mass_atdec = {dec: {} for dec in detectors}
    hist_true_mass = {dec: {} for dec in detectors}
    hist_true_deltamass = {dec: {} for dec in detectors}
    
    for dec in detectors:
        for iso in ISOTOPES[nuclei]:
            hist_true_mass_atdec[dec][iso] = WeightedHistogram(xbinning["Ekin"], fine_inversemassbinning, labels=[xlabel[variable], "mass"])
            hist_true_mass[dec][iso] = WeightedHistogram(xbinning["Ekin"], fine_inversemassbinning, labels=[xlabel[variable], "mass"])
            hist_true_deltamass[dec][iso] = WeightedHistogram(xbinning["Ekin"], mass_delta_binning, labels=[xlabel[variable], "mass"])

            
            for events in read_tree(filenames[iso], args.treename, chunk_size=args.chunk_size):
                #selections
                #events = remove_badrun_indst(events)
                events = SelectUnbiasL1LowerCut(events, NUCLEI_CHARGE[nuclei])
                #events = events[events.is_ub_l1 == 1]
                events = SelectCleanEvent(events)
                #events = events[events.is_richpass == 1]
                events = rich_selectors["CIEMAT"][dec](events, nuclei, iso, "MC", cutoff=False, rebin=args.isrebin)

                if dec == "Tof":
                    beta0 = ak.to_numpy(events.tof_betahmc)
                else:
                    beta0 = ak.to_numpy(events['rich_beta_cor'])
                    
                events = events[beta0 < 1]
               
                true_momentum = ak.to_numpy(events["mevmom1"][:, 0])
                true_rig = ak.to_numpy(events["mevmom1"][:, 0]/NUCLEI_CHARGE[nuclei])
                true_beta = calc_beta(true_rig, ISOTOPES_MASS[iso], ISOTOPES_CHARGE[iso])
                true_rig_inntrk = ak.to_numpy(events["mevmom1"][:, 12]/NUCLEI_CHARGE[nuclei])  
                true_rig_rich = ak.to_numpy(events["mevmom1"][:, 17]/NUCLEI_CHARGE[nuclei])
                # true_ekin = calc_ekin_from_rigidity_iso(true_rig, iso)        
                true_beta_rich = calc_beta(true_rig_rich, ISOTOPES_MASS[iso], ISOTOPES_CHARGE[iso])
                true_beta_inntrk = calc_beta(true_rig_inntrk, ISOTOPES_MASS[iso], ISOTOPES_CHARGE[iso])
                
                if dec == "Tof":
                    beta = ak.to_numpy(events.tof_betahmc)
                    true_mass_atdec = calc_inverse_mass(true_beta_inntrk, true_rig_inntrk, NUCLEI_CHARGE[nuclei])
                else:
                    beta = ak.to_numpy(events['rich_beta_cor'])
                    true_mass_atdec = calc_inverse_mass(true_beta_rich, true_rig_inntrk, NUCLEI_CHARGE[nuclei])

                true_mass = calc_inverse_mass(true_beta, true_rig, NUCLEI_CHARGE[nuclei])    
                delta_mass = true_mass - true_mass_atdec
                
                ekin = calc_ekin_from_beta(beta)
                                    
                rig_gen = ak.to_numpy(events.mmom/NUCLEI_CHARGE[nuclei])
                ekin_gen = calc_ekin_from_rigidity(rig_gen, MC_PARTICLE_IDS[iso])
                xval = {"Rigidity": rig_gen, "Ekin": ekin_gen}
                fluxweight = ak.to_numpy(events.ww)
                weight = np.ones_like(fluxweight)
                
                hist_true_mass_atdec[dec][iso].fill(ekin, true_mass_atdec, weights=weight)
                hist_true_mass[dec][iso].fill(ekin, true_mass, weights=weight)
                hist_true_deltamass[dec][iso].fill(ekin, delta_mass, weights=weight)
                
            #plot counts
            hist_true_mass_atdec[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_truemass_atdec")
            hist_true_mass[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_truemass")
            hist_true_deltamass[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_deltamass")
            

    
    if args.isrebin:
        np.savez(os.path.join(args.resultdir, f"{nuclei}MC_deltamass_{variable}_P8B1236_rebin_All.npz"), **dict_hist_mass)
    else:
        np.savez(os.path.join(args.resultdir, f"{nuclei}MC_deltamass_{variable}_P8B1236_finebin_All.npz"), **dict_hist_mass)
    
    plt.show()

if __name__ == "__main__":   
    main()

    
