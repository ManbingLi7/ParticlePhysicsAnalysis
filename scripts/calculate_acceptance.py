#!/usr/bin/env python3
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from tools.roottree import read_tree
from tools.selections import *
from tools.constants import MC_PARTICLE_CHARGES, MC_PARTICLE_IDS 
from tools.binnings_collection import mass_binning, fbinning_energy, LithiumRigidityBinningFullRange
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.calculator import calc_mass, calc_ekin_from_rigidity, calc_betafrommomentom
from collections.abc import MutableMapping
from tools.corrections import shift_correction
import uproot
from scipy import interpolate
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction
import ROOT
from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F
from ROOT import gROOT, gBenchmark, gRandom, gSystem
import array
import scipy.integrate as integrate
from tools.calculator import calculate_efficiency_and_error 
uproot.default_library
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID

rigiditybinning = LithiumRigidityBinningFullRange()
energybinning = fbinning_energy()
FONTSIZE = 30

xbinning = {"Rigidity" : rigiditybinning, "KineticEnergyPerNucleon": energybinning}

def flatten(d, parent_key='', sep='_'):                                                                                                                      
    items = []                                                                                                 
    for k, v in d.items():                                                                                                      
        new_key = parent_key + sep + k if parent_key else k                                                
        if isinstance(v, MutableMapping):               
            items.extend(flatten(v, new_key, sep=sep).items())     
        else:                                    
            items.append((new_key, v)) 
    return dict(items)                        

def get_acceptance(file_tree, treename, file_pgen, trigger, isotope, variable):
    root_pgen = TFile.Open(file_pgen[isotope], "READ")
    hist_pgen = root_pgen.Get("PGen_px")
    print("trigger from pgen: ", hist_pgen.Integral())
    nbins_pgen = hist_pgen.GetNbinsX()
    minLogMom = None
    maxLogMom = None
    charge = MC_PARTICLE_CHARGES[MC_PARTICLE_IDS[isotope]]
    for i in range(nbins_pgen):
        if hist_pgen.GetBinContent(i+1) > 0 :
            if minLogMom is None:
                minLogMom = hist_pgen.GetXaxis().GetBinLowEdge(i+1)
            maxLogMom = hist_pgen.GetXaxis().GetBinUpEdge(i+1)
            
    
    binning = xbinning[variable]
    print(charge)
    minMom = 10**(minLogMom)
    maxMom = 10**(maxLogMom)
    minLogMom_v1 = np.log10(4)
    maxLogMom_v1 = np.log10(8000)
    
    print(minMom, maxMom)
    minRigGen = 10**(minLogMom_v1)/charge
    maxRigGen = 10**(maxLogMom_v1)/charge
    minEkinGen = calc_ekin_from_rigidity(minRigGen, MC_PARTICLE_IDS[isotope])
    maxEkinGen = calc_ekin_from_rigidity(maxRigGen, MC_PARTICLE_IDS[isotope])
    print("minLogMom:", minLogMom_v1, "maxLogMom:", maxLogMom_v1)
    print("minLogMom:", minLogMom, "maxLogMom:", maxLogMom)
    print("minRigGen:", minRigGen, "maxRigGen:", maxRigGen)
        
    hist_total = ROOT.TH1F("hist_total", "hist_total",  len(binning)-1, binning)                                          
    rootfile = TFile.Open(file_tree, "READ")
    nucleitree = rootfile.Get(treename)
    hist_pass = TH1F("hist_pass", "hist_pass", len(binning)-1, binning)
    tot_trigger = trigger[isotope]
    
    if variable == "Rigidity":
        for ibin in range(1, len(binning)):
            print(ibin, binning[ibin], binning[ibin - 1])
            frac = (np.log(binning[ibin]) - np.log(binning[ibin - 1]))/(np.log(maxRigGen) - np.log(minRigGen))                                                                                  
            num = tot_trigger * frac                                                                                                                                     
            hist_total.SetBinContent(ibin, num)
            
        for entry in nucleitree:
            arr_evmom = entry.mmom
            rig_gen = arr_evmom/charge
            hist_pass.Fill(rig_gen)
            ekin_gen = calc_ekin_from_rigidity(rig_gen, MC_PARTICLE_IDS[isotope])
            
    elif (variable == "KineticEnergyPerNucleon"):
        for ibin in range(1, len(binning)):                                                                                                                                                       
            frac = (np.log(binning[ibin]) - np.log(binning[ibin - 1]))/(np.log(maxEkinGen) - np.log(minEkinGen))                                                                                  
            num = tot_trigger * frac                                                                                                                                     
            hist_total.SetBinContent(ibin, num)
        
        for entry in nucleitree:
            arr_evmom = entry.mmom
            weight = entry.ww
            rig_gen = arr_evmom/charge
            ekin_gen = calc_ekin_from_rigidity(rig_gen, MC_PARTICLE_IDS[isotope])
            hist_pass.Fill(ekin_gen)
    else:
        raise ValueError("Wrong variable is given")

    print("the total trigger: ", tot_trigger)
    print("the number of passed events: ",  hist_pass.Integral())
    arr_tot = np.zeros(len(binning) -1)
    arr_pass = np.zeros(len(binning) -1) 
    for i in range(len(binning) - 1):
        arr_tot[i] = hist_total.GetBinContent(i+1)
        arr_pass[i] = hist_pass.GetBinContent(i+1)

    eff, efferr = calculate_efficiency_and_error(arr_pass, arr_tot, "MC")
    acc = eff * 3.9 * 3.9 * np.pi
    accerr = efferr * 3.9 * 3.9 * np.pi
    xbincenter = get_bin_center(binning)
    for i in range(len(binning) - 1): 
        print(arr_pass[i],  arr_tot[i] , arr_pass[i]/arr_tot[i] * 3.9 * 3.9 * np.pi, acc[i])
        
    return acc, accerr, xbincenter             

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--filename", default="tree_results/LiISS_RichAgl_0.root",   help="Path to root file to read tree from")
    parser.add_argument("--treename", default="amstreea_cor", help="Name of the tree in the root file.")
    parser.add_argument("--resultdir", default="trees/acceptance", help="Directory to store plots and result files in.")
    parser.add_argument("--dataname", default="Be7Mc", help="Directory to store plots and result files in.")
    parser.add_argument("--variable", default="KineticEnergyPerNucleon", help="Directory to store plots and result files in.")
    parser.add_argument("--nuclei", default="Be", help="dataname for the output file and plots.")

    isotopes = {"Li": ["Li6", "Li7"], "Be": ["Be7", "Be9", "Be10"], "Boron": ["Bo10", "Bo11"]}
    #detectors = {"Tof", "Agl", "NaF"}
    detectors = {"Tof"}
    args = parser.parse_args()
    nuclei = args.nuclei
    os.makedirs(args.resultdir, exist_ok=True)
        
    color = ["tab:orange", "tab:blue", "tab:red"]
    file_pgen = {isotope: f"trees/acceptance/histPGen_{isotope}.root" for isotope in isotopes[nuclei]}
    trigger = {"Be7" : 6090705916, "Be9": 6484356101, "Be10": 2605554333}
    trigger_from_pgen = {"Be7" : 6999354260, "Be9": 8757003151, "Be10": 8581757698}
    trigger_P8MC1236 = {"Be7" : 10795242754, "Be9": 9941558890, "Be10":9022178304, 'Li6': 7572367848, 'Li7': 6713403468}
    
    #filenames = {"Tof": {"Be7": "trees/MC/Be7MC_Tof_CMAT_V1_0.root", "Be9":"trees/MC/Be9MC_Tof_CMAT_V1_0.root",  "Be10": "trees/MC/Be10MC_Tof_CMAT_V1_0.root"},
    #             "NaF": {"Be7": "trees/MC/Be7MC_RICHNaF_CMAT_V1_0.root", "Be9":"trees/MC/Be9MC_RICHNaF_CMAT_V1_0.root",  "Be10": "trees/MC/Be10MC_RICHNaF_CMAT_V1_0.root"},
    #             "Agl": {"Be7": "trees/MC/Be7MC_RICHAgl_CMAT_V1_0.root", "Be9":"trees/MC/Be9MC_RICHAgl_CMAT_V1_0.root",  "Be10": "trees/MC/Be10MC_RICHAgl_CMAT_V1_0.root"}}

    #filenames = {"Tof": {"Be7": "$DATA/data_mc/Be7MC_NucleiSelection_clean_0.root", "Be9":"$DATA/data_mc/Be9MC_NucleiSelection_clean_0.root",  "Be10": "$DATA/data_mc/Be10MC_NucleiSelection_clean_0.root"}}
    filenames = {"Tof": {"Li6": "$DATA/data_LiP8/rootfile/Li6MC_B1236P8_CIEBetaCor.root", "Li7":"$DATA/data_LiP8/rootfile/Li7MC_B1236P8_CIEBetaCor.root"}}
    #filenames = {"Tof": {"Be7": "$DATA/data_BeP8/rootfile/Be7MC_B1236P8_CIEBetaCor.root", "Be9":"$DATA/data_BeP8/rootfile/Be9MC_B1236P8_CIEBetaCor.root",  "Be10": "$DATA/data_BeP8/rootfile/Be10MC_B1236P8_CIEBetaCor.root"}}
    variable = args.variable
    
    acc = {"Tof":{}, "NaF": {}, "Agl":{}}
    accerr = {"Tof":{}, "NaF": {}, "Agl":{}}
    x_bincenter = {"Tof":{}, "NaF":{}, "Agl":{}}
    for detector in detectors:
        print(detector, ":")
        for isotope in isotopes[nuclei]:
            acc[detector][isotope], accerr[detector][isotope], x_bincenter[detector][isotope] = get_acceptance(filenames[detector][isotope], args.treename, file_pgen, trigger_P8MC1236, isotope, variable)
            
    xaxistitle = {"Rigidity": "Rigidity (GV)", "KineticEnergyPerNucleon": "Ekin/n (GeV)"}
    setplot_defaultstyle()
    acc_jiahuifile = {isotope :f'/home/manbing/Documents/Data/jiahui/average_flux/acc_isotope_sel_{isotope}.txt' for isotope in isotopes[nuclei]}
    acc_ref = pd.read_csv("/home/manbing/Documents/Data/jiahui/average_flux/acc_isotope_sel_Be7.txt",  sep='\s+', header=0)

    for detector in detectors:
        figure = plt.figure(figsize=FIGSIZE_BIG)
        figure.subplots_adjust(left= 0.13, right=0.96, bottom=0.1, top=0.95) 
        plot = figure.subplots(1,1)
        for i, isotope in enumerate(isotopes[nuclei]):
            plot1dhist(figure, plot, xbinning[variable], acc[detector][isotope], accerr[detector][isotope], xaxistitle[variable], r"$\mathrm{Acceptance (m^{2} sr)}$", isotope, color[i], FONTSIZE, 1, 0, 0, 1)

        savefig_tofile(figure, args.resultdir, f"Acceptance_{nuclei}_{detector}_{variable}", 1)
        
    plt.show()
    dict_acc = flatten(acc)
    dict_accerr = flatten(accerr)
    df_acc = pd.DataFrame(dict_acc, columns=dict_acc.keys())
    df_acc.to_csv(f'trees/acceptance/acc_nucleiselection_isclean.csv', index=False) 
    #np.savez("accbe_V2.npz", **dict_acc)

             
if __name__ == "__main__":
    main()


     
            

