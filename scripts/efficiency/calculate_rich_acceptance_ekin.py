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
from tools.calculator import calc_mass, calc_ekin_from_rigidity, calc_betafrommomentom, calc_rig_from_ekin
from tools.constants import ISOTOPES_MASSES, ISOTOPES_COLOR
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
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, plot_comparison_nphist
from tools.graphs import MGraph, slice_graph, concatenate_graphs

rigiditybinning = LithiumRigidityBinningFullRange()
energybinning = fbinning_energy()
FONTSIZE = 30

xbinning = {"Rigidity" : rigiditybinning, "Ekin": energybinning}

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
    minMom = 10**(minLogMom)
    maxMom = 10**(maxLogMom)
    minLogMom_v1 = np.log10(4)
    maxLogMom_v1 = np.log10(8000)

    #print(minMom, maxMom)
    minRigGen = 10**(minLogMom_v1)/charge
    maxRigGen = 10**(maxLogMom_v1)/charge
    minEkinGen = calc_ekin_from_rigidity(minRigGen, MC_PARTICLE_IDS[isotope])
    maxEkinGen = calc_ekin_from_rigidity(maxRigGen, MC_PARTICLE_IDS[isotope])
    #print("minLogMom:", minLogMom_v1, "maxLogMom:", maxLogMom_v1)
    #print("minLogMom:", minLogMom, "maxLogMom:", maxLogMom)
    #print("minRigGen:", minRigGen, "maxRigGen:", maxRigGen)
        
    hist_total = ROOT.TH1F("hist_total", "hist_total",  len(binning)-1, binning)                                          
    rootfile = TFile.Open(file_tree, "READ")
    nucleitree = rootfile.Get(treename)
    hist_pass = TH1F("hist_pass", "hist_pass", len(binning)-1, binning)
    tot_trigger = trigger[isotope]
    
    if variable == "Rigidity":
        for ibin in range(1, len(binning)):
            #print(ibin, binning[ibin], binning[ibin - 1])
            frac = (np.log(binning[ibin]) - np.log(binning[ibin - 1]))/(np.log(maxRigGen) - np.log(minRigGen))                                                                                  
            num = tot_trigger * frac                                                                                                                                     
            hist_total.SetBinContent(ibin, num)
            
        for entry in nucleitree:
            arr_evmom = entry.mmom
            rig_gen = arr_evmom/charge
            hist_pass.Fill(rig_gen)
            ekin_gen = calc_ekin_from_rigidity(rig_gen, MC_PARTICLE_IDS[isotope])
            
    elif (variable == "Ekin"):
        for ibin in range(1, len(binning)):
            upbinrig = calc_rig_from_ekin(binning[ibin], ISOTOPES_MASSES[isotope], charge)
            lowbinrig = calc_rig_from_ekin(binning[ibin-1], ISOTOPES_MASSES[isotope], charge)
            frac = (np.log(upbinrig) - np.log(lowbinrig))/(np.log(maxRigGen) - np.log(minRigGen))                                                                                  
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
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file.")
    parser.add_argument("--resultdir", default="plots/acceptance", help="Directory to store plots and result files in.")
    parser.add_argument("--dataname", default="Be7Mc", help="Directory to store plots and result files in.")
    parser.add_argument("--variable", default="Ekin", help="Directory to store plots and result files in.")
    parser.add_argument("--nuclei", default="Be", help="dataname for the output file and plots.")

    isotopes = {"Li": ["Li6", "Li7"], "Be": ["Be7", "Be9", "Be10"], "Boron": ["Bo10", "Bo11"]}
    detectors = {"Tof", "Agl", "NaF"}
    #detectors = {"NaF"}
    args = parser.parse_args()
    nuclei = args.nuclei
    os.makedirs(args.resultdir, exist_ok=True)
        
    
    file_pgen = {isotope: f"trees/acceptance/histPGen_{isotope}.root" for isotope in isotopes[nuclei]}
    trigger = {"Be7" : 6090705916, "Be9": 6484356101, "Be10": 2605554333}
    trigger_from_pgen = {"Be7" : 6999354260, "Be9": 8757003151, "Be10": 8581757698}
    
    filenames = {"Tof": {"Be7": "trees/MC2/Be7MC_Tof_CMAT_V1_0.root", "Be9":"trees/MC2/Be9MC_Tof_CMAT_V1_0.root",  "Be10": "trees/MC2/Be10MC_Tof_CMAT_V1_0.root"},
                 "NaF": {"Be7": "trees/MC2/Be7MC_RICHNaF_CMAT_V1_0.root", "Be9":"trees/MC2/Be9MC_RICHNaF_CMAT_V1_0.root",  "Be10": "trees/MC2/Be10MC_RICHNaF_CMAT_V1_0.root"},
                 "Agl": {"Be7": "trees/MC2/Be7MC_RICHAgl_CMAT_V1_0.root", "Be9":"trees/MC2/Be9MC_RICHAgl_CMAT_V1_0.root",  "Be10": "trees/MC2/Be10MC_RICHAgl_CMAT_V1_0.root"}}

    selector = {"CIEMAT": {"Tof":selector_tofevents, "NaF": selector_nafevents_ciematvar, "Agl": selector_aglevents_ciematvar}}
    #filenames = {"Tof": {"Be7": "$DATA/data_mc/Be7MC_NucleiSelection_ubl1_0.root", "Be9":"$DATA/data_mc/Be9MC_NucleiSelection_ubl1_0.root",  "Be10": "$DATA/data_mc/Be10MC_NucleiSelection_ubl1_0.root"}}
    variable = args.variable
    xaxistitle = {"Rigidity": "Rigidity (GV)", "Ekin": "Ekin/n (GeV)"}
    fit_range = {"Tof": [1.9, 1000], "NaF": [4, 100], "Agl": [10, 100]}
    xticks = {"Tof": [2, 5, 10, 30, 60, 100, 300, 1000], "NaF": [5, 10, 30, 60, 100], "Agl": [10, 20, 30, 40, 60, 100]}
    acc = {"Tof":{}, "NaF": {}, "Agl":{}}
    accerr = {"Tof":{}, "NaF": {}, "Agl":{}}
    x_bincenter = {"Tof":{}, "NaF":{}, "Agl":{}}
    graph_acc = {dec: dict() for dec in detectors}
    dict_graph_acc = dict()
    for detector in detectors:
        print(detector, ":")
        for isotope in isotopes[nuclei]:
            acc[detector][isotope], accerr[detector][isotope], x_bincenter[detector][isotope] = get_acceptance(filenames[detector][isotope], args.treename, file_pgen, trigger, isotope, variable)
            graph_acc[detector][isotope] = MGraph(x_bincenter[detector][isotope], acc[detector][isotope], accerr[detector][isotope], labels=[xaxistitle[variable], "Acceptance (m^{2} sr)"])
            graph_acc[detector][isotope].add_to_file(dict_graph_acc, f"raw_acc_{detector}_{isotope}")

    np.savez(os.path.join(args.resultdir, f"Be_dict_graph_rawacc_{variable}.npz"), **dict_graph_acc)  

    setplot_defaultstyle()

    detector_alias = {"Tof": "tof", "NaF": "naf", "Agl": "agl"}
    acc_jiahuifile = {detector: {isotope :f'/home/manbing/Documents/Data/jiahui/isotope_fluxes/Acceptance/{detector}_{isotope}.txt' for isotope in isotopes[nuclei]} for detector in detectors}
    acc_ref = {detector: dict() for detector in detectors}
    for detector in detectors:
         for isotope in isotopes[nuclei]:  
             acc_ref[detector][isotope] = pd.read_csv(acc_jiahuifile[detector][isotope],  sep='\s+', header=0)


    acc_jiahuifile_rig = {detector: {isotope :'/home/manbing/Documents/Data/jiahui/average_flux/acc_isotope_sel_{}_{}.txt'.format(isotope, detector_alias[detector]) for isotope in isotopes[nuclei]} for detector in detectors}
    acc_ref_rig = {detector: dict() for detector in detectors}
    for detector in detectors:
         for isotope in isotopes[nuclei]:  
             acc_ref_rig[detector][isotope] = pd.read_csv(acc_jiahuifile_rig[detector][isotope],  sep='\s+', header=0)

    if variable == "Rigidity":
        for detector in detectors:
            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(21, 14))    
            for i, isotope in enumerate(isotopes[nuclei]):
                plot_comparison_nphist(figure, ax1, ax2, x_binning=xbinning[variable], com=acc[detector][isotope], com_err=accerr[detector][isotope], ref=acc_ref_rig[detector][isotope]["EffAccRaw"][3:-1], ref_err=np.zeros(len(acc_ref_rig[detector][isotope]["EffAccRaw"][3:-1])), xlabel=xaxistitle[variable], ylabel=r"$\mathrm{Acceptance (m^{2} sr)}$", legendA=f"this {isotope}", legendB=f"jiahui {isotope}", color=ISOTOPES_COLOR[isotope])
            ax2.set_xlim(fit_range[dec])
            ax2.set_xticks(xticks[dec]) 
            ax2.set_ylim([0.9, 1.1])
            ax1.set_xscale("log")
            ax1.text(0.05, 0.98, f"{detector}", fontsize=30, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')  
            savefig_tofile(figure, args.resultdir, f"Acceptance_{nuclei}_{detector}_{variable}", 1)

    else:
        for detector in detectors:
            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))    
            for i, isotope in enumerate(isotopes[nuclei]):
                plot_comparison_nphist(figure, ax1, ax2, x_binning=xbinning[variable], com=acc[detector][isotope], com_err=accerr[detector][isotope], ref=acc_ref[detector][isotope]["Eff_Acc_raw"], ref_err=np.zeros(len(acc_ref[detector][isotope]["Eff_Acc_raw"])), xlabel=xaxistitle[variable], ylabel=r"$\mathrm{Acceptance (m^{2} sr)}$", legendA=f"this {isotope}", legendB=f"jiahui {isotope}", color=ISOTOPES_COLOR[isotope])
            ax2.set_ylim([0.9, 1.1])
            savefig_tofile(figure, args.resultdir, f"Acceptance_{nuclei}_{detector}_{variable}", 1)
        

    
    plt.show()
    dict_acc = flatten(acc)
    dict_accerr = flatten(accerr)
    df_acc = pd.DataFrame(dict_acc, columns=dict_acc.keys())
    df_acc.to_csv(os.path.join(args.resultdir, f'acc_detector_{variable}.csv'), index=False)

    df_accerr = pd.DataFrame(dict_accerr, columns=dict_accerr.keys())
    df_accerr.to_csv(os.path.join(args.resultdir, f'accerr_detector_{variable}.csv'), index=False)
    
    print(df_acc.head())

    
    #np.savez("accbe_V2.npz", **dict_acc)

             
if __name__ == "__main__":
    main()


     
            

