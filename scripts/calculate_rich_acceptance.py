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
from tools.constants import MC_PARTICLE_CHARGES, MC_PARTICLE_IDS, ISOTOPES
from tools.binnings_collection import mass_binning, fbinning_energy, LithiumRigidityBinningFullRange, Rigidity_Analysis_Binning_FullRange
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center, kinetic_energy_neculeon_binning
from tools.calculator import calc_mass, calc_ekin_from_rigidity, calc_betafrommomentom, calc_rig_from_ekin
from tools.constants import ISOTOPES_MASS, ISOTOPES_COLOR
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
from tools.histograms import Histogram

rigiditybinning = np.array([0.8,1.00,1.16,1.33,1.51,1.71,1.92,2.15,2.40,2.67,2.97,3.29,3.64,4.02,4.43,4.88,5.37,5.90,
                            6.47,7.09,7.76,8.48,9.26, 10.1,11.0,12.0,13.0,14.1,15.3,16.6,18.0,19.5,21.1,22.8,24.7,26.7,28.8,31.1,33.5,36.1,38.9,41.9,
                            45.1,48.5,52.2,56.1,60.3,64.8,69.7,74.9,80.5,86.5,93.0, 100.,108.,116.,125.,135.,147.,160.,175.,192.,211.,233.,259.,291.,
                            330.,379.,441.,525.,660.,880.,1300.,3300.])

#rigiditybinning = ()
energybinning = kinetic_energy_neculeon_binning()
FONTSIZE = 30

xbinning = {"Rigidity" : rigiditybinning, "Ekin": energybinning}
xaxistitle = {"Rigidity": "Rigidity (GV)", "Ekin": "Ekin/n (GeV)"}
    
def flatten(d, parent_key='', sep='_'):                   
    items = []                 
    for k, v in d.items():                                                
        new_key = parent_key + sep + k if parent_key else k                                                
        if isinstance(v, MutableMapping):               
            items.extend(flatten(v, new_key, sep=sep).items())     
        else:                                    
            items.append((new_key, v)) 
    return dict(items)                        

def get_acceptance_from_hist(hist_pass, treename, file_pgen, trigger, isotope, variable):
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
    tot_trigger = trigger[isotope]
    
    if variable == "Rigidity":
        for ibin in range(1, len(binning)):
            #print(ibin, binning[ibin], binning[ibin - 1])
            frac = (np.log(binning[ibin]) - np.log(binning[ibin - 1]))/(np.log(maxRigGen) - np.log(minRigGen))                                                                                  
            num = tot_trigger * frac                                                                                                                                     
            hist_total.SetBinContent(ibin, num)
        
    elif (variable == "Ekin"):
        for ibin in range(1, len(binning)):
            upbinrig = calc_rig_from_ekin(binning[ibin], ISOTOPES_MASS[isotope], charge)
            lowbinrig = calc_rig_from_ekin(binning[ibin-1], ISOTOPES_MASS[isotope], charge)
            frac = (np.log(upbinrig) - np.log(lowbinrig))/(np.log(maxRigGen) - np.log(minRigGen))
            num = tot_trigger * frac                                                       
            hist_total.SetBinContent(ibin, num)
        
    else:
        raise ValueError("Wrong variable is given")

    print("the total trigger: ", tot_trigger)
    print("the number of passed events: ",  np.sum(hist_pass.values[1:-1]))
    arr_tot = np.zeros(len(binning) -1)
    arr_pass = hist_pass.values[1:-1]
    for i in range(len(binning) - 1):
        arr_tot[i] = hist_total.GetBinContent(i+1)

    eff, efferr = calculate_efficiency_and_error(arr_pass, arr_tot, "MC")
    acc = eff * 3.9 * 3.9 * np.pi
    accerr = efferr * 3.9 * 3.9 * np.pi
    xbincenter = get_bin_center(binning)
    for i in range(len(binning) - 1): 
        print(arr_pass[i],  arr_tot[i] , arr_pass[i]/arr_tot[i] * 3.9 * 3.9 * np.pi, acc[i])

    graph_acc = MGraph(xbincenter, acc, accerr, labels=[xaxistitle[variable], "Acceptance (m^{2} sr)"])
    return graph_acc            

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file.")
    parser.add_argument("--resultdir", default="/home/manbing/Documents/Data/data_LiP8/acceptance", help="Directory to store plots and result files in.")
    parser.add_argument("--datadir", default="/home/manbing/Documents/Data/data_LiP8", help="Directory to store plots and result files in.")
    #parser.add_argument("--filename", default="acceptance/BeMC_hist_counts_finebin_nottofedge_final.npz", help="Directory to store plots and result files in.")
    #parser.add_argument("--filename", default="presel_flux/LiMC_hist_prsel_counts_vsGenR.npz", help="Directory to store plots and result files in.")
    #parser.add_argument("--filename", default="acceptance/LiMC_hist_Counts.npz", help="Directory to store plots and result files in.")
    parser.add_argument("--filename", default="/home/manbing/Documents/Data/data_LiP8/Hist2D/LiMC_hist_MassHist_B1308_FineNewBin_ShiftToF_CutToFEdge.npz", help="Directory to store plots and result files in.")    
    parser.add_argument("--variable", default="Ekin", help="Directory to store plots and result files in.")
    parser.add_argument("--nuclei", default="Li", help="dataname for the output file and plots.")

    isotopes = {"Li": ["Li6", "Li7"], "Be": ["Be7", "Be9", "Be10"], "Boron": ["Bo10", "Bo11"]}
    detectors = {"Tof", "NaF", "Agl"}
    #detectors = {"Tof"}
    args = parser.parse_args()
    nuclei = args.nuclei
    os.makedirs(args.resultdir, exist_ok=True)
        
    #file_pgen = {isotope: f"{args.datadir}/acceptance/histPGen_{isotope}P8.root" for isotope in isotopes[nuclei]}
    file_pgen = {isotope: f"{args.datadir}/acceptance/histPGen_Li6.root" for isotope in isotopes[nuclei]}
    
    triggerP7 = {"Be7" : 6090705916, "Be9": 6484356101, "Be10": 2605554333}
    trigger_from_pgenP7 = {"Be7" : 6999354260, "Be9": 8757003151, "Be10": 8581757698}
    trigger_P8MC1236 = {"Be7" : 10795242754, "Be9": 9941558890, "Be10":9051178304, 'Li6': 7572367848, 'Li7': 6713403468, 'B10': 7726627606, 'B11': 7940431312}
    trigger_P8MC1308 = {"Li6": 8917822200, 'Li7':9425263851}
    trigger_P8MC1236_Jiahui = {"Be7" : 10692146954, "Be9": 9763431490, "Be10":8943038252}

    variable = args.variable

    fit_range = {"Tof": [1.9, 1000], "NaF": [4, 110], "Agl": [10, 110]}
    xticks = {"Tof": [2, 5, 10, 30, 60, 100, 300, 1000], "NaF": [5, 10, 30, 60, 100], "Agl": [10, 20, 30, 40, 60, 100]}
    acc = {"Tof":{}, "NaF": {}, "Agl":{}}
    accerr = {"Tof":{}, "NaF": {}, "Agl":{}}
    x_bincenter = {"Tof":{}, "NaF":{}, "Agl":{}}
    graph_acc = {dec: dict() for dec in detectors}
    dict_graph_acc = dict()
    print(args.datadir)    
    with np.load(os.path.join(args.datadir, f"{args.filename}")) as file_counts:
        #with np.load(os.path.join(args.datadir, f"")) as file_counts:

        #hist_pass = {dec: {iso: Histogram.from_file(file_counts, f'hist_{iso}MC_{dec}_counts') for iso in ISOTOPES[nuclei]} for dec in detectors}
        hist_pass = {dec: {iso: Histogram.from_file(file_counts, f'hist_{iso}MC_{dec}_counts') for iso in ISOTOPES[nuclei]} for dec in detectors}
        
    for dec in detectors:
        print(dec, ":")
        for iso in ISOTOPES[nuclei]:
            graph_acc[dec][iso]  = get_acceptance_from_hist(hist_pass[dec][iso], args.treename, file_pgen, trigger_P8MC1308, iso, variable)
            graph_acc[dec][iso].add_to_file(dict_graph_acc, f"raw_acc_{dec}_{iso}")

    np.savez(os.path.join(args.datadir, f"LiMC_dict_graph_rawacc_{variable}P8_CutToFEdge"), **dict_graph_acc)  


             
if __name__ == "__main__":
    main()


    '''
    if variable == "Rigidity":
        
        for dec in detectors:
            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(21, 14))    
            for i, iso in enumerate(isotopes[nuclei]):
                plot_comparison_nphist(figure, ax1, ax2, x_binning=xbinning[variable], com=graph_acc[dec][iso].yvalues, com_err=graph_acc[dec][iso].yerrs,
                                       ref=acc_ref_rig[dec][iso]["EffAccRaw"], ref_err=np.zeros(len(acc_ref_rig[dec][iso]["EffAccRaw"])),
                                       xlabel=xaxistitle[variable], ylabel=r"$\mathrm{Acceptance (m^{2} sr)}$", legendA=f"this {iso}", legendB=f"Jiahui {iso}", color=ISOTOPES_COLOR[iso])
            ax2.set_xlim(fit_range[dec])
            ax2.set_xticks(xticks[dec])
            ax1.set_ylim(yaxes_lim[dec])
            ax2.set_ylim([0.9, 1.1])
            ax2.set_xscale("log")
            ax2.grid()
            ax1.text(0.05, 0.98, f"{dec}", fontsize=30, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')  
            savefig_tofile(figure, args.resultdir, f"Acceptance_{nuclei}_{dec}_{variable}", 1)

    else:
        for dec in detectors:
            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))    
            for i, iso in enumerate(isotopes[nuclei]):
                plot_comparison_nphist(figure, ax1, ax2, x_binning=xbinning[variable], com=graph_acc[dec][iso].yvalues, com_err=graph_acc[dec][iso].yerrs,
                                       ref=acc_ref[dec][iso]["Eff_Acc_raw"], ref_err=np.zeros(len(acc_ref[dec][iso]["Eff_Acc_raw"])),
                                       xlabel=xaxistitle[variable], ylabel=r"$\mathrm{Acceptance (m^{2} sr)}$", legendA=f"this {iso}", legendB=f"jiahui {iso}", color=ISOTOPES_COLOR[iso])
            ax2.set_ylim([0.9, 1.1])
            savefig_tofile(figure, args.resultdir, f"Acceptance_{nuclei}_{dec}_{variable}", 1)
        

    
    plt.show()
    dict_acc = flatten(acc)
    dict_accerr = flatten(accerr)
    df_acc = pd.DataFrame(dict_acc, columns=dict_acc.keys())
    df_acc.to_csv(os.path.join(args.resultdir, f'acc_dec_{variable}.csv'), index=False)

    df_accerr = pd.DataFrame(dict_accerr, columns=dict_accerr.keys())
    df_accerr.to_csv(os.path.join(args.resultdir, f'accerr_dec_{variable}.csv'), index=False)
    '''
     
            

