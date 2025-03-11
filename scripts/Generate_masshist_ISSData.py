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
from tools.binnings_collection import mass_binning, fbinning_energy, LithiumRigidityBinningFullRange, Rigidity_Analysis_Binning_FullRange, fbinning_inversemass, fbinning_energy_rebin, fbinning_energy_Li, kinetic_energy_neculeon_binning
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar_v2, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, FONTSIZE_BIG, FONTSIZE_MID, plot1d_errorbar, plot1d_step
from tools.calculator import calc_rig_from_ekin, calc_ratio_err, calc_ekin_from_beta, calc_mass, calc_inverse_mass, calc_rig_iso
from tools.constants import ISOTOPES_MASS, NUCLEI_CHARGE, NUCLEIS, MASS_NUCLEON_GEV, ISOTOPES
from tools.histograms import Histogram, plot_histogram_2d
from tools.binnings import Binning, make_lin_binning  
from tools.roottree import read_tree
from tools.selections import *
import pickle
from tools.graphs import MGraph, slice_graph

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


binning_rig_residual = make_lin_binning(-1.0, 1.0, 550)     
binning_rig_resolution = make_lin_binning(-1, 1, 200)

FIGNAME = 'newbin_P8_WithToFEdge'

def main():                                                                                 
    import argparse                                                                     
    parser = argparse.ArgumentParser()                                                      
    #parser.add_argument("--filename", default="/home/manbing/Documents/Data/data_iss/BeISS_NucleiSelection_BetaCor.root",help="Path to fils")
    #parser.add_argument("--filename", default="/home/manbing/Documents/Data/data_BeP8/rootfile/OxygenIss_P8_CIEBetaCor.root",help="Path to fils")
    parser.add_argument("--nuclei", default="Li", help="the analyzed nuclei")
    parser.add_argument("--filename", default="/home/manbing/Documents/Data/data_LiP8/rootfile/LiISS_P869.root",help="Path to fils")
    parser.add_argument("--treename", default="amstreea",help="Path to fils")                    
    parser.add_argument("--plotdir", default="plots/counts")
    parser.add_argument("--datadir", default="/home/manbing/Documents/Data/data_LiP8")
    parser.add_argument("--variable", default="Ekin", help="analysis in rigidity or in kinetic energy per nucleon")
    parser.add_argument("--isrebin", default=False, type=bool,  help="is rebin version")
    parser.add_argument("--isGBL", default=True, type=bool,  help="is rebin version")
    parser.add_argument("--cut_ubl1", default=True, type=bool,  help="cut ubl1 in ToF/RICH sel")
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
    xbinning_carbontoli = np.array([0.338, 1.1473,  6.0,  12.1303])
    
    xbinning = {"Rigidity": Binning(Rigidity_Analysis_Binning_FullRange()), "Ekin":Binning(fbinning_energy_rebin()) if args.isrebin else Binning(kinetic_energy_neculeon_binning())}
    #xbinning = {"Rigidity": Binning(Rigidity_Analysis_Binning_FullRange()), "Ekin":Binning(xbinning_carbontoli)}
    
    xlabel = {"Rigidity": "Rigidity(GV)", "Ekin": "Ekin/n (GeV/n)"}
    
    
    hist_counts = {dec: dict() for dec in detectors}
    dict_graph_counts = dict()
    dict_hist_counts = dict()
    
    hist_mass = {dec: dict() for dec in detectors}
    dict_hist_mass = dict()

    hist_rig_residual_vsEkin = {dec: dict() for dec in detectors}
    hist_rig_resolution_vsEkin = {dec: dict() for dec in detectors}   
    hist_rig_residual_vsR = {dec: dict() for dec in detectors}
    hist_rig_resolution_vsR = {dec: dict() for dec in detectors}
    hist_counts_preselection_vsR = Histogram(xbinning['Rigidity'], labels=[xlabel['Rigidity'], "counts"])
    
    for dec in detectors:
        for iso in isotopes:
            num = 0 
            hist_counts[dec][iso] = Histogram(xbinning[variable], labels=[xlabel[variable], "counts"])
            hist_mass[dec][iso] = Histogram(xbinning["Ekin"], inverse_mass_binning, labels=[xlabel[variable], "mass"])
            hist_rig_residual_vsEkin[dec][iso] = Histogram(xbinning["Ekin"], binning_rig_residual, labels=["Ekin/N (GeV/n)", rf"$\mathrm{{1/R_{{rec}}  - 1/R_{{ {dec}}} }}$"])
            hist_rig_resolution_vsEkin[dec][iso] = Histogram(xbinning["Ekin"], binning_rig_resolution, labels=["Ekin/N (GeV/n)", rf"$\mathrm{{ (1/R_{{rec}}  - 1/R_{{{dec}}})/1/R_{{{dec}}}}}$"])
            hist_rig_residual_vsR[dec][iso] = Histogram(xbinning["Rigidity"], binning_rig_residual, labels=["R (GV)", rf"$\mathrm{{1/R_{{rec}}  - 1/R_{{ {dec}}} }}$"])
            hist_rig_resolution_vsR[dec][iso] = Histogram(xbinning["Rigidity"], binning_rig_resolution, labels=["R (GV)", rf"$\mathrm{{ (1/R_{{rec}}  - 1/R_{{{dec}}})/1/R_{{{dec}}}}}$"]) 
  
            for events in read_tree(args.filename, args.treename, chunk_size=args.chunk_size):
                #selections
                events = remove_badrun_indst(events)
                print(events.tk_exqln)
                #events = SelectUnbiasL1Q(events, 8.0)                
                events = SelectCleanEvent(events)
                #remove the photon period
                #isphorun=(a.run>=1620025528 && a.run<1635856717);//check whether the run is using the photon-polarization trigger
                #events = events[(events.run < 1620025528) | (events.run>=1635856717)]
                #events = events[events.is_richpass == 1]
                runnum = events.run
                ####################################
                #select same time range for P7
                #events = events[runnum <= 1620028707]
                #####################################
                #print(runnum[(runnum >= 1620025528) & (runnum <1635856717)])
                #events = events[runnum > 1463616000]
                ######################################

                events_presel = geomagnetic_IGRF_cutoff(events, 1.2)
                hist_counts_preselection_vsR.fill(ak.to_numpy(events_presel.tk_rigidity1[:, 1, 2, 1]))
                
                if variable == "Ekin":
                    events = rich_selectors["CIEMAT"][dec](events, nuclei, iso, "ISS", cutoff=True, rebin=args.isrebin, cut_ubl1=args.cut_ubl1, cutTofEdge=False)
                else:
                    events = rich_selectors["CIEMAT"][dec](events, nuclei, iso, "ISS", cutoff=False, rebin=args.isrebin, cut_ubl1=args.cut_ubl1)
                    events = geomagnetic_IGRF_cutoff(events, 1.2)

                num = num + len(events)

                #fill hist
                if dec == "Tof":
                    beta = ak.to_numpy(events.tof_betah)
                else:
                    #beta = ak.to_numpy(events['rich_beta2'][:, 0])
                    beta = ak.to_numpy(events['rich_beta_cor'])
                         
                ekin = calc_ekin_from_beta(beta)
                rig_fromBeta = calc_rig_iso(beta, iso)     

                if args.isGBL:
                    rigidity = ak.to_numpy((events.tk_rigidity1)[:, 1, 2, 1])
                else:
                    rigidity = ak.to_numpy((events.tk_rigidity1)[:, 0, 2, 1])
                    
                xval = {"Rigidity": rigidity, "Ekin": ekin}

                mass = calc_mass(beta, rigidity, NUCLEI_CHARGE[nuclei])
                invmass = calc_inverse_mass(beta, rigidity, NUCLEI_CHARGE[nuclei])
                hist_mass[dec][iso].fill(ekin, invmass)   
                hist_counts[dec][iso].fill(xval[variable])
                hist_rig_resolution_vsEkin[dec][iso].fill(ekin, (1/rigidity - 1/rig_fromBeta)/(1/rig_fromBeta))                                                                            
                hist_rig_residual_vsEkin[dec][iso].fill(ekin, 1/rigidity - 1/rig_fromBeta)  
                
                hist_rig_resolution_vsR[dec][iso].fill(rig_fromBeta, (1/rigidity - 1/rig_fromBeta)/(1/rig_fromBeta))                                                                            
                hist_rig_residual_vsR[dec][iso].fill(rig_fromBeta, 1/rigidity - 1/rig_fromBeta)  

                                    
            print('num, opt', iso, num)
            if variable == 'Ekin':
                graph_counts = MGraph(xbinning[variable].bin_centers[1:-1], hist_counts[dec][iso].values[1:-1], hist_counts[dec][iso].get_errors()[1:-1])
                graph_counts.add_to_file(dict_hist_mass, f"graph_{dec}_Opt{iso}_counts")
                hist_counts[dec][iso].add_to_file(dict_hist_mass, f"hist_{nuclei}_{dec}Opt{iso}_counts")
                hist_mass[dec][iso].add_to_file(dict_hist_mass, f"{nuclei}_{dec}Opt{iso}_mass_ciemat")
                hist_rig_resolution_vsEkin[dec][iso].add_to_file(dict_hist_mass, f"{nuclei}_{dec}Opt{iso}_RigResoRefBeta_vsEkn")
                hist_rig_residual_vsEkin[dec][iso].add_to_file(dict_hist_mass, f"{nuclei}_{dec}Opt{iso}_RigResidualRefBeta_vsEkn") 
            
                hist_rig_resolution_vsR[dec][iso].add_to_file(dict_hist_mass, f"{nuclei}_{dec}Opt{iso}_RigResoRefBeta_vsR")
                hist_rig_residual_vsR[dec][iso].add_to_file(dict_hist_mass, f"{nuclei}_{dec}Opt{iso}_RigResidualRefBeta_vsR")

            if variable == 'Rigidity':
                hist_counts_preselection_vsR.add_to_file(dict_hist_mass, f"hist_{nuclei}_countsVsR")
                

    np.savez(os.path.join(args.datadir, f"{nuclei}ISS_{variable}_{FIGNAME}.npz"), **dict_hist_mass)
    plt.show()

if __name__ == "__main__":   
    main()

    
