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
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center, compute_dayfromtime, ring_position_binning
from tools.binnings_collection import mass_binning, fbinning_energy, LithiumRigidityBinningFullRange, Rigidity_Analysis_Binning_FullRange, fbinning_inversemass, fbinning_energy_rebin, LithiumBetaBinning, fbinning_beta
from tools.binnings_collection import BeRigidityBinningRICHRange, Rigidity_Analysis_Binning 
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar_v2, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, FONTSIZE_BIG, FONTSIZE_MID, plot1d_errorbar, plot1d_step
from tools.calculator import calc_rig_from_ekin, calc_ratio_err, calc_ekin_from_beta, calc_mass, calc_ekin_from_rigidity, calc_inverse_mass, calc_beta, calc_ekin_from_rigidity_iso
from tools.calculator import calc_rig_iso
from tools.constants import ISOTOPES_MASS, NUCLEI_CHARGE, NUCLEIS, ISOTOPES_CHARGE
from tools.histograms import Histogram, plot_histogram_2d, plot_histogram_1d, WeightedHistogram
from tools.binnings import Binning, make_lin_binning, make_log_binning, reduce_bins
from tools.roottree import read_tree
from tools.selections import *
import pickle
from tools.graphs import MGraph, slice_graph
from tools.studybeta import GetMCRICHTunedBeta_WithHighR, GetMCTofTunedBeta_WithSpline, ShiftToFBeta

setplot_defaultstyle()

def read_values_from_hist(hist):
    values = hist.values
    errors = hist.get_errors()
    return values, errors

rich_selectors = {"LIP": {"Tof": selector_tof, "NaF": selector_naf_lipvar, "Agl": selector_agl_lipvar},    
             "CIEMAT": {"Tof":selector_tof, "NaF": selector_naf_ciematvar, "Agl": selector_agl_ciematvar}}   

geomagnetic_cutoff_beta = {'Be10': geomagnetic_cutoff_Be10, 'Li7': geomagnetic_cutoff_Li7, 'Li6': geomagnetic_cutoff_Li7}
geomagnetic_cutoff_betaMC = {'Be10': geomagnetic_cutoff_Li7MC, 'Li7': geomagnetic_cutoff_Li7MC, 'Li6':geomagnetic_cutoff_Li7MC}

def get_bin_content(histogram, values):
    result = np.array([histogram.values[index] for index in histogram.binnings[0].get_indices(values)])
    return result

#Set if using up/low bound of the tuning parameter
ExtErr = '0'
#FigName = 'RigReso'
FigName = 'masscutoff_ISSbkt1_mcgen_B1308_1P5sigma'

binning_rig_residual = make_lin_binning(-1.0, 1.0, 550)
binning_rig_resolution = make_lin_binning(-1, 1, 200)
binning_beta_resolution = {"Tof": make_lin_binning(-0.2, 0.2, 200), "NaF": make_lin_binning(-0.05, 0.05, 300), "Agl": make_lin_binning(-0.02, 0.02, 400)}  
binning_beta_residual = {"Tof": make_lin_binning(-0.07, 0.04, 170), "NaF": make_lin_binning(-0.03, 0.01, 300), "Agl": make_lin_binning(-0.006, 0.003, 300)}    
binning_inversebeta_residual = {"Tof": make_lin_binning(-0.07, 0.15, 300), "NaF": make_lin_binning(-0.03, 0.03, 400), "Agl": make_lin_binning(-0.006, 0.006, 350)}  


max_Sigma_Scale = {'Be': {'Agl': 1.15, 'NaF': 1.2}}
min_Sigma_Scale = {'Be': {'Agl': 1.1, 'NaF': 1.22}}

nuclei = "Li"

def main():
    import argparse                                                                     
    parser = argparse.ArgumentParser()
    parser.add_argument("--treename", default="amstreea",help="Path to fils")                                                                       
    parser.add_argument("--plotdir", default=f"/home/manbing/Documents/Data/data_{nuclei}P8")
    parser.add_argument("--datadir", default=f"/home/manbing/Documents/Data/data_{nuclei}P8/rootfile")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    
    parser.add_argument("--variable", default="Ekin", help="analysis in rigidity or in kinetic energy per nucleon")
    parser.add_argument("--isrebin", default=False, type=bool,  help="if with rebin")
    parser.add_argument("--isP8GBL", default=True, type=bool,  help="if with Pass8")
    parser.add_argument("--cutl1up", default=True, type=bool,  help="if fill counts for acceptance")

    
    args = parser.parse_args()                                                    
    os.makedirs(args.plotdir, exist_ok=True)

    detectors = ['Tof', 'NaF', 'Agl']
    #detectors = ["Tof"]
    isotopes = ['Li7']
    #isotopes = ISOTOPES[nuclei]
    detectors_alias = {"Tof": "tof", "Agl":"agl", "NaF": "naf"}
    variable = args.variable
    massbinning = make_lin_binning(3, 12, 50)
    inverse_mass_binning = make_lin_binning(0.03, 0.3, 100)

    fine_inversemassbinning = make_lin_binning(0.08, 0.16, 2000)
    mass_delta_binning = make_lin_binning(-0.001, 0.006, 2000)
    #filenames = {iso: os.path.join(args.datadir, f"{iso}MC_BetaCor.root") for iso in ISOTOPES[nuclei]}

    filename_iss = os.path.join(args.datadir, f"{nuclei}ISS_IGRFCutoff.root")
    print(filename_iss)
  
    #filenames = {iso: os.path.join(args.datadir, f"{iso}MC_IGRFCutoff.root") for iso in isotopes}
    #filenames = {iso : os.path.join(args.datadir, f"{iso}MC_IGRFCutoff.root") for iso in isotopes}
    filenames = {iso : os.path.join(args.datadir, f"{iso}MC_B1308.root") for iso in isotopes}
    print(filenames)
    
    
    #xbinning = Binning(LithiumRigidityBinningFullRange())
    #xbinning = {"Rigidity": Binning(Rigidity_Analysis_Binning_FullRange()), "Ekin":  reduce_bins(Binning(fbinning_energy_rebin()), 2)}
    xbinning = {"Rigidity": make_lin_binning(0.5, 40, 100), "Ekin":  reduce_bins(Binning(fbinning_energy_rebin()), 2)}
    
    #xbinning = {"Rigidity": Binning(Rigidity_Analysis_Binning_FullRange()), "Ekin": make_lin_binning(2, 20, 3)}
    #xbinningNaF = Binning(np.array([0.4185, 1.0, 2.0085,  4.9146, 10.2502]))
    xbinningNaF = Binning(np.array([0.4185, 1.0, 4.9146, 10.2502]))
    
    xbinning_ekin = {'Tof': Binning(fbinning_energy_rebin()), 'NaF': xbinningNaF, 'Agl': xbinningNaF}
    xekinbinning_fromRig = Binning(calc_ekin_from_rigidity_iso(Rigidity_Analysis_Binning()[:], 'Be10'))
    xekin_finelogbin = make_log_binning(0.4, 20, 30)
    #xbinning_beta = Binning(fbinning_beta())
    xbinning_beta = {'Tof': make_lin_binning(0.6, 1.01, 60), 'NaF': make_lin_binning(0.6, 1.01, 100), 'Agl': make_lin_binning(0.9, 1.01, 80)}
    xbinning_rigidity_finebin = {'Tof': make_lin_binning(0.5, 40, 400), 'NaF': make_lin_binning(0.5, 40, 180), 'Agl': make_lin_binning(0.5, 40, 180)}
    
    xlabel = {"Rigidity": "Rigidity(GV)", "Ekin": "Ekin/n (GeV/n)"}
    
    mix_com = {'Tof':{'Be7': 1.0, 'Be9': 0.6, 'Be10': 0.15},
               'NaF':{'Be7': 1.0, 'Be9': 0.6, 'Be10': 0.2},
               'Agl':{'Be7': 1.0, 'Be9': 0.6, 'Be10': 0.3}}
    dict_hist_mass = {}

    weight_mix = {'Be7': 1.0, 'Be9': 1.0, 'Be10': 1.0, 'Li6': 1.0, 'Li7':1.0}        

    histmc_mass_selcutoff = {dec: {} for dec in detectors}
    histmc_inversemass_selcutoff = {dec: {} for dec in detectors}
    histmc_rigidity_selcutoff = {dec: {} for dec in detectors}                                                                                                                                               
    histmc_beta_selcutoff = {dec: {} for dec in detectors}

    histiss_mass_selcutoff = {dec: {} for dec in detectors}
    histiss_inversemass_selcutoff = {dec: {} for dec in detectors}
    histiss_rigidity_selcutoff = {dec: {} for dec in detectors}
    histiss_rigidity_vs_theta = {}
    histmc_rigidity_vs_theta = {}
    histiss_beta_selcutoff = {dec: {} for dec in detectors}

    rigcutoff_factor = {'Tof': 1.2,  'NaF': 1.2, 'Agl': 1.1}
    betacutoff_factor_up = {'Tof': 0.985,  'NaF': 0.999, 'Agl': 0.9995}
    betacutoff_factor_low = {'Tof': 1.015,  'NaF': 1.001, 'Agl': 1.000}
    betacutoff_factor = {'Tof': 1.016,  'NaF': 1.001, 'Agl': 1.0005}
    histiss_beta_vsR = {}
    histiss_beta_vsRCutoff = {}
    histiss_beta_vsRMaxCutoff = {}
    histmc_beta_vsR = {dec: {} for dec in detectors}
    histiss_maxcutoff = {}
    histiss_cutoff = {}
    histiss_betacutoffLi6 = {}
    histiss_betacutoffLi7 = {}
    histmc_cutoff = {dec: {} for dec in detectors}
    df_probpars_AglBe7 = np.load('/home/manbing/Documents/lithiumanalysis/scripts/plots/unfold/LiBeta/AglLi6_polypar_inversebeta.npz')         
    df_probpars_NaFBe7 = np.load('/home/manbing/Documents/lithiumanalysis/scripts/plots/unfold/LiBeta/NaFLi6_polypar_inversebeta.npz')          
    df_probpars_TofBe7 = np.load('/home/manbing/Documents/lithiumanalysis/scripts/plots/unfold/LiBeta/TofLi6_polypar_inversebeta.npz')
    df_probpars = {'Tof': df_probpars_TofBe7, 'NaF': df_probpars_NaFBe7, 'Agl': df_probpars_AglBe7}
    histiss_bkstat = {}
    histiss_bkstat_vs_rigcutoff = {}
    histiss_bkstat_vsbeta = {}
    histiss_timestamp = {}
    histmc_timestamp = {dec: {} for dec in detectors}
    binning_theta = make_lin_binning(0, 30, 30)
    binning_timestamp = make_lin_binning(1.3e9, 1.7e9, 500)
    binning_timestampMC = make_lin_binning(1.65e9, 1.7e9, 50)
    
    for dec in detectors:
        histiss_beta_vsR[dec] = Histogram(xbinning_rigidity_finebin[dec], xbinning_beta[dec], labels=['R (GV)', 'beta'])
        histiss_beta_vsRCutoff[dec] = Histogram(xbinning['Rigidity'], xbinning_beta[dec], labels=['R (GV)', 'beta'])
        histiss_beta_vsRMaxCutoff[dec] = Histogram(xbinning['Rigidity'], xbinning_beta[dec], labels=['R (GV)', 'beta'])
        
        histiss_maxcutoff[dec] = Histogram(xbinning['Rigidity'], labels=[r'$\mathrm{R_{maxcf}}$'])
        histiss_cutoff[dec] = Histogram(xbinning['Rigidity'], labels=[r'$\mathrm{R_{cf}}$'])
        histiss_bkstat[dec] = Histogram(make_lin_binning(-1, 8, 9), labels=['bkstat'])
        histiss_bkstat_vs_rigcutoff[dec] = Histogram(make_lin_binning(-1, 8, 9), make_lin_binning(0, 30, 200), labels=['bkstat', 'rigiditycutoff'])
        histiss_bkstat_vsbeta[dec] = Histogram(make_lin_binning(-1, 8, 9), make_lin_binning(0.6, 1.0, 500), labels=['bkstat', 'beta'])
        histiss_rigidity_vs_theta[dec] = Histogram(xbinning_rigidity_finebin[dec], binning_theta, labels=['costheta', 'R (GV)'])
        histmc_rigidity_vs_theta[dec] = WeightedHistogram(xbinning_rigidity_finebin[dec], binning_theta, labels=['costheta', 'R (GV)'])
        histiss_timestamp[dec] = Histogram(binning_timestamp, labels=['time stamp'])
        
        for iso in isotopes:
            histmc_beta_vsR[dec][iso] = WeightedHistogram(xbinning_rigidity_finebin[dec], xbinning_beta[dec], labels=['R (GV)', 'beta'])
            histmc_cutoff[dec][iso] = WeightedHistogram(xbinning['Rigidity'], labels=[r'$\mathrm{R_{cf}}$'])
            histmc_mass_selcutoff[dec][iso] = WeightedHistogram(xbinning_ekin[dec], massbinning, labels=[xlabel[variable], 'mass'])
            histmc_inversemass_selcutoff[dec][iso] = WeightedHistogram(xbinning_ekin[dec], inverse_mass_binning, labels=[xlabel[variable], 'mass'])
            histmc_rigidity_selcutoff[dec][iso] = WeightedHistogram(xbinning_ekin[dec], xbinning_rigidity_finebin[dec], labels=[xlabel['Ekin'], xlabel['Rigidity']])                     
            histmc_beta_selcutoff[dec][iso] = WeightedHistogram(xbinning_beta[dec], labels=['beta'])
            histmc_timestamp[dec][iso] = WeightedHistogram(binning_timestampMC, labels=['time stamp'])
            
            histiss_mass_selcutoff[dec][iso] = Histogram(xbinning_ekin[dec], massbinning, labels=[xlabel[variable], 'mass'])
            histiss_inversemass_selcutoff[dec][iso] = Histogram(xbinning_ekin[dec], inverse_mass_binning, labels=[xlabel[variable], 'mass'])
            histiss_rigidity_selcutoff[dec][iso] = Histogram(xbinning_ekin[dec], xbinning_rigidity_finebin[dec], labels=[xlabel['Ekin'], xlabel['Rigidity']])                     
            histiss_beta_selcutoff[dec][iso] = Histogram(xbinning_beta[dec], labels=['beta'])
            
            for events in read_tree(filename_iss, args.treename, chunk_size=args.chunk_size):
                events = SelectCleanEvent(events)
                events = SelectUnbiasL1LowerCut(events, NUCLEI_CHARGE[nuclei])
                events = rich_selectors["CIEMAT"][dec](events, nuclei, iso, "ISS", cutoff=False, rebin=args.isrebin, cut_ubl1=args.cutl1up)   
                beta0 =  ak.to_numpy(events.tof_betah) if dec == 'Tof' else ak.to_numpy(events['rich_beta'][:, 0])
                deltabeta = np.array(np.poly1d(df_probpars[dec]['mean'])(np.log(beta0)))
                beta_toi = beta0 + deltabeta 
                
                rigidity0 = ak.to_numpy((events.tk_rigidity1)[:, 1, 2, 1]) 
    
                histiss_maxcutoff[dec].fill((events.mcutoffi)[:, 1, 1])
                histiss_cutoff[dec].fill(events.cal_igrf)

                #events = geomagnetic_IGRF_cutoff(events, factor=rigcutoff_factor[dec], datatype='ISS')
                beta1 = ak.to_numpy(events.tof_betah) if dec == 'Tof' else ak.to_numpy(events['rich_beta'][:, 0])
                deltabeta1 = np.array(np.poly1d(df_probpars[dec]['mean'])(np.log(beta1)))  
                deltabeta1[deltabeta1 < 0] = 0.0000000001
                beta_toi1 = beta1 + deltabeta1 
                histiss_beta_vsRMaxCutoff[dec].fill(events.cal_igrf, beta_toi1)
                histiss_timestamp[dec].fill(events.run)
        
                events = events[events.bkt_stat == 1]
                events = geomagnetic_cutoff_beta[iso](events, dec, datatype='ISS', factor=betacutoff_factor[dec], englos_factor=1.1)
                histiss_bkstat[dec].fill(events.bkt_stat)

                          
                if dec == "Tof":
                    beta = ak.to_numpy(events.tof_betah)
                else:
                    beta = ak.to_numpy(events['rich_beta'][:, 0])
                
                if args.isP8GBL:
                    rigidity = ak.to_numpy((events.tk_rigidity1)[:, 1, 2, 1])
                else:
                    rigidity = ak.to_numpy((events.tk_rigidity1)[:, 0, 2, 1])

                ekin = calc_ekin_from_beta(beta)
                invmass = calc_inverse_mass(beta, rigidity, NUCLEI_CHARGE[nuclei])
                mass = calc_mass(beta, rigidity, NUCLEI_CHARGE[nuclei])

                acostheta = np.arccos(-ak.to_numpy(events.tk_dir[:, 0, 2]))
                adegrees = np.degrees(acostheta)
                histiss_mass_selcutoff[dec][iso].fill(ekin, mass)
                histiss_inversemass_selcutoff[dec][iso].fill(ekin, invmass)
                histiss_beta_selcutoff[dec][iso].fill(beta)
                histiss_rigidity_selcutoff[dec][iso].fill(ekin, rigidity)
                histiss_beta_vsR[dec].fill(rigidity, beta)
                histiss_rigidity_vs_theta[dec].fill(rigidity, adegrees)
                deltabeta2 = np.array(np.poly1d(df_probpars[dec]['mean'])(np.log(beta)))  

                beta_toi2 = beta + deltabeta2 
                histiss_beta_vsRCutoff[dec].fill(events.cal_igrf, beta_toi2)
                histiss_bkstat_vsbeta[dec].fill(events.bkt_stat, beta_toi2)
                histiss_bkstat_vs_rigcutoff[dec].fill(events.bkt_stat, events.cal_igrf)                                
            
            for events in read_tree(filenames[iso], args.treename, chunk_size=args.chunk_size):
                #selections
                #events = SelectUnbiasL1LowerCut(events, NUCLEI_CHARGE[nuclei])
                #events = events[events.is_ub_l1 == 1]
                #events = SelectCleanEvent(events)
                #events = events[events.is_richpass == 1]
                events = SelectCleanEvent(events)
                events = SelectUnbiasL1LowerCut(events, NUCLEI_CHARGE[nuclei])                            
                events = rich_selectors["CIEMAT"][dec](events, nuclei, iso, "MC", cutoff=False, rebin=args.isrebin, cut_ubl1=args.cutl1up)

                beta0 =  ak.to_numpy(events.tof_betah) if dec == 'Tof' else ak.to_numpy(events['rich_beta'][:, 0])
                rigidity0 = ak.to_numpy((events.tk_rigidity1)[:, 1, 2, 1])
                w0 = ak.to_numpy(events.ww) 

                histmc_cutoff[dec][iso].fill(events.cal_igrf, weights=w0)
                histmc_timestamp[dec][iso].fill(events.run, weights=w0)
                #events = events[events.bkt_stat == 1]
                events = geomagnetic_GenMCRig_Cutoff(events, nuclei, factor=1.0)
                events = geomagnetic_cutoff_betaMC[iso](events, dec, datatype='MC', factor=betacutoff_factor[dec])
                
                
                if dec == "Tof":
                    beta = ak.to_numpy(events.tof_betah) 
                else:
                    beta = ak.to_numpy(events['rich_beta'][:, 0])
                
                if args.isP8GBL:
                    rigidity = ak.to_numpy((events.tk_rigidity1)[:, 1, 2, 1])
                else:
                    rigidity = ak.to_numpy((events.tk_rigidity1)[:, 0, 2, 1])

                weight = ak.to_numpy(events.ww)                
                ekin = calc_ekin_from_beta(beta)
                invmass = calc_inverse_mass(beta, rigidity, NUCLEI_CHARGE[nuclei])
                mass = calc_mass(beta, rigidity, NUCLEI_CHARGE[nuclei])
                acostheta = np.arccos(-ak.to_numpy(events.tk_dir[:, 0, 2]))
                adegrees = np.degrees(acostheta)
                histmc_mass_selcutoff[dec][iso].fill(ekin, mass, weights=weight)
                histmc_inversemass_selcutoff[dec][iso].fill(ekin, invmass, weights=weight)
                histmc_beta_selcutoff[dec][iso].fill(beta, weights=weight)                                                                                                    
                histmc_rigidity_selcutoff[dec][iso].fill(ekin, rigidity, weights=weight)
                histmc_beta_vsR[dec][iso].fill(rigidity, beta, weights=weight)
                histmc_rigidity_vs_theta[dec].fill(rigidity, adegrees, weights=weight)
                
            histmc_mass_selcutoff[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_masscutoff")
            histmc_inversemass_selcutoff[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_inversemasscutoff")
            histmc_beta_selcutoff[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_beta_cutoff")                                                                                   
            histmc_rigidity_selcutoff[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_rigidity_cutoff")  
            histmc_beta_vsR[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_betavsR")
            histmc_cutoff[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_cutoff")
            histmc_timestamp[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_timestamp")
            histiss_timestamp[dec].add_to_file(dict_hist_mass, f"{nuclei}ISS_{dec}_timestamp")
            
            histiss_beta_vsR[dec].add_to_file(dict_hist_mass, f"{nuclei}ISS_{dec}_betavsR")
            histiss_bkstat[dec].add_to_file(dict_hist_mass, f"{nuclei}ISS_{dec}_bkstat")
            histiss_bkstat_vsbeta[dec].add_to_file(dict_hist_mass, f"{nuclei}ISS_{dec}_bkstat_vsbeta")
            histiss_bkstat_vs_rigcutoff[dec].add_to_file(dict_hist_mass, f"{nuclei}ISS_{dec}_bkstat_vsrigcutoff")
            
            histiss_beta_vsRCutoff[dec].add_to_file(dict_hist_mass, f"{nuclei}ISS_{dec}_betavsRCutoff")
            histiss_beta_vsRMaxCutoff[dec].add_to_file(dict_hist_mass, f"{nuclei}ISS_{dec}_betavsRMaxCutoff")
            
            histiss_maxcutoff[dec].add_to_file(dict_hist_mass, f"{nuclei}ISS_{dec}_maxcutoff")
            histiss_cutoff[dec].add_to_file(dict_hist_mass, f"{nuclei}ISS_{dec}_cutoff")
            histiss_mass_selcutoff[dec][iso].add_to_file(dict_hist_mass, f"{iso}ISS_{dec}_masscutoff")
            histiss_inversemass_selcutoff[dec][iso].add_to_file(dict_hist_mass, f"{iso}ISS_{dec}_inversemasscutoff")
            histiss_beta_selcutoff[dec][iso].add_to_file(dict_hist_mass, f"{iso}ISS_{dec}_beta_cutoff")                                                                                   
            histiss_rigidity_selcutoff[dec][iso].add_to_file(dict_hist_mass, f"{iso}ISS_{dec}_rigidity_cutoff")
            histmc_rigidity_vs_theta[dec].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_rigidity_vs_theta")
            histiss_rigidity_vs_theta[dec].add_to_file(dict_hist_mass, f"{iso}ISS_{dec}_rigidity_vs_theta")  

    np.savez(os.path.join(args.plotdir, f"Lihist_{FigName}.npz"), **dict_hist_mass)
    

    
    plt.show()

if __name__ == "__main__":   
    main()

    
