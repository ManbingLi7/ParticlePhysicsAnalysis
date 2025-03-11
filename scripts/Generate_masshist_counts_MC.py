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
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center, compute_dayfromtime, ring_position_binning, kinetic_energy_neculeon_binning
from tools.binnings_collection import mass_binning, fbinning_energy, LithiumRigidityBinningFullRange, Rigidity_Analysis_Binning_FullRange, fbinning_inversemass, fbinning_energy_rebin
from tools.binnings_collection import BeRigidityBinningRICHRange, Rigidity_Analysis_Binning 
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar_v2, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, FONTSIZE_BIG, FONTSIZE_MID, plot1d_errorbar, plot1d_step
from tools.calculator import calc_rig_from_ekin, calc_ratio_err, calc_ekin_from_beta, calc_mass, calc_ekin_from_rigidity, calc_inverse_mass, calc_beta, calc_ekin_from_rigidity_iso
from tools.calculator import calc_rig_iso
from tools.constants import ISOTOPES_MASS, NUCLEI_CHARGE, NUCLEIS, ISOTOPES_CHARGE
from tools.histograms import Histogram, plot_histogram_2d, plot_histogram_1d, WeightedHistogram
from tools.binnings import Binning, make_lin_binning, make_log_binning
from tools.roottree import read_tree
from tools.selections import *
import pickle
from tools.graphs import MGraph, slice_graph
from tools.studybeta import GetMCRICHTunedBeta_WithHighR, GetMCTofTunedBeta_WithSpline, GetMCRICHTunedInverseBeta_GivenTuneValues, ShiftToFBeta

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

#Set if using up/low bound of the tuning parameter
ExtErr = '0'
FigName = 'MassHist_B1308_FineNewBin_ShiftCutoff_CutToFEdge'
#FigName = 'Counts'

binning_rig_residual = make_lin_binning(-0.3, 0.3, 800)
binning_rig_resolution = make_lin_binning(-1, 1, 200)
binning_beta_resolution = {"Tof": make_lin_binning(-0.2, 0.2, 200), "NaF": make_lin_binning(-0.05, 0.05, 300), "Agl": make_lin_binning(-0.02, 0.02, 400)}  
binning_beta_residual = {"Tof": make_lin_binning(-0.07, 0.04, 170), "NaF": make_lin_binning(-0.03, 0.03, 400), "Agl": make_lin_binning(-0.006, 0.006, 300)}    
binning_inversebeta_residual = {"Tof": make_lin_binning(-0.07, 0.3, 350), "NaF": make_lin_binning(-0.03, 0.05, 400), "Agl": make_lin_binning(-0.006, 0.006, 350)}

binning_beta = {"Tof": make_lin_binning(0.5, 1.0, 150), "NaF": make_lin_binning(0.7, 1.0, 70), "Agl": make_lin_binning(0.955, 1.0, 30)}    
max_Sigma_Scale = {'Be': {'Agl': 1.15, 'NaF': 1.2}}
min_Sigma_Scale = {'Be': {'Agl': 1.1, 'NaF': 1.22}}

meanshift = {'Li': {'NaF': 0.0, 'Agl': -1.5e-5}}
scalesigma = {'Li': {'NaF': 1.22, 'Agl': 1.1}}
Nuclei = 'Li'
shiftMPVmass = {'Li': {'Tof': 0.0008 ,'NaF': -0.0002, 'Agl': 0.0}}
def main():
    import argparse                                                                     
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuclei", default=Nuclei, help="the analyzed nuclei")
    parser.add_argument("--treename", default="amstreea",help="Path to fils")                                                                       
    parser.add_argument("--plotdir", default=f"/home/manbing/Documents/Data/data_{Nuclei}P8")
    parser.add_argument("--datadir", default=f"/home/manbing/Documents/Data/data_{Nuclei}P8/rootfile")

    parser.add_argument("--variable", default="Ekin", help="analysis in rigidity or in kinetic energy per nucleon")
    parser.add_argument("--isrebin", default=False, type=bool,  help="if with rebin")
    parser.add_argument("--isP8GBL", default=True, type=bool,  help="if with Pass8")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")

    parser.add_argument("--fillCounts", default=True, type=bool,  help="if fill counts")
    parser.add_argument("--cutl1up", default=False, type=bool,  help="if fill counts for acceptance")
    
    parser.add_argument("--fillMass", default=True, type=bool,  help="if fill mass")

    parser.add_argument("--fillRig", default=False, type=bool,  help="if fill Rig")
    parser.add_argument("--rigResoRefBeta", default=False, type=bool,  help="if fill Beta")

    parser.add_argument("--useModifyRig", default=False, type=bool,  help="if fill Beta")
    
    parser.add_argument("--fillBetaReso", default=False, type=bool,  help="if fill Beta")
    parser.add_argument("--useTunedBeta", default=False, type=bool,  help="if fill Beta")
    parser.add_argument("--fillBetaResidual", default=False, type=bool,  help="if fill Beta")
    parser.add_argument("--useGivenSigmaScale", default=False, type=bool,  help="if fill Beta")

    parser.add_argument("--fillRICHPosition", default=False, type=bool,  help="if fill RICHPosition")
    parser.add_argument("--fillBetaRICHCom", default=False, type=bool,  help="if fill BetaRICHCompare")
    
    args = parser.parse_args()                                                    
    os.makedirs(args.plotdir, exist_ok=True)
    nuclei = args.nuclei
    detectors = ["Tof", "NaF", "Agl"]
    #detectors = ["Tof"]
    isotopes = ISOTOPES[nuclei]
    detectors_alias = {"Tof": "tof", "Agl":"agl", "NaF": "naf"}
    variable = args.variable
    massbinning = Binning(mass_binning())
    inverse_mass_binning = Binning(fbinning_inversemass(nuclei))

    
    fine_inversemassbinning = make_lin_binning(0.08, 0.16, 2000)
    mass_delta_binning = make_lin_binning(-0.001, 0.006, 2000)
    #filenames = {iso: os.path.join(args.datadir, f"{iso}MC_BetaCor.root") for iso in ISOTOPES[nuclei]}
    if (nuclei == 'Be'):
        filenames = {iso: os.path.join(args.datadir, f"{iso}MC_B1236P8_CIEBetaCor.root") for iso in ISOTOPES[nuclei]}
    else:
        filenames = {iso: os.path.join(args.datadir, f"{iso}MC_B1308.root") for iso in ISOTOPES[nuclei]}
    print(filenames)
    
    #xbinning = Binning(LithiumRigidityBinningFullRange())
    xbinning = {"Rigidity": Binning(Rigidity_Analysis_Binning_FullRange()), "Ekin":Binning(fbinning_energy_rebin()) if args.isrebin else Binning(kinetic_energy_neculeon_binning())}
    xekinbinning_fromRig = Binning(calc_ekin_from_rigidity_iso(Rigidity_Analysis_Binning_FullRange()[:], 'Li6'))
    xekin_finelogbin = make_log_binning(0.1, 20, 30)

    xlabel = {"Rigidity": "Rigidity(GV)", "Ekin": "Ekin/n (GeV/n)"}
    
    hist_counts_build = {dec: {} for dec in detectors}
    hist_counts_test = {dec: {} for dec in detectors}
    hist_counts_all = {dec: {} for dec in detectors}
    hist_counts_vsGenEkn = {dec: {} for dec in detectors}
    dict_graph_counts = {dec: {} for dec in detectors}
    dict_hist_counts = {dec: {} for dec in detectors}

    hist_mass = {dec: {} for dec in detectors}
    hist_mass_mixweight = {dec: {} for dec in detectors}
    hist_mass_build = {dec: {} for dec in detectors}
    hist_mass_build_extraweight = {dec: {} for dec in detectors}
    hist_mass_test = {dec: {} for dec in detectors}
    
    hist_mass_mcmix_test = dict()
    hist_mass_mcmix_build = dict()

    hist_rig_residual_vsEkin = {dec: {} for dec in detectors}
    hist_rig_resolution_vsEkin = {dec: {} for dec in detectors}

    hist_rig_residual_vsR = {dec: {} for dec in detectors}
    hist_rig_resolution_vsR = {dec: {} for dec in detectors}
  
    hist_beta_residual_vsbeta = {dec: {} for dec in detectors}     
    hist_beta_resolution = {dec: {} for dec in detectors}            

    hist_inversebeta_residual_vsEkin = {dec: {} for dec in detectors}
    hist_inversebeta_residual_vsBeta = {dec: {} for dec in detectors}
    hist_beta_residual_vsBeta = {dec: {} for dec in detectors}
    hist_beta_residual_vsEkin = {dec: {} for dec in detectors}
    hist_beta_resolution = {dec: {} for dec in detectors}

  
    hist_beta_reso_vsEkinGen = {dec: {} for dec in detectors}
    hist_inversebeta_reso_vsEkinGen = {dec: {} for dec in detectors}

    hist_counts_vs_truerigidity = {dec: {} for dec in detectors}
    
    mix_com = {'Tof':{'Be7': 1.0, 'Be9': 0.6, 'Be10': 0.15},
               'NaF':{'Be7': 1.0, 'Be9': 0.6, 'Be10': 0.2},
               'Agl':{'Be7': 1.0, 'Be9': 0.6, 'Be10': 0.3}}
    dict_hist_mass = {}

    weight_mix = {'Be7': 1.0, 'Be9': 1.0, 'Be10': 1.0, 'Li6': 1.0, 'Li7':1.0}        
    hist_rich_position = Histogram(Binning(ring_position_binning()), Binning(ring_position_binning()), labels=['X (cm)', 'Y (cm)'])
    
    for dec in detectors:
        hist_mass_mcmix_test[dec] = WeightedHistogram(xbinning["Ekin"], inverse_mass_binning, labels=[xlabel[variable], "mass"])
        hist_mass_mcmix_build[dec] = WeightedHistogram(xbinning["Ekin"], inverse_mass_binning, labels=[xlabel[variable], "mass"])  
        for iso in isotopes:
            
            hist_counts_vs_truerigidity[dec][iso] = Histogram(xbinning['Rigidity'], labels=['rigidity (GV)'])
            hist_counts_all[dec][iso] = WeightedHistogram(xbinning[variable], labels=[xlabel[variable]])
            hist_counts_test[dec][iso] = WeightedHistogram(xbinning[variable], labels=[xlabel[variable]])
            hist_counts_build[dec][iso] = WeightedHistogram(xbinning[variable], labels=[xlabel[variable]])
            hist_mass[dec][iso] = WeightedHistogram(xbinning["Ekin"], inverse_mass_binning, labels=[xlabel[variable], "mass"])
            hist_mass_mixweight[dec][iso] = WeightedHistogram(xbinning["Ekin"], inverse_mass_binning, labels=[xlabel[variable], "mass"])
            hist_mass_test[dec][iso] = WeightedHistogram(xbinning["Ekin"], inverse_mass_binning, labels=[xlabel[variable], "mass"])
            hist_mass_build[dec][iso] = WeightedHistogram(xbinning["Ekin"], inverse_mass_binning, labels=[xlabel[variable], "mass"])
            hist_mass_build_extraweight[dec][iso] = WeightedHistogram(xbinning["Ekin"], inverse_mass_binning, labels=[xlabel[variable], "mass"])            
            hist_rig_residual_vsEkin[dec][iso] = WeightedHistogram(xbinning["Ekin"], binning_rig_residual, labels=["Ekin/N (GeV/n)", r"$\mathrm{1/R_{rec} - 1/R_{gen}}$"])
            hist_rig_resolution_vsEkin[dec][iso] = WeightedHistogram(xbinning["Ekin"], binning_rig_resolution, labels=["Ekin/N (GeV/n)", r"$\mathrm{(1/R_{rec} - 1/R_{gen})/1/R_{gen}}$"])    

            hist_rig_residual_vsR[dec][iso] = WeightedHistogram(xbinning["Rigidity"], binning_rig_residual, labels=["R (GV)", r"$\mathrm{1/R_{rec} - 1/R_{gen}}$"])
            hist_rig_resolution_vsR[dec][iso] = WeightedHistogram(xbinning["Rigidity"], binning_rig_resolution, labels=["R (GV)", r"$\mathrm{(1/R_{rec} - 1/R_{gen})/1/R_{gen}}$"])    

            hist_beta_residual_vsEkin[dec][iso] = WeightedHistogram(xekin_finelogbin, binning_beta_residual[dec], labels=["Ekin/n (GeV/n)", r"$\mathrm{\beta_{m}-\beta{t}}$"])    
            hist_inversebeta_residual_vsEkin[dec][iso] = WeightedHistogram(xekin_finelogbin, binning_inversebeta_residual[dec], labels=["Ekin/n (GeV/n)", r"$\mathrm{1/\beta_{m}-1/\beta{t}}$"])
            hist_inversebeta_residual_vsBeta[dec][iso] = WeightedHistogram(binning_beta[dec], binning_inversebeta_residual[dec], labels=[r"$\beta_{gen}$", r"$\mathrm{1/\beta_{m}-1/\beta{t}}$"])
            hist_beta_residual_vsBeta[dec][iso] = WeightedHistogram(binning_beta[dec], binning_inversebeta_residual[dec], labels=[r"$\beta_{gen}$", r"$\mathrm{beta_{m}-beta{t}}$"])
            
            hist_beta_reso_vsEkinGen[dec][iso] = WeightedHistogram(xekin_finelogbin, binning_beta_resolution[dec], labels=["Ekin/n (GeV/n)", r"$\mathrm{(\beta_{m}-\beta{t})/\beta_{t}}$"])    
            hist_inversebeta_reso_vsEkinGen[dec][iso] = WeightedHistogram(xekin_finelogbin, binning_beta_resolution[dec], labels=["Ekin/n (GeV/n)", r"$\mathrm{\beta_{m}/\beta_{t}-1}$"])

            
            for events in read_tree(filenames[iso], args.treename, chunk_size=args.chunk_size):
                #selections
                events = SelectUnbiasL1LowerCut(events, NUCLEI_CHARGE[nuclei])
                events = SelectCleanEvent(events)
                #events = events[events.is_richpass == 1]
                events_presel = SelectCleanEvent(events)
                events_presel = SelectUnbiasL1LowerCut(events, NUCLEI_CHARGE[nuclei])
                true_rig_presel = ak.to_numpy(events_presel["mevmom1"][:, 0]/NUCLEI_CHARGE[nuclei]) 
                hist_counts_vs_truerigidity[dec][iso].fill(true_rig_presel)
                                            
                events = rich_selectors["CIEMAT"][dec](events, nuclei, iso, "MC", cutoff=False, rebin=args.isrebin, cut_ubl1=args.cutl1up, cutTofEdge=True)

                #################################################################
                #wrtie 1d histogram of counts before using the l1upper charge cut
                #################################################################
                if dec == "Tof":
                    beta0 = ak.to_numpy(events.tof_betah)
                else:
                    beta0 = ak.to_numpy(events['rich_beta_cor'])

                rig_gen0 = ak.to_numpy(events["mevmom1"][:, 0])/NUCLEI_CHARGE[nuclei]
                ekin_gen0 = calc_ekin_from_rigidity(rig_gen0, MC_PARTICLE_IDS[iso])
                xval0 = {"Rigidity": rig_gen0, "Ekin": ekin_gen0}
                hist_counts_all[dec][iso].fill(xval0[variable], weights=np.ones_like(ak.to_numpy(events.ww)))

                ##################################################################
                #start fill mass histogram fom MC
                ##################################################################
                events = events[events.is_ub_l1 == 1]

                #events = events[beta0 < 1]
                rig_gen = ak.to_numpy(events["mevmom1"][:, 0])/NUCLEI_CHARGE[nuclei]
                ekin_gen = calc_ekin_from_rigidity(rig_gen, MC_PARTICLE_IDS[iso])
                xval = {"Rigidity": rig_gen, "Ekin": ekin_gen}
                true_momentum = ak.to_numpy(events["mevmom1"][:, 0])
                true_beta = calc_beta(rig_gen, ISOTOPES_MASS[iso], ISOTOPES_CHARGE[iso])
                true_rig_inntrk = ak.to_numpy(events["mevmom1"][:, 12]/NUCLEI_CHARGE[nuclei])  
                true_rig_rich = ak.to_numpy(events["mevmom1"][:, 20]/NUCLEI_CHARGE[nuclei])
                true_beta_rich = calc_beta(true_rig_rich, ISOTOPES_MASS[iso], ISOTOPES_CHARGE[iso])
                true_beta_inntrk = calc_beta(true_rig_inntrk, ISOTOPES_MASS[iso], ISOTOPES_CHARGE[iso])

                
                if dec == "Tof":
                    beta = ak.to_numpy(events.tof_betah)
                    ######Not Tuning for Tof for B1308
                    beta_tuned = beta
                    true_beta_atdec = true_beta_inntrk
                    rig_fromBeta = calc_rig_iso(beta, iso)
                else:
                    #beta = ak.to_numpy(events['rich_beta'][:, 0])
                    beta = ak.to_numpy(events['rich_beta_cor'])
                    
                    if args.useGivenSigmaScale:
                        #beta_tuned = GetMCRICHTunedBeta_WithHighR(events, beta, dec, iso, exterr=ExtErr, given_korr_sigma=min_Sigma_Scale[nuclei][dec])
                        beta_tuned = GetMCRICHTunedInverseBeta_GivenTuneValues(events, beta, iso, meanshift[Nuclei][dec], scalesigma[Nuclei][dec])
                    else:
                        #beta_tuned = GetMCRICHTunedBeta_WithHighR(events, beta, dec, iso, exterr=ExtErr)
                        beta_tuned = beta #Tuning of Agl to be checked
                        
                    true_beta_atdec = true_beta_rich
                    rig_fromBeta = calc_rig_iso(beta_tuned, iso)

                ekin_gen_atdec = calc_ekin_from_beta(true_beta_atdec)
                if args.isP8GBL:
                    rigidity = ak.to_numpy((events.tk_rigidity1)[:, 1, 2, 1])
                else:
                    rigidity = ak.to_numpy((events.tk_rigidity1)[:, 0, 2, 1])

                #rig_modified = rigidity
                #condition = rigidity - true_rig_inntrk > 1.5 * (0.1 * true_rig_inntrk)
                #rig_modified[condition] = true_rig_inntrk[condition] + (rigidity[condition] - true_rig_inntrk[condition])  * 1.05
                rig_modified = true_rig_inntrk + (rigidity - true_rig_inntrk) * 1.05
                
                weight = ak.to_numpy(events.ww)
                #weight = np.ones_like(fluxweight)

                run = events['run']
                run_test = run%10 < 3
                run_build = run%10 >= 2
                
                

                ######################################################################
                ########use beta_tuned#################################################

                if args.useModifyRig:
                    rigidity = rig_modified
                
                if args.useTunedBeta:
                    ekin = calc_ekin_from_beta(beta_tuned)
                    invmass = calc_inverse_mass(beta_tuned, rigidity, NUCLEI_CHARGE[nuclei])
                else:
                    ekin = calc_ekin_from_beta(beta)
                    invmass = calc_inverse_mass(beta, rigidity, NUCLEI_CHARGE[nuclei])

                #################
                #shift inversemass accroding to cutoff effect study
                invmass = invmass + shiftMPVmass[nuclei][dec]
                #ekin = calc_ekin_from_beta(beta)
                #invmass = calc_inverse_mass(beta, rigidity, NUCLEI_CHARGE[nuclei])
                
                ########################################################################
                invmass_build = invmass[run_build]
                invmass_test = invmass[run_test]
                ekin_test = ekin[run_test]
                ekin_build = ekin[run_build]
                weight_test = weight[run_test]
                weight_build = weight[run_build]

                #mix_weight = hist_counts_forweight[dec][iso].values[hist_counts_forweight[dec][iso].get(ekin)]/hist_counts_forweight[dec]['Be7'].values[hist_counts_forweight[dec]['Be7'].get(ekin)] * weight_mix[iso]
                #mix_weight = get_bin_content(hist_counts_forweight[dec][iso], ekin)/get_bin_content(hist_counts_forweight[dec]['Be7'], ekin) * mix_com[dec][iso]
                #mix_weight_test = get_bin_content(hist_counts_forweight[dec][iso], ekin_test)/get_bin_content(hist_counts_forweight[dec]['Be7'], ekin_test) * mix_com[dec][iso]
                #mix_weight_build = get_bin_content(hist_counts_forweight[dec][iso], ekin_build)/get_bin_content(hist_counts_forweight[dec]['Be7'], ekin_build) * mix_com[dec][iso]
                #print(mix_weight)


                if args.fillCounts:
                    hist_counts_test[dec][iso].fill(ekin_test, weights=weight_test)
                    hist_counts_build[dec][iso].fill(ekin_build, weights=weight_build)

                if args.fillMass:
                    hist_mass[dec][iso].fill(ekin, invmass, weights=weight)
                    hist_mass_mixweight[dec][iso].fill(ekin, invmass, weights=weight * weight_mix[iso])
                    hist_mass_build[dec][iso].fill(ekin_build, invmass_build, weights=weight_build)
                    hist_mass_build_extraweight[dec][iso].fill(ekin_build, invmass_build, weights=weight_build * weight_mix[iso])
                    hist_mass_mcmix_build[dec].fill(ekin_build, invmass_build, weights=weight_build * weight_mix[iso])                
                    hist_mass_test[dec][iso].fill(ekin_test, invmass_test, weights=weight_test * weight_mix[iso])
                    hist_mass_mcmix_test[dec].fill(ekin_test, invmass_test, weights=weight_test * weight_mix[iso])

                if args.fillRig:
                    if args.rigResoRefBeta:
                        hist_rig_resolution_vsEkin[dec][iso].fill(rig_fromBeta, (1/rigidity - 1/rig_fromBeta)/(1/rig_fromBeta), weights=weight)  
                        hist_rig_residual_vsEkin[dec][iso].fill(rig_fromBeta, 1/rigidity - 1/rig_fromBeta, weights=weight)
                        hist_rig_resolution_vsR[dec][iso].fill(rig_fromBeta, (1/rigidity - 1/rig_fromBeta)/(1/rig_fromBeta), weights=weight)  
                        hist_rig_residual_vsR[dec][iso].fill(rig_fromBeta, 1/rigidity - 1/rig_fromBeta, weights=weight)  

                    else:
                        hist_rig_resolution_vsEkin[dec][iso].fill(ekin_gen, (1/rigidity - 1/rig_gen)/(1/rig_gen), weights=weight)  
                        hist_rig_residual_vsEkin[dec][iso].fill(ekin_gen, 1/rigidity - 1/rig_gen, weights=weight)  
                        hist_rig_resolution_vsR[dec][iso].fill(rig_gen, (1/rigidity - 1/rig_gen)/(1/rig_gen), weights=weight)  
                        hist_rig_residual_vsR[dec][iso].fill(rig_gen, 1/rigidity - 1/rig_gen, weights=weight)  

                if args.fillBetaResidual:
                    hist_beta_residual_vsEkin[dec][iso].fill(ekin_gen, beta-true_beta, weights=weight)              
                    hist_inversebeta_residual_vsEkin[dec][iso].fill(ekin_gen, 1/beta-1/true_beta, weights=weight)
                    hist_inversebeta_residual_vsBeta[dec][iso].fill(true_beta, 1/beta-1/true_beta, weights=weight)
                    hist_beta_residual_vsBeta[dec][iso].fill(true_beta, beta-true_beta, weights=weight)

                if args.fillBetaReso:
                    hist_beta_reso_vsEkinGen[dec][iso].fill(ekin_gen, (beta-true_beta)/true_beta, weights=weight)              
                    hist_inversebeta_reso_vsEkinGen[dec][iso].fill(ekin_gen, (1/beta-1/true_beta)/(1/true_beta), weights=weight)

                if args.fillRICHPosition:
                    richevents = events[events.irich >= 0]
                    xpos = ak.to_numpy((richevents.rich_pos)[:, 0])                                                    
                    ypos = ak.to_numpy((richevents.rich_pos)[:, 1])  
                    hist_rich_position.fill(xpos, ypos)
                    
            hist_counts_vs_truerigidity[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_presel_counts_vsGenR")
            #plot counts
            if args.fillCounts:
                graph_counts = MGraph(xbinning[variable].bin_centers[1:-1], np.sum(hist_mass[dec][iso].values[1:-1, 1:-1], axis=1), np.sqrt(np.sum(hist_mass[dec][iso].values[1:-1, 1:-1], axis=1)))
                graph_counts.add_to_file(dict_hist_mass, f"graph_{dec}_{iso}MC_counts")
                hist_counts_all[dec][iso].add_to_file(dict_hist_mass, f"hist_{iso}MC_{dec}_counts")
                hist_counts_test[dec][iso].add_to_file(dict_hist_mass, f"hist_{iso}MC_{dec}_counts_test")
                hist_counts_build[dec][iso].add_to_file(dict_hist_mass, f"hist_{iso}MC_{dec}_counts_build")

            if args.fillMass:
                hist_mass[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_mass")
                hist_mass_mixweight[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_mass_mixweight")
                hist_mass_test[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_mass_test")
                hist_mass_build[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_mass_build")
                hist_mass_build_extraweight[dec][iso].add_to_file(dict_hist_mass, f"{iso}MC_{dec}_mass_build_mixweight")

            if args.fillRig:
                hist_rig_resolution_vsEkin[dec][iso].add_to_file(dict_hist_mass, f"hist_rig_resolution_{dec}{iso}_vsEkn")
                hist_rig_residual_vsEkin[dec][iso].add_to_file(dict_hist_mass, f"hist_rig_residual_{dec}{iso}_vsEkn")     

                hist_rig_resolution_vsR[dec][iso].add_to_file(dict_hist_mass, f"hist_rig_resolution_{dec}{iso}_vsR")
                hist_rig_residual_vsR[dec][iso].add_to_file(dict_hist_mass, f"hist_rig_residual_{dec}{iso}_vsR")     

            if args.fillBetaResidual:
                hist_beta_residual_vsEkin[dec][iso].add_to_file(dict_hist_mass, f'hist2d_beta_residual_{dec}{iso}')   
                hist_inversebeta_residual_vsEkin[dec][iso].add_to_file(dict_hist_mass, f'hist2d_inversebeta_residual_{dec}{iso}')
                hist_inversebeta_residual_vsBeta[dec][iso].add_to_file(dict_hist_mass, f'hist2d_inversebeta_residual_vsBeta_{dec}{iso}')
                hist_beta_residual_vsBeta[dec][iso].add_to_file(dict_hist_mass, f'hist2d_beta_residual_vsBeta_{dec}{iso}') 

            if args.fillBetaReso:
                hist_beta_reso_vsEkinGen[dec][iso].add_to_file(dict_hist_mass, f'hist2d_beta_resolution_{dec}{iso}')   
                hist_inversebeta_reso_vsEkinGen[dec][iso].add_to_file(dict_hist_mass, f'hist2d_inversebeta_resolution_{dec}{iso}') 
            
                
                
        if args.fillMass:
            hist_mass_mcmix_test[dec].add_to_file(dict_hist_mass, f"{nuclei}MCMix_{dec}_mass_test")
            hist_mass_mcmix_build[dec].add_to_file(dict_hist_mass, f"{nuclei}MCMix_{dec}_mass_build")

            
    if args.fillRICHPosition:
        hist_rich_position.add_to_file(dict_hist_mass, f"RICH{iso}_richposition") 
                
    #np.savez(os.path.join(args.outdatadir, f"{nuclei}MC_2Dhist_{FigName}_{variable}_B1236tunedbetaHighR{ExtErr}_NaFAgl.npz"), **dict_hist_mass)
    np.savez(os.path.join(args.plotdir, f"{nuclei}MC_hist_{FigName}.npz"), **dict_hist_mass)
    

    
    plt.show()

if __name__ == "__main__":   
    main()

    
