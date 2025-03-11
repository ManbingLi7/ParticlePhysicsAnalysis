from pyunfold import iterative_unfold
from pyunfold.callbacks import Logger
import multiprocessing as mp
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from tools.roottree import read_tree
from tools.selections import *
import scipy.stats
from scipy.optimize import curve_fit
from tools.binnings_collection import LithiumRichAglBetaResolutionBinning, LithiumRigidityBinningFullRange, fbinning_energy, kinetic_energy_neculeon_binning
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center, BeRigidityBinningRICHRange, Rigidity_Analysis_Binning

from tools.binnings_collection import fbinning_energy

import matplotlib.pyplot as plt
import seaborn as sns
from tools.studybeta import hist1d, hist2d
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, plot1d_errorbar_v2, plot1d_step
from tools.calculator import calc_ratio_err, calculate_efficiency_and_error, calc_beta_from_ekin
from tools.histograms import WeightedHistogram, Histogram, plot_histogram_2d, plot_histogram_1d
from tools.binnings import Binning, make_lin_binning, make_log_binning, reduce_bins
from tools.calculator import calc_ekin_from_rigidity_iso, calc_gamma_from_momentum, calc_beta, calc_ekin_from_beta, calc_ekin_from_rigidity_iso
from tools.plottools import xaxistitle
from tools.constants import ISOTOPES_CHARGE, ISOTOPES_MASS, RICH_SIGMA_SCALE , RICH_MEAN_SHIFT

detectors = {"Tof"}

xbinning = {"Rigidity": Binning(BeRigidityBinningRICHRange()),
            "Ekin": {"Tof": Binning(fbinning_energy()), "NaF": Binning(fbinning_energy()), "Agl": Binning(fbinning_energy())},
            "Gamma": make_log_binning(1.6, 100, 100),
            'Beta': {'Tof': Binning(calc_beta_from_ekin(fbinning_energy())), 'NaF': Binning(calc_beta_from_ekin(fbinning_energy())), 'Agl': Binning(calc_beta_from_ekin(fbinning_energy()))}}
            
xekinbinning = reduce_bins(Binning(calc_ekin_from_rigidity_iso(Rigidity_Analysis_Binning()[:-1], 'Be7')), 2)
#xekinbinning = Binning(calc_ekin_from_rigidity_iso(Rigidity_Analysis_Binning()[:], 'Be7'))

print('xekinbinning:', xekinbinning)   

setplot_defaultstyle()
binning_rig_residual = make_lin_binning(-0.5, 0.7, 550)
binning_rig_resolution = make_lin_binning(-1, 1, 300)
binning_beta_resolution = {"Tof": make_lin_binning(-0.2, 0.2, 200), "NaF": make_lin_binning(-0.05, 0.05, 300), "Agl": make_lin_binning(-0.02, 0.02, 400)}
binning_beta_residual = {"Tof": make_lin_binning(-0.07, 0.04, 170), "NaF": make_lin_binning(-0.03, 0.01, 300), "Agl": make_lin_binning(-0.006, 0.003, 300)}
binning_inversebeta_residual = {"Tof": make_lin_binning(-0.07, 0.15, 300), "NaF": make_lin_binning(-0.01, 0.01, 70), "Agl": make_lin_binning(-0.003, 0.003, 100)}


binning_tof_naf_mius = make_lin_binning(-0.1, 0.05, 200)

binning_gamma = make_lin_binning(0, 10, 1000)
rich_selectors = {"LIP": {"Tof": selector_tof, "NaF": selector_naf_lipvar, "Agl": selector_agl_lipvar},
                  "CIEMAT": {"Tof":selector_tof, "NaF": selector_naf_ciematvar, "Agl": selector_agl_ciematvar}}

nuclei = 'Be'
isotopes = ISOTOPES[nuclei]
print('isotopes:', isotopes)

riglim = {'Tof': 200, 'NaF': 100, 'Agl': 200}
isoweight = {'Be7': 0.6, 'Be9': 0.3, 'Be10': 0.1,
             'O16': 1.0, 'C12': 1.0}

def handle_file(arg):    
    filename_iss, filename_mc, treename, chunk_size,  rank, nranks, kwargs = arg
    resultdir = kwargs["resultdir"]

    hist_issbetareso = {}
    hist_mcbetareso_mix = {}
    hist_mcbetareso = {dec: {} for dec in detectors}
    hist_beta_reso_dict= {}
    
    for dec in detectors:
        hist_issbetareso[dec] = WeightedHistogram(xekinbinning, binning_tof_naf_mius, labels=['Ekin/n (GeV/n)', r"$\mathrm{1/\beta}$"])
        hist_mcbetareso_mix[dec] = WeightedHistogram(xekinbinning, binning_tof_naf_mius, labels=['Ekin/n (GeV/n)', r"$\mathrm{1/\beta}$"])
        for iso in isotopes:
            hist_mcbetareso[dec][iso] = WeightedHistogram(xekinbinning, binning_tof_naf_mius, labels=['Ekin/n (GeV/n)', r"$\mathrm{1/\beta}$"])

    for dec in detectors: 
        for events in read_tree(filename_iss, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
            events = SelectUnbiasL1LowerCut(events, NUCLEI_CHARGE[nuclei])
            events = SelectCleanEvent(events)
            events = rich_selectors["CIEMAT"]['Tof'](events, nuclei, iso, "ISS", cutoff=False)
            events = rich_selectors["CIEMAT"][dec](events, nuclei, iso, "ISS", cutoff=False)
            events = geomagnetic_IGRF_cutoff(events)
            rigidity = ak.to_numpy((events.tk_rigidity1)[:, 0, 2, 1])
            beta_fromRig = calc_beta(rigidity, ISOTOPES_MASS['O16'], ISOTOPES_CHARGE['O16'])
            beta_tof = ak.to_numpy(events['tof_betah'])
            ekin = calc_ekin_from_beta(beta_tof)
            #beta_naf = ak.to_numpy(events['rich_beta_cor'])
            #beta_naf = ak.to_numpy(events['rich_beta'][:, 0])
            #ekin_naf = calc_ekin_from_beta(beta_naf)
    
            hist_issbetareso[dec].fill(ekin, beta_tof-beta_fromRig, weights=np.ones_like(beta_tof))        
        hist_issbetareso[dec].add_to_file(hist_beta_reso_dict, f"hist_issbetareso_{dec}")
        

    for dec in detectors: 
        for i_iso, iso in enumerate(isotopes):
            for events in read_tree(filename_mc[i_iso], treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
                events = SelectUnbiasL1LowerCut(events, NUCLEI_CHARGE[nuclei])
                events = SelectCleanEvent(events)
                events = rich_selectors["CIEMAT"]['Tof'](events, nuclei, iso, "MC", cutoff=False)
                #events = rich_selectors["CIEMAT"][dec](events, nuclei, iso, "MC", cutoff=False)
                #######################
                #select events in NaF geo
                #events = IsWithin_RICHAgl(events)
                
                rigidity = ak.to_numpy((events.tk_rigidity1)[:, 0, 2, 1])
                
                mc_weight = ak.to_numpy(events["ww"])
                weight_one = np.ones(len(mc_weight))

                true_momentum = ak.to_numpy(events["mevmom1"][:, 0])
                true_rigidity = ak.to_numpy(events["mevmom1"][:, 0]/NUCLEI_CHARGE[nuclei])
                true_ekin = calc_ekin_from_rigidity_iso(true_rigidity, iso)
                true_beta = calc_beta(true_rigidity, ISOTOPES_MASS[iso], ISOTOPES_CHARGE[iso])         
                
                true_momentum_atInnTrk = ak.to_numpy(events["mevmom1"][:, 12])
                true_rigidity_atInnTrk = ak.to_numpy(events["mevmom1"][:, 12]/NUCLEI_CHARGE[nuclei])
                true_beta_atInnTrk = calc_beta(true_rigidity_atInnTrk, ISOTOPES_MASS[iso], ISOTOPES_CHARGE[iso])
                true_ekin_atInnTrk = calc_ekin_from_rigidity_iso(true_rigidity_atInnTrk, iso)
                
                beta_tof = ak.to_numpy(events['tof_betahmc'])
                #beta_naf = ak.to_numpy(events['rich_beta_cor'])
                #beta_naf_tuned = RICH_MEAN_SHIFT['NaF'][nuclei] + (beta_naf - true_beta_atRICH) * RICH_SIGMA_SCALE['NaF'][nuclei] + true_beta_atRICH
                
                ekin = calc_ekin_from_beta(beta_tof)
                #ekin_naf = calc_ekin_from_beta(beta_naf)
                #ekin_naf_tuned = calc_ekin_from_beta(beta_naf_tuned)  

                weight_mix = isoweight[iso] * mc_weight
                hist_mcbetareso[dec][iso].fill(true_ekin_atInnTrk, beta_tof - true_beta_atInnTrk, weights=mc_weight)
                hist_mcbetareso_mix[dec].fill(true_ekin_atInnTrk, beta_tof - true_beta_atInnTrk, weights=weight_mix)

                #hist_mcbetareso[dec][iso].fill(ekin_naf_tuned, beta_tof - true_beta, weights=mc_weight)
                #hist_mcbetareso_mix[dec].fill(ekin_naf_tuned, beta_tof - true_beta, weights=weight_mix)
                 
            hist_mcbetareso[dec][iso].add_to_file(hist_beta_reso_dict, f'hist_mcbetareso_{dec}{iso}')
        hist_mcbetareso_mix[dec].add_to_file(hist_beta_reso_dict, f'hist_mcbetareso_mix_{dec}') 
            
    np.savez(os.path.join(resultdir, f"tofbeta_{rank}.npz"), **hist_beta_reso_dict)

def make_args(filename_iss, filename_mc, treename, chunk_size, nranks, **kwargs):
    for rank in range(nranks):
        yield (filename_iss, filename_mc, treename, chunk_size, rank, nranks, kwargs)

def main():
    import argparse    
    parser = argparse.ArgumentParser()    
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)") 
    #parser.add_argument("--filename_iss", default= "/home/manbing/Documents/Data/data_iss/BeISS_NucleiSelection_BetaCor.root",  help="(e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--filename_iss", default= "/home/manbing/Documents/Data/data_BeP8/rootfile/CarbonIss_P8_CIEBetaCor.root",  help="(e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--filename_mc", nargs='+', help="Path to root file to read tree from")
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file." )
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    #parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--nprocesses", type=int, default=2, help="Number of processes to use in parallel.")   
    parser.add_argument("--resultdir", default="/home/manbing/Documents/Data/data_BeP8", help="Directory to store plots and result files in.")
    args = parser.parse_args()

    #args.filename_mc = [f"/home/manbing/Documents/Data/data_mc/{iso}MC_NucleiSelection_clean_0.root" for iso in ISOTOPES[args.nuclei]]
    #args.filename_mc = [f"/home/manbing/Documents/Data/data_mc/dfile/{iso}MC_B1236_dst.root" for iso in isotopes]
    args.filename_mc = [f"/home/manbing/Documents/Data/data_BeP8/rootfile/{iso}MC_B1236P8_CIEBetaCor.root" for iso in isotopes]
    #args.filename_mc = [f"/home/manbing/Documents/Data/data_BeP8/rootfile/C12MC_B1236P8_addtof.root" for iso in isotopes]
    #args.filename_mc = [f"/home/manbing/Documents/Data/data_BeP8/rootfile/O16MC_B1236P8_CIECor_addtof.root" for iso in isotopes]
    
    #args.filename_mc = [f"/home/manbing/Documents/Data/data_mc/dfile/{iso}_B1220_rwth.root" for iso in isotopes]
    
    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.filename_iss, args.filename_mc, args.treename, args.chunk_size, args.nprocesses, resultdir=args.resultdir)
        for _ in pool.imap_unordered(handle_file, pool_args):
            pass
    
    hist_issbetareso = {}
    hist_mcbetareso_mix = {}
    hist_mcbetareso = {dec: {} for dec in detectors}
    hist_beta_reso_dict= {}
    
    for dec in detectors:
        hist_issbetareso[dec] = WeightedHistogram(xekinbinning, binning_tof_naf_mius, labels=['Ekin/n (GeV/n)', r"$\mathrm{beta_{tof} - beta_{naf}}$"])
        hist_mcbetareso_mix[dec] = WeightedHistogram(xekinbinning, binning_tof_naf_mius, labels=['Ekin/n (GeV/n)', r"$\mathrm{1/\beta}$"])
        for iso in isotopes:
            hist_mcbetareso[dec][iso] = WeightedHistogram(xekinbinning, binning_tof_naf_mius, labels=['Ekin/n (GeV/n)', r"$\mathrm{1/\beta}$"])

              
    for rank in range(args.nprocesses):
        filename_response =  os.path.join(args.resultdir, f"tofbeta_{rank}.npz")  
        with np.load(filename_response) as reso_file:
            for dec in detectors:
                hist_issbetareso[dec] += WeightedHistogram.from_file(reso_file, f"hist_issbetareso_{dec}")
                hist_mcbetareso_mix[dec] += WeightedHistogram.from_file(reso_file, f"hist_mcbetareso_mix_{dec}")
                for iso in isotopes:
                    hist_mcbetareso[dec][iso] += WeightedHistogram.from_file(reso_file, f"hist_mcbetareso_{dec}{iso}")

                    
    result_resolution = dict()
    for dec in detectors:
        hist_issbetareso[dec].add_to_file(result_resolution, f"hist_issbetareso_{dec}")
        hist_mcbetareso_mix[dec].add_to_file(result_resolution, f"hist_mcbetareso_mix_{dec}")
        for iso in isotopes: 
            hist_mcbetareso[dec][iso].add_to_file(result_resolution, f"hist_mcbetareso_{dec}{iso}")
        

    np.savez(os.path.join(args.resultdir, f"{nuclei}_tofbeta1Residual_refTrueBetaAtInnTrk_B1236P8_mcweight_TofGeo.npz"), **result_resolution)

    
if __name__ == "__main__":
    main()






    

