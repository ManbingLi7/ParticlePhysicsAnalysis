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
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center, BeRigidityBinningRICHRange
from tools.binnings_collection import fbinning_energy

import matplotlib.pyplot as plt
import seaborn as sns
from tools.studybeta import hist1d, hist2d, GetMCRICHTunedBeta_WithHighR, GetMCTofTunedBeta_WithSpline
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, plot1d_errorbar_v2, plot1d_step
from tools.calculator import calc_ratio_err, calculate_efficiency_and_error, calc_beta_from_ekin
from tools.histograms import WeightedHistogram, Histogram, plot_histogram_2d, plot_histogram_1d
from tools.binnings import Binning, make_lin_binning, make_log_binning
from tools.calculator import calc_ekin_from_rigidity_iso, calc_gamma_from_momentum, calc_beta, calc_ekin_from_beta
from tools.plottools import xaxistitle
from tools.constants import ISOTOPES_CHARGE, ISOTOPES_MASS, RICH_SIGMA_SCALE , RICH_MEAN_SHIFT

detectors = {"NaF", "Agl"}
xbinning = {"Rigidity": Binning(BeRigidityBinningRICHRange()),
            "Ekin": {"Tof": Binning(fbinning_energy()), "NaF": Binning(fbinning_energy()), "Agl": Binning(fbinning_energy())},
            "Gamma": make_log_binning(1.6, 100, 100),
            'Beta': {'Tof': Binning(calc_beta_from_ekin(fbinning_energy())), 'NaF': Binning(calc_beta_from_ekin(fbinning_energy())), 'Agl': Binning(calc_beta_from_ekin(fbinning_energy()))}}
            

binning_rig_residual = make_lin_binning(-0.5, 0.7, 550)
binning_rig_resolution = make_lin_binning(-1, 1, 300)
binning_beta_resolution = {"Tof": make_lin_binning(-0.2, 0.2, 200), "NaF": make_lin_binning(-0.05, 0.05, 300), "Agl": make_lin_binning(-0.02, 0.02, 400)}
binning_beta_residual = {"Tof": make_lin_binning(-0.07, 0.04, 170), "NaF": make_lin_binning(-0.03, 0.01, 300), "Agl": make_lin_binning(-0.006, 0.003, 300)}
binning_inversebeta_residual = {"Tof": make_lin_binning(-0.07, 0.07, 100), "NaF": make_lin_binning(-0.01, 0.01, 120), "Agl": make_lin_binning(-0.003, 0.003, 100)}

rigiditybinning =  make_lin_binning(50, 1000, 500)  
binning_gamma = make_lin_binning(0, 10, 1000)
rich_selectors = {"LIP": {"Tof": selector_tof, "NaF": selector_naf_lipvar, "Agl": selector_agl_lipvar},
                  "CIEMAT": {"Tof":selector_tof, "NaF": selector_naf_ciematvar, "Agl": selector_agl_ciematvar}}


riglim = {'Tof': 200, 'NaF': 100, 'Agl': 200}

isoweight = {'Be': {'Be7': 0.6, 'Be9': 0.4, 'Be10': 0.1},
             'O': {'O16': 1.0},
             'C': {'C12': 1.0},
             'B': {'B10': 0.5, 'B11': 0.5},
             'N': {'N14': 0.5, 'N15': 0.5},
             'Li': {'Li6': 0.5, 'Li7':0.5}}

nuclei = 'Be'
isotopes = ISOTOPES[nuclei]
useTunedBeta = False
#isotopes = ['B10']
useInverseBeta = True
FigName = 'BeforeTuneBeta'

def handle_file(arg):    
    filename_iss, filename_mc, treename, chunk_size,  rank, nranks, kwargs = arg
    resultdir = kwargs["resultdir"]
    #nuclei = kwargs['nuclei']

    hist_issbetareso = {}
    hist_mcbetareso_mix = {}
    hist_mcbetareso = {dec: {} for dec in detectors}
    hist_beta_reso_dict= {}
    hist_mcTrueR = {dec: {} for dec in detectors}
    
    for dec in detectors:
        hist_issbetareso[dec] = WeightedHistogram(binning_inversebeta_residual[dec], labels=[r"$\mathrm{1/\beta - 1}$"])
        hist_mcbetareso_mix[dec] = WeightedHistogram(binning_inversebeta_residual[dec], labels=[r"$\mathrm{1/\beta - 1}$"])

        for iso in isotopes:
            hist_mcbetareso[dec][iso] = WeightedHistogram(binning_inversebeta_residual[dec], labels=[r"$\mathrm{1/\beta - 1}$"])
            hist_mcTrueR[dec][iso] = WeightedHistogram(rigiditybinning, labels=[f"rigidity(GV)"])
            
    for dec in detectors: 
        for events in read_tree(filename_iss, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
            events = SelectUnbiasL1LowerCut(events, NUCLEI_CHARGE[nuclei])
            events = SelectEventsCharge(events, NUCLEI_CHARGE[nuclei])
            events = SelectCleanEvent(events)

            events = rich_selectors["CIEMAT"][dec](events, nuclei, iso, "ISS", cutoff=False)
            rigidity = ak.to_numpy((events.tk_rigidity1)[:, 0, 2, 1])
            events = events[rigidity > riglim[dec]]
            
            if dec == "Tof":
                beta = ak.to_numpy(events['tof_betah'])
            else:
                beta = ak.to_numpy(events['rich_beta_cor'])

            if useInverseBeta:
                hist_issbetareso[dec].fill(1/beta-1, weights=np.ones_like(beta))
            else:
                hist_issbetareso[dec].fill(beta-1, weights=np.ones_like(beta))
                
        hist_issbetareso[dec].add_to_file(hist_beta_reso_dict, f"hist_issbetareso_{dec}")
        

    for dec in detectors: 
        for i_iso, iso in enumerate(isotopes):
            for events in read_tree(filename_mc[i_iso], treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
                events = SelectUnbiasL1LowerCut(events, NUCLEI_CHARGE[nuclei])
                events = SelectCleanEvent(events)
                events = SelectEventsCharge(events, NUCLEI_CHARGE[nuclei])
                events = rich_selectors["CIEMAT"][dec](events, nuclei, iso, "MC", cutoff=False)
                rigidity = ak.to_numpy((events.tk_rigidity1)[:, 0, 2, 1])
                events = events[rigidity > riglim[dec]]
                mc_weight = ak.to_numpy(events["ww"])
                weight_one = np.ones(len(mc_weight))

                true_momentum = ak.to_numpy(events["mevmom1"][:, 0])
                true_rigidity = ak.to_numpy(events["mevmom1"][:, 0]/ISOTOPES_CHARGE[iso])
                
                true_ekin = calc_ekin_from_rigidity_iso(true_rigidity, iso)
                true_beta = calc_beta(true_rigidity, ISOTOPES_MASS[iso], ISOTOPES_CHARGE[iso])
                 
                if dec == "Tof":
                    beta = ak.to_numpy(events['tof_betahmc'])
                    beta_tuned = GetMCTofTunedBeta_WithSpline(events, nuclei, iso)
                    
                else:                    
                    beta = ak.to_numpy(events['rich_beta_cor'])
                    #beta = ak.to_numpy(events['rich_beta'][:, 0])
                    beta_tuned = GetMCRICHTunedBeta_WithHighR(events, beta, dec, iso, exterr=0)
                
                weight_mix = isoweight[nuclei][iso] * mc_weight

                if useTunedBeta:
                    useBeta = beta_tuned
                else:
                    useBeta = beta
                    
                if useInverseBeta:
                    hist_mcbetareso[dec][iso].fill(1/useBeta-1.0, weights=mc_weight)
                    hist_mcbetareso_mix[dec].fill(1/useBeta-1.0, weights=weight_mix)
                else:
                    hist_mcbetareso[dec][iso].fill(useBeta-1.0, weights=mc_weight)
                    hist_mcbetareso_mix[dec].fill(useBeta-1.0, weights=weight_mix)

                hist_mcTrueR[dec][iso].fill(true_rigidity, weights=mc_weight)
                
            hist_mcbetareso[dec][iso].add_to_file(hist_beta_reso_dict, f'hist_mcbetareso_{dec}{iso}')
            hist_mcTrueR[dec][iso].add_to_file(hist_beta_reso_dict, f'hist_mcTrueR_{dec}{iso}')
        hist_mcbetareso_mix[dec].add_to_file(hist_beta_reso_dict, f'hist_mcbetareso_mix_{dec}') 
            
    np.savez(os.path.join(resultdir, f"betahighR_{rank}.npz"), **hist_beta_reso_dict)

def make_args(filename_iss, filename_mc, treename, chunk_size, nranks, **kwargs):
    for rank in range(nranks):
        yield (filename_iss, filename_mc, treename, chunk_size, rank, nranks, kwargs)

def main():
    import argparse    
    parser = argparse.ArgumentParser()    
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)") 
    #parser.add_argument("--filename_iss", default= "/home/manbing/Documents/Data/data_iss/BeISS_NucleiSelection_BetaCor.root",  help="(e.g. results/ExampleAnalysisTree*.root)")
    #parser.add_argument("--filename_iss", default= "/home/manbing/Documents/Data/data_BeP8/rootfile/BeISS_P8_CIEBeta.root",  help="(e.g. results/ExampleAnalysisTree*.root)")
    #parser.add_argument("--filename_iss", default= f"/home/manbing/Documents/Data/data_BeP8/rootfile/{nuclei}ISSP8HighR_CIEBetaCor.root",  help="(e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--filename_iss", default= "/home/manbing/Documents/Data/data_BeP8/rootfile/ISSP8HighR_CIEBetaCor.root",  help="(e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--filename_mc", nargs='+', help="Path to root file to read tree from")
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file." )
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    #parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--nprocesses", type=int, default=2, help="Number of processes to use in parallel.")   
    parser.add_argument("--resultdir", default="/home/manbing/Documents/Data/data_BeP8", help="Directory to store plots and result files in.")
    #parser.add_argument("--nuclei", default="C", help="the nuclei to be analyzed")
    args = parser.parse_args()

    #nuclei = args.nuclei
    #args.filename_mc = [f"/home/manbing/Documents/Data/data_mc/{iso}MC_NucleiSelection_clean_0.root" for iso in ISOTOPES[args.nuclei]]
    #args.filename_mc = [f"/home/manbing/Documents/Data/data_BeP8/rootfile/{iso}MC_B1236P8_CIEBetaCor_Tof2.root" for iso in isotopes]
    args.filename_mc = [f"/home/manbing/Documents/Data/data_BeP8/rootfile/{iso}MC_B1236P8_CIEBetaCor.root" for iso in isotopes]
   
    #args.filename_mc = [f"/home/manbing/Documents/Data/data_LiP8/rootfile/{iso}MC_B1236P8_CIEBetaCor.root" for iso in isotopes]
    #args.filename_mc = [f"/home/manbing/Documents/Data/data_BeP8/rootfile/{iso}MC_B1236P8_addtof.root" for iso in isotopes]
    
    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.filename_iss, args.filename_mc, args.treename, args.chunk_size, args.nprocesses, resultdir=args.resultdir)
        for _ in pool.imap_unordered(handle_file, pool_args):
            pass
    

    hist_issbetareso = {}
    hist_mcbetareso_mix = {}
    hist_mcbetareso = {dec: {} for dec in detectors}
    hist_beta_reso_dict= {}
    hist_mcTrueR = {dec: {} for dec in detectors}
    
    for dec in detectors:
        hist_issbetareso[dec] = WeightedHistogram(binning_inversebeta_residual[dec], labels=[r"$\mathrm{1/\beta - 1}$"])
        hist_mcbetareso_mix[dec] = WeightedHistogram(binning_inversebeta_residual[dec], labels=[r"$\mathrm{1/\beta - 1}$"])
        for iso in isotopes:                           
            hist_mcbetareso[dec][iso] = WeightedHistogram(binning_inversebeta_residual[dec], labels=[r"$\mathrm{1/\beta - 1}$"])
            hist_mcTrueR[dec][iso] = WeightedHistogram(rigiditybinning, labels=[f"rigidity(GV)"])
    for rank in range(args.nprocesses):
        filename_response =  os.path.join(args.resultdir, f"betahighR_{rank}.npz")  
        with np.load(filename_response) as reso_file:
            for dec in detectors:
                hist_issbetareso[dec] += WeightedHistogram.from_file(reso_file, f"hist_issbetareso_{dec}")
                hist_mcbetareso_mix[dec] += WeightedHistogram.from_file(reso_file, f"hist_mcbetareso_mix_{dec}")
                for iso in isotopes:
                    hist_mcbetareso[dec][iso] += WeightedHistogram.from_file(reso_file, f"hist_mcbetareso_{dec}{iso}")
                    hist_mcTrueR[dec][iso] += WeightedHistogram.from_file(reso_file, f"hist_mcTrueR_{dec}{iso}")
                    
    result_resolution = dict()
    for dec in detectors:
        hist_issbetareso[dec].add_to_file(result_resolution, f"hist_issbetareso_{dec}")
        hist_mcbetareso_mix[dec].add_to_file(result_resolution, f"hist_mcbetareso_mix_{dec}")
        for iso in isotopes: 
            hist_mcbetareso[dec][iso].add_to_file(result_resolution, f"hist_mcbetareso_{dec}{iso}")
            hist_mcTrueR[dec][iso].add_to_file(result_resolution, f"hist_mcTrueR_{dec}{iso}")
        


    np.savez(os.path.join(args.resultdir, f"{nuclei}ISS_{FigName}HighR_B1236P8_tofBeta1.npz"), **result_resolution)


    
if __name__ == "__main__":
    main()






    

