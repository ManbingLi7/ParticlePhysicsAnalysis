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
from tools.studybeta import hist1d, hist2d
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, plot1d_errorbar_v2, plot1d_step
from tools.calculator import calc_ratio_err, calculate_efficiency_and_error, calc_beta_from_ekin
from tools.histograms import WeightedHistogram, Histogram, plot_histogram_2d, plot_histogram_1d
from tools.binnings import Binning, make_lin_binning, make_log_binning
from tools.calculator import calc_ekin_from_rigidity_iso, calc_gamma_from_momentum, calc_beta, calc_ekin_from_beta
from tools.plottools import xaxistitle
from tools.constants import ISOTOPES_CHARGE, ISOTOPES_MASS

detectors = {"Tof", "NaF", "Agl"}
xbinning = {"Rigidity": Binning(BeRigidityBinningRICHRange()),
            "Ekin": {"Tof": Binning(fbinning_energy()), "NaF": Binning(fbinning_energy()), "Agl": Binning(fbinning_energy())},
            "Gamma": make_log_binning(1.6, 100, 100),
            'Beta': {'Tof': Binning(calc_beta_from_ekin(fbinning_energy())), 'NaF': Binning(calc_beta_from_ekin(fbinning_energy())), 'Agl': Binning(calc_beta_from_ekin(fbinning_energy()))}}
            
for dec in detectors:
    print(xbinning['Ekin'][dec].edges)
    print(xbinning['Beta'][dec].edges)

print(xbinning['Gamma'].edges)
norminal_charge = 4.0
setplot_defaultstyle()
binning_rig_residual = make_lin_binning(-0.5, 0.7, 550)
binning_rig_resolution = make_lin_binning(-1, 1, 300)
binning_beta_resolution = {"Tof": make_lin_binning(-0.2, 0.2, 200), "NaF": make_lin_binning(-0.05, 0.05, 300), "Agl": make_lin_binning(-0.02, 0.02, 400)}
binning_beta_residual = {"Tof": make_lin_binning(-0.07, 0.04, 170), "NaF": make_lin_binning(-0.03, 0.01, 300), "Agl": make_lin_binning(-0.006, 0.003, 300)}
binning_inversebeta_residual = {"Tof": make_lin_binning(-0.07, 0.15, 300), "NaF": make_lin_binning(-0.03, 0.03, 400), "Agl": make_lin_binning(-0.006, 0.006, 350)}
            
binning_gamma = make_lin_binning(0, 10, 1000)
rich_selectors = {"LIP": {"Tof": selector_tof, "NaF": selector_naf_lipvar, "Agl": selector_agl_lipvar},
                  "CIEMAT": {"Tof":selector_tof, "NaF": selector_naf_ciematvar, "Agl": selector_agl_ciematvar}}


isotopes = ["Be7", "Be9", "Be10"]

def handle_file(arg):    
    filename_iss, filename_mc, treename, chunk_size,  rank, nranks, kwargs = arg
    resultdir = kwargs["resultdir"]
    nuclei = kwargs['nuclei']
    hist_rig_resolution = dict()
    hist_beta_residual_vsbeta = {dec: {} for dec in detectors}
    hist_beta_resolution = {dec: {} for dec in detectors}
    hist_ekin_responsematrix = {dec: {} for dec in detectors}
    hist_beta_responsematrix = {dec: {} for dec in detectors}
    hist_inversebeta_residual_vsbeta = {dec: {} for dec in detectors}
    hist_rigidity_reso_dict = {}
    hist_beta_reso_dict= {}
    hist_gamma = {dec: {} for dec in detectors}
    hist_mcEkinRec = {dec: {} for dec in detectors}
    hist_mcEkinTrue = {dec: {} for dec in detectors}
    hist_issEkin = {}
    
    for dec in detectors:
        hist_issEkin[dec] = Histogram(xbinning['Ekin'][dec], labels=[xaxistitle['Ekinn']])
        for iso in isotopes:
            hist_beta_resolution[dec][iso] = WeightedHistogram(xbinning['Gamma'], binning_beta_resolution[dec], labels=[r"$\mathrm{\gamma}$", r"$\mathrm{\beta_{m}/\beta{t}-1}$"])
            hist_beta_residual_vsbeta[dec][iso] = WeightedHistogram(xbinning['Beta'][dec], binning_beta_residual[dec], labels=[r"$\mathrm{\beta}$", r"$\mathrm{\beta_{m}-\beta{t}}$"])
            hist_inversebeta_residual_vsbeta[dec][iso] = WeightedHistogram(xbinning['Beta'][dec], binning_inversebeta_residual[dec], labels=[r"$\mathrm{\beta}$", r"$\mathrm{1/\beta_{m}-1/\beta{t}}$"])
            hist_ekin_responsematrix[dec][iso] = WeightedHistogram(xbinning['Ekin'][dec], xbinning['Ekin'][dec], labels=["Ekin/n", "Ekin/n"])
            hist_beta_responsematrix[dec][iso] = WeightedHistogram(xbinning['Beta'][dec], xbinning['Beta'][dec], labels=["beta", "beta"])
            hist_gamma[dec][iso] = WeightedHistogram(xbinning['Ekin'][dec], binning_gamma, labels=[xaxistitle['Ekinn'], r"$\gamma$"])
            hist_mcEkinRec[dec][iso] = WeightedHistogram(xbinning['Ekin'][dec], labels=[xaxistitle['Ekinn']])
            hist_mcEkinTrue[dec][iso] = WeightedHistogram(xbinning['Ekin'][dec], labels=[xaxistitle['Ekinn']])
            
    for dec in detectors: 
        for i_iso, iso in enumerate(isotopes):
            for events in read_tree(filename_mc[i_iso], treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
                events = SelectUnbiasL1LowerCut(events, 4.0)
                events = SelectCleanEvent(events)
                events = rich_selectors["CIEMAT"][dec](events, nuclei, iso, "MC", cutoff=False)
                rigidity = ak.to_numpy((events.tk_rigidity1)[:, 0, 2, 1])
                mc_weight = ak.to_numpy(events["ww"])
                weight_one = np.ones(len(mc_weight))

                if dec == "Tof":
                    true_momentum = ak.to_numpy(events["mevmom1"][:, 0])
                    true_rigidity = ak.to_numpy(events["mevmom1"][:, 0]/norminal_charge)
                    true_ekin = calc_ekin_from_rigidity_iso(true_rigidity, iso)
                    true_beta = calc_beta(true_rigidity, ISOTOPES_MASS[iso], ISOTOPES_CHARGE[iso])
                    true_gamma = calc_gamma_from_momentum(true_momentum, ISOTOPES_MASS[iso])
                    beta = ak.to_numpy(events['tof_betahmc'])
                else:
                    true_momentum = ak.to_numpy(events["mevmom1"][:, 0])
                    true_rigidity = ak.to_numpy(events["mevmom1"][:, 0]/norminal_charge)
                    true_ekin = calc_ekin_from_rigidity_iso(true_rigidity, iso)
                    true_beta = calc_beta(true_rigidity, ISOTOPES_MASS[iso], ISOTOPES_CHARGE[iso])
                    true_gamma = calc_gamma_from_momentum(true_momentum, ISOTOPES_MASS[iso])
                    beta = ak.to_numpy(events['rich_beta'][:, 0])
                    

                ekin = calc_ekin_from_beta(beta)
                np.nan_to_num(ekin, nan=np.inf)
                weight_mom = 1/true_momentum
                ekin_new = np.nan_to_num(ekin, nan=np.inf, posinf=np.inf, neginf=-np.inf)
                
                hist_beta_residual_vsbeta[dec][iso].fill(true_beta, beta-true_beta, weights=weight_one)
                hist_inversebeta_residual_vsbeta[dec][iso].fill(true_beta, 1/beta-1/true_beta, weights=weight_one)   
                                                                
            hist_beta_residual_vsbeta[dec][iso].add_to_file(hist_beta_reso_dict, f'hist2d_beta_residual_{dec}_{iso}')
            hist_inversebeta_residual_vsbeta[dec][iso].add_to_file(hist_beta_reso_dict, f'hist2d_inversebeta_residual_{dec}_{iso}') 
            
    np.savez(os.path.join(resultdir, f"betaresolution_{rank}.npz"), **hist_beta_reso_dict)

def make_args(filename_iss, filename_mc, treename, chunk_size, nranks, **kwargs):
    for rank in range(nranks):
        yield (filename_iss, filename_mc, treename, chunk_size, rank, nranks, kwargs)

def main():
    import argparse    
    parser = argparse.ArgumentParser()    
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)") 
    parser.add_argument("--filename_iss", default= "/home/manbing/Documents/Data/data_iss/BeISS_NucleiSelection_BetaCor.root",  help="(e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--filename_mc", nargs='+', help="Path to root file to read tree from")
    parser.add_argument("--dataname", default="heiss", help="give a name to describe the dataset")
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file." )
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    #parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--nprocesses", type=int, default=4, help="Number of processes to use in parallel.")   
    parser.add_argument("--resultdir", default="/home/manbing/Documents/Data/data_unfold/dfile", help="Directory to store plots and result files in.")
    parser.add_argument("--nuclei", default="Be", help="the nuclei to be analyzed")
    parser.add_argument("--qsel",  default=4.0, type=float, help="the charge of the particle to be selected")                
    args = parser.parse_args()

    nuclei = args.nuclei
    #args.filename_mc = [f"/home/manbing/Documents/Data/data_mc/{iso}MC_NucleiSelection_clean_0.root" for iso in ISOTOPES[args.nuclei]]
    args.filename_mc = [f"/home/manbing/Documents/Data/data_mc/dfile/{iso}_B1220_rwth.root" for iso in isotopes]
    
    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.filename_iss, args.filename_mc, args.treename, args.chunk_size, args.nprocesses, resultdir=args.resultdir, nuclei=args.nuclei)
        for _ in pool.imap_unordered(handle_file, pool_args):
            pass
    

    hist_beta_residual_vsbeta = {dec: {} for dec in detectors}
    hist_inversebeta_residual_vsbeta = {dec: {} for dec in detectors}


    for dec in detectors:
        for iso in isotopes:
            hist_beta_residual_vsbeta[dec][iso] = WeightedHistogram(xbinning['Beta'][dec], binning_beta_residual[dec], labels=[r"$\mathrm{\beta}$", r"$\mathrm{\beta_{m}-\beta{t}}$"])
            hist_inversebeta_residual_vsbeta[dec][iso] = WeightedHistogram(xbinning['Beta'][dec], binning_inversebeta_residual[dec], labels=[r"$\mathrm{\beta}$", r"$\mathrm{1/\beta_{m}-1/\beta{t}}$"])
            
    for rank in range(args.nprocesses):
        filename_response =  os.path.join(args.resultdir, f"betaresolution_{rank}.npz")  
        with np.load(filename_response) as reso_file:
            for dec in detectors:
                for iso in isotopes:
                    hist_beta_residual_vsbeta[dec][iso] += WeightedHistogram.from_file(reso_file, f"hist2d_beta_residual_{dec}_{iso}")
                    hist_inversebeta_residual_vsbeta[dec][iso] += WeightedHistogram.from_file(reso_file, f"hist2d_inversebeta_residual_{dec}_{iso}")
                    
    result_resolution = dict()
    for dec in detectors:
        for iso in isotopes: 
            hist_beta_residual_vsbeta[dec][iso].add_to_file(result_resolution, f"hist2d_beta_residual_{dec}_{iso}")
            hist_inversebeta_residual_vsbeta[dec][iso].add_to_file(result_resolution, f"hist2d_inversebeta_residual_{dec}_{iso}")

    np.savez(os.path.join(args.resultdir, f"BeMC_beta_residual_rwthB1220raw_v1.npz"), **result_resolution)

    
if __name__ == "__main__":
    main()






    

