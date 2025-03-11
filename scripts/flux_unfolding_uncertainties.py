import pyunfold
from pyunfold import iterative_unfold
from pyunfold.callbacks import Logger
from pyunfold.callbacks import SplineRegularizer
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
from tools.binnings_collection import LithiumRichAglBetaResolutionBinning, LithiumRigidityBinningFullRange
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
import matplotlib.pyplot as plt
import seaborn as sns
from tools.studybeta import hist1d, hist2d
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, plot1d_errorbar_v2, plot1d_step
from tools.calculator import calc_ratio_err, calculate_efficiency_and_error
from tools.histograms import WeightedHistogram, Histogram, plot_histogram_2d, plot_histogram_1d
from tools.binnings import Binning, make_lin_binning

plt.rcParams['figure.figsize'] = (16, 12)
#plt.rcParams['line.markeredgewidth'] = 2
#input of the measured counts and migration matrix
rigidity_binning = LithiumRigidityBinningFullRange()
xbinning = Binning(rigidity_binning)
num_bins = len(rigidity_binning) - 1
norminal_charge = 4.0
setplot_defaultstyle()

binning_rig_residual = make_lin_binning(-0.5, 0.7, 550)
binning_rig_resolution = make_lin_binning(-1, 1, 550)
def handle_file(arg):
    
    filename_iss, filename_mc, treename, chunk_size,  rank, nranks, kwargs = arg
    resultdir = kwargs["resultdir"]
    

    hist_response = WeightedHistogram(xbinning, xbinning, labels=["Rigidity (GV)", f"Gen-Rigidity (GV)"])
    hist_datacounts = WeightedHistogram(xbinning, labels=["Rigidity (GV)"])
    hist_mcTrueRig = WeightedHistogram(xbinning, labels=["Gen-Rigidity (GV)"])
    hist_mcRecRig = WeightedHistogram(xbinning, labels=["Rec-Rigidity (GV)"])
    
    for events in read_tree(filename_iss, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
        rigidity = ak.to_numpy((events.tk_rigidity1)[:, 0, 2, 1])
        mc_weight = ak.to_numpy(events["ww"])
        weight_one = np.ones(len(mc_weight))
        hist_datacounts.fill(rigidity, weights=weight_one)
        
    for events in read_tree(filename_mc, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
        rigidity = ak.to_numpy((events.tk_rigidity1)[:, 0, 2, 1])
        true_rigidity = ak.to_numpy(events["mevmom1"][:, 0]/norminal_charge)
        mc_weight = ak.to_numpy(events["ww"])
        weight_one = np.ones(len(mc_weight))
        weight_rig = 1/true_rigidity
        hist_response.fill(rigidity, true_rigidity, weights=weight_one)
        hist_mcTrueRig.fill(true_rigidity, weights=weight_rig)
        hist_mcRecRig.fill(rigidity, weights=weight_rig)
        
    hist_response_dict = {}
    hist_response.add_to_file(hist_response_dict, "response")
    hist_datacounts.add_to_file(hist_response_dict, "datacounts")
    hist_mcTrueRig.add_to_file(hist_response_dict, "true")
    hist_mcRecRig.add_to_file(hist_response_dict, "rec")
    np.savez(os.path.join(resultdir, f"response_{rank}.npz"), **hist_response_dict)

def make_args(filename_iss, filename_mc, treename, chunk_size, nranks, **kwargs):
    for rank in range(nranks):
        yield (filename_iss, filename_mc, treename, chunk_size, rank, nranks, kwargs)

def main():
    import argparse    
    parser = argparse.ArgumentParser()    
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)") 
    parser.add_argument("--filename_iss", default= "/home/manbing/Documents/Data/data_iss/BeISS_NucleiSelection_BetaCor.root",  help="(e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--filename_mc", default="/home/manbing/Documents/Data/data_mc/dfile/Be7MC_NucleiSelection_clean_0.root", help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--dataname", default="heiss", help="give a name to describe the dataset")
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file." )
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")   
    parser.add_argument("--resultdir", default="plots/unfold", help="Directory to store plots and result files in.")  
    parser.add_argument("--qsel",  default=2.0, type=float, help="the charge of the particle to be selected")                
    args = parser.parse_args()

    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.filename_iss, args.filename_mc, args.treename, args.chunk_size, args.nprocesses, resultdir=args.resultdir)
        for _ in pool.imap_unordered(handle_file, pool_args):
            pass

    ####################################################################################
    #Step1 read the rigidity matrix
    ####################################################################################
    hist_response = WeightedHistogram(xbinning, xbinning, labels=["Rigidity (GV)", f"Gen-Rigidity (GV)"])
    hist_datacounts = WeightedHistogram(xbinning, labels=["Rigidity (GV)"])
    hist_mcTrueRig = WeightedHistogram(xbinning, labels=["Gen-Rigidity (GV)"])
    hist_mcRecRig = WeightedHistogram(xbinning, labels=["Rec-Rigidity (GV)"])

    
    for rank in range(args.nprocesses):
        filename_response =  os.path.join(args.resultdir, f"response_{rank}.npz")  
        with np.load(filename_response) as response_file:
            hist_response += WeightedHistogram.from_file(response_file, "response")
            hist_datacounts += WeightedHistogram.from_file(response_file, "datacounts")
            hist_mcTrueRig += WeightedHistogram.from_file(response_file, "true")
            hist_mcRecRig += WeightedHistogram.from_file(response_file, "rec")
            
    result_response = dict()
    hist_response.add_to_file(result_response, "response")
    np.savez(os.path.join(args.resultdir, f"rig_response.npz"), **result_response)


    histcolor = {'rec': "tab:orange", 'true': "tab:blue"}
    showfig = True    
    fig, ax = plt.subplots()
    plot_histogram_1d(ax, hist_mcRecRig, color="tab:orange", label="measured", scale=None, gamma=None, xlog=True, ylog=True)
    plot_histogram_1d(ax, hist_mcTrueRig, color="tab:blue", label="true", scale=None, gamma=None, xlog=True, ylog=True)
    ax.legend()

    fig, plot = plt.subplots()
    plot_histogram_2d(plot, hist_response, scale=None, transpose=False, show_overflow=False, show_overflow_x=None, show_overflow_y=None, label=None, xlog=True, ylog=True, log=True)
    
    response_mat = hist_response.values[1:-1, :]    
    column_sums = hist_response.values.sum(axis=0)
    normalization_factor = 1.0 / column_sums
    normalized_response = response_mat * normalization_factor[None, :]
    response_err = np.sqrt(hist_response.squared_values[1:-1, :])
    response_err[response_err==0] = 1
    response_err *= normalization_factor[None, :]
    #response_err[:, -1] = 1
    
    print("response_err=", response_err)
    effi_test = hist_response.values[1:-1, :].sum(axis=0)/hist_response.values[:, :].sum(axis=0)
    print(effi_test)
    effi, effi_err = calculate_efficiency_and_error(hist_response.values[1:-1, :].sum(axis=0), hist_response.values[:, :].sum(axis=0), "ISS")
    
    print(effi)
    fig, plot = plt.subplots()
    plot2dhist(fig, plot, xbinning=xbinning.edges[1:-1], ybinning=xbinning.edges[1:-1], counts=normalized_response[:, 1:-1], xlabel="Rigidity(GV)", ylabel="Gen-Rig(GV)", zlabel="counts", setlogx=True, setlogy=True, setscilabelx=False, setscilabely=False,  setlogz=True)    

    # Setup Callbacks
    logger = Logger()
    regularizer = SplineRegularizer(smooth=1.25)
    
    unfolded_results = iterative_unfold(data=hist_mcRecRig.values[1:-1], data_err=hist_mcRecRig.get_errors()[1:-1], response=normalized_response, response_err = response_err, efficiencies = effi, efficiencies_err = effi_err, callbacks = [logger, regularizer])
    #print(unfolded_results)

    results = unfolded_results
    unflod_flux = unfolded_results['unfolded'][1:-1]
    unflod_flux_err = unfolded_results['stat_err'][1:-1]
    

    
    figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))   
    plot1dhist(figure, ax1, xbinning=rigidity_binning, counts=hist_mcTrueRig.values[1:-1], err=hist_mcTrueRig.get_errors()[1:-1], label_x="Rigidity (GV)", label_y="counts",  legend="true", col="tab:orange", setlogx=True, setlogy=False, setscilabelx=False, setscilabely=True)
    plot1dhist(figure, ax1, xbinning=rigidity_binning, counts=unflod_flux, err=unflod_flux_err, label_x="Rigidity (GV)", label_y="counts",  legend="unfold", col="black", setlogx=True, setlogy=False, setscilabelx=False, setscilabely=True)
    ratio = hist_mcTrueRig.values[1:-1]/unflod_flux
    ratio_errs = calc_ratio_err(hist_mcTrueRig.values[1:-1], unflod_flux , hist_mcTrueRig.get_errors()[1:-1], unflod_flux_err)
    plot1dhist(figure, ax2, xbinning=rigidity_binning, counts=ratio, err=ratio_errs, label_x="Rigidity (GV)", label_y="true/unfold",  legend=None, col="tab:blue", setlogx=True, setlogy=False, setscilabelx=False, setscilabely=True)
    ax2.set_ylim([0.7, 1.3])
    ax1.set_yscale("log")
    ax1.legend()
    savefig_tofile(figure, args.resultdir, f"{args.dataname}_unfold_true_rigidity", showfig)

    figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))   
    plot1dhist(figure, ax1, xbinning=rigidity_binning, counts=hist_mcRecRig.values[1:-1], err=hist_mcRecRig.get_errors()[1:-1], label_x="Rigidity (GV)", label_y="counts",  legend="Rec", col="tab:blue", setlogx=True, setlogy=False, setscilabelx=False, setscilabely=True)
    plot1dhist(figure, ax1, xbinning=rigidity_binning, counts=unflod_flux, err=unflod_flux_err, label_x="Rigidity (GV)", label_y="counts",  legend="unfold", col="black", setlogx=True, setlogy=False, setscilabelx=False, setscilabely=True)
    ratio = hist_mcRecRig.values[1:-1]/unflod_flux
    ratio_errs = calc_ratio_err(hist_mcRecRig.values[1:-1], unflod_flux , hist_mcRecRig.get_errors()[1:-1], unflod_flux_err)
    plot1dhist(figure, ax2, xbinning=rigidity_binning, counts=ratio, err=ratio_errs, label_x="Rigidity (GV)", label_y="rec/unfold",  legend=None, col="black", setlogx=True, setlogy=False, setscilabelx=False, setscilabely=True)
    ax2.set_ylim([0.7, 1.3])
    ax1.set_yscale("log")
    ax1.legend()
    savefig_tofile(figure, args.resultdir, f"{args.dataname}_unfold_true_rigidity", showfig)

    
    plt.show()

    
if __name__ == "__main__":
    main()






    

