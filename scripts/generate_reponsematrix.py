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
from tools.binnings_collection import LithiumRichAglBetaResolutionBinning, LithiumRigidityBinningFullRange, BeRigidityBinningUnfold
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
#rigidity_binning = LithiumRigidityBinningFullRange()
rigidity_binning = BeRigidityBinningUnfold()
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
    parser.add_argument("--filename_mc", default="/home/manbing/Documents/Data/data_unfold/dfile/Be7_B1220_rwth_411.root", help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--dataname", default="heiss", help="give a name to describe the dataset")
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file." )
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
   # parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--nprocesses", type=int, default=4, help="Number of processes to use in parallel.")   
    parser.add_argument("--resultdir", default="/home/manbing/Documents/Data/data_unfold/dfile", help="Directory to store plots and result files in.")  
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
    hist_response.add_to_file(result_response, "hist_response")
    hist_datacounts.add_to_file(result_response, "hist_datacounts")
    hist_mcTrueRig.add_to_file(result_response, "hist_mcTrueRig")
    hist_mcRecRig.add_to_file(result_response, "hist_mcRecRig")
    np.savez(os.path.join(args.resultdir, f"unfold_data_mcrwth.npz"), **result_response)
   
    plt.show()

    
if __name__ == "__main__":
    main()






    

