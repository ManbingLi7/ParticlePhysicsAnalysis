#!/usr/bin/env python3

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
from tools.binnings_collection import LithiumRichAglBetaResolutionBinning, LithiumRigidityBinningFullRange, BeRigidityBinningUnfold, kinetic_energy_neculeon_binning
from tools.binnings_collection import kinetic_energy_binning_NaF, kinetic_energy_binning_Agl, kinetic_energy_binning_Tof
from tools.binnings_collection import fbinning_energy_Tof, fbinning_energy_NaF, fbinning_energy_Agl  
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
import matplotlib.pyplot as plt
import seaborn as sns
from tools.studybeta import hist1d, hist2d
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FONTSIZE, FIGSIZE_MID, FIGSIZE_WID, plot1d_errorbar_v2, plot1d_step
from tools.calculator import calc_ratio_err, calculate_efficiency_and_error
from tools.histograms import WeightedHistogram, Histogram, plot_histogram_2d, plot_histogram_1d
from tools.binnings import Binning, make_lin_binning
from tools.graphs import MGraph

plt.rcParams['figure.figsize'] = (16, 12)
#ekin_binning = {'Tof': kinetic_energy_binning_Tof(), "NaF": kinetic_energy_binning_NaF(), "Agl": kinetic_energy_binning_Agl()}
#xbinning = {"Tof": Binning(kinetic_energy_binning_Tof()), "NaF": Binning(kinetic_energy_binning_NaF()), 'Agl': Binning(kinetic_energy_binning_Agl())}

ekin_binning = {'Tof': fbinning_energy_Tof(), "NaF": fbinning_energy_NaF(), "Agl": fbinning_energy_Agl()}
xbinning = {"Tof": Binning(fbinning_energy_Tof()), "NaF": Binning(fbinning_energy_NaF()), 'Agl': Binning(fbinning_energy_Agl())}

norminal_charge = 4.0
setplot_defaultstyle()

binning_rig_residual = make_lin_binning(-0.5, 0.7, 550)
binning_rig_resolution = make_lin_binning(-1, 1, 550)

def main():
    import argparse    
    parser = argparse.ArgumentParser()    
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)") 
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file." )
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")   
    parser.add_argument("--resultdir", default="/home/manbing/Documents/Data/data_unfold/dfile", help="Directory to store plots and result files in.")  
    parser.add_argument("--qsel",  default=2.0, type=float, help="the charge of the particle to be selected")
    parser.add_argument("--nuclei",  default="Be", help="the particle to be analyzed")
    args = parser.parse_args()
    nuclei = args.nuclei
    detectors = ["Tof"]
    #df_unfold = np.load(os.path.join(args.resultdir, 'unfold_data_mcrwth.npz'))
    unfold_factor = dict()
    for dec in detectors:
        df_unfold = np.load(os.path.join(args.resultdir, 'Be7MC_beta_resolution_rawrwth.npz'))
        hist_issCounts = Histogram.from_file(df_unfold, f'histIss_counts_{dec}')
        hist_response = WeightedHistogram.from_file(df_unfold, f'hist_beta_response_{dec}_Be7')
        hist_datacounts = WeightedHistogram.from_file(df_unfold, f'histRec_{dec}_Be7')
        hist_mcTrueRig = WeightedHistogram.from_file(df_unfold, f'histTrue_{dec}_Be7')
        hist_mcRecRig = WeightedHistogram.from_file(df_unfold, f'histRec_{dec}_Be7')    

        histcolor = {'rec': "tab:orange", 'true': "tab:blue", "unfold": "black"}
        showfig = True    
                
        fig, plot = plt.subplots()
        plot2dhist(fig, plot, xbinning=hist_response.binnings[0].edges[1:-1],
                       ybinning=hist_response.binnings[1].edges[1:-1],
                       counts=hist_response.values[1:-1, 1:-1], 
                       xlabel=None, ylabel=None, zlabel="counts", zmin=None, zmax=None, 
                       setlogx=False, setlogy=False, setscilabelx=False, setscilabely=False,  setlogz=False)
        
        #plot_histogram_2d(plot, hist_response, scale=None, transpose=False, show_overflow=True, show_overflow_x=None, show_overflow_y=None, label=None, xlog=False, ylog=False, log=False)

     
    plt.show()

    
if __name__ == "__main__":
    main()






    

