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
        hist_response = WeightedHistogram.from_file(df_unfold, f'hist_ekin_response_{dec}_Be7')
        hist_datacounts = WeightedHistogram.from_file(df_unfold, f'histRec_{dec}_Be7')
        hist_mcTrueRig = WeightedHistogram.from_file(df_unfold, f'histTrue_{dec}_Be7')
        hist_mcRecRig = WeightedHistogram.from_file(df_unfold, f'histRec_{dec}_Be7')    

        histcolor = {'rec': "tab:orange", 'true': "tab:blue", "unfold": "black"}
        showfig = True    
        #fig, ax = plt.subplots()
        #plot_histogram_1d(ax, hist_mcRecRig, color="tab:orange", label="measured", scale=None, gamma=None, xlog=True, ylog=True)
        #plot_histogram_1d(ax, hist_mcTrueRig, color="tab:blue", label="true", scale=None, gamma=None, xlog=True, ylog=True)
        #ax.legend()
        
        fig, plot = plt.subplots()
        plot2dhist(fig, plot, xbinning=hist_response.binnings[0].edges[1:-1],
                   ybinning=hist_response.binnings[1].edges[1:-1],
                   counts=hist_response.values[1:-1:, 1:-1], 
                   xlabel=None, ylabel=None, zlabel="counts", zmin=None, zmax=None, 
                   setlogx=False, setlogy=False, setscilabelx=False, setscilabely=False,  setlogz=False)  

        #plot_histogram_2d(plot, hist_response, scale=None, transpose=False, show_overflow=True, show_overflow_x=None, show_overflow_y=None, label=None, xlog=False, ylog=False, log=False)

        response_mat = hist_response.values[1:-1, :]    
        column_sums = hist_response.values.sum(axis=0)
        normalization_factor = 1.0 / column_sums
        normalized_response = response_mat * normalization_factor[None, :]
        response_err = np.sqrt(hist_response.squared_values[1:-1, :])
        response_err[response_err==0] = 1
        response_err *= normalization_factor[None, :]
        response_err[:, -1] = 1
        print("response_err:", response_err)
        effi, effi_err = calculate_efficiency_and_error(hist_response.values[1:-1, :].sum(axis=0), hist_response.values[:, :].sum(axis=0), "ISS")
        effi_err[0] = 1
        effi_err[-1] = 1
        
        
        fig, plot = plt.subplots()
        #plot2dhist(fig, plot, xbinning=xbinning[dec].edges[1:-1], ybinning=xbinning[dec].edges[1:-1], counts=normalized_response[:, 1:-1], xlabel="Ekin/n(GeV/n)", ylabel="Gen-Ekin(GV)", zlabel="counts", setlogx=True, setlogy=True, setscilabelx=False, setscilabely=False,  setlogz=True)
        plot2dhist(fig, plot, xbinning=hist_response.binnings[0].edges[1:-1],
                       ybinning=hist_response.binnings[1].edges[1:-1],
                       counts=normalized_response[:, 1:-1], 
                       xlabel=None, ylabel=None, zlabel="counts", zmin=None, zmax=None, 
                       setlogx=False, setlogy=False, setscilabelx=False, setscilabely=False,  setlogz=False)  
        plot.text(0.03, 0.98, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", weight='bold')

        #########unfloding mc#####################
        unfolded_results_mc = iterative_unfold(data=hist_mcRecRig.values[1:-1], data_err=hist_mcRecRig.get_errors()[1:-1], response=normalized_response, response_err = response_err, efficiencies = effi, efficiencies_err = effi_err, callbacks = [Logger()])
        unflod_flux = unfolded_results_mc['unfolded']
        unflod_flux_err = unfolded_results_mc['stat_err']
        
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))   
        plot_histogram_1d(ax1, hist_mcRecRig, style="mc", color=histcolor['rec'], label='mc Rec', scale=None, gamma=None, xlog=True, ylog=False, shade_errors=False, show_overflow=True)
        plot_histogram_1d(ax1, hist_mcTrueRig, style="mc", color=histcolor['true'], label='mc True', scale=None, gamma=None, xlog=True, ylog=False, shade_errors=False, show_overflow=True)
        hist_unfold_mc = WeightedHistogram(xbinning[dec], values=unflod_flux, squared_values=unflod_flux_err**2)
        plot_histogram_1d(ax1, hist_unfold_mc, style="mc", color=histcolor['unfold'], label='mc unfold', scale=None, gamma=None, xlog=True, ylog=False, shade_errors=False, show_overflow=True, adjust_limits=None, adjust_limits_x=None, adjust_limits_y=None, flip_axes=False, override_limits=False, use_approximate_poisson_errors=False, draw_zeros=True, adjust_figure=True,  setscilabelx=False, setscilabely=False)
        ratio = unflod_flux/hist_mcTrueRig.values
        ratio_errs = calc_ratio_err(hist_mcTrueRig.values, unflod_flux , hist_mcTrueRig.get_errors(), unflod_flux_err)
        hist_unfold_mc_ratio = WeightedHistogram(xbinning[dec], values=ratio, squared_values=ratio_errs**2)
        plot_histogram_1d(ax2, hist_unfold_mc_ratio, style="mc", color=histcolor['unfold'], label='ratio', scale=None, gamma=None, xlog=True, ylog=False, shade_errors=False, show_overflow=True)
        ax1.text(0.03, 0.98, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')
        ax2.set_ylim([0.7, 1.3])
        ax1.set_yscale("log")
        ax1.legend()
        savefig_tofile(figure, args.resultdir, f"{nuclei}_unfoldvsTrue_{dec}", showfig)
        
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))
        #plot1dhist(figure, ax1, xbinning=ekin_binning[dec], counts=hist_mcRecRig.values[1:-1], err=hist_mcRecRig.get_errors()[1:-1], label_x="Rigidity (GV)", label_y="counts",  legend="Rec", col="tab:blue", setlogx=True, setlogy=False, setscilabelx=False, setscilabely=True)
        #plot1dhist(figure, ax1, xbinning=ekin_binning[dec], counts=unflod_flux, err=unflod_flux_err, label_x="Ekin/n (GeV/n)", label_y="counts",  legend="unfold", col="black", setlogx=True, setlogy=False, setscilabelx=False, setscilabely=True)
        #ratio = unflod_flux/hist_mcRecRig.values[1:-1]
        #graph_ratio = MGraph(get_bin_center(ekin_binning[dec]), ratio, np.zeros_like(ratio))
        #graph_ratio.add_to_file(unfold_factor, f'graph_unfold_{dec}_mc')
        #ax1.text(0.03, 0.98, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')
        #ratio_errs = calc_ratio_err(hist_mcRecRig.values[1:-1], unflod_flux , hist_mcRecRig.get_errors()[1:-1], unflod_flux_err)
        #plot1dhist(figure, ax2, xbinning=ekin_binning[dec], counts=ratio, err=ratio_errs, label_x="Ekin/n (GeV/n)", label_y="unfold/rec",  legend=None, col="black", setlogx=True, setlogy=False, setscilabelx=False, setscilabely=True)
        ax2.set_ylim([0.7, 1.3])
        ax1.set_yscale("log")
        ax1.legend()
        savefig_tofile(figure, args.resultdir, f"{nuclei}_unfoldVsRec_MC{dec}", showfig)

        unfolded_results_iss = iterative_unfold(data=hist_issCounts.values[1:-1], data_err=hist_issCounts.get_errors()[1:-1], response=normalized_response, response_err = response_err, efficiencies = effi, efficiencies_err = effi_err, callbacks = [Logger()])
        unflod_counts_iss = unfolded_results_iss['unfolded'][1:-1]
        unflod_countserr_iss = unfolded_results_iss['stat_err'][1:-1]        
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))   
        plot1dhist(figure, ax1, xbinning=ekin_binning[dec], counts=hist_issCounts.values[1:-1], err=hist_issCounts.get_errors()[1:-1], label_x="Ekin/n (GeV/n)", label_y="counts",  legend="Rec", col="tab:blue", setlogx=True, setlogy=False, setscilabelx=False, setscilabely=True)
        plot1dhist(figure, ax1, xbinning=ekin_binning[dec], counts=unflod_counts_iss, err=unflod_countserr_iss, label_x="Ekin/n (GeV/n)", label_y="counts",  legend="unfold", col="black", setlogx=True, setlogy=False, setscilabelx=False, setscilabely=True)
        ratio = unflod_counts_iss/hist_issCounts.values[1:-1]
        graph_ratio = MGraph(get_bin_center(ekin_binning[dec]), ratio, np.zeros_like(ratio))
        graph_ratio.add_to_file(unfold_factor, f'graph_unfold_{dec}_iss')
        print(dec, "unfold/raw")
        print(graph_ratio)
        ax1.text(0.03, 0.98, f"{dec}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')
        ratio_errs = calc_ratio_err(hist_issCounts.values[1:-1], unflod_counts_iss , hist_issCounts.get_errors()[1:-1], unflod_countserr_iss)
        plot1dhist(figure, ax2, xbinning=ekin_binning[dec], counts=ratio, err=ratio_errs, label_x="Ekin/n (GeV/n)", label_y="unfold/raw",  legend=None, col="black", setlogx=True, setlogy=False, setscilabelx=False, setscilabely=True)
        ax2.set_ylim([0.7, 1.3])
        ax1.set_yscale("log")
        ax1.legend()
        savefig_tofile(figure, args.resultdir, f"{nuclei}_unfoldVsRec_ISS{dec}", showfig)

    np.savez(os.path.join(args.resultdir, f"graph_unfoldfactor.npz"), **unfold_factor)
    plt.show()

    
if __name__ == "__main__":
    main()






    

