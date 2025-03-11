import multiprocessing as mp
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import uproot
import uproot3
from tools.roottree import read_tree
from tools.selections import *
import scipy.stats
from scipy.optimize import curve_fit
from tools.studybeta import hist1d, hist2d, hist_beta, getbeta, hist_betabias, compute_moment                                                                                                              
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, FIGSIZE_MID, FIGSIZE_BIG, setplot_defaultstyle, format_order_of_magnitude, FONTSIZE, savefig_tofile, FONTSIZE_BIG, flatten, plot1d_errorbar_v2
from tools.studybeta import calc_signal_fraction, hist1d, hist1d_weighted
from tools.binnings_collection import fbinning_energy_agl, fbinning_energy, BeRigidityBinningRICHRange, LithiumRigidityBinning
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, cdf_double_gaus, double_gaus
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d
from tools.binnings import Binning
from tools.constants import NUCLEI_CHARGE, ISOTOPES_COLOR, ISOTOPES
from tools.calculator import calc_ekin_from_beta, calc_ratio_err
from tools.calculator import calculate_efficiency_and_error
from tools.graphs import MGraph, slice_graph, concatenate_graphs
from scipy.interpolate import UnivariateSpline                         
import pickle
from tools.statistics import poly_func 

def RigidityBinningRICHRange():
     RigidityBinningRICHRange = np.array([1.92, 2.15, 2.4, 2.67, 2.97, 3.29, 3.64, 4.02, 4.43, 4.88, 5.37, 5.9,
                                         6.47, 7.09, 7.76, 8.48, 9.26, 10.1, 11, 12, 13, 14.1, 15.3, 16.6, 18, 19.5, 21.1, 22.8,
                                          24.7, 26.7, 28.8, 31.1, 33.5, 36.1, 38.9, 41.9, 45.1, 48.5, 52.2,  60.3,  69.7,  
                                          80.5,  93,  108])
     return RigidityBinningRICHRange
xbinning = Binning(RigidityBinningRICHRange())

setplot_defaultstyle()

def getpars_curvefit_poly(datagraph, deg):
    initial_guess = np.zeros(deg) # Initial guess for the polynomial coefficients
    fit_coeffs, _ = curve_fit(poly_func, np.log(datagraph.getx()[:]), datagraph.gety()[:], sigma=datagraph.get_yerrors()[:], p0=initial_guess)
    return fit_coeffs


def plot_comparison_nphist(figure=None, ax1=None, ax2=None, x_binning=None, com=None, com_err=None, ref=None, ref_err=None, xlabel=None, ylabel=None, legendA=None, legendB=None, colorA="black", colorB="tab:green", colorpull="black"):
    if figure == None: 
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 13))

    plot1d_errorbar(figure, ax1, x_binning, ref, err=ref_err, label_x=xlabel, label_y=ylabel, col=colorB, legend=legendB)        


    #plot1d_step(figure, ax1,  x_binning, ref, err=ref_err, label_x=xlabel, label_y=ylabel, col=colorB, legend=legendB)
    pull = np.array(com/ref)
    
    #pull_err = ratioerr(pull, com, ref, com_err, ref_err)
    pull_err = np.zeros(len(pull))   
    plot1d_errorbar(figure, ax2, x_binning, counts=pull, err=pull_err,  label_x=xlabel, label_y="ISS/MC", legend=None,  col=colorpull, setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
    plt.subplots_adjust(hspace=.0)                             
    ax1.legend()                                         
    ax2.sharex(ax1)


def fill_counts(events, xbinning, isdata=True, ialgo=1):
    #richbeta = get_richbeta(events, is_data=isdata)  # with LIP analysis then change to this
     
    rigidity = (events.tk_rigidity1)[:, ialgo, 2, 1]  
    weight = ak.to_numpy(events['ww'])
    if isdata:
        hist = Histogram(xbinning, labels=["Rigidity (GV)"])
        hist.fill(rigidity)
    else:
        hist = WeightedHistogram(xbinning, labels=["Rigidity (GV)"])
        hist.fill(rigidity, weights=np.ones_like(weight))
    return hist

detectors = {"Tof"}

#selector_denominator_event = {"LIP": {"Tof": selector_istof, "NaF": selector_isnaf_lip, "Agl": selector_isagl_lip},
selector_denominator_event = {"CIEMAT": {"Tof":selector_istof, "NaF": selector_isnaf_ciemat, "Agl": selector_isagl_ciemat}}

#selector_numerator_event = {"LIP": {"Tof": selector_tofevents, "NaF": selector_nafevents_lipvar, "Agl": selector_aglevents_lipvar},
selector_numerator_event = {"CIEMAT": {"Tof":selector_tofevents, "NaF": selector_nafevents_ciematvar, "Agl": selector_aglevents_ciematvar}}

nuclei = 'Li'

def main():
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument("--filename_iss", default="/home/manbing/Documents/Data/data_iss/BeISS_NucleiSelection.root", help="Path to root file to read tree from")
    #parser.add_argument("--filenames_mc", default=["/home/manbing/Documents/Data/data_mc/Be7MC_BetaCor.root", 
    #                                               "/home/manbing/Documents/Data/data_mc/Be9MC_BetaCor.root",
    #                                               "/home/manbing/Documents/Data/data_mc/Be10MC_BetaCor.root"], help="Path to root file to read tree from")
    parser.add_argument("--filename_iss", default=f"/home/manbing/Documents/Data/data_{nuclei}P8/rootfile/{nuclei}ISS_P8_CIEBeta.root", help="Path to root file to read tree from")
    parser.add_argument("--filenames_mc", default=[f"/home/manbing/Documents/Data/data_{nuclei}P8/rootfile/{iso}MC_B1236P8_CIEBetaCor.root" for iso in ISOTOPES[nuclei]], help="Path to root file to read tree from")
    
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--resultdir", default="plots/tof", help="Directory to store plots and result files in.")
    parser.add_argument("--dataname", default=f"{nuclei}Tof", help="dataname for the output file and plots.")
    parser.add_argument("--betatype", default="CIEMAT", help="dataname for the output file and plots.")
    parser.add_argument("--ialgo", default=1, help="choutko:0, GBL:1.")
    
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    betatype = args.betatype
    charge = NUCLEI_CHARGE[nuclei]
    
    #I am going to calculate the RICH efficiency of data and mc, and compare the efficiency to get the efficiency correction for RICH cuts
    #I have different isotopes for mc, be: be7, be9, be10

    #ISS: get beta, calculate EkinN, fill in Histogram
    hist_total = dict()
    hist_pass = dict()
    eff_iss = dict()
    efferr_iss = dict()
    graph_eff_iss = dict()
    for detector in detectors:
        hist_total[detector] = Histogram(xbinning, labels=["Rigidity (GV)"])
        hist_pass[detector] = Histogram(xbinning, labels=["Rigidity (GV)"])
        for events in read_tree(args.filename_iss, args.treename, chunk_size=args.chunk_size):
      
            events_tot = selector_denominator_event[betatype][detector](events)
            
            hist_total[detector] += fill_counts(events_tot, xbinning,  isdata=True, ialgo=args.ialgo)  
            eventspass = selector_numerator_event[betatype][detector](events, nuclei, "ISS")
            hist_pass[detector] += fill_counts(eventspass, xbinning, isdata=True, ialgo=args.ialgo)
            #print(hist_pass[detector].values)
            #print(hist_pass[detector].values)

        figure = plt.figure(figsize=FIGSIZE_BIG)  
        plot = figure.subplots(1, 1)    
        plot_histogram_1d(plot, hist_total[detector], style="iss", color="tab:orange", label='iss total', scale=None, gamma=None,
                          xlog=True, ylog=False, shade_errors=False, show_overflow=False, adjust_limits=None, adjust_limits_x=None,
                          adjust_limits_y=None, flip_axes=False, override_limits=False, use_approximate_poisson_errors=False, draw_zeros=True)
        plot_histogram_1d(plot, hist_pass[detector], style="iss", color="tab:blue", label='iss pass')
        
        plot.legend()
        figure.savefig("plots/tof/hist_issevents_passtot{}.pdf".format(detector), dpi=250)    
        eff_iss[detector], efferr_iss[detector] = calculate_efficiency_and_error(hist_pass[detector].values, hist_total[detector].values, "ISS")
        
        graph_eff_iss[detector] = MGraph(xbinning.bin_centers, eff_iss[detector], efferr_iss[detector])
        #print(len(eff_beiss))
        #print(len(hist_pass[detector].binnings[0].edges))
        

    fit_range = {"Tof":[2, 1000], "NaF": [4, 110], "Agl": [10, 110]}                                                                                                                                       
    xticks = {"Tof": [5, 10, 30, 60, 100, 300, 800], "NaF": [5, 10, 30, 60, 100], "Agl": [10, 20, 30, 40, 60, 100]}
    dict_graph_tof_effcor = {}
    for dec in detectors:
        graph_eff_iss[dec].add_to_file(dict_graph_tof_effcor, f"graph_{nuclei}_tof_effiss")

    np.savez(os.path.join(args.resultdir, f"graph_{nuclei}_tof_effcor.npz"), **dict_graph_tof_effcor)
    

if __name__ == "__main__":
    main()

