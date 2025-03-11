import multiprocessing as mp
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import uproot
import uproot3
from tools.roottree import read_tree
from tools.selectors import *
import scipy.stats
from scipy.optimize import curve_fit
from tools.studybeta import hist1d, hist2d, hist_beta, getbeta, hist_betabias, compute_moment                                                                                                              
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, FIGSIZE_MID, FIGSIZE_BIG, setplot_defaultstyle, format_order_of_magnitude, FONTSIZE
from tools.studybeta import calc_signal_fraction, hist1d, hist1d_weighted
from tools.binnings_collection import  fbinning_energy_agl
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, cdf_double_gaus, double_gaus
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d
from tools.binnings import Binning
from tools.constants import NUCLEI_CHARGE
from tools.calculator import calc_ekin_from_beta
from tools.calculator import calculate_efficiency_and_error

xbinning = fbinning_energy_agl()
setplot_defaultstyle()

def fill_hist_ekin_counts(events, ekinbinning, isdata=True):
    #richbeta = get_richbeta(events, is_data=isdata)
    richbeta = events['rich_beta_cor']
    ekin_rich = calc_ekin_from_beta(richbeta)
    hist = Histogram(ekinbinning, labels=["Ekin/n (GeV/n)", "events"]) 
    hist.fill(ekin_rich)
    return hist

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--filename_iss", default="/home/manbing/Documents/Data/data_BeP8/rootfile/BeISS_P8_CIEBeta.root", help="Path to root file to read tree from")
    parser.add_argument("--filenames_mc", default=["/home/manbing/Documents/Data/data_BeP8/rootfile/Be7MC_B1236P8_CIEBetaCor.root", 
                                                   "/home/manbing/Documents/Data/data_BeP8/rootfile/Be9MC_B1236P8_CIEBetaCor.root",
                                                   "/home/manbing/Documents/Data/data_BeP8/rootfile/Be10MC_B1236P8_CIEBetaCor.root"], help="Path to root file to read tree from")
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--resultdir", default="plots/agl", help="Directory to store plots and result files in.")
    parser.add_argument("--dataname", default="AglISS", help="dataname for the output file and plots.")
    parser.add_argument("--nuclei", default="Be", help="dataname for the output file and plots.")
    
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    nuclei = args.nuclei
    charge = NUCLEI_CHARGE[nuclei]
    isotopes = {"Li": ["Li6", "Li7"], "Be": ["Be7", "Be9", "Be10"], "Boron": ["Bo10", "Bo11"]}
    #I am going to calculate the RICH efficiency of data and mc, and compare the efficiency to get the efficiency correction for RICH cuts
    #I have different isotopes for mc, be: be7, be9, be10

    #ISS: get beta, calculate EkinN, fill in Histogram
    with uproot3.open(args.filename_iss) as isstreefile:
         isstree = isstreefile["amstreea"]                                                                                                                                               
         issbeta = isstree.array('rich_beta_cor')
         issekin = calc_ekin_from_beta(issbeta)
         ekinbinning = Binning(xbinning)
         hist_ekin_total = Histogram(ekinbinning, labels=["Ekin/n (GeV/n)", "events"])
         hist_ekin_total.fill(issekin)
         print(hist_ekin_total.binnings[0].edges)
         print(hist_ekin_total.values)

    for events in read_tree(args.filename_iss, args.treename, chunk_size=args.chunk_size):
             events_tot = selector_agl_event(events)
             hist_ekin_tot = fill_hist_ekin_counts(events_tot, ekinbinning, isdata=True)

             eventspass = selector_agl_ciematvar(events, nuclei, isotopes[nuclei][0], isdata=False) 
             hist_ekin_pass = fill_hist_ekin_counts(eventspass, ekinbinning, isdata=True)

    effagl_beiss, efferragl_beiss = calculate_efficiency_and_error(hist_ekin_pass.values, hist_ekin_tot.values, datatype='ISS')
    figure = plt.figure(figsize=FIGSIZE_BIG)                                                                                                                                                        
    plot = figure.subplots(1, 1)    
    plot_histogram_1d(plot, hist_ekin_tot, style="iss", color="tab:orange", label='iss total', scale=None, gamma=None, ylog=False, shade_errors=False, show_overflow=False, adjust_limits=None, adjust_limits_x=None, adjust_limits_y=None, flip_axes=False, override_limits=False, use_approximate_poisson_errors=False, draw_zeros=True)
    plot_histogram_1d(plot, hist_ekin_pass, style="iss", color="tab:blue", label='iss pass')
    plot.legend()

    dict_eff_cor = dict()
    for i, isotope in enumerate(isotopes[nuclei]):
        filename_mc = args.filenames_mc[i]
        for events in read_tree(filename_mc, "amstreea", chunk_size=args.chunk_size):
             eventsmctot = selector_agl_event(events)
             hist_ekin_mctot = fill_hist_ekin_counts(eventsmctot, ekinbinning, isdata=False)
             
             eventsmcpass = selector_agl_ciematvar(events, nuclei, isotope, isdata=False) 
             hist_ekin_mcpass = fill_hist_ekin_counts(eventsmcpass, ekinbinning, isdata=False)  


        effagl_mcbe, efferragl_mcbe = calculate_efficiency_and_error(hist_ekin_mcpass.values, hist_ekin_mctot.values, datatype='MC')
        figure = plt.figure(figsize=FIGSIZE_BIG)                                                                                                                                                        
        plot = figure.subplots(1, 1)    
        plot_histogram_1d(plot, hist_ekin_mctot, style="mc", color="tab:orange", label=f'{isotope}mc total', scale=None, gamma=None, ylog=False, shade_errors=False, show_overflow=False, adjust_limits=None, adjust_limits_x=None, adjust_limits_y=None, flip_axes=False, override_limits=False, use_approximate_poisson_errors=False, draw_zeros=True)
        plot_histogram_1d(plot, hist_ekin_mcpass, style="mc", color="tab:blue", label=f'{isotope}mc pass')
        plot.legend()
        eff_cor = effagl_beiss/effagl_mcbe
        dict_eff_cor[isotope] = eff_cor
        #hist_effcor = Histogram(ekinbinning, values=eff_cor, labels=["Ekin/n (GeV/n)", "Eff Correction"])
        np.savez(os.path.join(args.resultdir, f"{isotope}eff_correction_agl_V2.npz"), xbincenter=ekinbinning.bin_centers,  eff_cor=eff_cor)
        print(eff_cor)                               
        figure = plt.figure(figsize=FIGSIZE_BIG)                                                                                         
        plot = figure.subplots(1, 1)                                                                                                                                                                       
        plot1dhist(figure, plot, hist_ekin_pass.binnings[0].edges, effagl_beiss,  efferragl_beiss, "Ekin/n (GeV/n)", "Efficiency", "ISS",  "black", 30, 1, 0, 0, 1)
        plot1dhist(figure, plot, hist_ekin_mcpass.binnings[0].edges, effagl_mcbe,  efferragl_mcbe, "Ekin/n (GeV/n)", "Efficiency", f"{isotope}MC",  "blue", 30, 1, 0, 0, 1)                                
        figure.savefig("plots/agl/hist_{}_ekin_efficiency.pdf".format(args.dataname), dpi=250)    

    color = {"Be7": "tab:orange", "Be9": "tab:blue", "Be10": 'tab:red'}
    figure = plt.figure(figsize=FIGSIZE_BIG)                                                                                         
    plot = figure.subplots(1, 1)
    for isotope in isotopes[nuclei]:
        plot1dhist(figure, plot, hist_ekin_mcpass.binnings[0].edges, dict_eff_cor[isotope],  np.zeros(len(dict_eff_cor[isotope])), "Ekin/n (GeV/n)", "Eff Correction", f"{isotope}",  color[isotope], 30, 1, 0, 0, 1)    
    figure.savefig("plots/agl/hist_{}_ekin_efficiencycorrection.pdf".format(args.dataname), dpi=250)    

    plt.show()
if __name__ == "__main__":
    main()
