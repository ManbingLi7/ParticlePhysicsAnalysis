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
from tools.binnings_collection import fbinning_energy_agl, fbinning_energy, BeRigidityBinningRICHRange
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


def fill_counts(events, xbinning, isdata=True):
    #richbeta = get_richbeta(events, is_data=isdata)  # with LIP analysis then change to this 
    rigidity = (events.tk_rigidity1)[:, 0, 2, 1]  
    weight = ak.to_numpy(events['ww'])
    if isdata:
        hist = Histogram(xbinning, labels=["Rigidity (GV)"])
        hist.fill(rigidity)
    else:
        hist = WeightedHistogram(xbinning, labels=["Rigidity (GV)"])
        hist.fill(rigidity, weights=weight)
    return hist

detectors = {"Tof", "NaF", "Agl"}

#selector_denominator_event = {"LIP": {"Tof": selector_istof, "NaF": selector_isnaf_lip, "Agl": selector_isagl_lip},
selector_denominator_event = {"CIEMAT": {"Tof":selector_istof, "NaF": selector_isnaf_ciemat, "Agl": selector_isagl_ciemat}}

#selector_numerator_event = {"LIP": {"Tof": selector_tofevents, "NaF": selector_nafevents_lipvar, "Agl": selector_aglevents_lipvar},
selector_numerator_event = {"CIEMAT": {"Tof":selector_tofevents, "NaF": selector_nafevents_ciematvar, "Agl": selector_aglevents_ciematvar}}


nuclei = 'Li'

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename_iss", default=f"/home/manbing/Documents/Data/data_{nuclei}P8/rootfile/{nuclei}ISS_P8_CIEBeta.root", help="Path to root file to read tree from")
    #parser.add_argument("--filenames_mc", default=[f"/home/manbing/Documents/Data/data_{nuclei}P8/rootfile/{iso}MC_B1236P8_CIEBetaCor.root" for iso in ISOTOPES[nuclei]], help="Path to root file to read tree from")
    parser.add_argument("--filenames_mc", default=[f"/home/manbing/Documents/Data/data_{nuclei}P8/rootfile/{iso}MC_B1308.root" for iso in ISOTOPES[nuclei]], help="Path to root file to read tree from")
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--resultdir", default="plots/rich", help="Directory to store plots and result files in.")
    parser.add_argument("--betatype", default="CIEMAT", help="dataname for the output file and plots.")
    
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
            
            hist_total[detector] += fill_counts(events_tot, xbinning,  isdata=True)  
            eventspass = selector_numerator_event[betatype][detector](events, nuclei, "ISS")
            hist_pass[detector] += fill_counts(eventspass, xbinning, isdata=True)
            #print(hist_pass[detector].values)
            #print(hist_pass[detector].values)

        figure = plt.figure(figsize=FIGSIZE_BIG)  
        plot = figure.subplots(1, 1)    
        plot_histogram_1d(plot, hist_total[detector], style="iss", color="tab:orange", label='iss total', scale=None, gamma=None,
                          xlog=True, ylog=False, shade_errors=False, show_overflow=False, adjust_limits=None, adjust_limits_x=None,
                          adjust_limits_y=None, flip_axes=False, override_limits=False, use_approximate_poisson_errors=False, draw_zeros=True)
        plot_histogram_1d(plot, hist_pass[detector], style="iss", color="tab:blue", label='iss pass')
        
        plot.legend()
        figure.savefig("plots/rich/hist_issevents_passtot{}.pdf".format(detector), dpi=250)    
        eff_iss[detector], efferr_iss[detector] = calculate_efficiency_and_error(hist_pass[detector].values, hist_total[detector].values, "ISS")
        
        graph_eff_iss[detector] = MGraph(xbinning.bin_centers, eff_iss[detector], efferr_iss[detector])
        #print(len(eff_beiss))
        #print(len(hist_pass[detector].binnings[0].edges))

        figure = plt.figure(figsize=(16, 13))                                                                                         
        plot = figure.subplots(1, 1)                                                                                                                                                                       
        plot1dhist(figure, plot, hist_pass[detector].binnings[0].edges, graph_eff_iss[detector].yvalues,  graph_eff_iss[detector].yerrs, "Rigidity(GV)", "Efficiency", "ISS",  "black", 30, 1, 0, 0, 1)
        plot.set_ylim([0.5, 1.05])
        plot.set_xscale("log")
        savefig_tofile(figure, args.resultdir, f"hist_be_efficiency_{detector}.pdf", 1)  

    dict_effcor ={dec: dict() for dec in detectors}
    hist_total_mc = {dec: dict() for dec in detectors}
    hist_pass_mc = {dec: dict() for dec in detectors}
    eff_mc = {dec: dict() for dec in detectors}
    efferr_mc = {dec: dict() for dec in detectors}
    graph_eff_mc = {dec: dict() for dec in detectors}
    dict_spline_effcor = {dec: dict() for dec in detectors}
    dict_fit_poly =  {dec: dict() for dec in detectors}
    graph_rich_effcor =  {dec: dict() for dec in detectors}
    eff_cor =  {dec: dict() for dec in detectors}
    eff_cor_err =  {dec: dict() for dec in detectors}
    dict_graph_rich_eff = dict()
    
    for i, isotope in enumerate(ISOTOPES[nuclei]):
    
        filename_mc = args.filenames_mc[i]
        for detector in detectors:
            hist_total_mc[detector][isotope] = Histogram(xbinning, labels=["Rigidity (GV)"])
            hist_pass_mc[detector][isotope] = Histogram(xbinning, labels=["Rigidity (GV)"])
            for events in read_tree(filename_mc, "amstreea", chunk_size=args.chunk_size):
                eventsmctot = selector_denominator_event[betatype][detector](events)
                hist_total_mc[detector][isotope] += fill_counts(eventsmctot, xbinning, isdata=False)                
                eventsmcpass = selector_numerator_event[betatype][detector](events, nuclei, "MC")
                hist_pass_mc[detector][isotope] += fill_counts(eventsmcpass, xbinning, isdata=False) 
            eff_mc[detector][isotope], efferr_mc[detector][isotope] = calculate_efficiency_and_error(hist_pass_mc[detector][isotope].values, hist_total_mc[detector][isotope].values, "MC")
            eff_mcbe_test = hist_pass_mc[detector][isotope].values/hist_total_mc[detector][isotope].values

            graph_eff_mc[detector][isotope] = MGraph(xbinning.bin_centers, eff_mc[detector][isotope], efferr_mc[detector][isotope])
            #eff_cor[detector][isotope] = graph_eff_iss[detector].yvalues/graph_eff_mc[detector][isotope].yvalues
            eff_cor[detector][isotope] = eff_iss[detector]/eff_mc[detector][isotope]
            eff_cor_err[detector][isotope] = calc_ratio_err(graph_eff_iss[detector].yvalues, graph_eff_mc[detector][isotope].yvalues, graph_eff_iss[detector].yerrs, graph_eff_mc[detector][isotope].yerrs)
            graph_rich_effcor[detector][isotope] = MGraph(xbinning.bin_centers, eff_cor[detector][isotope], eff_cor_err[detector][isotope])
            dict_effcor[detector][isotope] = eff_cor[detector][isotope] 
            print("test_0", dict_effcor[detector][isotope])

            #fit_coeffs, _ = curve_fit(poly_func, np.log(xbinning.bin_centers), eff_cor, sigma=1/eff_cor_err, p0=np.zeros(2))
            #dict_fit_poly[detector][isotope] = fit_coeffs
            dict_spline_effcor[detector][isotope] = UnivariateSpline(xbinning.bin_centers, eff_cor[detector][isotope] ,  w=1/eff_cor_err[detector][isotope], k=3, s=3)
            #print("test_1:", dict_spline_effcor[detector][isotope](xbinning.bin_centers))
            
            
    np.savez(os.path.join(args.resultdir, f"{nuclei}_rich_eff_correction.npz"), xbincenter=xbinning.bin_centers, **dict_effcor)
    with open(os.path.join(args.resultdir, f'{nuclei}_spline_rich_eff.pickle'), 'wb') as f:
        pickle.dump(dict_spline_effcor, f)

    with open(os.path.join(args.resultdir, f'{nuclei}_spline_rich_eff.pickle'), "rb") as pf:
        loaded_dict = pickle.load(pf)



    fit_range = {"Tof":[2, 110], "NaF": [4, 110], "Agl": [10, 110]}                                                                                                                                        
    xticks = {"Tof": [5, 10, 30, 60, 100], "NaF": [5, 10, 30, 60, 100], "Agl": [10, 20, 30, 40, 60, 100]}                  
    for dec in detectors:
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.5, 0.5]}, figsize=(16, 13))
        graph_eff_iss[dec].add_to_file(dict_graph_rich_eff, f'grapheff_{nuclei}ISS_rich{dec}')
        plot1d_errorbar(figure, ax1, xbinning.edges, graph_eff_iss[dec].yvalues, err=graph_eff_iss[dec].yerrs, label_x="Rigidity(GV)", label_y="Efficiency", col="black", legend="ISS", setlogx=True)
        for iso in ISOTOPES[nuclei]:
            graph_eff_mc[dec][iso].add_to_file(dict_graph_rich_eff, f'grapheff_{iso}MC_rich{dec}')
            graph_rich_effcor[dec][iso].add_to_file(dict_graph_rich_eff, f'grapheffcor_{iso}MC_rich{dec}')
            plot1d_errorbar(figure, ax1, xbinning.edges, graph_eff_mc[dec][iso].yvalues, err=graph_eff_mc[dec][iso].yerrs, label_x="Rigidity(GV)", label_y="Efficiency", col=ISOTOPES_COLOR[iso], legend=f"{iso} MC")
            plot1d_errorbar(figure, ax2, xbinning.edges, graph_rich_effcor[dec][iso].yvalues, err=graph_rich_effcor[dec][iso].yerrs, label_x="Rigidity(GV)", label_y="ISS/MC", col=ISOTOPES_COLOR[iso], setlogx=True)
            #plot_comparison_nphist(figure, ax1, ax2, x_binning=xbinning.edges, com=eff_iss[dec],
            #                       com_err=np.zeros(len(eff_iss[dec])), ref=eff_mc[dec][iso], ref_err=np.zeros(len(eff_mc[dec][iso])),
            #                       ylabel=r"Efficiency", xlabel="Rigidity(GV)", legendA=f"ISS", legendB=f"MC {iso}",  colorB=ISOTOPES_COLOR[iso], colorpull=ISOTOPES_COLOR[iso])
        ax2.set_xscale("log")
        ax2.set_ylim([0.85, 1.05])
        ax1.set_ylim([0.3, 0.9])
        #ax1.set_xlim([5, 100])
        ax2.set_xlim(fit_range[dec])             
        ax2.set_xticks(xticks[dec])  
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.get_yticklabels()[0].set_visible(False)
        ax1.sharex(ax2)
        plt.subplots_adjust(hspace=.0)
        ax1.text(0.05, 0.95, f"{dec}", fontsize=30, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold') 
        #ax2.plot(xbinning.edges, [1]*len(xbinning.edges), 'b--')
        savefig_tofile(figure, args.resultdir, f"eff_correction_{dec}", 1)  


    iso_ratio = [0.6, 0.3, 0.1]
    average_eff_cor = dict()
    average_eff_cor_err = dict()
    graph_average_eff_cor = dict()
    dict_graph_rich_effcor = dict()
    dict_fit_pars = dict()

    
    for dec in detectors:
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 13))
        average_eff_cor[dec] = np.zeros_like(graph_rich_effcor[dec]["Li7"].yvalues)
        average_eff_cor_err[dec] = np.zeros_like(graph_rich_effcor[dec]["Li7"].yvalues)
        for i, iso in enumerate(ISOTOPES[nuclei]):
            average_eff_cor[dec] += iso_ratio[i] * graph_rich_effcor[dec][iso].yvalues
            average_eff_cor_err[dec] += iso_ratio[i] * graph_rich_effcor[dec][iso].yerrs 

        graph_average_eff_cor[dec] = MGraph(xbinning.bin_centers, average_eff_cor[dec], average_eff_cor_err[dec], labels=["Rigidity (GV)", "Efficiency correction"])
        fit_min = graph_average_eff_cor[dec].get_index(fit_range[dec][0])
        fit_max = graph_average_eff_cor[dec].get_index(fit_range[dec][1])
        slice_graph_average_eff_cor = slice_graph(graph_average_eff_cor[dec], fit_min, fit_max)
        dict_fit_pars[dec], _ = curve_fit(poly_func, np.log(slice_graph_average_eff_cor.xvalues), slice_graph_average_eff_cor.yvalues, sigma=1/slice_graph_average_eff_cor.yerrs, p0=np.zeros(3))

        fity = poly_func(np.log(slice_graph_average_eff_cor.xvalues), *dict_fit_pars[dec])
        ax1.plot(slice_graph_average_eff_cor.xvalues, fity, '-', label="fit", color="tab:orange")

        
        graph_average_eff_cor[dec].add_to_file(dict_graph_rich_eff, f"graph_rich{dec}_avgeffcor")
        
        plot1d_errorbar(figure, ax1, xbinning.edges, average_eff_cor[dec], err=average_eff_cor_err[dec],
                        label_x="Rigidity(GV)", label_y="ISS/MC",  legend="Li", col="tab:orange", style=".", legendfontsize=FONTSIZE, setlogx=True)
            
        
        ax2.set_xscale("log")
        ax1.set_ylim([0.8, 1.0])
        #ax1.set_xlim([1.9, 350])
        ax2.set_ylabel("this/J.W")
        ax1.text(0.1, 0.95, f"RICH-{dec}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
        ax1.legend(loc='lower right')
        ax2.set_xlim(fit_range[dec])             
        ax2.set_xticks(xticks[dec])  
        ax2.set_ylim([0.95, 1.05])
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.get_yticklabels()[0].set_visible(False)
        ax1.sharex(ax2)
        plt.subplots_adjust(hspace=.0)
        plt.grid()
        ax2.grid()

    np.savez(os.path.join(args.resultdir, f"graph_{nuclei}_rich_eff_B1308.npz"), **dict_graph_rich_eff)
     
    plt.show()
if __name__ == "__main__":
    main()


