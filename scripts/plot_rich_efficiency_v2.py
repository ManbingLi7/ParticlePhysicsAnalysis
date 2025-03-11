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
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, FIGSIZE_MID, FIGSIZE_BIG, setplot_defaultstyle, format_order_of_magnitude, FONTSIZE, savefig_tofile, plot1d_errorbar_v2, FONTSIZE_BIG,set_plot_defaultstyle
from tools.studybeta import calc_signal_fraction, hist1d, hist1d_weighted
from tools.binnings_collection import fbinning_energy_agl, fbinning_energy, BeRigidityBinningFullRange, BeRigidityBinningRICHRange
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, cdf_double_gaus, double_gaus
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d
from tools.binnings import Binning
from tools.constants import NUCLEI_CHARGE, ISOTOPES_COLOR, ISOTOPES
from tools.calculator import calc_ekin_from_beta
from tools.calculator import calculate_efficiency_and_error
from tools.graphs import MGraph, slice_graph, concatenate_graphs
from scipy.interpolate import UnivariateSpline                         
import pickle
from tools.utilities import get_spline_from_graph, save_spline_to_file, get_graph_from_spline, get_spline_from_file
import pickle
from scipy.interpolate import make_interp_spline, BSpline
from scipy.interpolate import UnivariateSpline
from tools.statistics import poly_func 
from tools.graphs import MGraph, slice_graph, concatenate_graphs

xbinning = Binning(BeRigidityBinningRICHRange())
setplot_defaultstyle()


def plot_comparison_nphist(figure=None, ax1=None, ax2=None, x_binning=None, com=None, com_err=None, ref=None, ref_err=None, xlabel=None, ylabel=None, legendA=None, legendB=None, colorA="black", colorB="tab:green", colorpull="black"):
    if figure == None: 
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))

    plot1d_errorbar(figure, ax1, x_binning, ref, err=ref_err, label_x=xlabel, label_y=ylabel, col=colorB, legend=legendB)        
    plot1d_errorbar(figure, ax1, x_binning, com, err=com_err, label_x=xlabel, label_y=ylabel, col=colorA, legend=legendA)

    #plot1d_step(figure, ax1,  x_binning, ref, err=ref_err, label_x=xlabel, label_y=ylabel, col=colorB, legend=legendB)
    pull = np.array(com/ref)
    
    #pull_err = ratioerr(pull, com, ref, com_err, ref_err)
    pull_err = np.zeros(len(pull))   
    plot1d_errorbar(figure, ax2, x_binning, counts=pull, err=pull_err,  label_x=xlabel, label_y=r"$\mathrm{this/ref}$", legend=None,  col=colorpull, setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
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

detectors = {"Agl"}

#selector_denominator_event = {"LIP": {"Tof": selector_istof, "NaF": selector_isnaf_lip, "Agl": selector_isagl_lip},
selector_denominator_event = {"CIEMAT": {"Tof":selector_istof, "NaF": selector_isnaf_ciemat, "Agl": selector_isagl_ciemat}}

#selector_numerator_event = {"LIP": {"Tof": selector_tofevents, "NaF": selector_nafevents_lipvar, "Agl": selector_aglevents_lipvar},
selector_numerator_event = {"CIEMAT": {"Tof":selector_tofevents, "NaF": selector_nafevents_ciematvar, "Agl": selector_aglevents_ciematvar}}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--resultdir", default="plots/effcor/rich", help="Directory to store plots and result files in.")
    parser.add_argument("--dataname", default="AglISS", help="dataname for the output file and plots.")
    parser.add_argument("--nuclei", default="Be", help="dataname for the output file and plots.")
    parser.add_argument("--betatype", default="CIEMAT", help="dataname for the output file and plots.")
    
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    nuclei = args.nuclei
    betatype = args.betatype
    charge = NUCLEI_CHARGE[nuclei]
    isoratio = [0.6, 0.3, 0.1]

    eff_cor_jw = dict()
    with open('/home/manbing/Documents/Data/jiahui/efficiency/eff_cor_be_10yr_updated.pkl', 'rb') as f:
        data_jiahui_eff = pickle.load(f)
        eff_cor_jw["Tof"] = data_jiahui_eff[4][0]
        eff_cor_jw["NaF"] = data_jiahui_eff[4][1]
        eff_cor_jw["Agl"] = data_jiahui_eff[4][2]

    graph_average_eff_cor = {dec: dict() for dec in detectors}
    graph_average_eff_cor_nobetacut = {dec: dict() for dec in detectors}
    with np.load("/home/manbing/Documents/lithiumanalysis/scripts/plots/effcor/rich/graph_rich_effcor_v0.npz") as datafile:
        for dec in detectors:
            graph_average_eff_cor[dec] = MGraph.from_file(datafile, f"{dec}")
            
    with np.load("/home/manbing/Documents/lithiumanalysis/scripts/plots/effcor/rich_nobeta/graph_rich_effcor.npz") as datafile:
        for dec in detectors:
            graph_average_eff_cor_nobetacut[dec] = MGraph.from_file(datafile, f"{dec}")

    dict_fit_pars = dict()
    dict_fit_pars_nobeta = dict()
    fit_range = {"NaF": [4, 110], "Agl": [9, 110]}
    xticks = {"NaF": [5, 10, 30, 60, 100], "Agl": [10, 20, 30, 40, 60, 100]}
    diff = {"NaF": "2%", "Agl": "0.5%"}
    ylim_compare = {"NaF": [0.97, 1.03], "Agl": [0.99, 1.01]}
    for dec in detectors:
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 13))
        
        fit_min = graph_average_eff_cor[dec].get_index(fit_range[dec][0])
        fit_max = graph_average_eff_cor[dec].get_index(fit_range[dec][1])
        slice_graph_average_eff_cor = slice_graph(graph_average_eff_cor[dec], fit_min, fit_max)
        slice_graph_average_eff_cor_nobeta = slice_graph(graph_average_eff_cor_nobetacut[dec], fit_min, fit_max)
        dict_fit_pars[dec], _ = curve_fit(poly_func, np.log(slice_graph_average_eff_cor.xvalues), slice_graph_average_eff_cor.yvalues, sigma=1/slice_graph_average_eff_cor.yerrs, p0=np.zeros(3))
        fity = poly_func(np.log(slice_graph_average_eff_cor.xvalues), *dict_fit_pars[dec])
  
        ax1.plot(slice_graph_average_eff_cor.xvalues, fity, '-', label="fit", color="tab:orange")

        
        plot1d_errorbar_v2(figure, ax1, slice_graph_average_eff_cor.xvalues, slice_graph_average_eff_cor.yvalues, err=slice_graph_average_eff_cor.yerrs,
                           label_x="Rigidity(GV)", label_y="ISS/MC",  label="Be", color="tab:orange", setlogx=True)

        #plot1d_errorbar_v2(figure, ax1, slice_graph_average_eff_cor_nobeta.xvalues, slice_graph_average_eff_cor_nobeta.yvalues, err=slice_graph_average_eff_cor_nobeta.yerrs,
        #                   label_x="Rigidity(GV)", label_y="ISS/MC",  label="No Tof-RICH Consistency cut", color="tab:blue", setlogx=True)

        ax1.plot(slice_graph_average_eff_cor.xvalues, eff_cor_jw[dec](np.log(slice_graph_average_eff_cor.xvalues)), "-", color="black", label="J.W")

        #plot1d_errorbar_v2(figure, ax2, slice_graph_average_eff_cor.xvalues, fity/eff_cor_jw[dec](np.log(slice_graph_average_eff_cor.xvalues)), err=np.zeros(len(slice_graph_average_eff_cor.xvalues)),
        #                   label_x="Rigidity(GV)", label_y="this/J.W", setlogx=True, color="black")
        ax2.plot(slice_graph_average_eff_cor.xvalues, fity/eff_cor_jw[dec](np.log(slice_graph_average_eff_cor.xvalues)), "-", color="black")   
        set_plot_defaultstyle(ax2)
        ax2.set_xscale("log")
        ax1.set_ylim([0.8, 1.0])
        #ax1.set_xlim([1.9, 350])
        ax2.set_ylabel("this/J.W")
        ax1.text(0.1, 0.95, f"{dec}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
        ax2.text(0.1, 0.95, f"Difference within {diff[dec]}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes, color="black", fontweight="bold")
        ax1.legend(loc='lower right')
        ax2.set_ylim(ylim_compare[dec])
        ax2.set_xlim(fit_range[dec])
        ax2.set_xticks(xticks[dec])
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.get_yticklabels()[0].set_visible(False)
        ax1.sharex(ax2)
        ax1.legend(loc="lower left")
        plt.subplots_adjust(hspace=.0)
        ax2.grid()
        savefig_tofile(figure, args.resultdir, f"compafity_nobeta,re_efficiency_rich_{dec}", 1)

    for dec in detectors:
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 13))

        fit_min = graph_average_eff_cor[dec].get_index(fit_range[dec][0])
        fit_max = graph_average_eff_cor[dec].get_index(fit_range[dec][1])
        slice_graph_average_eff_cor = slice_graph(graph_average_eff_cor[dec], fit_min, fit_max)
        slice_graph_average_eff_cor_nobeta = slice_graph(graph_average_eff_cor_nobetacut[dec], fit_min, fit_max)
        dict_fit_pars[dec], _ = curve_fit(poly_func, np.log(slice_graph_average_eff_cor.xvalues), slice_graph_average_eff_cor.yvalues, sigma=1/slice_graph_average_eff_cor.yerrs, p0=np.zeros(3))
        fity = poly_func(np.log(slice_graph_average_eff_cor.xvalues), *dict_fit_pars[dec])
        fity_nobeta = poly_func(np.log(slice_graph_average_eff_cor_nobeta.xvalues), *dict_fit_pars_nobeta[dec])
        ax1.plot(slice_graph_average_eff_cor.xvalues, fity, '-', label="fit", color="tab:orange")
        ax1.plot(slice_graph_average_eff_cor_nobeta.xvalues, fity_nobeta, '-', label="fit No beta cut", color="tab:blue")

        
        plot1d_errorbar_v2(figure, ax1, slice_graph_average_eff_cor.xvalues, slice_graph_average_eff_cor.yvalues, err=slice_graph_average_eff_cor.yerrs,
                           label_x="Rigidity(GV)", label_y="ISS/MC",  label="BeMC Mix: All quality cuts", color="tab:orange", setlogx=True)

        plot1d_errorbar_v2(figure, ax1, slice_graph_average_eff_cor_nobeta.xvalues, slice_graph_average_eff_cor_nobeta.yvalues, err=slice_graph_average_eff_cor_nobeta.yerrs,
                           label_x="Rigidity(GV)", label_y="ISS/MC",  label="BeMC Mix: No Tof-RICH Consistency cut", color="tab:blue", setlogx=True)

        #ax1.plot(slice_graph_average_eff_cor.xvalues, eff_cor_jw[dec](np.log(slice_graph_average_eff_cor.xvalues)), "-", color="black", label="J.W")

        #plot1d_errorbar_v2(figure, ax2, slice_graph_average_eff_cor.xvalues, fity/eff_cor_jw[dec](np.log(slice_graph_average_eff_cor.xvalues)), err=np.zeros(len(slice_graph_average_eff_cor.xvalues)),
        #                   label_x="Rigidity(GV)", label_y="this/J.W", setlogx=True, color="black")
        ax2.plot(slice_graph_average_eff_cor.xvalues, fity_nobeta/fity, "-", color="black")   
        set_plot_defaultstyle(ax2)
        ax2.set_xscale("log")
        ax1.set_ylim([0.7, 1.0])
        #ax1.set_xlim([1.9, 350])
        ax2.set_ylabel("this/J.W")
        ax1.text(0.1, 0.95, f"{dec}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
        ax2.text(0.1, 0.95, "Difference 1% to 4%", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes, color="black", fontweight="bold")
        ax1.legend(loc='lower right')
        ax2.set_ylim([1.0, 1.04])
        ax2.set_xlim(fit_range[dec])
        ax2.set_xticks(xticks[dec])
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.get_yticklabels()[0].set_visible(False)
        ax1.sharex(ax2)
        ax1.legend(loc="lower left")
        plt.subplots_adjust(hspace=.0)
        ax2.grid()
        savefig_tofile(figure, args.resultdir, f"compare_efficiency_rich_{dec}_betacut", 1)    


    plt.show()
    


if __name__ == "__main__":
    main()

