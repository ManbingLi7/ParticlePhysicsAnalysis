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
from tools.binnings_collection import fbinning_energy_agl, fbinning_energy, BeRigidityBinningRICHRange, Rigidity_Analysis_Binning_FullRange
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, cdf_double_gaus, double_gaus
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d
from tools.binnings import Binning
from tools.constants import NUCLEI_CHARGE, ISOTOPES_COLOR, ISOTOPES, ANALYSIS_RANGE_RIG
from tools.calculator import calc_ekin_from_beta, calc_ratio_err
from tools.calculator import calculate_efficiency_and_error
from tools.graphs import MGraph, slice_graph, concatenate_graphs, scale_graph
from scipy.interpolate import UnivariateSpline                         
import pickle
from tools.utilities import get_spline_from_graph, save_spline_to_file, get_graph_from_spline, get_spline_from_file
import pickle
from scipy.interpolate import make_interp_spline, BSpline
from scipy.interpolate import UnivariateSpline
from tools.statistics import poly_func 
from tools.graphs import MGraph, slice_graph, concatenate_graphs, compute_pull_graphs, plot_graph, slice_graph_by_value
import pandas as pd

xbinning = Binning(Rigidity_Analysis_Binning_FullRange())
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

detectors = {"NaF", "Agl"}

#selector_denominator_event = {"LIP": {"Tof": selector_istof, "NaF": selector_isnaf_lip, "Agl": selector_isagl_lip},
selector_denominator_event = {"CIEMAT": {"Tof":selector_istof, "NaF": selector_isnaf_ciemat, "Agl": selector_isagl_ciemat}}

#selector_numerator_event = {"LIP": {"Tof": selector_tofevents, "NaF": selector_nafevents_lipvar, "Agl": selector_aglevents_lipvar},
selector_numerator_event = {"CIEMAT": {"Tof":selector_tofevents, "NaF": selector_nafevents_ciematvar, "Agl": selector_aglevents_ciematvar}}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--resultdir", default="plots/rich", help="Directory to store plots and result files in.")
    parser.add_argument("--dataname", default="AglISS", help="dataname for the output file and plots.")
    parser.add_argument("--nuclei", default="Be", help="dataname for the output file and plots.")
    parser.add_argument("--datadir", default="/home/manbing/Documents/Data/data_effacc/eff_corrections", help="Directory to store plots and result files in.")
    
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    nuclei = args.nuclei
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
    with np.load("plots/rich/graph_rich_effcor.npz") as datafile:
        for dec in detectors:
            graph_average_eff_cor[dec] = MGraph.from_file(datafile, f"{dec}")
            
    with np.load("/home/manbing/Documents/lithiumanalysis/scripts/efficiency/plots/rich/graph_rich_effcor.npz") as datafile:
        for dec in detectors:
            graph_average_eff_cor_nobetacut[dec] = MGraph.from_file(datafile, f"{dec}")



    graph_rich_jiahui = dict()
    graph_rich_jiahui_iss = dict()
    graph_rich_jiahui_mc = dict()
    #open jiahui rich_agl efficiency csv file:
    pd_jiahui_agl = pd.read_csv("/home/manbing/Documents/Data/jiahui/efficiencies/rich_agl_eff_mix.csv",  sep='\s+', header=0)

    graph_rich_jiahui["Agl"] = MGraph(xvalues=pd_jiahui_agl["bin_center[GV]"], yvalues=pd_jiahui_agl["dt"]/pd_jiahui_agl["Be_mix"],
                             yerrs=calc_ratio_err(pd_jiahui_agl["dt"], pd_jiahui_agl["Be_mix"], pd_jiahui_agl["dt_err"], pd_jiahui_agl["Be_mix_err"]))

    graph_rich_jiahui_iss["Agl"] = MGraph(xvalues=pd_jiahui_agl["bin_center[GV]"], yvalues=pd_jiahui_agl["dt"], yerrs=pd_jiahui_agl["dt_err"])
    graph_rich_jiahui_mc["Agl"] = MGraph(xvalues=pd_jiahui_agl["bin_center[GV]"], yvalues=pd_jiahui_agl["Be_mix"], yerrs=pd_jiahui_agl["Be_mix_err"])

        #open jiahui rich_naf efficiency csv file:
    pd_jiahui_naf = pd.read_csv("/home/manbing/Documents/Data/jiahui/efficiencies/rich_naf_eff_mix.csv",  sep='\s+', header=0)
    graph_rich_jiahui["NaF"] = MGraph(xvalues=pd_jiahui_naf["bin_center[GV]"], yvalues=pd_jiahui_naf["dt"]/pd_jiahui_naf["Be_mix"],
                             yerrs=calc_ratio_err(pd_jiahui_naf["dt"], pd_jiahui_naf["Be_mix"], pd_jiahui_naf["dt_err"], pd_jiahui_naf["Be_mix_err"]))

    graph_rich_jiahui_iss["NaF"] = MGraph(xvalues=pd_jiahui_naf["bin_center[GV]"], yvalues=pd_jiahui_naf["dt"], yerrs=pd_jiahui_naf["dt_err"])
    graph_rich_jiahui_mc["NaF"] = MGraph(xvalues=pd_jiahui_naf["bin_center[GV]"], yvalues=pd_jiahui_naf["Be_mix"], yerrs=pd_jiahui_naf["Be_mix_err"])


    fit_range = {"NaF": [3.0, 30], "Agl": [9, 90]}
    xticks = {"NaF": [2.0, 4.0, 5, 7, 10, 20 ,30, 40], "Agl": [10, 20, 30, 40, 60, 80]}
    diff = {"NaF": "1%", "Agl": "0.5%"}
    ylim_compare = {"NaF": [0.98, 1.02], "Agl": [0.98, 1.02]}
    xlim_range = {"NaF": [3.0, 45], "Agl": [9.0, 90]}

    graph_eff_iss = dict()
    graph_eff_mc = {dec: dict() for dec in detectors}
    graph_rich_effcor = {dec: dict() for dec in detectors}
    graph_eff_mc_average = dict()
    xticks = {"NaF": [2.0, 3.0, 4.0, 5, 7, 10, 20 ,30, 40], "Agl": [10, 20, 30, 40, 60, 80]}
    init_ratio = {'Be7': 0.6, 'Be9': 0.3, 'Be10': 0.1}

    graph_effcor_average_new = dict()
    graph_effcor_average_slice = dict()
    spline_rich_effcor_new = dict()
    rich_effcor_from_spline = dict()
    with np.load("/home/manbing/Documents/lithiumanalysis/scripts/efficiency/plots/rich/graph_rich_eff.npz") as file_graph_rich_eff:
        for dec in detectors:
            graph_eff_iss[dec] = MGraph.from_file(file_graph_rich_eff, f'{dec}_iss_eff')
            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 13))
            plot1d_errorbar_v2(figure, ax1, graph_eff_iss[dec].xvalues, graph_eff_iss[dec].yvalues, err=graph_eff_iss[dec].yerrs, label_x="Rigidity(GV)", label_y="Efficiency", setlogx=True, color="black", label="ISS Be", markersize=20)
            graph_eff_mc_average[dec] = MGraph(graph_eff_iss[dec].xvalues, np.zeros_like(graph_eff_iss[dec].xvalues), yerrs=np.zeros_like(graph_eff_iss[dec].xvalues))
            
            for iso in ISOTOPES[nuclei]:
                graph_eff_mc[dec][iso] = MGraph.from_file(file_graph_rich_eff, f'{dec}_mc_{iso}_eff')
                graph_rich_effcor[dec][iso]= MGraph.from_file(file_graph_rich_eff, f'{dec}_{iso}_effcor')
                plot1d_errorbar_v2(figure, ax1, graph_eff_mc[dec][iso].xvalues, graph_eff_mc[dec][iso].yvalues, err=graph_eff_mc[dec][iso].yerrs, label_x="Rigidity(GV)", label_y="Efficiency", setlogx=True, color=ISOTOPES_COLOR[iso], label=f"MC {iso}", markersize=20)
                plot1d_errorbar_v2(figure, ax2, graph_rich_effcor[dec][iso].xvalues, graph_rich_effcor[dec][iso].yvalues, err=graph_rich_effcor[dec][iso].yerrs, label_x="Rigidity(GV)", label_y="ISS/MC", setlogx=True, color=ISOTOPES_COLOR[iso], markersize=20)
                graph_eff_mc_average[dec] = graph_eff_mc_average[dec] + scale_graph(graph_eff_mc[dec][iso], init_ratio[iso])
                
                #plot_graph(figure, ax1, graph_eff_mc[dec][iso],  color=ISOTOPES_COLOR[iso], label=f"MC {iso}", style="EP", xlog=True, markersize=20)
                #plot_graph(figure, ax2, graph_rich_effcor[dec][iso],  color=ISOTOPES_COLOR[iso], label=f"MC {iso}", style="EP", xlog=True, markersize=20)


            #plot1d_errorbar_v2(figure, ax1, graph_eff_mc_average[dec].xvalues[1:-1], graph_eff_mc_average[dec].yvalues[1:-1], err=graph_eff_mc_average[dec].yerrs[1:-1], setlogx=True, color="red", markersize=20)
            effcor_errs =  calc_ratio_err(graph_eff_iss[dec].yvalues[1:-1], graph_eff_mc_average[dec].yvalues[1:-1], graph_eff_iss[dec].yerrs[1:-1], graph_eff_mc_average[dec].yerrs[1:-1])
            #plot1d_errorbar_v2(figure, ax2, graph_eff_iss[dec].xvalues[1:-1], graph_eff_iss[dec].yvalues[1:-1]/graph_eff_mc_average[dec].yvalues[1:-1], err=effcor_errs, setlogx=True, color="red", markersize=20)

            plot_graph(figure, ax1, graph_rich_jiahui_iss[dec],  color="grey", label=f"Jiahui ISS", style="EP", xlog=True, markersize=20, linewidth=2)
            plot_graph(figure, ax1, graph_rich_jiahui_mc[dec],  color="magenta", label=f"Jiahui MC Mix", style="EP", xlog=True, markersize=20)
            plot_graph(figure, ax2, graph_rich_jiahui[dec],  color="grey", label=f"Jiahui", style="EP", xlog=True, markersize=20, linewidth=2)
            ax2.set_xscale("log")
            ax1.set_xlabel("Rigidity(GV)")

            ax1.set_ylim([0.26, 0.82])
            ax2.set_ylim([0.85, 1.01])
            #ax1.set_xlim([5, 100])
            ax2.set_xlim(xlim_range[dec])             
            ax2.set_xticks(xticks[dec])
            ax1.set_xticks([])  
            ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax1.get_yticklabels()[0].set_visible(False)
            ax1.sharex(ax2)
            ax1.legend(fontsize=FONTSIZE-2, loc="lower right" if dec is "Agl" else "lower center")
            ax2.legend(fontsize=FONTSIZE-1)
            plt.subplots_adjust(hspace=.0)
            ax1.text(0.05, 0.95, f"{dec}", fontsize=30, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold') 
            #ax2.plot(xbinning.edges, [1]*len(xbinning.edges), 'b--')
            savefig_tofile(figure, args.resultdir, f"eff_correction_{dec}", 1)

            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 13))
            plot1d_errorbar_v2(figure, ax1, graph_eff_iss[dec].xvalues, graph_eff_iss[dec].yvalues, err=graph_eff_iss[dec].yerrs, label_x="Rigidity(GV)", label_y="Efficiency", setlogx=True, color="black", label="ISS Be", markersize=20)
            plot1d_errorbar_v2(figure, ax1, graph_eff_mc_average[dec].xvalues[1:-1], graph_eff_mc_average[dec].yvalues[1:-1], err=graph_eff_mc_average[dec].yerrs[1:-1], label_x="Rigidity(GV)", label_y="Efficiency", setlogx=True, color="red", markersize=20, label="MCMix Be")
            effcor_errs =  calc_ratio_err(graph_eff_iss[dec].yvalues[1:-1], graph_eff_mc_average[dec].yvalues[1:-1], graph_eff_iss[dec].yerrs[1:-1], graph_eff_mc_average[dec].yerrs[1:-1])
            plot1d_errorbar_v2(figure, ax2, graph_eff_iss[dec].xvalues[1:-1], graph_eff_iss[dec].yvalues[1:-1]/graph_eff_mc_average[dec].yvalues[1:-1], err=effcor_errs, label_x="Rigidity(GV)", label_y="ISS/MC", setlogx=True, color="red", markersize=20, label="this")
            
            plot_graph(figure, ax1, graph_rich_jiahui_iss[dec],  color="grey", label=f"Jiahui ISS", style="EP", xlog=True, markersize=21, markerfacecolor='none', linewidth=2)
            plot_graph(figure, ax1, graph_rich_jiahui_mc[dec],  color="magenta", label=f"Jiahui MC Mix", style="EP", xlog=True, markersize=22, markerfacecolor='none')
            plot_graph(figure, ax2, graph_rich_jiahui[dec],  color="grey", label=f"Jiahui", style="EP", xlog=True, markersize=21, markerfacecolor='none', linewidth=2)
            ax2.set_xscale("log")
            ax1.set_xlabel("Rigidity(GV)")

            ax1.set_ylim([0.26, 0.82])
            ax2.set_ylim([0.85, 1.01])
            #ax1.set_xlim([5, 100])
            ax2.set_xlim(xlim_range[dec])             
            ax2.set_xticks(xticks[dec])  
            ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax1.get_yticklabels()[0].set_visible(False)
            ax1.sharex(ax2)
            ax1.legend(fontsize=FONTSIZE-2, loc="lower right" if dec is "Agl" else "lower center")
            ax2.legend(fontsize=FONTSIZE-1)
            plt.subplots_adjust(hspace=.0)
            ax1.text(0.05, 0.95, f"{dec}", fontsize=30, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold') 
            #ax2.plot(xbinning.edges, [1]*len(xbinning.edges), 'b--')
            savefig_tofile(figure, args.resultdir, f"eff_correction_{dec}_mix", 1)

            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.5, 0.5]}, figsize=(16, 13))
            graph_effcor_average_new[dec] = MGraph(graph_eff_iss[dec].xvalues[1:-1], graph_eff_iss[dec].yvalues[1:-1]/graph_eff_mc_average[dec].yvalues[1:-1], yerrs= effcor_errs)
            print(dec, graph_effcor_average_new[dec].xvalues)
            print(dec, graph_effcor_average_new[dec].yvalues)
            graph_effcor_average_slice[dec] = slice_graph_by_value(graph_effcor_average_new[dec], fit_range[dec])
            spline_rich_effcor_new[dec] = UnivariateSpline(np.log(graph_effcor_average_slice[dec].xvalues[:]), graph_effcor_average_slice[dec].yvalues[:], k=3, s=5)
            rich_effcor_from_spline[dec] = spline_rich_effcor_new[dec](np.log(graph_effcor_average_new[dec].xvalues))

            plot1d_errorbar_v2(figure, ax1, graph_effcor_average_new[dec].xvalues, graph_effcor_average_new[dec].yvalues, err=graph_effcor_average_new[dec].yerrs,
                               label_x="Rigidity(GV)", label_y="ISS/MC(Be Mix)",  label="this", color="red", setlogx=True, markersize=20)
            plot1d_errorbar_v2(figure, ax1, graph_rich_jiahui[dec].xvalues, graph_rich_jiahui[dec].yvalues, err=graph_rich_jiahui[dec].yerrs,
                               label_x="Rigidity(GV)", label_y="ISS/MC(Be Mix)",  label="Jiahui", color="grey", setlogx=True, markersize=21, markerfacecolor='none')
        
            ax1.plot(graph_effcor_average_new[dec].xvalues, rich_effcor_from_spline[dec], '-', color="red")
            ax1.plot(graph_effcor_average_new[dec].xvalues, eff_cor_jw[dec](np.log(graph_effcor_average_new[dec].xvalues)), "-", color="grey", linewidth=2)
            ax2.plot(graph_effcor_average_new[dec].xvalues, rich_effcor_from_spline[dec]/eff_cor_jw[dec](np.log(graph_effcor_average_new[dec].xvalues)), "-", color="red")

            diff_effcor = compute_pull_graphs(graph_effcor_average_new[dec], graph_rich_jiahui[dec], slice_range=[2, 90] if dec is "NaF" else [10, 90], show_error=False)
            plot_graph(figure, ax2, diff_effcor, color="black", label="ratio: data", style="EP", xlog=True, markersize=20)
        
            set_plot_defaultstyle(ax2)
            ax2.set_xscale("log")
            ax1.set_ylim([0.8, 1.018])
        
            ax2.set_ylabel("this/J.W")
            ax1.text(0.9, 0.95, f"{dec}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
            if dec=="Agl":
                ax2.text(0.5, 0.95, f"Difference within {diff[dec]}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes, color="red", fontweight="bold")
            if dec=="NaF":
                ax2.text(0.6, 0.95, r"Difference < 1%", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes, color="red", fontweight="bold")

            ax1.fill_betweenx(np.linspace(0.8, 1.018, 100), ANALYSIS_RANGE_RIG[nuclei][dec][0], ANALYSIS_RANGE_RIG[nuclei][dec][1], alpha=0.1, color='tab:blue')
            ax2.fill_betweenx(np.linspace(ylim_compare[dec][0], ylim_compare[dec][1], 100), ANALYSIS_RANGE_RIG[nuclei][dec][0], ANALYSIS_RANGE_RIG[nuclei][dec][1], alpha=0.1, color='tab:blue')
            ax2.set_ylim(ylim_compare[dec])
            ax2.set_xlim(xlim_range[dec])
            ax2.set_xticks(xticks[dec])
            ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax2.set_xlabel("Rigidity(GV)", fontsize=FONTSIZE)
            ax1.get_yticklabels()[0].set_visible(False)
            ax1.sharex(ax2)
            ax1.set_xticks([])  
            ax1.legend(loc="lower right")
            plt.subplots_adjust(hspace=.0)
            ax2.grid()
            savefig_tofile(figure, args.resultdir, f"compare_efficiency_rich_{dec}_new", 1)

    with open(os.path.join(args.datadir, 'spline_NaF_effcor_v1.pickle'), 'wb') as f:
        pickle.dump(spline_rich_effcor_new['NaF'], f)
    
    plt.show()
    
'''        
    for dec in detectors:
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 12))
        #for i, iso in enumerate(ISOTOPES[nuclei]):
        #plot1d_errorbar(figure, ax1, xbinning.edges, dict_effcor[dec], err=None,
        #                label_x="Rigidity(GV)", label_y="ISS/MC",  legend="this", col="tab:orange", style=".", legendfontsize=FONTSIZE, setlogx=True)
            
        ax1.plot(xbinning.bin_centers, eff_cor_jw[dec](np.log(xbinning.bin_centers)), "-", color="black", label="J.W")
        #ax2.plot(xbinning.bin_centers, fit_effcor[dec](np.log(xbinning.bin_centers))/eff_cor_jw[dec](np.log(xbinning.bin_centers)), ".", color="tab:orange", label="this")
                 
        ax1.set_ylim([0.8, 1.0])
        ax1.set_xlim([1.9, 350])
        ax1.text(0.1, 0.95, f"RICH-{dec}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold")
        ax1.legend(loc='lower right')
        ax2.set_ylim([0.95, 1.05])
        ax2.set_xticks([2, 5, 10, 30,  100,  300])
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.set_ylabel("this/J.W")
        ax1.get_yticklabels()[0].set_visible(False)
        ax1.sharex(ax2)
        plt.subplots_adjust(hspace=.0)
        plt.grid()
        ax2.grid()
'''

if __name__ == "__main__":
    main()

