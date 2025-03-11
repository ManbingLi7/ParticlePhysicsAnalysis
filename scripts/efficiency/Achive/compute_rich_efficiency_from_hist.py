import multiprocessing as mp
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import uproot
import uproot3
import pickle
from tools.roottree import read_tree
from tools.selections import *
import scipy.stats
from scipy.optimize import curve_fit
from tools.studybeta import hist1d, hist2d, hist_beta, getbeta, hist_betabias, compute_moment   
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, FIGSIZE_MID, FIGSIZE_BIG, setplot_defaultstyle, format_order_of_magnitude, FONTSIZE, savefig_tofile, FONTSIZE_BIG, FONTSIZE_MID, plot1d_errorbar_v2
from tools.studybeta import calc_signal_fraction, hist1d, hist1d_weighted
from tools.binnings_collection import  fbinning_energy_agl
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, cdf_double_gaus, double_gaus
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d
from tools.binnings import Binning
from tools.constants import NUCLEI_CHARGE
from tools.calculator import calc_ekin_from_beta
from tools.calculator import calculate_efficiency_and_error, calculate_efficiency_and_error_weighted, calculate_efficiency_weighted

#import uproot_methods.classes.TGraphErrors as TGraphErrors
from tools.graphs import MGraph, slice_graph, concatenate_graphs
from scipy.interpolate import UnivariateSpline 
from tools.utilities import get_spline_from_graph, save_spline_to_file, get_graph_from_spline, get_spline_from_file

#xbinning = fbinning_energy_agl()
setplot_defaultstyle()
kNucleiBinsRebin = np.array([0.8,1.00,1.16,1.33,1.51,1.71,1.92,2.15,2.40,2.67,2.97,3.29,3.64,4.02,4.43,4.88,
                             5.37,5.90,6.47,7.09,7.76,8.48,9.26, 10.1,11.0,12.0,13.0,14.1,15.3,16.6,18.0,19.5,21.1,22.8,24.7,26.7,28.8,31.1,33.5,36.1,    
                             38.9,41.9,45.1,48.5,52.2,60.3,69.7,80.5,93.0,108., 116.,147.,192.,259.,379.,660., 1300, 3300])
kNucleiBinsRebin_center = get_bin_center(kNucleiBinsRebin)

def read_roothist(rootfile, histname=None, labelx=None):
    hist = rootfile[histname]
    bincontent, bindeges = hist.to_numpy(flow=True)
    binerrors = hist.errors(flow=True)
    xbinning = Binning(bindeges)
    hist1dpy = WeightedHistogram(xbinning, values=np.array(bincontent), squared_values=binerrors**2, labels=[labelx])
    return hist1dpy

def calculate_efficiency_from_hist(histpass, histtot, datatype, flow=False, asGraph=True):
    npass, binedges = histpass.to_numpy(flow=flow)
    ntot, binedges = histtot.to_numpy(flow=flow)
    eff, efferr = calculate_efficiency_and_error(npass, ntot, datatype)
    if asGraph:
        xvalue = get_bin_center(binedges)
        return MGraph(xvalue, eff, efferr)
    else:
        return eff, efferr 

def fill_hist_ekin_counts(events, ekinbinning, isdata=True):
    richbeta = get_richbeta(events, is_data=isdata)
    ekin_rich = calc_ekin_from_beta(richbeta)
    hist = Histogram(ekinbinning, labels=["Ekin/n (GeV/n)", "events"]) 
    hist.fill(ekin_rich)
    return hist

def plot_comparison_twohist(figure=None, ax1=None, ax2=None, histA=None, histB=None, ratio=None, ratioerr=None, xlog=True, ylog=False, isotope=None, xlabel=None, ylabel=None, figname=None, save=False, cutname=None):
    #if figure==None:        
    #   figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))
    if cutname == "trigger":
        labelA = f"{isotope} Unbias"
        labelB = f"{isotope} Physics"
    else:
        labelA= f'{isotope} total'
        labelB = f'{isotope} pass'
    plot_histogram_1d(ax1, histA, style="iss", color="tab:blue", label=labelA, xlog=xlog, ylog=ylog,  setscilabely=True, markersize=20, show_overflow=False)
    plot_histogram_1d(ax1, histB, style="iss", color="tab:orange", label=labelB, scale=None, gamma=None, xlog=True, ylog=False, shade_errors=False, show_overflow=False, adjust_limits=None, adjust_limits_x=None, adjust_limits_y=None, flip_axes=False, override_limits=False, use_approximate_poisson_errors=False, draw_zeros=True, setscilabelx=False, setscilabely=True, markersize=20)
    xbinning = histA.binnings[0] 
    plot1dhist(figure, ax2, xbinning=xbinning.edges[1:-1], counts=ratio, err=ratioerr,  label_x=xlabel, label_y="ISS/MC", legend=None,  col="black", setlogx=True, setlogy=False, setscilabelx=False,  setscilabely=False)
    plt.subplots_adjust(hspace=.0)                                                               
    ax1.legend()
    ax2.sharex(ax1)
    if (save):
        savefig_tofile(figure, "trees/acceptance", f"hist_{figname}_rig_passtotalevent", show=False)
     
isotopes = {"Li": ["Li6", "Li7"], "Be": ["Be7", "Be9", "Be10"], "Boron": ["Bo10", "Bo11"], "Carbon": ["C12"]}
color_iso = {"ISS": "black", "Be7": "tab:orange", "Be9": "tab:blue", "Be10": 'tab:green', "C12": 'tab:orange'}
color_cut = {"l1q": "magenta", "tof": "tab:blue", "inntof": "tab:orange", "inncutoff": "tab:orange",
             "pk": "brown", "bz": "tab:red", "trigger": "green", "richAgl": "red", "richNaF": "blue", "total": "black"}
xlabel = {"Rigidity" : "Rigidity (GV)", "KineticEnergyPerNucleon": "Ekin/n (GeV/n)"}   
chargename = {"Li": "z3", "Be": "z4", "Boron": "z5", "Carbon": "z6"}

studyeff = ["richAgl", "richNaF"]

cutname = {"l1q": "L1 unbiased charge", "tof": "Tof charge", "inntof": "Inner Track(Rtof)", "inncutoff": "Inner Track(Rcutoff)",
           "pk": "L1 picking up", "bz" : "L1 big charge", "trigger": "trigger", "richAgl": "RICH-Agl", "richNaF":"RICH-NaF"}
cutplotlim = {"richAgl": [0.2, 1.0], "richNaF":[0.2, 1.0]}
effcorplotlim = {"richAgl":[0.92, 1.02], "richNaF":[0.85, 1.0]}

def get_efficiency(nuclei, filename, datatype, variable):
    znum = chargename[nuclei]
    histnameden = {"richAgl": f"h_rich_den0_{znum}_rich2", "richNaF": f"h_rich_den0_{znum}_rich1"}
    histnamenum = {"richAgl": f"h_rich_num0_{znum}_rich2", "richNaF": f"h_rich_num0_{znum}_rich1"}

    hist_den = dict()
    hist_num = dict()
    eff = dict()
    efferr = dict()
    dict_hist_eff = dict()

    with uproot.open(filename) as rootfile:
        for i, cut in enumerate(studyeff):
            hist_den[cut] = read_roothist(rootfile, histnameden[cut], labelx=xlabel[variable])
            hist_num[cut] = read_roothist(rootfile, histnamenum[cut], labelx=xlabel[variable])
            xbinning = hist_den[cut].binnings[0]
            hist_tot = rootfile[histnameden[cut]]
            hist_pass = rootfile[histnamenum[cut]]
            xvalues = xbinning.bin_centers[1:-1]
            grapheff = calculate_efficiency_from_hist(hist_pass , hist_tot, datatype)
            #eff[cut], efferr[cut] = calculate_efficiency_and_error(hist_num[cut].values, hist_den[cut].values, datatype)
                
            dict_hist_eff[cut]  = grapheff
            
            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.55, 0.45]}, figsize=(16, 12))
            plot_comparison_twohist(figure, ax1, ax2, histA=hist_den[cut], histB=hist_num[cut], ratio=grapheff.gety(), ratioerr=grapheff.get_yerrors(), xlog=True, ylog=False, isotope=datatype, xlabel=xlabel[variable], ylabel="Efficiency", figname=f"{nuclei}{datatype}_{cut}", cutname=cut)
            ax1.text(0.03, 1.0, f"{cutname[cut]}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold") 
            plt.subplots_adjust(hspace=.0)
            #if (cut != "inn"):
            #    ax1.set_ylim([0, 1.14 * max(hist_den[cut].values)])
            ax1.get_yticklabels()[0].set_visible(False)
            ax2.set_ylim(cutplotlim[cut])                                                                     
            savefig_tofile(figure, "trees/acceptance", f"hist_{nuclei}{datatype}_{cut}_passevent", show=False)
    return dict_hist_eff

def plot_graph_eff(figure, plot, graph_eff, isotope=None, label=None, drawlabel=True, label_y="Efficiency", **kwargs):
    if drawlabel:
        plot1d_errorbar_v2(figure, plot,  graph_eff.getx(), graph_eff.gety(), err=graph_eff.get_yerrors(), label_x="Rigidity", label_y="Efficiency", style=".", legendfontsize=FONTSIZE_MID, setlogx=True, setlogy=False, setscilabelx=False, setscilabely=False, label=label if label is not None else f'{isotope}', **kwargs)
    else:
        plot1d_errorbar_v2(figure, plot,  graph_eff.getx(), graph_eff.gety(), err=graph_eff.get_yerrors(), label_x="Rigidity", label_y="ISS/MC", style=".", legendfontsize=FONTSIZE_BIG, setlogx=True, setlogy=False, setscilabelx=False, setscilabely=False,  **kwargs)



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--filename_iss", default="/home/manbing/Documents/Data/data_effacc/eff/BeISS_eff_part.root", help="Path to root file to read tree from")
    parser.add_argument("--filenames_mc", default=["/home/manbing/Documents/Data/data_effacc/eff/Be7_eff.root", 
                                                   "/home/manbing/Documents/Data/data_effacc/rich/Be9_rich_eff.root",
                                                   "/home/manbing/Documents/Data/data_effacc/rich/Be10_rich_eff.root"], help="Path to root file to read tree from")
    parser.add_argument("--treename", default="amstreea_cor", help="Name of the tree in the root file.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--resultdir", default="trees/acceptance", help="Directory to store plots and result files in.")
    parser.add_argument("--dataname", default="AglISS", help="dataname for the output file and plots.")
    parser.add_argument("--nuclei", default="Be", help="dataname for the output file and plots.")
    parser.add_argument("--variable", default="Rigidity", help="dataname for the output file and plots.")
    
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    nucleus = ["Be"]
    variable = args.variable
    dictgraph_eff_iss = {nuclei: dict() for nuclei in nucleus}
    dictgraph_eff_mc = {nuclei: {iso: dict() for iso in isotopes[nuclei]} for nuclei in nucleus}
    dict_eff_cor = {nuclei: {iso: dict() for iso in isotopes[nuclei]} for nuclei in nucleus}
    dict_polyfitpars_effcor = {nuclei: {iso: dict() for iso in isotopes[nuclei]} for nuclei in nucleus}
    filename_mc = {nuclei: {isotope: f'/home/manbing/Documents/Data/data_effacc/eff/{isotope}_eff.root' for isotope in isotopes[nuclei]} for nuclei in nucleus}

    graph_richagl_eff_iss = dict()
    graph_richagl_eff_mc = {nuclei: dict() for nuclei in nucleus}
    graph_richagl_effcor = {nuclei: dict() for nuclei in nucleus}
    graph_richnaf_eff_iss = dict()
    graph_richnaf_eff_mc = {nuclei: dict() for nuclei in nucleus}
    graph_richnaf_effcor = {nuclei: dict() for nuclei in nucleus}
    
    for nuclei in nucleus:
        dictgraph_eff_iss[nuclei] =  get_efficiency(nuclei, args.filename_iss, "ISS", variable)
        for i, isotope in enumerate(isotopes[nuclei]):        
            dictgraph_eff_mc[nuclei][isotope] = get_efficiency(nuclei, filename_mc[nuclei][isotope], f"{isotope}MC", variable)
            for cut in studyeff:
                dict_eff_cor[nuclei][isotope][cut] = dictgraph_eff_iss[nuclei][cut]/dictgraph_eff_mc[nuclei][isotope][cut]
                         
        for cut in studyeff:  
            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 12))
            plot_graph_eff(figure, ax1, dictgraph_eff_iss[nuclei][cut], "ISS", color=color_iso["ISS"])
            for i, isotope in enumerate(isotopes[nuclei]):
                plot_graph_eff(figure, ax1, dictgraph_eff_mc[nuclei][isotope][cut], isotope, color=color_iso[isotope])
                plot_graph_eff(figure, ax2, dict_eff_cor[nuclei][isotope][cut], isotope, color=color_iso[isotope], label_y ="ISS/MC")
                            
            ax1.set_ylim(cutplotlim[cut])
            ax2.set_ylim(effcorplotlim[cut])
            ax1.text(0.1, 0.95, f"{cutname[cut]}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold") 
            ax1.legend(loc='lower right')                       
            ax2.sharex(ax1)
            plt.subplots_adjust(hspace=.0)
            plt.grid()
            ax1.grid()
            ax1.get_yticklabels()[0].set_visible(False)
            ax2.set_ylabel("ISS/MC")
            savefig_tofile(figure, args.resultdir, f"hist_{nuclei}_{cut}_rig_efficiency", show=True)        

        #plot the richagl efficiency         
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 12))
        graph_richagl_eff_iss[nuclei] = slice_graph(dictgraph_eff_iss[nuclei]["richAgl"], dictgraph_eff_iss[nuclei]["richAgl"].get_index(5.0), dictgraph_eff_iss[nuclei]["richAgl"].get_index(52.0))
        plot_graph_eff(figure, ax1, graph_richagl_eff_iss[nuclei], "ISS", color=color_iso["ISS"])
        for i, isotope in enumerate(isotopes[nuclei]):
            #get the spline function of the efficiency correction and save it to pickel file:
            graph_richagl_eff_mc[nuclei][isotope] = slice_graph(dictgraph_eff_mc[nuclei][isotope]["richAgl"], dictgraph_eff_mc[nuclei][isotope]["richAgl"].get_index(5.0),
                                                       dictgraph_eff_mc[nuclei][isotope]["richAgl"].get_index(52.0))
            graph_richagl_effcor[nuclei][isotope] = graph_richagl_eff_iss[nuclei] /graph_richagl_eff_mc[nuclei][isotope]  
            plot_graph_eff(figure, ax1, graph_richagl_eff_mc[nuclei][isotope], isotope, color=color_iso[isotope])         
            plot_graph_eff(figure, ax2, graph_richagl_effcor[nuclei][isotope], isotope, color=color_iso[isotope], label_y="ISS/MC")
            
            #plot_curvefit(figure, ax2, graph_richagl_effcor[nuclei][isotope], poly_func, fit_coeffs, col=color_cut["richAgl"], label="RICH Agl")
            spline_richagl = get_spline_from_graph(graph_richagl_effcor[nuclei][isotope])
            splinefit_richagl_eff = spline_richagl(graph_richagl_effcor[nuclei][isotope].getx())
            save_spline_to_file(spline_richagl, args.resultdir, "spline_richagl_eff.pickle")
            print("splinefit_richagl_eff", splinefit_richagl_eff)

        ax2.sharex(ax1)
        plt.subplots_adjust(hspace=.0)
        plt.grid()
        ax1.grid()
        ax1.set_ylim([0.4, 0.85])
        ax1.text(0.03, 0.96, "RICH-Agl", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold") 
        ax2.set_ylim([0.9, 1.05])
        ax1.get_yticklabels()[0].set_visible(False)
        ax2.set_ylabel("ISS/MC")
        savefig_tofile(figure, args.resultdir, f"hist_{nuclei}_richagl_rig_efficiency", show=True)

        #plot the richnaf efficiency         
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 12))
        graph_richnaf_eff_iss[nuclei] = slice_graph(dictgraph_eff_iss[nuclei]["richNaF"], dictgraph_eff_iss[nuclei]["richNaF"].get_index(3.0), dictgraph_eff_iss[nuclei]["richNaF"].get_index(52.0))
        plot_graph_eff(figure, ax1, graph_richnaf_eff_iss[nuclei], "ISS", color=color_iso["ISS"])
        for i, isotope in enumerate(isotopes[nuclei]):
            graph_richnaf_eff_mc[nuclei][isotope] = slice_graph(dictgraph_eff_mc[nuclei][isotope]["richNaF"], dictgraph_eff_mc[nuclei][isotope]["richNaF"].get_index(3.0),
                                                       dictgraph_eff_mc[nuclei][isotope]["richNaF"].get_index(52.0))
            graph_richnaf_effcor[nuclei][isotope] = graph_richnaf_eff_iss[nuclei] /graph_richnaf_eff_mc[nuclei][isotope]  
            plot_graph_eff(figure, ax1, graph_richnaf_eff_mc[nuclei][isotope], isotope, color=color_iso[isotope])           
            plot_graph_eff(figure, ax2, graph_richnaf_effcor[nuclei][isotope], isotope, color=color_iso[isotope], label_y="ISS/MC")                                                                                                          
            spline_richnaf = get_spline_from_graph(graph_richnaf_effcor[nuclei][isotope])
            splinefit_richnaf_eff = spline_richagl(graph_richnaf_effcor[nuclei][isotope].getx())
            save_spline_to_file(spline_richnaf, args.resultdir, "spline_richnaf_eff.pickle")
            
        ax2.sharex(ax1)
        ax1.text(0.03, 0.96, "RICH-NaF", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold") 
        plt.subplots_adjust(hspace=.0)
        plt.grid()
        ax1.grid()
        ax1.set_ylim([0.4, 0.85])
        ax2.set_ylim([0.8, 1.05])
        ax1.get_yticklabels()[0].set_visible(False)
        ax2.set_ylabel("ISS/MC")
        savefig_tofile(figure, args.resultdir, f"hist_{nuclei}_richnaf_rig_efficiency", show=True)

        #plot the mc mixture 

    plt.show()
        
if __name__ == "__main__":
    main()

        
