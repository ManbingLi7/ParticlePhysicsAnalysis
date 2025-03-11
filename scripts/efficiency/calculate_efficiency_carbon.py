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
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, FIGSIZE_MID, FIGSIZE_BIG, setplot_defaultstyle, format_order_of_magnitude, FONTSIZE, savefig_tofile, FONTSIZE_BIG
from tools.studybeta import calc_signal_fraction, hist1d, hist1d_weighted
from tools.binnings_collection import  fbinning_energy_agl
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, cdf_double_gaus, double_gaus
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d
from tools.binnings import Binning
from tools.constants import NUCLEI_CHARGE
from tools.calculator import calc_ekin_from_beta
from tools.calculator import calculate_efficiency_and_error, calculate_efficiency_and_error_weighted

#xbinning = fbinning_energy_agl()
setplot_defaultstyle()

def get_triggereff(hist_phystrigger, hist_unbiastrigger):
    print(hist_phystrigger.values)
    print(hist_unbiastrigger.values)
    trigger_eff = hist_phystrigger.values/(hist_phystrigger.values + 100 * hist_unbiastrigger.values)
    err = np.zeros(len(trigger_eff)) 
    return trigger_eff, err

def get_triggereff_mc(hist_phystrigger, hist_unbiastrigger):
    print(hist_phystrigger.values)
    print(hist_unbiastrigger.values)
    trigger_eff = hist_phystrigger.values/(hist_phystrigger.values + hist_unbiastrigger.values)
    err = np.zeros(len(trigger_eff)) 
    return trigger_eff, err

def read_roothist(rootfile, histname=None, labelx=None):
    hist = rootfile[histname]
    bincontent, bindeges = hist.to_numpy(flow=True)
    binerrors = hist.errors(flow=True)
    xbinning = Binning(bindeges)
    hist1dpy = WeightedHistogram(xbinning, values=np.array(bincontent), squared_values=binerrors**2, labels=[labelx])
    return hist1dpy

def fill_hist_ekin_counts(events, ekinbinning, isdata=True):
    richbeta = get_richbeta(events, is_data=isdata)
    ekin_rich = calc_ekin_from_beta(richbeta)
    hist = Histogram(ekinbinning, labels=["Ekin/n (GeV/n)", "events"]) 
    hist.fill(ekin_rich)
    return hist

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--filename_iss", default="trees/eff_corr/BeISS_anaflux.root", help="Path to root file to read tree from")
    parser.add_argument("--filenames_mc", default=["trees/eff_corr/C12MC_anaflux.root"], help="Path to root file to read tree from")
    parser.add_argument("--treename", default="amstreea_cor", help="Name of the tree in the root file.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--resultdir", default="trees/acceptance", help="Directory to store plots and result files in.")
    parser.add_argument("--dataname", default="AglISS", help="dataname for the output file and plots.")
    parser.add_argument("--nuclei", default="Carbon", help="dataname for the output file and plots.")
    parser.add_argument("--variable", default="Rigidity", help="dataname for the output file and plots.")
    
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    nuclei = args.nuclei
    charge = NUCLEI_CHARGE[nuclei]
    variable = args.variable
    isotopes = {"Li": ["Li6", "Li7"], "Be": ["Be7", "Be9", "Be10"], "Boron": ["Bo10", "Bo11"], "Carbon": ["C12"]}
    color = {"Be7": "tab:blue", "Be9": "tab:orange", "Be10": 'tab:green', "C12": 'tab:orange'}
    color_cut = {"l1q": "tab:orange", "tof": "tab:blue", "inn": "tab:red", "pk": "tab:green", "bz": "yellow", "trigger": "black"}
    #1.I need to get the total number events before and after L1unbias charge
    #2. get the efficiency of data and MC
    #3. calculate the efficiency corrections
    xlabel = {"Rigidity" : "Rigidity (GV)", "KineticEnergyPerNucleon": "Ekin/n (GeV/n)"}   
    chargename = {"Li": "z3", "Be": "z4", "Boron": "z5", "Carbon": "z6"};
    znum = chargename[nuclei]
    
    studyeff = ["tof", "trigger"]
    #studyeff = ["inn"]
    histnameden = {"l1q" : f"h_l1q_den_{znum}_0", "tof": f"h_tof_den_{znum}_0", "inn":f"h_inn_den_ek_{znum}_4_0", "pk": f"h_pk_den_{znum}_0", "bz":f"h_bz_den_{znum}_0", "trigger": f"h_trigger_{znum}_0_ub"}
    histnamenum = {"l1q" : f"h_l1q_num_{znum}_0", "tof": f"h_tof_num_{znum}_0", "inn":f"h_inn_num_ek_{znum}_4_0", "pk": f"h_pk_num_{znum}_0", "bz":f"h_bz_num_{znum}_0", "trigger": f"h_trigger_{znum}_0_phy"}
    cutname = {"l1q": "L1 unbiased charge", "tof": "Tof charge", "inn": "Inner Track", "pk": "L1 picking up", "bz" : "L1 big charge", "trigger": "trigger"}
    cutplotlim = {"l1q": [0.95, 1.02], "tof": [0.9, 1.02], "inn": [0.7, 0.95], "pk": [0.7, 1.0], "bz" : [0.6, 0.9], "trigger": [0.98, 1.02]}
    effcorplotlim = {"l1q": [0.9, 1.1], "tof": [0.9, 1.1], "inn": [0.9, 1.1], "pk": [0.9, 1.1], "bz" : [0.9, 1.1], "trigger": [0.98, 1.01]}
    #effplotlim = {"l1q": [0.95, 1.02], "tof": [0.9, 1.02], "inn": [0.7, 0.95], "pk": [0.7, 1.0], "bz" : [0.6, 0.9], "trigger": [0.9, 1.02]}
    h_l1q_den = dict()
    h_l1q_num = dict()
    eff_issbe = dict()
    efferr_issbe = dict()
    hiss_eff_l1q = dict()
    with uproot.open(args.filename_iss) as issfile:
        for i, cut in enumerate(studyeff):
            
            h_l1q_den[cut] = read_roothist(issfile, histnameden[cut], labelx=xlabel[variable])
            h_l1q_num[cut] = read_roothist(issfile, histnamenum[cut], labelx=xlabel[variable])
            xbinning = h_l1q_num[cut].binnings[0]
            if cut == "trigger":
                eff_issbe[cut], efferr_issbe[cut] = get_triggereff(h_l1q_num[cut], h_l1q_den[cut])
            else:
                eff_issbe[cut], efferr_issbe[cut] = calculate_efficiency_and_error(h_l1q_num[cut].values, h_l1q_den[cut].values)
            hiss_eff_l1q[cut] = Histogram(h_l1q_num[cut].binnings[0], values=np.array(eff_issbe[cut]))

            figure = plt.figure(figsize=FIGSIZE_BIG)
            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.55, 0.45]}, figsize=(16, 12))
            plot_histogram_1d(ax1, h_l1q_den[cut], style="iss", color="tab:blue", label='iss total', xlog=True, ylog=False,  setscilabely=True, markersize=20, show_overflow=False)
            plot_histogram_1d(ax1, h_l1q_num[cut], style="iss", color="tab:orange", label='iss pass', scale=None, gamma=None, xlog=True, ylog=False, shade_errors=False, show_overflow=False, adjust_limits=None, adjust_limits_x=None, adjust_limits_y=None, flip_axes=False, override_limits=False, use_approximate_poisson_errors=False, draw_zeros=True, setscilabelx=False, setscilabely=True, markersize=20)
            #plot1dhist(figure, ax2, xbinning.edges, eff_issbe, efferr_issbe,  "Rigidity(GV)", "Efficiency", "ISS: L1 unbiased charge",  "black", 30, 1, 0, 0, 1)
            plot1dhist(figure, ax2, xbinning=xbinning.edges, counts=eff_issbe[cut], err=efferr_issbe[cut],  label_x=xlabel[variable], label_y="Efficiency", legend=None,  col="black", setlogx=True, setlogy=False, setscilabelx=False,  setscilabely=False)
            ax1.legend()
            ax1.text(0.03, 1.0, f"{cutname[cut]}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold") 
            plt.subplots_adjust(hspace=.0)
            if (cut != "inn"):
                ax1.set_ylim([0, 1.14 * max(h_l1q_den[cut].values)])
            ax1.get_yticklabels()[0].set_visible(False)
            ax2.set_ylim(cutplotlim[cut])                                                                     
            ax2.sharex(ax1)
            savefig_tofile(figure, "trees/acceptance", f"hist_{nuclei}ISS_{cut}_passevent", show=True)
            
    dict_eff_cor = {cut: dict() for cut in studyeff}
    eff_mcbe = {cut: dict() for cut in studyeff}
    efferr_mcbe = {cut: dict() for cut in studyeff}
    hmc_eff_l1q = {cut: dict() for cut in studyeff}
    hist_effcor = {cut: dict() for cut in studyeff} 
    hmc_l1q_den = {cut: dict() for cut in studyeff} 
    hmc_l1q_num = {cut: dict() for cut in studyeff} 
    for icut, cut in enumerate(studyeff):
        for i, isotope in enumerate(isotopes[nuclei]):
            filename_mc = args.filenames_mc[i]
            with uproot.open(filename_mc) as mcfile:
                hmc_l1q_den[cut][isotope] = read_roothist(mcfile, histnameden[cut], labelx=xlabel[variable])
                hmc_l1q_num[cut][isotope] = read_roothist(mcfile, histnamenum[cut], labelx=xlabel[variable])
                if cut == "trigger":
                    eff_mcbe[cut][isotope], efferr_mcbe[cut][isotope] = get_triggereff_mc(hmc_l1q_num[cut][isotope], hmc_l1q_den[cut][isotope])
                else:
                    eff_mcbe[cut][isotope], efferr_mcbe[cut][isotope] = calculate_efficiency_and_error_weighted(hmc_l1q_num[cut][isotope].values, hmc_l1q_den[cut][isotope].values, hmc_l1q_num[cut][isotope].squared_values, hmc_l1q_den[cut][isotope].squared_values)
                xbinning = hmc_l1q_num[cut][isotope].binnings[0]

                hmc_eff_l1q[cut][isotope] = Histogram(xbinning, values=np.array(eff_mcbe[cut][isotope]))


            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 12))
            plot_histogram_1d(ax1, hmc_l1q_den[cut][isotope], style="iss", color="tab:blue", label=f'{isotope} total', xlog=True, ylog=False,  setscilabely=True, markersize=20, show_overflow=False)
            plot_histogram_1d(ax1, hmc_l1q_num[cut][isotope], style="iss", color="tab:orange", label=f'{isotope} pass', scale=None, gamma=None, xlog=True, ylog=False, shade_errors=False, show_overflow=False, adjust_limits=None, adjust_limits_x=None, adjust_limits_y=None, flip_axes=False, override_limits=False, use_approximate_poisson_errors=False, draw_zeros=True, setscilabelx=False, setscilabely=True, markersize=20)
            plot1dhist(figure, ax2, xbinning=xbinning.edges, counts=eff_mcbe[cut][isotope], err=efferr_mcbe[cut][isotope],  label_x=xlabel[variable], label_y="Efficiency", legend=None,  col="black", setlogx=True, setlogy=False, setscilabelx=False,  setscilabely=False)
            ax1.text(0.03, 1.0, f"{cutname[cut]}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold") 
            plt.subplots_adjust(hspace=.0)                                                               
            ax1.legend()
            ax2.sharex(ax1)

            dict_eff_cor[cut][isotope] = hiss_eff_l1q[cut].values/hmc_eff_l1q[cut][isotope].values
            hist_effcor[cut][isotope] = Histogram(xbinning, values=np.array(dict_eff_cor[cut][isotope]), labels=[xlabel[variable]])
            savefig_tofile(figure, "trees/acceptance", f"hist_{isotope}_{cut}_rig_passtotalevent", show=True)
            #np.savez(os.path.join(args.resultdir, f"{isotope}eff_correction_agl_V2.npz"), xbincenter=xbinning.bin_centers,  eff_cor=eff_cor)
       
     
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 12))
        plot1dhist(figure, ax1, xbinning=xbinning.edges, counts=eff_issbe[cut], err=efferr_issbe[cut],  label_x=xlabel[variable], label_y="Efficiency", legend=f"ISS",  col="black", setlogx=True, setlogy=False, setscilabelx=False,  setscilabely=False)
        for isotope in isotopes[nuclei]:
            plot1dhist(figure, ax1, xbinning.edges, eff_mcbe[cut][isotope],  efferr_mcbe[cut][isotope], xlabel[variable], "Efficiency", f"{isotope}",  color[isotope], 30, 1, 0, 0, 0)
            plot1dhist(figure, ax2, xbinning=hmc_eff_l1q[cut][isotope].binnings[0].edges, counts=dict_eff_cor[cut][isotope], err=np.zeros(len(dict_eff_cor[cut][isotope])),  label_x=xlabel[variable], label_y="ISS/MC", legend=None,  col=color[isotope], setlogx=True, setlogy=False, setscilabelx=False,  setscilabely=False)
        ax1.set_ylim(cutplotlim[cut])
        
        ax2.set_ylim(effcorplotlim[cut])
        ax1.text(0.1, 0.95, f"{cutname[cut]}", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold") 
        ax1.legend(loc='lower right')                       
        ax2.sharex(ax1)
        plt.subplots_adjust(hspace=.0)
        plt.grid()
        ax1.grid()
        ax1.get_yticklabels()[0].set_visible(False)
        savefig_tofile(figure, "trees/acceptance", f"hist_{cut}_rig_efficiency", show=True)
        
    for isotope in isotopes[nuclei]:
        figure = plt.figure(figsize=FIGSIZE_BIG)                              
        plot = figure.subplots(1, 1)
        plot.text(0.8, 0.95, f"{isotope}", fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color=color[isotope], weight='normal') 
        for cut in studyeff:
            plot1dhist(figure, plot, hmc_eff_l1q[cut][isotope].binnings[0].edges, dict_eff_cor[cut][isotope],  np.zeros(len(dict_eff_cor[cut][isotope])), xlabel[variable], "Eff Correction", f"{cut}",  color_cut[cut], 30, 1, 0, 0, 1) 
        plot.set_ylim([0.95, 1.05])
        savefig_tofile(figure, "trees/acceptance", f"hist_{cut}_rig_efficiencycorrection", show=True)
    plt.show()
    
if __name__ == "__main__":
    main()
