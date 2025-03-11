import multiprocessing as mp
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import uproot
import uproot3 
from tools.roottree import read_tree
from tools.selections import *
import scipy.stats
from scipy.optimize import curve_fit
from tools.studybeta import hist1d, hist2d, hist_beta, getbeta, hist_betabias, compute_moment                                                                                                              
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, FIGSIZE_MID, FIGSIZE_BIG, setplot_defaultstyle, format_order_of_magnitude, FONTSIZE, savefig_tofile, FONTSIZE_BIG, plot1d_errorbar_v2, tick_length, tick_labelsize, tick_width, set_plot_defaultstyle
from tools.studybeta import calc_signal_fraction, hist1d, hist1d_weighted
from tools.binnings_collection import  fbinning_energy_agl
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, cdf_double_gaus, double_gaus
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d
from tools.binnings import Binning
from tools.constants import NUCLEI_CHARGE, ISOTOPES_CHARGE
from tools.calculator import calc_ekin_from_beta, calc_ekin_from_rigidity
from tools.calculator import calculate_efficiency_and_error, calculate_efficiency_and_error_weighted, calculate_efficiency_weighted
from tools.statistics import poly_func
from tools.graphs import MGraph, slice_graph, concatenate_graphs, plot_graph
from tools.utilities import get_spline_from_graph, save_spline_to_file, get_graph_from_spline, get_spline_from_file
import pickle
from scipy.interpolate import make_interp_spline, BSpline
from scipy.interpolate import UnivariateSpline
import pandas as pd
import ROOT
from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F
from tools.jupytertools import *
#xbinning = fbinning_energy_agl()

tick_length = 14                                                                                         
tick_width=1.5                                                                                           
tick_labelsize = 30                                                                                      
legendfontsize = 40 
def set_plot_style(plot):                                                                         
    plt.rcParams["font.weight"] = "bold"                                                                 
    plt.rcParams["axes.labelweight"] = "bold"                                                            
    plt.rcParams['font.size']= 35                                                                 
    plt.rcParams['xtick.top'] = True                                                                    
    plt.rcParams['ytick.right'] = True                                                                  
    plot.tick_params(axis='both', which="major",direction='in', length=tick_length, width=tick_width, labelsize=tick_labelsize, pad=10)                                       
    plot.tick_params(axis='both', which="minor",direction='in', length=tick_length/2.0, width=tick_width, labelsize=tick_labelsize, pad=10)                                        
    for axis in ['top','bottom','left','right']:                                                    
        plot.spines[axis].set_linewidth(3)                                                               
    plt.minorticks_on() 

rigiditybinning = np.array([0.8,1.00,1.16,1.33,1.51,1.71,1.92,2.15,2.40,2.67,2.97,3.29,3.64,4.02,4.43,4.88,5.37,5.90,
                            6.47,7.09,7.76,8.48,9.26, 10.1,11.0,12.0,13.0,14.1,15.3,16.6,18.0,19.5,21.1,22.8,24.7,26.7,28.8,31.1,33.5,36.1,38.9,41.9,
                            45.1,48.5,52.2,56.1,60.3,64.8,69.7,74.9,80.5,86.5,93.0, 100.,108.,116.,125.,135.,147.,160.,175.,192.,211.,233.,259.,291.,
                            330.,379.,441.,525.,660.,880.,1300.,3300.])

energybinning = fbinning_energy()
xbinning = {"Rigidity" : rigiditybinning, "Ekin": energybinning}
xaxistitle = {"Rigidity": "Rigidity (GV)", "Ekin": "Ekin/n (GeV)"}

RIG_XLIM =[1.8, 1000]
RIG_XLABEL = "Rigidity (GV)"

def read_roothist(rootfile, histname=None, labelx=None):
    hist = rootfile[histname]
    bincontent, bindeges = hist.to_numpy(flow=True)
    binerrors = hist.errors(flow=True)
    xbinning = Binning(bindeges)
    hist1dpy = WeightedHistogram(xbinning, values=np.array(bincontent), squared_values=binerrors**2, labels=[labelx])
    return hist1dpy

isotopes = {"He": ["He4"], "Li": ["Li6", "Li7"], "Be": ["Be7", "Be9", "Be10"], "Boron": ["Bo10", "Bo11"], "C": ["C12"]}
color_iso = {"ISS": "black", "Be7": "tab:orange", "Be9": "tab:blue", "Be10": 'tab:green', "C12": 'tab:orange', 'He4':"tab:blue", 'O16':"tab:green"}
color_cut = {"l1q": "magenta", "tof": "tab:blue", "inntof": "tab:orange", "inncutoff": "tab:orange", "pk": "brown", "bz": "tab:red", "trigger": "green", "total": "black", "background": "pink"}
xlabel = {"Rigidity" : "Rigidity (GV)", "KineticEnergyPerNucleon": "Ekin/n (GeV/n)"}   
chargename = {"He": "z2", "Li": "z3", "Be": "z4", "Boron": "z5", "C": "z6"}

studyeff = ["background"]

cutname = {"background": "Background Reduction Cut"}
cutplotlim = {"background": {"He": [0.96, 1.02], "C": [0.89, 1.01]}}
effcorplotlim = {"background": {"He": [0.98, 1.01], "C": [0.94, 1.0]}}


#histnameden =  f"h_event_{znum}_2_1"
#histnamenum =  f"h_event_clean4_{znum}_2_1"
MC_GEN_MOMENTOM_RANGE = {"C12": [6, 12000], "B10": [5, 10000], "B11": [5, 10000], "Be7":[4, 8000],
                         "Be9": [4, 8000],  "Be10": [4, 8000], 'O16': [8, 16000], "N14":[7, 14000], "N15":[7, 14000]}

def plot_curvefit(figure, plot, graph, func, pars, col=None, label=None):
    plot.plot(graph.getx()[:], func(np.log(graph.getx()[:]), *pars), '-', label=label, color=col)

def get_nppolyfit(x, y, deg):
    coeffs1 = np.polyfit(x, y, deg=deg)
    fit1 = np.poly1d(coeffs1)
    return fit1

def getpars_curvefit_poly(datagraph, deg):
    initial_guess = np.zeros(deg) # Initial guess for the polynomial coefficients
    fit_coeffs, _ = curve_fit(poly_func, np.log(datagraph.getx()[:]), datagraph.gety()[:], sigma=datagraph.get_yerrors()[:], p0=initial_guess)
    return fit_coeffs


def get_acceptance_input_hist(events_pass, file_pgen, trigger, isotope, variable):
    #root_pgen = TFile.Open(file_pgen[isotope], "READ")
    #hist_pgen = root_pgen.Get("PGen_px")
    #print("trigger from pgen: ", hist_pgen.Integral())
    #nbins_pgen = hist_pgen.GetNbinsX()
    #minLogMom = None
    #maxLogMom = None
    charge = ISOTOPES_CHARGE[isotope]
    #for i in range(nbins_pgen):
    #    if hist_pgen.GetBinContent(i+1) > 0 :
    #        if minLogMom is None:
    #            minLogMom = hist_pgen.GetXaxis().GetBinLowEdge(i+1)
    #        maxLogMom = hist_pgen.GetXaxis().GetBinUpEdge(i+1)
            
    binning = xbinning[variable]
    #minMom = 10**(minLogMom)
    #maxMom = 10**(maxLogMom)
    minLogMom_v1 = np.log10(MC_GEN_MOMENTOM_RANGE[isotope][0])
    maxLogMom_v1 = np.log10(MC_GEN_MOMENTOM_RANGE[isotope][1])
    
    minRigGen = 10**(minLogMom_v1)/charge
    maxRigGen = 10**(maxLogMom_v1)/charge
    minEkinGen = calc_ekin_from_rigidity(minRigGen, MC_PARTICLE_IDS[isotope])
    maxEkinGen = calc_ekin_from_rigidity(maxRigGen, MC_PARTICLE_IDS[isotope])
        
    hist_total = ROOT.TH1F("hist_total", "hist_total",  len(binning)-1, binning)                                          
    tot_trigger = trigger[isotope]
    
    #arr_pass = hist_events_pass.values()
    arr_pass = events_pass
    #arr_pass_err = hist_events_pass.errors()
    arr_pass_err = np.sqrt(events_pass)
    print(isotope, len(arr_pass), len(arr_pass))
    print(isotope, 'binning:', binning)
    for ibin in range(1, len(binning)):
        #print(ibin, binning[ibin], binning[ibin - 1])
        frac = (np.log(binning[ibin]) - np.log(binning[ibin - 1]))/(np.log(maxRigGen) - np.log(minRigGen))                                                                                  
        num = tot_trigger * frac                                                                                                                                     
        hist_total.SetBinContent(ibin, num)
    #print("the total trigger: ", tot_trigger)

    arr_tot = np.zeros(len(binning) -1)
    for i in range(len(binning) - 1):
        arr_tot[i] = hist_total.GetBinContent(i+1)

    eff, efferr = calculate_efficiency_and_error(arr_pass, arr_tot, "MC")
    acc = eff * 3.9 * 3.9 * np.pi
    accerr = efferr * 3.9 * 3.9 * np.pi
    xbincenter = get_bin_center(binning)
    #for i in range(len(binning) - 1): 
    #    print(arr_pass[i],  arr_tot[i] , arr_pass[i]/arr_tot[i] * 3.9 * 3.9 * np.pi, acc[i])

    graph_acc = MGraph(xvalues = xbincenter, yvalues=acc, yerrs=accerr)
    return graph_acc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--filenames_mc", default=["/home/manbing/Documents/Data/data_effacc/annflux/C12MC_anaflux.root", 
                                                   "/home/manbing/Documents/Data/data_effacc/annflux/Be9MC_anaflux.root",
                                                   "/home/manbing/Documents/Data/data_effacc/annflux/Be10MC_anaflux.root"], help="Path to root file to read tree from")
    parser.add_argument("--treename", default="amstreea_cor", help="Name of the tree in the root file.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--resultdir", default="plots/effcor/total", help="Directory to store plots and result files in.")
    parser.add_argument("--datadir", default="/home/manbing/Documents/Data/data_effacc/eff_corrections", help="Directory to store and take data.")    
    parser.add_argument("--dataname", default="AglISS", help="dataname for the output file and plots.")
    parser.add_argument("--nuclei", default="Be", help="dataname for the output file and plots.")
    parser.add_argument("--variable", default="Rigidity", help="dataname for the output file and plots.")
    
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    nuclei_alias = {"He": "He", "C": "Carbon"}
    variable = args.variable 

    nuclei = args.nuclei
    #file_pgen = {isotope: f"trees/acceptance/histPGen_{isotope}.root" for isotope in isotopes[nuclei]}
    file_pgen = {"C12": "trees/acceptance/histPGen_Be7.root"}         
    trigger = {"Be7" : 6090705916, "Be9": 6484356101, "Be10": 2605554333, "C12": 8589106140, 'B10':8373181574, 'B11':8008225318, 'O16':8373181574, "N14":8402971473, "N15":8690555915}
    trigger_from_pgen = {"Be7" : 6999354260, "Be9": 8757003151, "Be10": 8581757698}                                        #get correct number for N

    znum = 'z4'
    file_name_iss = "/home/manbing/Documents/Data/data_BeP8/efficiency/anaflux_ISSP8_GBL.root"
    with uproot.open(file_name_iss) as file_iss:
        hist_event_be = file_iss["h_event_z4_2_1"]
        hist_event_clean_be = file_iss["h_event_clean4_z4_2_1"]
        
    file_names = {"B10": "/home/manbing/Documents/Data/data_BeP8/efficiency/B10MC_B1236_anaflux_GBL.root",
                  "B11": "/home/manbing/Documents/Data/data_BeP8/efficiency/B11MC_B1236_anaflux_GBL.root",
                  "C12": "/home/manbing/Documents/Data/data_BeP8/efficiency/C12MC_B1236_anaflux_GBL.root",
                  "N14": "/home/manbing/Documents/Data/data_BeP8/efficiency/N14MC_B1236_anaflux_GBL.root",
                  "N15": "/home/manbing/Documents/Data/data_BeP8/efficiency/N15MC_B1236_anaflux_GBL.root",
                  "O16": "/home/manbing/Documents/Data/data_BeP8/efficiency/O16MC_B1236_anaflux_GBL.root"}
    
    file_flux_qy = {"C12": '/home/manbing/Documents/Data/data_flux_qy/carbon64_20200420V2N_B1215401R4MCKY10COMBUNFOLDTOINB_totalQYAN.root',
                    'B10': '/home/manbing/Documents/Data/data_flux_qy/boron64_20200420V2N_B1218401R4MCKY10COMBUNFOLDTOINB_totalQYAN.root',
                    'B11': '/home/manbing/Documents/Data/data_flux_qy/boron64_20200420V2N_B1218401R4MCKY10COMBUNFOLDTOINB_totalQYAN.root',
                    'O16': '/home/manbing/Documents/Data/data_flux_qy/oxygen64_20200420V2N_B1220402R4MCKY10COMBUNFOLDNB_totalQYAN.root',
                    'N14': '/home/manbing/Documents/Data/data_flux_qy/nitrogen64_20200420V2N_B1220402R4MCKY10COMBUNFOLDTOINB_totalQYAN.root',
                    'N15': '/home/manbing/Documents/Data/data_flux_qy/nitrogen64_20200420V2N_B1220402R4MCKY10COMBUNFOLDTOINB_totalQYAN.root'}
    
    #fragment_nuclei = {"B10": "z5", "B11": "z5", "C12": "z6"}
    fragment_nuclei = {"B10": "Z5", "B11": "Z5","C12": "Z6", "N14": "Z7", "N15": "Z7", "O16": "Z8"}
    #fragment_nuclei = {"O16": "Z8"}
    rig_binning = Binning(rigiditybinning)
    hist_ref_flux = dict()
    hist_time_ref = dict()

    hist_event = dict()
    graph_acc = dict()
    counts_zxtoz4 = dict()
    spline_frag_ratio = dict()
    dict_counts = dict()

    hist_event_clean = dict()
    graph_acc_clean = dict()
    counts_zxtoz4_clean = dict()
    spline_frag_ratio_clean = dict()
    dict_counts_clean = dict()

    hist_event_nofragl1 = dict()
    hist_event_clean_nofragl1 = dict()
    event_fragl1 = dict()
    event_fragl1_clean = dict()
    for iso, z_name in fragment_nuclei.items():
        with uproot.open(file_flux_qy[iso]) as file_ref_flux:
            hist_ref_flux[iso] = file_ref_flux[f"{z_name}fluxh_totalN"]
            hist_time_ref[iso] = file_ref_flux["ExpoTimeN"]

        print('hist_ref_flux[iso].axes[0].edges():', hist_ref_flux[iso].axes[0].edges())
        
        with uproot.open(file_names[iso]) as rootfile:
            hist_event[iso] =  rootfile[f"h_event_gen_z4_2"]
            hist_event_nofragl1[iso] =  rootfile[f"h_event_gen_l1_z4_2"]

            hist_event_clean[iso] =  rootfile[f"h_event_gen_clean_z4_2"] 
            hist_event_clean_nofragl1[iso] =  rootfile[f"h_event_gen_clean_l1_z4_2"] 

            event_fragl1[iso]  = hist_event[iso].values() - hist_event_nofragl1[iso].values()
            event_fragl1_clean[iso]  = hist_event_clean[iso].values() - hist_event_clean_nofragl1[iso].values()
            print(iso, 'hist_event_clean[iso].axes[0].edges():', hist_event_clean[iso].axes[0].edges())
            print(iso, 'hist_event[iso].axes[0].edges():', hist_event[iso].axes[0].edges())
        graph_acc[iso] = get_acceptance_input_hist(event_fragl1[iso], file_pgen, trigger, f"{iso}", variable)
        graph_acc_clean[iso] = get_acceptance_input_hist(event_fragl1_clean[iso], file_pgen, trigger, f"{iso}", variable)
            

        counts_zxtoz4[iso] = hist_ref_flux[iso].values()[:-1] * graph_acc[iso].yvalues * rig_binning.bin_widths[1:-1] * hist_time_ref[iso].values()[:-1]
        counts_zxtoz4_clean[iso] = hist_ref_flux[iso].values()[:-1] * graph_acc_clean[iso].yvalues * rig_binning.bin_widths[1:-1] * hist_time_ref[iso].values()[:-1]
            
        dict_counts[iso] = {"rig_low": rig_binning.edges[1:-2],
                       "rig_ip": rig_binning.edges[2:-1],
                       "bin_width": rig_binning.bin_widths[1:-1],
                       "flux_carbon": hist_ref_flux[iso].values()[:-1],
                       "acc_zxtoz4": graph_acc[iso].yvalues,
                       "time": hist_time_ref[iso].values()[:-1],
                       "counts": counts_zxtoz4[iso]}
        df_counts = pd.DataFrame(dict_counts[iso], columns=dict_counts.keys())


    nuclei = ["B", "C", "N", "O"]
    #nuclei = ["B"]
    iso_ratio_guess = {"B10": 0.5, "B11":0.5, "C12":1.0, "O16": 1.0, "N14": 0.5, "N15": 0.5}
    COLOR_NUCLEI = {"B": "tab:blue", "C": "tab:green", "N":"magenta", "O": "tab:red"}

    #plot the background before the second trk cut
    frag_ratio = {nucleon: np.zeros(len(rig_binning.bin_centers[1:-1])) for nucleon in nuclei}
    bkg_abovel1_total = np.zeros(len(rig_binning.bin_centers[1:-1]))    
    yfit_bkg_total = np.zeros(len(rig_binning.bin_centers[1:-1]))    
    fig = plt.figure(figsize=FIGSIZE_BIG)
    ax1 = fig.subplots(1,1)
    fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95) 
    for i_nuclei in nuclei:
        for iso in ISOTOPES[i_nuclei]:
            frag_ratio[i_nuclei] += counts_zxtoz4[iso] * iso_ratio_guess[iso]/hist_event_be.values()
            bkg_abovel1_total += counts_zxtoz4[iso] * iso_ratio_guess[iso]/hist_event_be.values()
        spline_frag_ratio[i_nuclei] = UnivariateSpline(np.log(rig_binning.bin_centers[4:-2]), frag_ratio[i_nuclei][3:-1], k=3, s=100)
        yfit_frag_ratio = spline_frag_ratio[i_nuclei](np.log(rig_binning.bin_centers[1:-1]))
        ax1.plot(rig_binning.bin_centers[1:-1], frag_ratio[i_nuclei], ".", markersize=18, color=COLOR_NUCLEI[i_nuclei])
        ax1.plot(rig_binning.bin_centers[1:-1], yfit_frag_ratio, "-", color=COLOR_NUCLEI[i_nuclei], label=f'{i_nuclei}')
        yfit_bkg_total += yfit_frag_ratio
        
    ax1.plot(rig_binning.bin_centers[1:-1], bkg_abovel1_total, ".", color="black")
    spline_bkg_abovel1_total = UnivariateSpline(np.log(rig_binning.bin_centers[2:-2]), bkg_abovel1_total[1:-1], k=3, s=100)
    yfit_bkg = spline_bkg_abovel1_total(np.log(rig_binning.bin_centers[1:-1]))
    
    ax1.plot(rig_binning.bin_centers[1:-1], yfit_bkg, "--", color="magenta", linewidth=3, label="Total")
    ax1.plot(rig_binning.bin_centers[1:-1], yfit_bkg_total, "-", color="black", linewidth=3, label="Total")
    ax1.set_ylabel(r"$\mathrm{N_{x, Be}/N_{Be^{'}}}$", fontsize=FONTSIZE)
    ax1.set_xlabel(r"$\mathrm{Rigidity (GV)}$", fontsize=FONTSIZE)
    ax1.set_xscale('log')
    ax1.legend(fontsize=FONTSIZE)
    setplot_defaultstyle()
    ax1.set_xlim([1.8, 1300])
    ax1.set_xticks([2, 5, 10, 30,  100,  300, 1000])
    set_plot_defaultstyle(ax1)
    ax1.set_ylim([0.0, 0.2])
    savefig_tofile(fig, args.resultdir, "background_abovel1_before_cut", 1)

    #plot the background pile up 
    fig = plt.figure(figsize=FIGSIZE_BIG)
    ax1 = fig.subplots(1,1)
    fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)
    bkg_stackup = np.zeros(len(rig_binning.bin_centers[1:-1]))
    yfit_bkg_stackup = np.zeros((len(nuclei)+1, len(rig_binning.bin_centers[1:-1])))
    for i, i_nuclei in enumerate(nuclei):
        for iso in ISOTOPES[i_nuclei]:
            bkg_stackup += counts_zxtoz4[iso] * iso_ratio_guess[iso]/hist_event_be.values()

        #ax1.plot(rig_binning.bin_centers[1:-1], bkg_stackup, ".", markersize=18, color=COLOR_NUCLEI[i_nuclei])
        spline_bkg_stackup = UnivariateSpline(np.log(rig_binning.bin_centers[4:-2]), bkg_stackup[3:-1], k=3, s=100)
        yfit_bkg_stackup[i+1, :] = spline_bkg_stackup(np.log(rig_binning.bin_centers[1:-1]))
        ax1.plot(rig_binning.bin_centers[1:-1], yfit_bkg_stackup[i+1, :], "-", color=COLOR_NUCLEI[i_nuclei], linewidth=3, label=f'{i_nuclei}')
        ax1.fill_between(rig_binning.bin_centers[1:-1], yfit_bkg_stackup[i, :], yfit_bkg_stackup[i+1, :], interpolate=True, color=COLOR_NUCLEI[i_nuclei], alpha=0.5)
    ax1.set_ylabel(r"$\mathrm{N_{x, Be}/N_{Be^{'}}}$", fontsize=FONTSIZE)
    ax1.set_xlabel(r"$\mathrm{Rigidity (GV)}$", fontsize=FONTSIZE)
    ax1.set_xscale('log')
    setplot_defaultstyle()
    set_plot_defaultstyle(ax1)
    ax1.set_xlim([1.8, 1300])
    ax1.set_xticks([2, 5, 10, 30,  100,  300, 1000])
    ax1.set_ylim([0.0, 0.2])
    handles = [mpatches.Patch(color=c) for c in COLOR_NUCLEI.values()]
    ax1.legend(handles, nuclei, fontsize=FONTSIZE)
    set_plot_style(ax1)
    savefig_tofile(fig, args.resultdir, "background_abovel1_before_cut_stackup", 1)

    #plot the background after the second trk cut
    frag_ratio_clean = {nucleon: np.zeros(len(rig_binning.bin_centers[1:-1])) for nucleon in nuclei}
    bkg_abovel1_total_clean = np.zeros(len(rig_binning.bin_centers[1:-1]))    
    spline_frag_ratio_clean = dict()
    
    fig = plt.figure(figsize=FIGSIZE_BIG)
    ax1 = fig.subplots(1,1)
    fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)
    yfit_bkg_clean_total = np.zeros_like(rig_binning.bin_centers[1:-1])
    for i_nuclei in nuclei:
        for iso in ISOTOPES[i_nuclei]:
            frag_ratio_clean[i_nuclei] += counts_zxtoz4_clean[iso] * iso_ratio_guess[iso]/hist_event_clean_be.values()
            bkg_abovel1_total_clean += counts_zxtoz4_clean[iso] * iso_ratio_guess[iso]/hist_event_clean_be.values()

        print(frag_ratio_clean[i_nuclei], len(frag_ratio_clean[i_nuclei]))
        print(rig_binning.bin_centers, len(rig_binning.bin_centers))
        spline_frag_ratio_clean[i_nuclei] = UnivariateSpline(np.log(rig_binning.bin_centers[1:-1][5:-1]), frag_ratio_clean[i_nuclei][5:-1], k=5, s=10)
        yfit_frag_ratio_clean = spline_frag_ratio_clean[i_nuclei](np.log(rig_binning.bin_centers[1:-1]))
        ax1.plot(rig_binning.bin_centers[1:-1], frag_ratio_clean[i_nuclei], ".", markersize=18, color=COLOR_NUCLEI[i_nuclei])
        ax1.plot(rig_binning.bin_centers[1:-1], yfit_frag_ratio_clean, "-", color=COLOR_NUCLEI[i_nuclei], label=f'{i_nuclei}')
        yfit_bkg_clean_total = yfit_bkg_clean_total +  yfit_frag_ratio_clean
        #print(frag_ratio[i_nuclei])
    ax1.plot(rig_binning.bin_centers[1:-1], bkg_abovel1_total_clean, ".", color="black")
    spline_bkg_abovel1_total_clean = UnivariateSpline(np.log(rig_binning.bin_centers[6:-1]), bkg_abovel1_total_clean[5:], k=3, s=10)
    yfit_bkg_clean = spline_bkg_abovel1_total_clean(np.log(rig_binning.bin_centers[1:-1]))
    ax1.plot(rig_binning.bin_centers[1:-1], yfit_bkg_clean, "--", color="black", linewidth=3, label="Total")
    ax1.plot(rig_binning.bin_centers[1:-1], yfit_bkg_clean_total, "-", color="black", linewidth=3, label="Total")
    ax1.set_ylabel(r"$\mathrm{N_{x, Be}/N_{Be^{'}}}$", fontsize=FONTSIZE)
    ax1.set_xlabel(r"$\mathrm{Rigidity (GV)}$", fontsize=FONTSIZE)
    ax1.set_xscale('log')
    ax1.legend(fontsize=FONTSIZE)
    set_plot_style(ax1)
    ax1.set_xlim([1.8, 1300])
    ax1.set_ylim([0.0, 0.1])
    ax1.set_xticks([2, 5, 10, 30,  100,  300, 1000])
    set_plot_style(ax1)
    savefig_tofile(fig, args.resultdir, "background_abovel1_after_cut", 1)


    fig = plt.figure(figsize=FIGSIZE_BIG)
    ax1 = fig.subplots(1,1)
    fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)
    bkg_stackup_clean = np.zeros(len(rig_binning.bin_centers[1:-1]))
    yfit_bkg_stackup_clean = np.zeros((len(nuclei)+1, len(rig_binning.bin_centers[1:-1])))
    for i, i_nuclei in enumerate(nuclei):
        for iso in ISOTOPES[i_nuclei]:
            bkg_stackup_clean += counts_zxtoz4_clean[iso] * iso_ratio_guess[iso]/hist_event_clean_be.values()

        #ax1.plot(rig_binning.bin_centers[1:-1], bkg_stackup, ".", markersize=18, color=COLOR_NUCLEI[i_nuclei])
        spline_bkg_stackup_clean = UnivariateSpline(np.log(rig_binning.bin_centers[5:-1]), bkg_stackup_clean[4:], k=3, s=10)
        yfit_bkg_stackup_clean[i+1, :] = spline_bkg_stackup_clean(np.log(rig_binning.bin_centers[1:-1]))
        ax1.fill_between(rig_binning.bin_centers[1:-1], yfit_bkg_stackup_clean[i, :], yfit_bkg_stackup_clean[i+1, :], interpolate=True, color=COLOR_NUCLEI[i_nuclei], alpha=0.5)
        ax1.plot(rig_binning.bin_centers[1:-1], yfit_bkg_stackup_clean[i, :], "-", color=COLOR_NUCLEI[i_nuclei], linewidth=3, label=f'{i_nuclei}')

    
    ax1.set_ylabel(r"$\mathrm{N_{x, Be}/N_{Be^{'}}}$", fontsize=FONTSIZE)
    ax1.set_xlabel(r"$\mathrm{Rigidity (GV)}$", fontsize=FONTSIZE)
    ax1.set_xscale('log')
    setplot_defaultstyle()
    set_plot_defaultstyle(ax1)
    handles = [mpatches.Patch(color=c) for c in COLOR_NUCLEI.values()]
    ax1.legend(handles, nuclei, fontsize=FONTSIZE)
    ax1.set_xlim([1.8, 1300])
    ax1.set_ylim([0.0, 0.1])
    set_plot_style(ax1)
    ax1.set_xticks([2, 5, 10, 30,  100,  300, 1000])
    savefig_tofile(fig, args.resultdir, "background_abovel1_after_cut_stackup", 1)


    #compare the total bkg before and after the cut, and calculate the term (1+R_n)/(1+R_d)
    fig = plt.figure(figsize=FIGSIZE_BIG)
    ax1 = fig.subplots(1,1)
    fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)
    ax1.plot(rig_binning.bin_centers[1:-1], yfit_bkg_clean_total, "-", color="black", linewidth=3, label="Total After Cut")
    ax1.plot(rig_binning.bin_centers[1:-1], yfit_bkg, "-", color="red", linewidth=3, label="Total Before Cut")
    ax1.legend(fontsize=FONTSIZE)
    ax1.set_ylabel(r"$\mathrm{N_{x, Be}/N_{Be^{'}}}$", fontsize=FONTSIZE)
    ax1.set_xlabel(r"$\mathrm{Rigidity (GV)}$", fontsize=FONTSIZE)
    ax1.set_xscale('log')
    setplot_defaultstyle()
    ax1.set_xlim([1.8, 1300])
    ax1.set_xticks([2, 5, 10, 30,  100,  300, 1000])
    set_plot_defaultstyle(ax1)
    savefig_tofile(fig, args.resultdir, "compare_background_abovel1", 1)

    fig = plt.figure(figsize=FIGSIZE_BIG)
    ax1 = fig.subplots(1,1)
    fig.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)

    #factor = (1 + yfit_bkg_clean_total) /(1+yfit_bkg_total)
    factor = (1 -yfit_bkg_total) /(1 - yfit_bkg_clean_total)
    ax1.plot(rig_binning.bin_centers[1:-1], factor, "-", color="black", linewidth=3, label="Total After Cut")
    ax1.legend(fontsize=FONTSIZE)
    ax1.set_ylabel(r"$\mathrm{(1+R_{num})/(1+R_{den})}$", fontsize=FONTSIZE)
    ax1.set_xlabel(r"$\mathrm{Rigidity (GV)}$", fontsize=FONTSIZE)
    set_plot_style(ax1)
    #setplot_defaultstyle()
    ax1.set_xlim([1.8, 1300])
    ax1.set_xticks([2, 5, 10, 30,  100,  300, 1000])
    ax1.set_xscale('log')
    #set_plot_defaultstyle(ax1)
    set_plot_style(ax1)
    savefig_tofile(fig, args.resultdir, "background_abovel1_acc_factor", 1)

    dict_graph_factor = dict()
    graph_factor = MGraph(rig_binning.bin_centers[1:-1], factor, yerrs=np.zeros_like(factor))
    graph_factor.add_to_file(dict_graph_factor, 'graph_issbkgeff_factor')
    np.savez(os.path.join(args.datadir, "graph_issbkgeff_factor.npz"), **dict_graph_factor) 

    plt.show()
        
if __name__ == "__main__":
    main()

    
