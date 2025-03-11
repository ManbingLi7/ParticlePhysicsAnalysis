import multiprocessing as mp
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from tools.roottree import read_tree
from tools.selections import *
from tools.constants import MC_PARTICLE_CHARGES, MC_PARTICLE_IDS 
from tools.binnings_collection import mass_binning, Rigidity_Analysis_Binning_FullRange
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.calculator import calc_mass, calc_ekin_from_beta, calc_betafrommomentom
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, plot1d_step
from collections.abc import MutableMapping
from tools.corrections import shift_correction
import uproot
from scipy import interpolate
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, get_corrected_lipbeta_agl, get_index_correction_agl
import ROOT
from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F
from ROOT import gROOT, gBenchmark, gRandom, gSystem
from tools.functions import gaussian, asy_gaussian, poly, asy_gaussian_1d
uproot.default_library
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, LeastSquares
from iminuit.util import describe, make_func_code 
from tools.constants import ISOTOPES, NUCLEI_NUMBER, ISOTOPES_COLOR
from tools.histograms import WeightedHistogram, Histogram, plot_histogram_2d, plot_histogram_1d
from tools.binnings import Binning, make_lin_binning
from tools.graphs import MGraph, plot_graph, get_nppolyfit
detectors = ["RichAgl"]
datasets = {"LiIss": {detector: 0 for detector in detectors},
            "Li6Mc": {detector: 1 for detector in detectors}, 
            "Li7Mc":{detector: 2 for detector in detectors}}

mass_binning = mass_binning()
inverse_mass_binning = np.linspace(0.05, 0.3, 100)
energy_per_neculeon_binning = fbinning_energy()
bin1_num = len(energy_per_neculeon_binning) -1
bin2_num = len(inverse_mass_binning) - 1
qsel = 3.0

particleID = {"Li6Mc": MC_PARTICLE_IDS["Li6"], "Li7Mc": MC_PARTICLE_IDS["Li7"], "Be7Mc": MC_PARTICLE_IDS["Be7"]}
var_rigidity = "tk_rigidity1"

#par[0], par[1], par[2]: mean
#sigma: par[3], 4, 5
#fraction_coregaus: 6
#sigma_sec_to_prim: 7
#ays factor: 8
#norm7: 9
#norm9: 10
#norm10: 11
rigidityBinning = Binning(Rigidity_Analysis_Binning_FullRange(), overflow=False)

def make_rigidity_resolution_function(num_bins, resolution_binning):
    reso_binwidth = resolution_binning[1:] - resolution_binning[:-1]    
    def rig_reso_function(x, *pars):
        energy = x[0,:].reshape((num_bins, -1))
        rig_reso = x[1,:].reshape((num_bins, -1))
        mua, mub, muc = pars[0:3]
        siga, sigb, sigc = pars[3:6]
        fraccore = pars[6]
        sigma_ratio = pars[7]
        asy_factor = pars[8]
        norm_a, norm_b, norm_c, norm_d = pars[9:13]
        mean = poly(energy, muc, mub, mua)
        sigma = poly(energy, sigc, sigb, siga)
        norm = poly(energy, norm_d, norm_c, norm_b, norm_a)
        pdf = np.zeros(energy.shape)
        coregaus = gaussian(rig_reso, mean, sigma)
        asygaus = asy_gaussian(rig_reso, mean,  sigma_ratio * sigma, asy_factor)
        pdf = norm * (fraccore * coregaus + (1 - fraccore) * asygaus)  * reso_binwidth[None, :]
        return pdf.reshape((-1, ))
    parnames = ['x', 'mua', 'mub', 'muc', 'siga', 'sigb', 'sigc', 'fraccore', 'sigma_ratio', 'asy_factor', 'norm_a', 'norm_b', 'norm_c', 'norm_d'] 
    rig_reso_function.func_code = make_func_code(parnames)
    return rig_reso_function

def make_rig_function():
    def rig_resolution(x, mean, sigma, sigma_ratio, asy_factor, fraccore):
        coregaus = gaussian(x, mean, sigma)
        asygaus = asy_gaussian(x, mean,  sigma_ratio * sigma, asy_factor)
        pdf += norm * (fraccore * coregaus + (1 - fraccore) * asygaus)  * rigbinwidth
        return pdf
    parnames = ['x', 'mean', 'sigma', 'sigma_ratio', 'asy_factor', 'fraccore']
    rig_resolution.func_code = make_func_code(parnames)
    return rig_resolution

def rig_resolution(x, norm, mean, sigma, sigma_ratio, asy_factor, fraccore):
    coregaus = gaussian(x, mean, sigma)
    asygaus = asy_gaussian_1d(x, mean,  sigma_ratio * sigma, asy_factor)
    pdf = norm * (fraccore * coregaus + (1 - fraccore) * asygaus)
    return pdf

def cumulative_rig_resolution(edges, norm, mean, sigma, sigma_ratio, asy_factor, fraccore):
    x = (edges[1:] + edges[:-1])/2
    pdf = rig_resolution(x, norm, mean, sigma, sigma_ratio, asy_factor, fraccore)
    cpdf = np.cumsum(pdf)
    return np.concatenate(([0], cpdf))

    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--filename", default="/home/manbing/Documents/Data/data_rig/rigidity_resolution.npz", help="Path to root file to read tree from")
    parser.add_argument("--resultdir", default="plots/unfold", help="Directory to store plots and result files in.")
    parser.add_argument("--datadir", default="/home/manbing/Documents/Data/data_rig", help="Directory to store plots and result files in.")
    parser.add_argument("--nuclei", default="Be", help="Directory to store plots and result files in.")
    parser.add_argument("--detectors", nargs="+", default=["RichAgl"], help="Directory to store plots and result files in.")
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)

    filename = args.filename
    #get the TH2 histogram
    nuclei =args.nuclei
    isotopes = ISOTOPES[nuclei]
    detectors_energyrange = {"Tof": [0.4, 0.9], "RichNaF": [0.7, 5.5], "RichAgl": [4.0, 10]}
    range_rig_reso = [0.06, 0.25]
    hist_resolution = dict()
    fit_range = [-0.6, 0.7]
    par_names = ['norm', 'mean', 'sigma', 'sigma_ratio', 'asy_factor', 'fraccore']
    dict_pars = {iso: {} for iso in isotopes}
    with np.load(filename) as response_file:
        for iso in isotopes:
            hist_resolution[iso] = WeightedHistogram.from_file(response_file, f"rig_resolution_{iso}")
            fig = plt.figure(figsize=(20, 15))
            plot = fig.subplots()
            plot_histogram_2d(plot, hist_resolution[iso], scale=None, transpose=False, show_overflow=False, show_overflow_x=None, show_overflow_y=None, label=None, xlog=True, ylog=False, log=True)

            energy_binning = hist_resolution[iso].binnings[0]
            resolution_binning = hist_resolution[iso].binnings[1]
            rig_bin_edges = energy_binning.edges[1:-1]
            rig_bin_centers = energy_binning.bin_centers[1:28]
            dict_pars[iso] = {par: MGraph(rig_bin_centers, np.zeros(len(rig_bin_centers)), np.zeros(len(rig_bin_centers)), labels=["Rigidity(GV)", f"{par}"]) for par in par_names}
            
            # fit bin by bin 
            for i in range(1, 28):
                counts = hist_resolution[iso].values[i, :]
                hist_reso_ibin = hist_resolution[iso].project(i)
                fitrange_binindex = hist_reso_ibin.binnings[0].get_indices(fit_range)
                yvalues = hist_reso_ibin.values[fitrange_binindex[0]: fitrange_binindex[1]]
                yvalues_err = np.sqrt(hist_reso_ibin.squared_values[fitrange_binindex[0]: fitrange_binindex[1]])
                xvalues = hist_reso_ibin.binnings[0].bin_centers[fitrange_binindex[0]: fitrange_binindex[1]] 
                loss = ExtendedBinnedNLL(hist_reso_ibin.values[fitrange_binindex[0]: fitrange_binindex[1]],
                                         hist_reso_ibin.binnings[0].edges[fitrange_binindex[0]: fitrange_binindex[1]+1], cumulative_rig_resolution)
                guess = dict(norm=400, mean=0.001, sigma=0.11, sigma_ratio=1.0, asy_factor=1.0, fraccore=0.8)
                m = Minuit(loss, **guess)
                #############################################
                this set of constraint fit only the core part of the rigidity resolution
                m.limits["norm"]=(100, 1000)
                m.limits["mean"]=(-0.2, 0.2)
                m.limits["sigma"]=(0.08, 0.2)
                m.fixed['asy_factor'] = True   #1.0
                m.fixed['sigma_ratio'] = True  # 1.0 
                m.fixed['fraccore']=True        #1.0
                #############################################

                #this set of constraint fit to fit also the tile of the rigidity resolution
                #m.limits["norm"]=(100, 1000)
                #m.limits["mean"]=(-0.2, 0.2)
                #m.limits["sigma"]=(0.08, 0.2)
                #m.limits['sigma_ratio'] = (1.3, 1.32)
                #m.limits['fraccore']= (0.88, 0.9)
                

                
                m.migrad()                                                                                                                                                                            
                print(m)
                for par in dict_pars[iso].keys():
                    dict_pars[iso][par].yvalues[i-1] = m.values[par]
                    dict_pars[iso][par].yerrs[i-1] = m.errors[par]
                
                fit_results = rig_resolution(xvalues, *m.values)
                fit_guess = rig_resolution(xvalues, **guess)
                figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14)) 
                ax1.set_title(f"{i}_bin")
                plot_histogram_1d(ax1, hist_reso_ibin, style="iss", color="black", label="data", scale=None, gamma=None, xlog=False, ylog=True, shade_errors=False)
                ax1.set_xlabel('')
                ax2.set_xlabel('(1/R - 1/R_{rec}')
                ax1.plot(xvalues, fit_results, "-", color="tab:green", label="fit")
                #ax1.plot(xvalues, fit_guess, "-", color="tab:blue", label="guess")
                pull =(yvalues - fit_results)/yvalues_err
                ax2.plot(xvalues, pull, '.', markersize=20, color='black')
                #plot1d_errorbar(figure, ax2, rig_resobinedges[min_rig_resobin: max_rig_resobin + 2], counts=pull, err=np.zeros(len(pull)),  label_x="1/rig_reso (1/GeV)", label_y="pull", legend=None,  col="black", setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
                plt.subplots_adjust(hspace=.0)                             
                ax1.legend()                                         
                ax2.sharex(ax1)
                savefig_tofile(figure, args.resultdir, f"fit/fit_rigidity_resolution_{iso}_{i}", show=False)
                
        pars_poly = {iso: {} for iso in isotopes}
        polyfit ={iso: {} for iso in isotopes}
        deg = {'norm': 3, 'mean': 2, 'sigma':2, 'sigma_ratio':1, 'asy_factor':1, 'fraccore':1}
        for par in par_names:
            fig = plt.figure(figsize=(20, 15))
            plot = fig.subplots()
            for iso in isotopes:
                plot_graph(figure, plot, dict_pars[iso][par], color=ISOTOPES_COLOR[iso], label=f'{iso}', style="EP", xlog=True, ylog=False, scale=None, markersize=25)
                pars_poly[iso][par] = np.polyfit(np.log(dict_pars[iso][par].getx()), dict_pars[iso][par].gety(), deg=deg[par])
                polyfit[iso][par] = np.poly1d(pars_poly[iso][par])(np.log(dict_pars[iso][par].getx()))
                print(iso, par, pars_poly[iso][par])
                print(iso, polyfit[iso][par])
                plot.plot(rig_bin_centers, polyfit[iso][par], "-", color=ISOTOPES_COLOR[iso])
            savefig_tofile(fig, args.resultdir, f"fit/Be_Rig_{par}", show=True)
        # end fitting bin by bin

        #plot the ratio of sigma
        fig = plt.figure(figsize=(20, 15))
        plot = fig.subplots()
        graph_ratio_sigma_7to9 = dict_pars['Be7']['sigma']/dict_pars['Be9']['sigma']
        print(graph_ratio_sigma_7to9)
        graph_ratio_sigma_7to10 = dict_pars['Be7']['sigma']/dict_pars['Be10']['sigma']
        print(graph_ratio_sigma_7to10)
        plot_graph(figure, plot, graph_ratio_sigma_7to9, color=ISOTOPES_COLOR["Be9"], label=None, style="EP", xlog=False, ylog=False, scale=None, markersize=25)
        plot_graph(figure, plot, graph_ratio_sigma_7to10, color=ISOTOPES_COLOR["Be10"], label=None, style="EP", xlog=False, ylog=False, scale=None, markersize=25)
        plot.set_ylabel(r"$\mathrm{\sigma_{1/R} ~ ratio}$")
        plot.set_xlabel("Ekin/N")
        plot.set_ylim([0.93, 1.02])
        savefig_tofile(fig, args.resultdir, f"fit/Be_rig_sigma_ratio", show=True)

        #save graph to file:
        dict_graph_factor = dict()
        graph_ratio_sigma_7to9.add_to_file(dict_graph_factor, 'graph_rig_reso_sigma_7to9')
        graph_ratio_sigma_7to10.add_to_file(dict_graph_factor, 'graph_rig_reso_sigma_7to10')
        np.savez(os.path.join(args.datadir, f"graph_rigsigma_scale_factor.npz"), **dict_graph_factor)
        
    plt.show()
        
if __name__ == "__main__":
    main()

            

