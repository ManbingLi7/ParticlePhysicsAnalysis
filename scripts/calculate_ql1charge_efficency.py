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
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, FIGSIZE_MID, FIGSIZE_BIG, setplot_defaultstyle, format_order_of_magnitude, FONTSIZE, savefig_tofile, FONTSIZE_BIG, plot1d_errorbar_v2, plot1d_step, FONTSIZE_MID
from tools.studybeta import calc_signal_fraction, hist1d, hist1d_weighted
from tools.binnings_collection import  fbinning_energy_agl
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, cdf_double_gaus, double_gaus
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from tools.binnings import Binning
from tools.constants import NUCLEI_CHARGE
from tools.calculator import calc_ekin_from_beta
from tools.calculator import calculate_efficiency_and_error, calculate_efficiency_and_error_weighted, calculate_efficiency_weighted
from tools.statistics import poly_func
from tools.graphs import MGraph, slice_graph, concatenate_graphs
import ROOT
from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F
from iminuit import Minuit     
from iminuit.cost import ExtendedBinnedNLL, LeastSquares, NormalConstraint, ExtendedUnbinnedNLL
from iminuit.util import describe, make_func_code 
from scipy.interpolate import UnivariateSpline
import pickle
from tools.utilities import save_spline_to_file
from tools.functions import normalized_gaussian, cumulative_norm_gaus
#xbinning = fbinning_energy_agl()
setplot_defaultstyle()

kNucleiBinsRebin = np.array([0.8,1.00,1.16,1.33,1.51,1.71,1.92,2.15,2.40,2.67,2.97,3.29,3.64,4.02,4.43,4.88, 5.37,5.90,6.47,7.09,7.76,8.48,9.26, 10.1,11.0,12.0,13.0,14.1,15.3,16.6,18.0,19.5,21.1,22.8,24.7,26.7,28.8,31.1,33.5,36.1, 38.9, 41.9,45.1,48.5,52.2,60.3,69.7,80.5,93.0,108., 116.,147.,192.,259.,379.,660., 1300, 3300.])
kNucleiBinsRebin_center = get_bin_center(kNucleiBinsRebin)
kNucleiNbinRebin = 57

def make_charge_template_fit(nT, templates):
    def template_fit(x, *pars):
        #assert(len(pars) == len(templates))
        #assert(len(x) == len(templates[0]))
        pdf = np.zeros(x.shape)
        for i, ipar in enumerate(pars):
            pdf += ipar * templates[i]
        return pdf
    parnames = ['x'] +  [f"T_{i}" for i in range(nT)]
    template_fit.func_code = make_func_code(parnames)
    return template_fit



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--filename", default="trees/massfit/Be9iss_Agl_masshist.root",   help="Path to root file to read tree from")
    parser.add_argument("--resultdir", default="plots/effcor/l1q", help="Directory to store plots and result files in.")
    parser.add_argument("--nuclei", default="Be", help="Directory to store plots and result files in.")
    parser.add_argument("--detectors", nargs="+", default=["RichAgl"], help="Directory to store plots and result files in.")
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    
    hist_l1q = []
    nT = 3
    templates_name = ["Be", "B", "C"]
    tem_color = ["tab:green", "tab:blue", "red"]
    xbinning = Binning(kNucleiBinsRebin)
    ybinning = Binning(np.linspace(2, 14, 241))
    fitminbin = 30
    fitmaxbin = 100
    hist2d_qz = dict()
    with uproot.open("/home/manbing/Documents/Data/data_effacc/lq_template.root") as rootfile:
        for i, tem in enumerate(templates_name):
            qtem = i + 4
            print(templates_name[i], qtem)
            tree = rootfile[f'ql1_z{qtem}']
            qz = tree['q_template2'].array()
            qrig = tree['tk_rig'].array()
            hist2d_qz[tem] = Histogram(xbinning, ybinning, labels=["Rigidity (GV)", f"Z_{tem}"])
            hist2d_qz[tem].fill(qrig, qz)
            fig = plt.figure(figsize=(20, 15))
            plot = fig.subplots(1, 1)
            plot_histogram_2d(plot, hist2d_qz[tem], scale=None, transpose=False, show_overflow=True, show_overflow_x=None, show_overflow_y=None, label=None)

        cutlim_index = ybinning.get_indices([3.5, 4.7])
        boron_background = np.zeros(kNucleiNbinRebin-1)
        
        for i in range(kNucleiNbinRebin-1):
            hist_l1q.append(rootfile[f'h_l1q_{i}'])
            xvalues = get_bin_center(ybinning.edges[1:-1])[fitminbin:fitmaxbin]
            yvalues = hist_l1q[i].values()[fitminbin:fitmaxbin]
                        
            yvalues[yvalues == 0] = 1
            yvalueserr = hist_l1q[i].errors()[fitminbin:fitmaxbin]
            yvalueserr[yvalueserr == 0.0] = 1

            templates = np.zeros([nT, 240])

            for iT, temp in enumerate(templates_name):
                templates[iT] = hist2d_qz[temp].values[i+1, 1:-1]

                loss = ExtendedBinnedNLL(templates[iT], ybinning.edges[1:-1], cumulative_norm_gaus)
                guess = dict(norm=100, mu=iT+4, sigma=0.05)
                m_tem_z = Minuit(loss, **guess)
                m_tem_z.migrad()
                print(m_tem_z)
                
                templates[iT] = templates[iT]/np.sum(templates[iT] * ybinning.bin_widths[1:-1])
                
            fitfunc = make_charge_template_fit(nT, templates[:, fitminbin:fitmaxbin])
            cost = LeastSquares(xvalues, yvalues, yvalueserr, fitfunc)
            guess = {"T_0": 0.999, "T_1": 0.001, "T_2": 0.001}
            #guess = {"T_0": 0.8}
            m = Minuit(cost, **guess)
            m.migrad()
            print(m.values)

            fitresult = fitfunc(xvalues, *m.values)
            fig = plt.figure(figsize=(20, 15))
            ax1 = fig.subplots()
            plot1dhist(fig, ax1, xbinning=ybinning.edges[1:-1][fitminbin:fitmaxbin+1], counts=yvalues,
                       err=yvalueserr,  label_x="L1 Z", label_y="counts",  legend="l1q", col="tab:orange", setlogy=1)
            for iT, temp in enumerate(templates_name):
                plot1d_step(fig, ax1, xbinning=ybinning.edges[1:-1][fitminbin:fitmaxbin+1], counts= m.values[iT] * templates[iT][fitminbin:fitmaxbin],
                            err=None, label_x="L1 Z", label_y="counts",  legend=f"l2q {temp}", col=tem_color[iT], setlogy=1, alpha=0.8)
                ax1.fill_between(get_bin_center(ybinning.edges[1:-1][fitminbin:fitmaxbin+1]), y1=m.values[iT] * templates[iT][fitminbin:fitmaxbin], y2=0, where=None, interpolate=False, step=None, alpha=0.5, color=tem_color[iT])
                
            plot1d_step(fig, ax1, xbinning=ybinning.edges[1:-1][fitminbin:fitmaxbin+1], counts=fitresult, label_x="L1 Z", label_y="counts",
                        err=np.zeros(len(fitresult)), legend="fit", col="black", setlogy=1, linewidth=2.0)
            y_t = np.zeros([nT, len(templates[0][fitminbin:fitmaxbin+1])])
            for iT, temp in enumerate(templates_name):
                y_t[iT] = m.values[iT] * templates[iT][fitminbin:fitmaxbin+1]
            boron_background[i] = y_t[1][:cutlim_index[1]-fitminbin].sum()/fitresult[:cutlim_index[1]-fitminbin].sum()
            
            ax1.set_ylim([10, 2 * max(yvalues)])
            ax1.text(0.7, 0.7, r"$R = ({}, {})GV$".format(xbinning.edges[i+1], xbinning.edges[i+2]), fontsize=FONTSIZE_MID, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='normal')
            savefig_tofile(fig, args.resultdir, f"l1q_templatefit_{i}_log", show=False)  

            fig = plt.figure(figsize=(20, 15))
            ax1 = fig.subplots()
            plot1dhist(fig, ax1, xbinning=ybinning.edges[1:-1][fitminbin:fitmaxbin+1], counts=yvalues,
                       err=yvalueserr,  label_x="L1 Z", label_y="counts",  legend="l1q", col="tab:orange", setlogy=0)
            plot1d_step(fig, ax1, xbinning=ybinning.edges[1:-1][fitminbin:fitmaxbin+1], counts=fitresult,
                        label_x="L1 Z", label_y="counts", err=None, legend="fit", col="black", setlogy=0)
            for iT, temp in enumerate(templates_name):
                plot1d_step(fig, ax1, xbinning=ybinning.edges[1:-1][fitminbin:fitmaxbin+1], counts= m.values[iT] * templates[iT][fitminbin:fitmaxbin],
                            err=None, label_x="L1 Z", label_y="counts",  legend=f"l2q {temp}", col=tem_color[iT], setlogy=0)
            ax1.set_ylim([1, 1.1 * max(yvalues)])
            savefig_tofile(fig, args.resultdir, f"l1q_templatefit_{i}", show=False)

        print(len(boron_background), len(get_bin_center(hist2d_qz["Be"].binnings[0].edges[1:-1])))

        print(boron_background)
        be_signal = 1 - boron_background
        fig = plt.figure(figsize=(16, 10))
        ax1 = fig.subplots()
        plot1dhist(fig, ax1, xbinning=hist2d_qz["Be"].binnings[0].edges[1:-2], counts=boron_background,
                   err=np.zeros(len(boron_background)),  label_x="Rigidity (GV)", label_y="Background", col="tab:orange", setlogx=1)
        ax1.text(0.05, 0.95, r"$\mathrm{L1Q > 4.7}$", fontsize=FONTSIZE_MID, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='normal')
        #ax1.set_ylim([0.95, 1.01])
        savefig_tofile(fig, args.resultdir, f"l1q_upppercut_background", show=True)

        fig = plt.figure(figsize=(16, 10))
        ax1 = fig.subplots()
        plot1dhist(fig, ax1, xbinning=hist2d_qz["Be"].binnings[0].edges[1:-2], counts=be_signal,
                   err=np.zeros(len(be_signal)),  label_x="Rigidity (GV)", label_y="Purity", col="tab:orange", setlogx=1)
        ax1.text(0.05, 0.95, r"$\mathrm{L1Q > 4.7}$", fontsize=FONTSIZE_MID, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='normal')
        ax1.set_ylim([0.98, 1.01])
        savefig_tofile(fig, args.resultdir, f"l1q_upppercut_purity", show=True)

        #calculate efficiency of the l1 charge upper limit cut using hist2d_qz["Be"]
        eff_1 = hist2d_qz["Be"].values[:,:cutlim_index[1]].sum(axis=1) / hist2d_qz["Be"].values.sum(axis=1)
        n_pass = hist2d_qz["Be"].values[:,:cutlim_index[1]].sum(axis=1)
        n_tot = hist2d_qz["Be"].values.sum(axis=1)  
        eff, efferr = calculate_efficiency_and_error(n_pass, n_tot, "ISS")
        fig = plt.figure(figsize=(16, 14))
        ax1 = fig.subplots()
        plot1dhist(fig, ax1, xbinning= hist2d_qz["Be"].binnings[0].edges[1:-1], counts=eff[1:-1],
                   err=efferr[1:-1],  label_x="Rigidity (GV)", label_y="Efficiency", col="tab:orange", setlogx=1)
        ax1.set_ylim([0.95, 1.01])
        ax1.text(0.05, 0.95, r"$\mathrm{L1Q > 4.7}$", fontsize=FONTSIZE_MID, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='normal')
        print(len(hist2d_qz["Be"].binnings[0].bin_centers[1:-1]), len(eff[1:-1]), len(1/efferr[1:-1]))
        
        spline_l1q_eff = UnivariateSpline(get_bin_center(hist2d_qz["Be"].binnings[0].edges[1:-1]), eff[1:-1], w=1/efferr[1:-1], k=3, s=3)
        save_spline_to_file(spline_l1q_eff, args.resultdir, "spline_l1q_eff.pickle")
        savefig_tofile(fig, args.resultdir, f"l1q_upppercut_efficiency", show=True)
        
    plt.show()
                        
if __name__ == "__main__":
    main()
    
