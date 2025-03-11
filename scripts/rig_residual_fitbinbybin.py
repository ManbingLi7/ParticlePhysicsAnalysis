from tools.MassFunction import expo_func
from tools.massfunction_Tof import TofInverseMassFunctionFit
from tools.massfunction_NaF import NaFInverseMassFunctionFit
from tools.massfunction_Agl import AglInverseMassFunctionFit

import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import tools.roottree as read_tree
from tools.calculator import calc_mass, calc_ekin_from_beta, calc_betafrommomentom, calc_ratio_err

import uproot
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, LeastSquares, NormalConstraint
from iminuit.util import describe, make_func_code
from tools.constants import ISOTOPES, NUCLEI_NUMBER, ISOTOPES_COLOR, ISO_LABELS
from tools.histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d

from scipy import interpolate
from tools.graphs import MGraph, plot_graph
import uncertainties
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle
from tools.plottools import FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, plot1d_step, FONTSIZE, set_plot_defaultstyle, FONTSIZE_BIG
import uncertainties    
from uncertainties import unumpy 
from tools.constants import DETECTOR_LABEL, DETECTOR_COLOR, ISOTOPES_MASS
from scipy.optimize import curve_fit
from tools.functions import gaussian, asy_gaussian, poly, asy_gaussian_1d, upoly
from tools.constants import expofunc_rig_sigmafactor_7to9, expofunc_rig_sigmafactor_7to10
from tools.massfit_tools import get_fitpdf_witherr, get_fitpdferrorband, get_unp_pars_binbybin, plot_fitmc_pars, plot_fitdata_pars, plot_fitmc_compare_isopars, plot_fitMCISO_pars, plot_fitmc_pars_betareso
from tools.statistics import row_mean_and_std
import scipy.integrate as integrate
from tools.jupytertools import *

def funcbe10(total_counts, be7, be9):
    ybe10 = total_counts - be7 - be9
    return ybe10

def unorm_gaussian(x, counts, mu, sigma):                                                                                           
    return counts * np.exp(-(x - mu)**2 /(2 * sigma**2))

def make_rig_function():
    def rig_resolution(x, mean, sigma, sigma_ratio, asy_factor, fraccore):
        coregaus = gaussian(x, mean, sigma)
        asygaus = asy_gaussian(x, mean,  sigma_ratio * sigma, asy_factor)
        pdf += norm * (fraccore * coregaus + (1 - fraccore) * asygaus)  * rigbinwidth
        return pdf
    parnames = ['x', 'mean', 'sigma', 'sigma_ratio', 'asy_factor', 'fraccore']
    rig_resolution.func_code = make_func_code(parnames)
    return rig_resolution


def rig_resolution_gaus(x, norm, mean, sigma, sigma_ratio, asy_factor, fraccore):
    coregaus = gaussian(x, mean, sigma)
    asygaus = asy_gaussian_1d(x, mean,  sigma_ratio * sigma, asy_factor)
    pdfgaus = norm * fraccore * coregaus
    pdf = norm * (fraccore * coregaus + (1 - fraccore) * asygaus)
    return pdfgaus

def rig_resolution_asygaus(x, norm, mean, sigma, sigma_ratio, asy_factor, fraccore):
    coregaus = gaussian(x, mean, sigma)
    asygaus = asy_gaussian_1d(x, mean,  sigma_ratio * sigma, asy_factor)   
    pdf = norm * ((1 - fraccore) * asygaus)
    return pdf


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


detectors = ["Tof"]

iso_ratio_guess = {"Be7": 0.6, "Be9": 0.3, "Be10": 0.1}
par_names = {"norm": 500, "mean": 0.02, "sigma": 0.06, "fraccore":0.87, "sigma_ratio":1.2, "asy_factor":1.4}


scale_factor = {"Be7": 1, "Be9": 7./9., "Be10": 7.0/10.0}
mean_scale_factor_fromMC = {"Be7": 1, "Be9": 0.77949789, "Be10": 0.70255866}


mcpars_initlims_inversebeta = {'Tof': {'norm': (0.000001, 100), 'mean': (-0.00001, 0.02), 'sigma': (0.0005, 0.1), 'fraccore': (0.85, 0.96), 'sigma_ratio':(1.0, 2.5), 'asy_factor':(0.9, 1.5)},
                               'NaF': {'norm': (1, 20), 'mean': (-0.02, 0.04), 'sigma': (0.0001, 0.01), 'fraccore': (0.3, 1.0), 'sigma_ratio':(1.0, 2.0), 'asy_factor':(0.4, 2.0)},
                               'Agl': {'norm': (1, 50), 'mean': (-0.002, 0.005), 'sigma': (0.00001, 0.005), 'fraccore': (0.5, 1.0), 'sigma_ratio':(0.8, 3.0), 'asy_factor':(0.8, 2.0)}}

#sigma_scale_factor_fromMC = {"Be7": 1, "Be9": 7./9., "Be10": 7./10.}
par_names_axes = {'mean': '$\mathrm{\mu}$',
                  'sigma': '$\mathrm{\sigma_{p}}$',
                  "sigma_ratio": '$\mathrm{ \epsilon(\sigma ratio)}$',
                  "asy_factor":'alpha',
                  "fraccore":'$\mathrm{f_{core}}$',
                  'norm':'Norm'}

poly_deg = {'mean':3, 'sigma':6, 'sigma_ratio':3, 'asy_factor':3, 'fraccore': 3, 'norm':6}

ylim_range = {'mean':        [0.08, 0.16],
              'sigma':       [0.01, 0.3],
              'sigma_ratio': [0.7, 2.5],
              'asy_factor' : [0.8, 2.0],
              'fraccore'   : [0.6, 1.0],
              "norm"       : [0, 40]}

ylim_range_be7 = {'Tof': {'mean'  :     [-0.03, 0.03], 'sigma' : [0.007, 0.012], 'sigma_ratio':[1.0, 6.0], 'asy_factor': [1.0, 1.7], 'fraccore':   [0.95, 1.0], "norm":       [0, 10000]},
                  'NaF': {'mean'  :     [-0.01, 0.01], 'sigma' : [0.001, 0.002], 'sigma_ratio':[1.0, 6.0], 'asy_factor': [1.0, 1.7], 'fraccore':   [0.95, 1.0], "norm":       [0, 10000]},
                  'Agl': {'mean'  :     [-0.003, 0.01], 'sigma' : [0.0003, 0.0007], 'sigma_ratio':[1.0, 6.0], 'asy_factor': [1.0, 1.7], 'fraccore':   [0.95, 1.0], "norm":       [0, 10000]}}

#detectors_energyrange = {"Tof": [0.4, 1.5], "NaF": [1.0, 3.4], "Agl": [3.8, 11]} #Agl: 4.0, 10
beta_range = {"Tof": [2, 1000], "NaF": [0.88, 0.986], "Agl": [0.97, 0.9999]}


def update_guess(df_probbeta_pars, guess, xbeta):
    for name in par_names.keys():
        guess[name] = np.poly1d(df_probbeta_pars[name])(np.log(xbeta))
    return guess


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    #parser.add_argument("--filename", default="trees/massfit/Be9iss_Agl_masshist.root",   help="Path to root file to read tree from")
    parser.add_argument("--filename_mc", default="/home/manbing/Documents/Data/data_BeP8/Be2DHist/BeMC_hist_Tof_RigResoRefGen2.npz",  help="Path to root file to read tree from")
    parser.add_argument("--plotdir", default="plots/unfold/rig", help="Directory to store plots and result files in.")
    parser.add_argument("--datadir", default="/home/manbing/Documents/Data/data_BeP8", help="Directory to store plots and result files in.")
    parser.add_argument("--nuclei", default="Be", help="Directory to store plots and result files in.")
    parser.add_argument("--isinverse", default=True, help="Directory to store plots and result files in.")
    #parser.add_argument("--detectors", nargs="+", default=["NaF", "Agl"], help="Directory to store plots and result files in.")
    args = parser.parse_args()
    os.makedirs(args.plotdir, exist_ok=True)

    nuclei = args.nuclei
    #get the TH2 histogram
    isotopes_atom_num = [NUCLEI_NUMBER[iso] for iso in ISOTOPES[args.nuclei]]
    #isotopes = ISOTOPES[args.nuclei]
    isotopes = ['Be7']

    fit_range = {"Tof": [-0.15, 0.15], "NaF":[-0.02, 0.005], "Agl": [-0.002, 0.001]}
    init_ms = {"Tof": [100, 0.003, 0.016], "NaF":[10, -0.001, 0.001], "Agl": [10, -0.0001, 0.0005]}
    initguess = {'Tof': dict(norm=50, mean=0.003, sigma=0.01, sigma_ratio=1.7, asy_factor=1.1, fraccore=0.89),
                 'NaF': dict(norm=10, mean=0.001, sigma=0.001, sigma_ratio=1.5, asy_factor=1.0, fraccore=0.8),
                 'Agl': dict(norm=10, mean=0.0001, sigma=0.0005, sigma_ratio=1.5, asy_factor=1.0, fraccore=0.8)}
    # get mc histogram 
    hist_resolution = {dec: {} for dec in detectors}
    df_probbeta_pars = {dec: {} for dec in detectors}
    #for dec in detectors:
        #for iso in ISOTOPES[nuclei]:
        #        df_probbeta_pars[dec][iso] = np.load(f'/home/manbing/Documents/lithiumanalysis/scripts/plots/unfold/rig/{dec}{iso}_polypar_inversebetav2.npz')
    

    with np.load(args.filename_mc) as mc_file:
        for dec in detectors:
            for iso in isotopes:
                hist_resolution[dec][iso] = WeightedHistogram.from_file(mc_file, f"hist_rig_residual_Tof{iso}_vsR")

                fig = plt.figure(figsize=(20, 15))
                plot = fig.subplots(1, 1)
                #plot2dhist(fig, plot, xbinning=hist_resolution[dec][iso].binnings[0].edges[1:-1],
                #           ybinning=hist_resolution[dec][iso].binnings[1].edges[1:-1],
                #           counts=hist_resolution[dec][iso].values[1:-1, 1:-1],
                #           xlabel=None, ylabel=None, zlabel="counts", zmin=None, zmax=None,
                #           setlogx=False, setlogy=False, setscilabelx=True, setscilabely=True,  setlogz=True)
                plot.text(0.05, 0.98, f"{dec}_{iso}_betareso", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", fontweight="bold")
                plot.set_ylabel(r"R Reidual", fontsize=30)
                plot.set_xlabel("R_gen", fontsize=30)
                #savefig_tofile(fig, args.plotdir, f"hist2d_betareso_{dec}_{iso}", show=False)

                # fit bin by bin
                beta_binning = hist_resolution[dec][iso].binnings[0]
                beta_residual_binning = hist_resolution[dec][iso].binnings[1]
                xbincenters = beta_binning.bin_centers[1:-1]

                binrange = beta_binning.get_indices(beta_range[dec])
                xvalue_beta = beta_binning.bin_centers[binrange[0]: binrange[1]]
                dict_pars = {par: MGraph(xvalue_beta, np.zeros_like(xvalue_beta), yerrs=np.zeros_like(xvalue_beta), labels=["Beta", f"{par}"]) for par in par_names}
                print('binrange:', binrange)
                
                chisquare = np.zeros_like(xvalue_beta)


                for ip, ibin in enumerate(range(binrange[0], binrange[1])):
                    print(ibin, ip, beta_binning.bin_centers[ibin])
                    ibinedge = [hist_resolution[dec][iso].binnings[0].edges[ibin], hist_resolution[dec][iso].binnings[0].edges[ibin+1]]
                    
                    counts = hist_resolution[dec][iso].values[ibin, :]
                    hist_reso_ibin = hist_resolution[dec][iso].project(ibin)
                    guess = initguess[dec]

                    mean_iter0 = np.sum(hist_reso_ibin.binnings[0].bin_centers[1:-1] * hist_reso_ibin.values[1:-1])/np.sum(hist_reso_ibin.values[1:-1])
                    std_iter0 = np.sqrt(((hist_reso_ibin.binnings[0].bin_centers[1:-1] - mean_iter0)**2)*hist_reso_ibin.values[1:-1]).sum(axis=0)/np.sum(hist_reso_ibin.values[1:-1])
                    init_iter0 = [np.sum(hist_reso_ibin.values), mean_iter0,  std_iter0]
                    
                    popt, pcov = curve_fit(unorm_gaussian, hist_reso_ibin.binnings[0].bin_centers[1:-1], hist_reso_ibin.values[1:-1], p0 = init_iter0)

                    if ibin < 54:
                        guess['norm'] = 1
                        guess['mean'] = 0.001
                        guess['sigma'] = abs(popt[2])
                    else:
                        guess['norm'] = 0.1
                        guess['mean'] = 0.0001
                        guess['sigma'] = abs(popt[2])

                    #guess_update = update_guess(df_probbeta_pars[dec][iso], guess, beta_binning.bin_centers[ibin])
                    guess_update = guess
                    print('guess:', guess_update)
                    
                    fit_residual_range = hist_reso_ibin.binnings[0].get_indices([popt[1] -abs(4 * popt[2]), popt[1] + abs(4 *popt[2])])
                    
                    
                    
                    

                    print(guess_update)
                    #print('fit_residual_range:', fit_residual_range)
                    
                    fity = hist_reso_ibin.values[fit_residual_range[0]: fit_residual_range[1]]
                    
                    #fity[fity==0] = 1
                    fitbinedges = hist_reso_ibin.binnings[0].edges[fit_residual_range[0]: fit_residual_range[1]+1]
                    fitbincenters = hist_reso_ibin.binnings[0].bin_centers[fit_residual_range[0]: fit_residual_range[1]]
                    loss = ExtendedBinnedNLL(fity, fitbinedges, cumulative_rig_resolution)

                    m = Minuit(loss, **guess_update)
                    #for par in dict_pars.keys():
                    #    if args.isinverse:
                    #        m.limits[par]=mcpars_initlims_inversebeta[dec][par]
                    #    else:
                    #        m.limits[par]=mcpars_initlims[dec][par]

                    #m.limits['norm']=mcpars_initlims_inversebeta[dec]['norm']
                    m.limits['mean']=mcpars_initlims_inversebeta[dec]['mean']
                    m.limits['sigma']= [guess_update['sigma'] * 0.95, guess_update['sigma'] * 1.1]
                    m.limits['sigma_ratio']=mcpars_initlims_inversebeta[dec]['sigma_ratio']
                    m.limits['asy_factor']=mcpars_initlims_inversebeta[dec]['asy_factor']
                    #m.limits['fraccore']=mcpars_initlims_inversebeta[dec]['fraccore']
                    guess_update['fraccore']= 0.852
                    
                    m.fixed['fraccore']=True
                    m.migrad()
                    print(m)
                    for par in dict_pars.keys():
                        #dict_pars[par].xvalues[ip] = hist_reso_ibin.binnings[0].bin_centers[ibin]
                        
                        dict_pars[par].yvalues[ip] = m.values[par]
                        dict_pars[par].yerrs[ip] = m.errors[par]
                        
                    xvalues = hist_reso_ibin.binnings[0].edges[1:-1]
                    fit_results = rig_resolution(fitbincenters, *m.values)
                    fit_results_gaus = rig_resolution_gaus(fitbincenters, *m.values)
                    fit_results_asygaus = rig_resolution_asygaus(fitbincenters, *m.values)

                    
                    #print(*m.values)
                    #int_result = integrate.quad(rig_resolution, beta_binning.edges[ibin], beta_binning.edges[ibin+1], args=(m.values['norm'], m.values['mean'], m.values['sigma'], m.values['sigma_ratio'], m.values['asy_factor'], m.values['fraccore']))
                    #print(f'integrate in {ibin}', int_result)
                    
                    fit_guess = rig_resolution(fitbincenters, **guess_update)
                    
                    figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14)) 
                    ax1.errorbar(fitbincenters, fity, np.sqrt(fity), fmt=".", markersize=18, label='data', color='black')   
                    #plot_histogram_1d(ax1, hist_reso_ibin, style="iss", color="black", label="data", scale=None, gamma=None, xlog=False, ylog=False, shade_errors=False)
                    ax1.plot(fitbincenters, fit_results, "-", color="tab:green", label="fit")
                    ax1.plot(fitbincenters, fit_results_gaus, "--", color="tab:blue", label="fit")
                    ax1.plot(fitbincenters, fit_results_asygaus, "--", color="tab:red", label="fit")
                    #ax1.plot(fitbincenters, fit_guess, "-", color="tab:blue", label="guess")
                    #plot1d_errorbar(figure, ax1, hist_, counts, err=countserr, label_x="1/rig_reso (1/GeV)", label_y="counts")
                    #plot1d_step(figure, ax1,  rig_resobinedges[min_rig_resobin: max_rig_resobin + 2], rig_reso_function_fit[i], err=None, label_x="1/rig_reso (1/GeV)", label_y="counts", col="tab:orange")

                    fity[fity==0] = 1
                    pull =(fity - fit_results)/np.sqrt(fity)

                    handles = []
                    labels = []
                    chisquare[ip] = np.sum(pull**2)/(len(pull))
                    #fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {chisquare:.1f}",]
                    fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {chisquare[ip]:.1f}",
                                f"$\\mu$ ={m.values['mean']:.4f}$\\pm$ {m.errors['mean']:.4f}",
                                f"$\\sigma$ ={m.values['sigma']:.4f}$\\pm$ {m.errors['sigma']:.4f}",
                                f"$f_{{c}}$ ={m.values['fraccore']:.3f}$\\pm$ {m.errors['fraccore']:.3f}",
                                f"$\\epsilon$ ={m.values['sigma_ratio']:.3f}$\\pm$ {m.errors['sigma_ratio']:.3f}",
                                f"$\\alpha$ ={m.values['asy_factor']:.3f}$\\pm$ {m.errors['asy_factor']:.3f}",]
                    ax1.legend(labels, handles, title="\n".join(fit_info), frameon=False, title_fontsize=28)               
                    
                    fit_info_formatted = [rf"\\fontsize{FONTSIZE}\\selectfont {info}" for info in fit_info]

            
                    plot1d_errorbar(figure, ax2, fitbinedges, counts=pull, err=np.zeros(len(pull)),  label_x="1/rig_reso (1/GeV)", label_y="pull", legend=None,  col="black", setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
                    plt.subplots_adjust(hspace=.0)                             

                    ax1.set_ylim([0.1, 10*max(fity)])
                    ax1.set_yscale('log')
                    
                    ax1.set_xlim([m.values['mean'] - 7.0 * m.values['sigma'], m.values['mean'] + 6.8 * m.values['sigma']])
                    #else:
                    #    ax1.set_xlim([m.values['mean'] - 2.0 * m.values['sigma'], m.values['mean'] + 2.0 * m.values['sigma']])
                        
                    ax1.text(0.43, 0.98, f"R = [{ibinedge[0]:.2f}, {ibinedge[1]:.2f}] GV", fontsize=FONTSIZE_BIG, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color="black", fontweight="bold")
                    set_plot_style(ax1)
                    
                    ax2.sharex(ax1)
                    ax2.set_ylim([-2.5, 2.5])
                    set_plot_style(ax2)
                    
                    if args.isinverse:
                        savefig_tofile(figure, args.plotdir, f"fit_inverseRig_residual_{dec}{iso}_{ibin}_log2", show=False)
                    else:
                        savefig_tofile(figure, args.plotdir, f"fit_beta_residual_{dec}{iso}_{ibin}_log2", show=False)
                        

                pars_poly = dict()
                polyfit = dict()
                
                deg = {'norm': 8, 'mean': 5, 'sigma':8, 'sigma_ratio':3, 'asy_factor':3, 'fraccore':4}
                graph_rigfit = {}
                for par in dict_pars.keys():
                    dict_pars[par].add_to_file(graph_rigfit, f'graph_{par}_rig')
                    fig = plt.figure(figsize=(20, 15))
                    plot = fig.subplots()
                    print(par)
                    print(dict_pars[par])
                    
                    plot_graph(figure, plot, dict_pars[par], color="tab:orange", label=None, style="EP", xlog=False, ylog=False, scale=None, markersize=25)
                    pars_poly[par] = np.polyfit(np.log(dict_pars[par].getx()), dict_pars[par].gety(), deg=deg[par])
                    polyfit[par] = np.poly1d(pars_poly[par])(np.log(dict_pars[par].getx()))
                    plot.plot(dict_pars[par].getx(), polyfit[par], "-", color="tab:blue", label="fit")
                    plot.set_xscale('log')
                    plot.set_yscale('log')
                    
                    if args.isinverse:
                        savefig_tofile(fig, args.plotdir, f"{dec}{iso}_{par}_inversebeta", show=False)
                    else:
                        savefig_tofile(fig, args.plotdir, f"{dec}{iso}_{par}_beta", show=False)
                print(pars_poly)
                if args.isinverse:
                    np.savez(os.path.join(args.plotdir, f'{dec}{iso}_polypar_inversebeta.npz'), **pars_poly)
                else:
                    np.savez(os.path.join(args.plotdir, f'{dec}{iso}_polypar_beta.npz'), **pars_poly)
                np.savez(os.path.join(args.plotdir, f'graph_rigfitpar.npz'), **graph_rigfit)
                    
        # end fitting bin by bin

    
                


if __name__ == "__main__":
    main()


