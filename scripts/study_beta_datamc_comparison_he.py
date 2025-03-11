import multiprocessing as mp
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from tools.roottree import read_tree
from tools.selections import *
import scipy.stats
from scipy.optimize import curve_fit
from tools.studybeta import hist1d, hist2d, hist_beta, getbeta, hist_betabias, compute_moment                                                                                                              
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, FIGSIZE_MID, FIGSIZE_BIG, setplot_defaultstyle, format_order_of_magnitude, FONTSIZE
from tools.studybeta import calc_signal_fraction, get_corrected_lipbeta, get_corrected_lipbeta_mc, hist1d, hist1d_weighted
from tools.binnings_collection import fbinning_fraction, fbinning_RICHnpe
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, cdf_double_gaus, double_gaus

binning_npe = fbinning_RICHnpe()
beta_resolution_binning = np.linspace(-0.004, 0.004, 150)
figsize=FIGSIZE_MID
fontsize=FONTSIZE
FONTSIZE_MID = 28
setplot_defaultstyle()

def as_si(x, ndp):
    s = '{x:.{ndp:d}e}'.format(x=x, ndp=ndp)
    print(s)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))

def handle_file(arg):
    
    filename_iss, filename_mc, treename, chunk_size,  rank, nranks, kwargs = arg
    resultdir = kwargs["resultdir"]
    qsel = kwargs["qsel"]
    var_beta = "rich_betap"
    bin_content_iss = np.zeros(len(beta_resolution_binning) -1)
    bin_content_mc = np.zeros(len(beta_resolution_binning) -1)

    hist1d_npe_iss = np.zeros(len(binning_npe) -1 )
    hist1d_npe_mc = np.zeros(len(binning_npe) -1 )    
    riglim = 200
    print(qsel)

    for events in read_tree(filename_iss, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
        # create histogram of the charge values of this chunk of events:
        #events = selector_agl_lipvar_forHe(events, qsel) 
        events_iss = CutHighRig(events, riglim)
        richp_npe_iss = ak.to_numpy(events_iss["richp_npe"])
        rich_isNaF = ak.to_numpy(events_iss["richp_isNaF"])
        beta = ak.to_numpy(events_iss["rich_betap"])
        rich_position_lip = ak.to_numpy(events["richp_trackrec"])
        richx = rich_position_lip[:, 0]
        richy = rich_position_lip[:, 2]        
        beta_iss = get_corrected_lipbeta(beta, richx, richy, rich_isNaF)
        
        delta_beta = beta_iss -1.0
        delta_beta_raw = beta -1.0
        hist_values, hist_edges = np.histogram(delta_beta, bins= beta_resolution_binning)
        bin_edge = hist_edges
        bin_content_iss += hist_values
        hist1d_npe_iss += hist1d(richp_npe_iss, binning_npe)

    for events in read_tree(filename_mc, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
        #events = selector_agl_lipvar_forHe(events, qsel) 
        events_mc = CutHighRig(events, riglim)
        beta_mc = ak.to_numpy(events_mc["rich_betap"])
        richp_npe_mc = ak.to_numpy(events_mc["richp_npe"]) 
        rich_position_lip_mc = ak.to_numpy(events_mc["richp_trackrec"])
        richx_mc = rich_position_lip_mc[:, 0]
        richy_mc = rich_position_lip_mc[:, 2]
        beta_mc_corr = get_corrected_lipbeta_mc(beta_mc, richx_mc, richy_mc)
        
        weight = ak.to_numpy(events_mc["ww"])
#        re_weight = weight * 12205.0/2.6367194823905625e-06 
#        true_beta_mc = ak.to_numpy(events_mc[""])
        delta_beta_mc = beta_mc_corr - 1.0
        delta_beta_mc_raw = beta_mc - 1.0
        hist_values_mc, hist_edges_mc = np.histogram(delta_beta_mc_raw, bins= beta_resolution_binning, weights=weight)
        bin_edges_mc = hist_edges_mc
        bin_content_mc += hist_values_mc
        hist1d_npe_mc += hist1d_weighted(richp_npe_mc, binning_npe, weight) 

    np.savez(os.path.join(resultdir, f"results_{rank}.npz"), beta_resolution_binning=beta_resolution_binning, bin_content_iss=bin_content_iss, bin_content_mc=bin_content_mc, hist1d_npe_iss=hist1d_npe_iss, hist1d_npe_mc=hist1d_npe_mc)

def make_args(filename_iss, filename_mc,  treename, chunk_size, nranks, **kwargs):
    # this function creates the arguments for handle_file for each process
    for rank in range(nranks):
        yield (filename_iss, filename_mc,  treename, chunk_size, rank, nranks, kwargs)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--filename_iss", default="/home/manbing/Documents/lithiumanalysis/script/tree_results/HeISS_RICHAgl_V2.root", help="Path to root file to read tree from")
    parser.add_argument("--filename_mc", default="/home/manbing/Documents/lithiumanalysis/script/tree_results/He4MC_RICHAgl.root", help="Path to root file to read tree from")
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--resultdir", default="results", help="Directory to store plots and result files in.")
    parser.add_argument("--qsel", type=float, default=2.0,  help="charge of selected particle.")
    parser.add_argument("--dataname", default="ISS", help="dataname for the output file and plots.")
    
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    
    with mp.Pool(args.nprocesses) as pool:
        # create arguments for the individual processes
        pool_args = make_args(args.filename_iss, args.filename_mc, args.treename, args.chunk_size, args.nprocesses, resultdir=args.resultdir, qsel=args.qsel)
        # and execute handle_file for each process
        for _ in pool.imap_unordered(handle_file, pool_args):
            pass
     # now all processes are done, load the results and merge them                                                                   
    beta_iss = np.zeros(len(beta_resolution_binning) -1)
    beta_mc = np.zeros(len(beta_resolution_binning) -1)

    hist1d_npe_iss = np.zeros(len(binning_npe) -1 )
    hist1d_npe_mc = np.zeros(len(binning_npe) -1 )    

    for rank in range(args.nprocesses):
        filename = os.path.join(args.resultdir, f"results_{rank}.npz")
        with np.load(filename) as result_file:
            beta_iss += result_file["bin_content_iss"]
            beta_mc += result_file["bin_content_mc"]
            hist1d_npe_iss += result_file["hist1d_npe_iss"]
            hist1d_npe_mc += result_file["hist1d_npe_mc"]
            
    # now save merged result                
    integral = np.sum(beta_iss)* np.diff(beta_resolution_binning)[0]

    beta_mc = beta_mc / (np.sum(beta_mc * np.diff(beta_resolution_binning))) * (np.sum(beta_iss * np.diff(beta_resolution_binning)))
    hist1d_npe_mc = hist1d_npe_mc / (np.sum(hist1d_npe_mc * np.diff(binning_npe))) * (np.sum(hist1d_npe_iss * np.diff(binning_npe)))
    np.savez(os.path.join(args.resultdir, "{}_beta.npz".format(args.dataname)), beta_resolution_binning=beta_resolution_binning, beta_iss=beta_iss, beta_mc=beta_mc)

    var_bin_center = get_bin_center(beta_resolution_binning)
    guess = dict(counts=100, mu=0.00001, sigma=0.0005, sigma_ratio=2.0, fraction_sec=0.1)
    guess_mc = dict(counts=100, mu=0.0001, sigma=0.0005, sigma_ratio=2.0, fraction_sec=0.1)  
    par_iss, parerr_iss = minuitfit_LL(beta_iss, beta_resolution_binning, cdf_double_gaus, guess)
    par_mc, parerr_mc = minuitfit_LL(beta_mc, beta_resolution_binning, cdf_double_gaus, guess_mc)
    print("iss:", par_iss)
    print("iss err:", parerr_iss)
    print("mc:", par_mc)
    print("mc err:", parerr_mc)

    fit_y_iss = double_gaus(var_bin_center, *par_iss)
    fit_y_mc = double_gaus(var_bin_center, *par_mc)

    color_data = "green"
    color_mc = "tab:blue"
    figure = plt.figure(figsize=FIGSIZE_BIG)
    plot = figure.subplots(1, 1)
    figure.tight_layout()
    plt.plot(var_bin_center, fit_y_iss, '-', label=None, color=color_data)
    plt.plot(var_bin_center, fit_y_mc, '-', label=None, color="tab:blue") 
    plot1d_errorbar(figure, plot, beta_resolution_binning, beta_mc, np.sqrt(beta_mc), r"$\mathrm{\beta_{LIP} - 1}$", "counts", None, color_mc, ".", FONTSIZE,  0, 0, 1, 1, 0)
    plot1d_errorbar(figure, plot, beta_resolution_binning, beta_iss, np.sqrt(beta_iss), r"$\mathrm{\beta_{LIP} - 1}$", "counts", None, color_data, ".", FONTSIZE,  0, 0, 1, 1, 0)
    mu_iss = "{:.2e}".format(par_iss[1])
    print(mu_iss)
    
    #plot.text(0.05, 0.93, r"$\mathrm{\mu:}$" + format_order_of_magnitude(par_mc[1], 1) + r"$\pm$" + format_order_of_magnitude(parerr_mc[1], 1), fontsize=25, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color='tab:blue')
    #plot.text(0.05, 0.88, r"$\mathrm{\sigma:}$" + format_order_of_magnitude(par_mc[2], 1) + r"$\pm$" + format_order_of_magnitude(parerr_mc[2], 1), fontsize=25, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color='tab:blue')
    #plot.text(0.95, 0.9, r"$\mathrm{\mu=}$" + format_order_of_magnitude(par_iss[1]), fontsize=25, verticalalignment='top', horizontalalignment='right', transform=plot.transAxes, color='black')

    plot.text(0.05, 0.93, r"$\mathrm{\mu:}$" + r"$\mathrm{(1.11\pm0.01)\times10^{-4}}$", fontsize=FONTSIZE_MID, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color=color_mc, weight='normal')
    plot.text(0.05, 0.865, r"$\mathrm{\sigma:}$" + r"$\mathrm{(6.53\pm0.01)\times10^{-4}}$", fontsize=FONTSIZE_MID, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color=color_mc, weight='normal')
   # plot.text(0.05, 0.87, r"$\mathrm{\sigma:}$" + format_order_of_magnitude(par_mc[2], 1) + r"$\pm$" + format_order_of_magnitude(parerr_mc[2], 1), fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color=color_mc)
   
    plot.text(0.6, 0.93, r"$\mathrm{\mu:(-0.03\pm0.00)\times10^{-4}}$",  fontsize=FONTSIZE_MID, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color=color_data)
    plot.text(0.6, 0.865, r"$\mathrm{\sigma:(6.64\pm0.01)\times10^{-4}}$",  fontsize=FONTSIZE_MID, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color=color_data, weight='normal')
#    plot.text(0.6, 0.85, r"$\mathrm{\sigma:}$" + format_order_of_magnitude(par_iss[2], 1) + r"$\pm$" + format_order_of_magnitude(parerr_iss[2], 1), fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='left', transform=plot.transAxes, color='black')

    #plt.text(0.01, 0.28, r"$mu = {0:0.2e}$".format(par_iss[1]), size=20, fontsize=FONTSIZE, verticalalignment='top', horizontalalignment='right', transform=plot.transAxes, color='black')
    #plt.text(0.01, 0.23, r"$a = {0:s}$".format(as_si(mu_iss, 2)), size=20)
    plot.text(0.05, 0.985, r"$\mathrm{^{4}He}$ MC",
              verticalalignment='top', horizontalalignment='left',
              transform=plot.transAxes,
              color=color_mc, fontsize=FONTSIZE)
    
    plot.text(0.6, 0.985, r"$\mathrm{He}$ Data",
              verticalalignment='top', horizontalalignment='left',
              transform=plot.transAxes,
              color= color_data, fontsize=FONTSIZE)

    figure.savefig("plots/lipbeta/{}_betabias_athighR_agl.pdf".format(args.dataname), dpi=250)
    
    figure = plt.figure(figsize=FIGSIZE_MID)      
    plot = figure.subplots(1, 1)                                                                                                                                                                       
    plot1dhist(figure, plot, binning_npe, hist1d_npe_iss, np.sqrt(hist1d_npe_iss), r"$\mathrm{N_{pe}}$", "counts", "iss", "tab:orange", FONTSIZE,  0, 0)
    plot1dhist(figure, plot, binning_npe, hist1d_npe_mc, np.sqrt(hist1d_npe_mc), r"$\mathrm{N_{pe}}$", "counts", "mc", color_mc, FONTSIZE,  0, 0)                    
    figure.savefig("plots/lipbeta/hist_{}_npe.pdf".format(args.dataname), dpi=250)                                                    
    
    plt.show()

if __name__ == "__main__":
    main()
