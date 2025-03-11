import multiprocessing as mp
from collections import Counter
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from tools.roottree import read_tree
from tools.selections import *
from scipy.optimize import curve_fit
from tools.binnings_collection import fbinning_RichNumberHits, fbinning_RichAcceptance, fbinning_charge, fbinning_BetaConsistency, fbinning_RichProb, fbinning_energy_per_neculeon, fbinning_tofRichBetaMatch, fbinning_fraction, fbinning_Flatness, fbinning_RICHindex, fbinning_RICHAngle, fbinning_RICHlik, fbinning_RICHnpe, LithiumRichAglBetaResolutionBinning, LithiumRigidityBinningRICH
from tools.calculator import calc_mass, calc_kinetic_eng_nucleon
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center, get_binindices
from iminuit.cost import ExtendedBinnedNLL, LeastSquares
from iminuit import Minuit
import uproot
from tools.binnings import Binning
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID 
from tools.calculator import calculate_efficiency_and_error
from tools.studybeta import hist1d, hist_beta, getbeta, hist_betabias

energy_binning = fbinning_energy_per_neculeon()
binning_rig = LithiumRigidityBinningRICH()
binning_energy = fbinning_energy_per_neculeon()
binning_beta = LithiumRichAglBetaResolutionBinning()
binning_x = binning_beta
binning_resolution = np.linspace(-0.005, 0.005, 120)
qsel = 6.0
riglim = 200

def handle_file(arg):
    
    filename, treename, dataname, chunk_size,  rank, nranks, kwargs = arg
    resultdir = kwargs["resultdir"]
    qsel = kwargs["qsel"]
    
    hist1d_rig_before = np.zeros(len(binning_rig) - 1)
    hist1d_before_lip = np.zeros(len(binning_x) - 1)
    hist1d_before_ciemat = np.zeros(len(binning_x) - 1)
    hist1d_after_lip = np.zeros(len(binning_x) - 1)
    hist1d_after_ciemat = np.zeros(len(binning_x) - 1)
    histbetabias_lip = np.zeros(len(binning_resolution) - 1)
    histbetabias_ciemat = np.zeros(len(binning_resolution) - 1) 
    hist1d_rig_lip = np.zeros(len(binning_rig) - 1)
    hist1d_rig_ciemat = np.zeros(len(binning_rig) - 1)
    
    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):

        #selection except for the variable
        event_num = ak.to_numpy(events['event'])                                                                                                                                                           
        print(event_num)
        events = cut_background_reduce(events)
        events = CutTrackInRichAglAcceptance_LIP(events)
        events = CutTrackInRichAglAcceptance(events)
        events_lip = selector_agl_lipvar(events, qsel)
        events_ciemat = selector_agl_ciematvar(events, qsel)        
        richp_beta_lip = ak.to_numpy(getbeta(events_lip, 1))
        richp_beta_ciemat = ak.to_numpy(getbeta(events_ciemat, 0))
        
        rig_before = ak.to_numpy(events["tk_rigidity1"][:, 0, 0, 1])
        rig_lip = ak.to_numpy(events_lip["tk_rigidity1"][:, 0, 0, 1])
        rig_ciemat = ak.to_numpy(events_ciemat["tk_rigidity1"][:, 0, 0, 1])
        hist1d_rig_before += hist1d(rig_before, binning_rig)
        hist1d_rig_lip += hist1d(rig_lip , binning_rig)
        hist1d_rig_ciemat += hist1d(rig_ciemat, binning_rig)
        #plot 1d hist of the variable
        hist1d_before_lip += hist_beta(events, binning_x, 1)
        hist1d_after_lip += hist_beta(events_lip, binning_x, 1)
        hist1d_before_ciemat += hist_beta(events,  binning_x, 0)
        hist1d_after_ciemat += hist_beta(events_ciemat, binning_x, 0)
       
        # histbetabias_lip += hist_betabias(events_lip, riglim, binning_resolution, 1)
        # histbetabias_ciemat += hist_betabias(events_ciemat, riglim, binning_resolution, 0)

    np.savez(os.path.join(resultdir, f"hist_{rank}.npz"), hist1d_before_lip=hist1d_before_lip, hist1d_before_ciemat=hist1d_before_ciemat, hist1d_after_lip=hist1d_after_lip, hist1d_after_ciemat=hist1d_after_ciemat,  histbetabias_lip=histbetabias_lip, histbetabias_ciemat=histbetabias_ciemat, hist1d_rig_lip=hist1d_rig_lip, hist1d_rig_ciemat=hist1d_rig_ciemat, hist1d_rig_before=hist1d_rig_before)
            
def make_args(filename, treename, dataname, chunk_size, nranks, **kwargs):
    for rank in range(nranks):
        yield (filename, treename, dataname, chunk_size, rank, nranks, kwargs)
        
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--filename", nargs="+", help="(e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--dataname", default="data", help="give a name to describe the dataset")
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--resultdir", default="results", help="Directory to store plots and result files in.")
    parser.add_argument("--qsel", default=6.0, type=float, help="Directory to store plots and result files in.")
    
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    
    with mp.Pool(args.nprocesses) as pool:
        pool_args = make_args(args.filename, args.treename, args.dataname, args.chunk_size, args.nprocesses, resultdir=args.resultdir, qsel=args.qsel)
        for _ in pool.imap_unordered(handle_file, pool_args):
            pass
    
    hist1d_before_lip = np.zeros(len(binning_x) - 1)
    hist1d_before_ciemat = np.zeros(len(binning_x) - 1)
    hist1d_after_lip = np.zeros(len(binning_x) - 1)
    hist1d_after_ciemat = np.zeros(len(binning_x) - 1)
    #histbetabias_lip = np.zeros(len(binning_resolution) - 1)
    #histbetabias_ciemat = np.zeros(len(binning_resolution) - 1) 
    hist1d_rig_before = np.zeros(len(binning_rig) - 1)
    hist1d_rig_lip = np.zeros(len(binning_rig) - 1)
    hist1d_rig_ciemat = np.zeros(len(binning_rig) - 1)

    for rank in range(args.nprocesses):
        filename = os.path.join(args.resultdir, f"hist_{rank}.npz")
        with np.load(filename) as result_file:
            
            hist1d_before_lip += result_file["hist1d_before_lip"]
            hist1d_after_lip += result_file["hist1d_after_lip"]
            hist1d_before_ciemat += result_file["hist1d_before_ciemat"]
            hist1d_after_ciemat += result_file["hist1d_after_ciemat"]
            #histbetabias_lip += result_file["histbetabias_lip"]
            #histbetabias_ciemat += result_file["histbetabias_ciemat"]
            hist1d_rig_lip += result_file["hist1d_rig_lip"]
            hist1d_rig_ciemat += result_file["hist1d_rig_ciemat"]
            hist1d_rig_before += result_file["hist1d_rig_before"]

    #print("rigsum:", np.sum(hist1d_rig_before), "lip:", np.sum(hist1d_rig_lip), np.sum(hist1d_rig_lip)/np.sum(hist1d_rig_before),  "ciemat:", np.sum(hist1d_rig_ciemat), np.sum(hist1d_rig_ciemat)/np.sum(hist1d_rig_before))

    eff_lip, efferr_lip = calculate_efficiency_and_error(hist1d_after_lip, hist1d_before_lip)
    eff_ciemat, efferr_ciemat = calculate_efficiency_and_error(hist1d_after_ciemat, hist1d_before_ciemat)
    np.savez(os.path.join(args.resultdir, "{}_histcounts.npz".format(args.dataname)), hist1d_before_lip=hist1d_before_lip, hist1d_after_lip=hist1d_after_lip, hist1d_after_ciemat=hist1d_after_ciemat)

    eff_riglip, efferr_riglip = calculate_efficiency_and_error(hist1d_rig_lip, hist1d_rig_before)
    eff_rigciemat, efferr_rigciemat = calculate_efficiency_and_error(hist1d_rig_ciemat, hist1d_rig_before)
    
    #set color for drawing
    color_before_lip = "black"
    color_before_cmat = "red"
    color_lip = "tab:green"
    color_cmat = "tab:orange"
    color_nuclei = "black"
    figsize = FIGSIZE_BIG
    legendfontsize = 35
    setplot_defaultstyle()

    figure = plt.figure(figsize=figsize)
    plot = figure.subplots(1, 1)
    plot1dhist(figure, plot, binning_x, hist1d_before_lip, np.sqrt(hist1d_before_lip), r"$\mathrm{\beta}$", "counts", "lip:before", color_before_lip, legendfontsize, 0, 0)
    plot1dhist(figure, plot, binning_x, hist1d_before_ciemat, np.sqrt(hist1d_before_ciemat), r"$\mathrm{\beta}$", "counts", "ciemat:before", color_before_cmat, legendfontsize, 0, 0)
    plot1dhist(figure, plot, binning_x, hist1d_after_lip, np.sqrt(hist1d_after_lip), r" ", "counts", "lip:after", color_lip, legendfontsize, 0, 0)
    plot1dhist(figure, plot, binning_x, hist1d_after_ciemat, np.sqrt(hist1d_after_ciemat), r"$\mathrm{\beta}$", "counts", "ciemat:after", color_cmat, legendfontsize, 0, 0)
    figure.savefig("plots/eff/hist_{}_numberevents_vs_beta.pdf".format(args.dataname), dpi=250)    

    figure = plt.figure(figsize=figsize)
    plot = figure.subplots(1, 1)
    plot1dhist(figure, plot, binning_x, eff_lip, efferr_lip, r"$\mathrm{\beta}$", "efficiency", "LIP",  color_lip, legendfontsize, 1, 0, 0, 0) 
    plot1dhist(figure, plot, binning_x, eff_ciemat, efferr_ciemat, r"$\mathrm{\beta}$", "efficiency", "CIEMAT",   color_cmat, legendfontsize, 1, 0, 0, 0) 
    figure.savefig("plots/eff/hist_{}_beta_efficiency.pdf".format(args.dataname), dpi=250) 

    '''
    figure = plt.figure(figsize=figsize)
    plot = figure.subplots(1, 1)
    plot1dhist(figure, plot, binning_resolution, histbetabias_lip, np.sqrt(histbetabias_lip), r"$\mathrm{\beta - 1}$", "counts", "lip",   color_lip, legendfontsize, 0, 0) 
    plot1dhist(figure, plot, binning_resolution, histbetabias_ciemat, np.sqrt(histbetabias_ciemat), r"$\mathrm{\beta - 1}$", "counts", "ciemat", color_cmat, legendfontsize, 0, 0) 
    figure.savefig("plots/eff/hist_{}_betabias.pdf".format(args.dataname), dpi=250)
    '''
    
    figure = plt.figure(figsize=figsize)
    plot = figure.subplots(1, 1)
    plot1dhist(figure, plot, binning_rig, hist1d_rig_before, np.sqrt(hist1d_rig_before), "rigidity (GV)", "counts", "Nuclei Selections", color_nuclei, legendfontsize, 1, 0, 0, 1)    
    plot1dhist(figure, plot, binning_rig, hist1d_rig_lip, np.sqrt(hist1d_rig_lip), "rigidity (GV)", "counts", "LIP", color_lip, legendfontsize, 1, 0, 0 ,1)
    plot1dhist(figure, plot, binning_rig, hist1d_rig_ciemat, np.sqrt(hist1d_rig_ciemat), "rigidity (GV)", "counts", "CIEMAT", color_cmat, legendfontsize, 1, 0, 0, 1)
    figure.savefig("plots/eff/hist_{}_rig.pdf".format(args.dataname), dpi=250)

    figure = plt.figure(figsize=figsize)
    plot = figure.subplots(1, 1)
    plot1dhist(figure, plot, binning_rig, hist1d_rig_before, np.sqrt(hist1d_rig_before), "Rigidity (GV)", "counts", "Nuclei Selections", color_nuclei, legendfontsize, 1, 1, 0, 0) 
    plot1dhist(figure, plot, binning_rig, hist1d_rig_lip, np.sqrt(hist1d_rig_lip), "Rigidity (GV)", "counts", "LIP", color_lip, legendfontsize, 1, 1, 0, 0)  
    plot1dhist(figure, plot, binning_rig, hist1d_rig_ciemat, np.sqrt(hist1d_rig_ciemat), "Rigidity (GV)", "counts", "CIEMAT", color_cmat, legendfontsize, 1, 1, 0, 0) 
    figure.savefig("plots/eff/hist_{}_rig_log.pdf".format(args.dataname), dpi=250)

    figure = plt.figure(figsize=figsize)
    plot = figure.subplots(1, 1)
    plot1dhist(figure, plot, binning_rig, eff_riglip, efferr_riglip, "Rigidity (GV)", "Efficiency", "LIP",  color_lip, legendfontsize, 1, 0, 0, 1)
    plot1dhist(figure, plot, binning_rig, eff_rigciemat, efferr_rigciemat, "Rigidity (GV)", "Efficiency", "CIEMAT",  color_cmat, legendfontsize, 1, 0, 0, 1)   
    figure.savefig("plots/eff/hist_{}_rig_efficiency.pdf".format(args.dataname), dpi=250) 
    
    plt.show()
    
        
if __name__ == "__main__":
    main()

'''
    figure = plt.figure(figsize=figsize)
    plot = figure.subplots(1, 1)
    plot2dhist(figure, plot, binning_x, binning_ringeff, hist2d_ringeff, label_x=r"rigidity(GV)", label_y="ring efficiency")
'''
