import ROOT
import uproot
import uproot.behaviors.TGraph
import uproot3                                                                              
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center, compute_dayfromtime
from tools.binnings_collection import mass_binning, fbinning_energy, LithiumRigidityBinningFullRange
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar_v2, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, FONTSIZE_BIG, FONTSIZE_MID, plot_comparison_nphist, plot1d_errorbar, plot1d_step
from tools.calculator import calc_rig_from_ekin
from tools.constants import ISOTOPES_MASS, NUCLEI_CHARGE, NUCLEIS, ISOTOPES_COLOR
xbinning = fbinning_energy()
xbincenter = get_bin_center(xbinning)


def ratioerr(c, a, b, delta_a, delta_b):
  return c * np.sqrt((delta_a / a)**2 + (delta_b / b)**2)

isotopes = ["Be7", "Be9", "Be10"]
isotopesLow = {"Be7": "be7", "Be9":"be9", "Be10":"be10"}
detectors = ["Tof", "Agl", "NaF"]                                                                    
colordec = {"Agl": "tab:blue", "NaF": "tab:green", "Tof": "tab:red"}



def main():                                                                                 
    import argparse                                                                     
    parser = argparse.ArgumentParser()
    parser.add_argument("--filenameA", default="/home/manbing/Documents/Data/expo_time/run_list_85yr.root",help="Path to fils")
    parser.add_argument("--filetimeB", default="/home/manbing/Documents/Data/expo_time/run_list_10yr.root", help="Path to file measuring time")
    parser.add_argument("--resultdir", default="trees")
    
    args = parser.parse_args()                                                    
    os.makedirs(args.resultdir, exist_ok=True)

    #open file with measuring time
    histexpotime = dict()
    histexpotime_jiahui = dict()    
    with uproot.open("/home/manbing/Documents/Data/Be_fluxes/expotime/expo_time_sf_finebin_10yr.root") as meastime:
      hist_time_rig = meastime["h_expo_sf_1"]
      for isotope in isotopes:
        histexpotime[isotope] = meastime[f"h_expo_rig_1_{isotopesLow[isotope]}_bina7"]

    with uproot.open("/home/manbing/Documents/Data/Be_fluxes/expotime/expo_time_sf_finebin_10yr.root") as meastime_jiahui:
      hist_time_rig_jiahui = meastime_jiahui["h_expo_sf_1"]
      for isotope in isotopes:
        histexpotime_jiahui[isotope] = meastime_jiahui[f"h_expo_rig_1_{isotopesLow[isotope]}_bina7"]

    setplot_defaultstyle()
    print(max(hist_time_rig.values()))
    figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(21, 14))
    plot1d_errorbar(figure, ax1, hist_time_rig.axes[0].edges() , hist_time_rig.values(), err=hist_time_rig.errors(),
                    label_x="Rigidity (GV)", label_y="Measuring Time (s)", col="tab:orange", legend="this")
    plot1d_step(figure, ax1,  hist_time_rig.axes[0].edges() , hist_time_rig_jiahui.values(), err=hist_time_rig_jiahui.errors(),
                label_x="Rigidity (GV)", label_y="Exposure Time (s)", col="black", legend="J.W")
    pull = np.array(hist_time_rig.values()/ hist_time_rig_jiahui.values())
    #pull_err = ratioerr(pull, com, ref, com_err, ref_err)
    pull_err = np.zeros(len(pull))   
    plot1d_errorbar(figure, ax2, hist_time_rig.axes[0].edges(), counts=pull, err=pull_err,
                    label_x="Rigidity (GV)", label_y=r"$\mathrm{this/ref}$", legend=None,  col="black", setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
    plt.subplots_adjust(hspace=.0)                             
    ax1.legend()                                         
    ax2.set_ylim([0.95, 1.05])
    ax1.set_xscale("log")
    ax2.set_xscale("log")
    ax1.set_xlim([1.9, 1000])
    ax2.set_xlim([1.9, 1000])
    ax1.set_xticklabels([])
    ax1.text(0.65, 0.83, r"$\mathrm{T_{max} = (2.18)\times10^{8} s}$", fontsize=30, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')  
    savefig_tofile(figure, args.resultdir, f"measuring_time_rig_compare", 1)
    plt.show()
    
    '''        
    figure = plt.figure(figsize=FIGSIZE_BIG)                         
    plot = figure.subplots(1, 1)
    for isotope in isotopes:
        yexpotime, expotime_binedges = histexpotime[isotope].allnumpy()
        yexpotime_jiahui, expotime_binedges_jiahui = histexpotime_jiahui[isotope].allnumpy()    
        plot1dhist(figure, plot, expotime_binedges,  yexpotime, np.zeros(len(yexpotime)), "Ekin/n (GeV/n)", "Exposure Time(s)",  isotope, ISOTOPES_COLOR[isotope], FONTSIZE_MID, 0, 0)
        plot1dhist(figure, plot, expotime_binedges_jiahui,  yexpotime_jiahui, np.zeros(len(yexpotime_jiahui)), "Ekin/n (GeV/n)", "Exposure Time(s)",  isotope, "black", FONTSIZE_MID, 0, 0)
        ax1.text(0.05, 0.98, f"{detector}", fontsize=30, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", weight='bold')   
    savefig_tofile(figure, "trees/beflux/plots", f"time", 1)
'''
   
    

                                                                                                                                                                                                          
if __name__ == "__main__":                                                                                                                                                                                 
    main()



