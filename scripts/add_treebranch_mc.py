import multiprocessing as mp
from array import array
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from tools.roottree import read_tree
from tools.selections import *
from tools.constants import MC_PARTICLE_CHARGES, MC_PARTICLE_IDS 
from tools.binnings_collection import mass_binning, kinetic_energy_neculeon_binning
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.calculator import calc_mass, calc_ekin_from_beta, calc_betafrommomentom
from collections.abc import MutableMapping
from tools.corrections import shift_correction
import uproot
from scipy import interpolate
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, get_corrected_lipbeta_naf, get_index_correction_naf, get_corrected_lipbeta_agl, get_index_correction_agl 
import ROOT
from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F
from ROOT import gROOT, gBenchmark, gRandom, gSystem

uproot.default_library

mass_binning = mass_binning()
inverse_mass_binning = np.linspace(0.05, 0.3, 100)
energy_per_neculeon_binning = kinetic_energy_neculeon_binning()
bin1_num = len(energy_per_neculeon_binning) -1
bin2_num = len(inverse_mass_binning) - 1
qsel = 3.0

particleID = {"Li6Mc": MC_PARTICLE_IDS["Li6"], "Li7Mc": MC_PARTICLE_IDS["Li7"]}
var_rigidity = "tk_rigidity1"
#isotope = {"Li": "Li6", "Be":"Be7", "Bo": "Bo10"}

def addleaves(filename, outputfile, new_var1, new_var2, leavesvalues, leavesvalues2, newtreename):
    rootfile = TFile.Open(filename, "READ")                                                                                                                                                     
    tree = rootfile.Get("amstreea")   
    events = tree.GetEntries()

    outputfile.cd()
    newtree = tree.CloneTree(0)
    newtree.SetName(newtreename)
    myvar = array('d', [0])
    myvar2 = array('d', [0])
    newtree.Branch(f'{new_var1}', myvar, 'myvar/D')
    newtree.Branch(f'{new_var2}', myvar2, 'myvar2/D')
    for i in range(events):
        #returnValue = tree.GetEntry(i, 1) #1 get all branches, 0 only active branches
        tree.GetEntry(i)
        myvar[0] = leavesvalues[i]
        myvar2[0] = leavesvalues2[i]
        newtree.Fill()
        
        if i % 10000 == 0:
            print("%s of %s" % (i,events))
    print("Saved tree with %s events . . ." % (events))
    outputfile.Write()

def write_corrected_beta(tree):
    richp_beta_cor = np.full(tree.GetEntries(), -1.0, dtype=float)
    rich_beta_cor = np.full(tree.GetEntries(), -1.0, dtype=float)
    for i, entry in enumerate(tree):
            lipbeta = entry.rich_betap
            cmatbeta = entry.rich_beta[0]
            richp_beta_cor[i] = lipbeta
            rich_beta_cor[i] = cmatbeta
    return rich_beta_cor, richp_beta_cor


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--filename", default="", help="Path to root file to read tree from")
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file.")
    parser.add_argument("--resultdir", default="trees/MC", help="Directory to store plots and result files in.")
    parser.add_argument("--nuclei", default="Li", help="give an isotope for determine the geomagnetic cut off")
    parser.add_argument("--isotope", default="Be7", required=True, help="give an isotope for determine the geomagnetic cut off")

    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    nuclei = args.nuclei
    isotope = args.isotope
    newfile = TFile(f"{args.resultdir}/{isotope}{args.dataname}_BetaCor.root","RECREATE")
    
    filename_lip = args.filename
    rootfile_lip = TFile.Open(filename_lip, "READ")
    tree_lip = rootfile_lip.Get("amstreea")    
    rich_beta_cor, richp_beta_cor = write_corrected_beta(tree_lip)    
    addleaves(filename_lip, newfile, "rich_beta_cor", "richp_beta_cor", rich_beta_cor, richp_beta_cor, "amstreea")
    newfile.cd()

if __name__ == "__main__":
    main()


     
            

