import multiprocessing as mp
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from tools.roottree import read_tree
from tools.selections import *
from tools.constants import MC_PARTICLE_CHARGES, NUCLEI_CHARGE, ISOTOPES, ISOTOPES_MASSES, ISOTOPES_COLOR
from tools.constants import NAF_FractionPE_LIM, AGL_FractionPE_LIM, NAF_NPMT_LIM, AGL_NPMT_LIM, NAF_BETACONSIS_LIM, TOFNAF_CONSIS_LIM, TOFAGL_CONSIS_LIM, RICHAGL_INNER_LIM, RICHAGL_RADIUS_UPPLIM
from tools.binnings_collection import mass_binning, fbinning_energy, BeRigidityBinningRICHRange
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.calculator import calc_mass, calc_ekin_from_beta, calc_betafrommomentom
from collections.abc import MutableMapping
from tools.corrections import shift_correction
import uproot
from scipy import interpolate
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, get_corrected_lipbeta_agl, get_index_correction_agl
import ROOT
from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F
from ROOT import gROOT, gBenchmark, gRandom, gSystem

uproot.default_library

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


energy_per_neculeon_binning = fbinning_energy()
bin_num_ekin = len(energy_per_neculeon_binning) - 1
rigidity_binning = BeRigidityBinningRICHRange()
bin_num_rig = len(rigidity_binning) -1

particleID = {"Li6Mc": MC_PARTICLE_IDS["Li6"], "Li7Mc": MC_PARTICLE_IDS["Li7"], "Be7Mc": MC_PARTICLE_IDS["Be7"]}
var_rigidity = "tk_rigidity1"


def sel_denominator_naf(entry, nuclei):
    is_good = True
    is_good = entry.ntrack == 1
    is_good = is_good and entry.irich >= 0
    is_good = is_good and entry.rich_NaF == 1
    is_good = is_good and entry.is_ub_l1 == True
    is_good = is_good and (abs(entry.rich_pos[0]) < 17.0) and abs(entry.rich_pos[1]) < 17.0
    return is_good

def sel_numerator_naf(entry, nuclei):
    is_good = sel_denominator_naf(entry, nuclei)
    is_good = is_good and entry.rich_good and entry.rich_clean
    is_good = is_good and entry.rich_npe[0]/entry.rich_npe[2] > NAF_FractionPE_LIM
    is_good = is_good and entry.rich_pmt > NAF_NPMT_LIM
    is_good = is_good and np.sqrt(entry.rich_q[0]) > NUCLEI_CHARGE[nuclei] - 1.5
    is_good = is_good and np.sqrt(entry.rich_q[0]) < NUCLEI_CHARGE[nuclei] + 2.0
    is_good = is_good and entry.rich_pb >= 0.02
    return is_good

def sel_denominator_agl(entry, nuclei):
    xpos = entry.rich_pos[0]
    ypos = entry.rich_pos[1] 
    is_good = True
    is_good = entry.ntrack == 1
    is_good = is_good and entry.irich >= 0
    is_good = is_good and entry.rich_NaF == 0
    is_good = is_good and entry.is_ub_l1 == True
    is_good = is_good and (abs(xpos) > RICHAGL_INNER_LIM or abs(ypos) > RICHAGL_INNER_LIM)
    is_good = (xpos * xpos + ypos * ypos) < RICHAGL_RADIUS_UPPLIM
    return is_good

def sel_numerator_agl(entry, nuclei):
    is_good = sel_denominator_naf(entry, nuclei)
    is_good = is_good and entry.rich_good and entry.rich_clean
    is_good = is_good and entry.rich_npe[0]/entry.rich_npe[2] > AGL_FractionPE_LIM
    is_good = is_good and entry.rich_pmt > AGL_NPMT_LIM
    is_good = is_good and np.sqrt(entry.rich_q[0]) > NUCLEI_CHARGE[nuclei] - 1.5
    is_good = is_good and np.sqrt(entry.rich_q[0]) < NUCLEI_CHARGE[nuclei] + 2.0
    is_good = is_good and entry.rich_pb >= 0.02
    return is_good



detectors = ["NaF", "Agl"]
sel_denominator = {"NaF": sel_denominator_naf, "Agl": sel_denominator_agl}
sel_numerator = {"NaF": sel_numerator_naf, "Agl": sel_numerator_agl}
               
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="ISS", help="title of the input data (e.g. ISS)")
    parser.add_argument("--filename", default="tree_results/LiISS_RichAgl_0.root",   help="Path to root file to read tree from")
    parser.add_argument("--treename", default="amstreea", help="Name of the tree in the root file.")
    parser.add_argument("--resultdir", default="results", help="Directory to store plots and result files in.")
    parser.add_argument("--nuclei", default="Be", help="Directory to store plots and result files in.")
    args = parser.parse_args()
    os.makedirs(args.resultdir, exist_ok=True)
    nuclei = args.nuclei

    isotopes = [f"{nuclei}ISS"] + [iso for iso in ISOTOPES[nuclei]]
    print(isotopes)
    
    filename_mc = {"BeISS": "/home/manbing/Documents/Data/data_iss/BeISS_NucleiSelection_Clean.root",
                   "Be7": "/home/manbing/Documents/Data/data_mc/Be7MC_BetaCor.root",
                   "Be9": "/home/manbing/Documents/Data/data_mc/Be9MC_BetaCor.root",
                   "Be10": "/home/manbing/Documents/Data/data_mc/Be10MC_BetaCor.root"}        


    hist_den_rig  = {iso: dict() for iso in ISOTOPES[nuclei]}
    hist_num_rig  = {iso: dict() for iso in ISOTOPES[nuclei]}
    
    for iso in ISOTOPES[nuclei]:
        for dec in detectors:
            hist_den_rig[iso][dec] = TH1F(f"hist_event_den_rig_{iso}{dec}", f"hist_event_den_rig_{iso}{dec}", bin_num_rig, rigidity_binning)
            hist_num_rig[iso][dec] = TH1F(f"hist_event_num_rig_{iso}{dec}", f"hist_event_num_rig_{iso}{dec}", bin_num_rig, rigidity_binning)
            
            rootfile = TFile.Open(filename_mc[iso], "READ")
            tree = rootfile.Get(args.treename)
            for entry in tree:
                weight = entry.ww
                rigidity = entry.tk_rigidity1[2]
                if sel_denominator[dec](entry, nuclei):
                    hist_den_rig[iso][dec].Fill(rigidity, weight)
                    if sel_numerator[dec](entry, nuclei):
                        hist_num_rig[iso][dec].Fill(rigidity, weight)
        
            rootfile.Close()

    file_hist = TFile("hist_number_events.root", "RECREATE")
    for iso in ISOTOPES[nuclei]:
        for dec in detectors:
            hist_den_rig[iso][dec].Write()
            hist_num_rig[iso][dec].Write()

            
    
    
if __name__ == "__main__":
    main()


     
            

