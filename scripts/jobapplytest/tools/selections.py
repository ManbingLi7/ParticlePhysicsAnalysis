import json
import os

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from .calculator import calc_rig, calc_ekin_from_beta
from .constants import MC_PARTICLE_MASSES, MC_PARTICLE_CHARGES, MC_PARTICLE_IDS, CUTOFF_FACTOR, ISOTOPES, NUCLEI_CHARGE, ISOTOPES_MASS
from .binnings_collection import fbinning_beta, fbinning_energy, Rigidity_Analysis_Binning, Rigidity_Analysis_Binning_FullRange, fbinning_beta_rebin, fbinning_energy_rebin
from .binnings import Binning

NAF_FractionPE_LIM = 0.45
AGL_FractionPE_LIM = 0.4

NAF_NPMT_LIM = 10
AGL_NPMT_LIM = 2

NAF_BETACONSIS_LIM = 0.01
#AGL_BETACONSIS_LIM = 0.01

TOFNAF_CONSIS_LIM = 0.06
TOFAGL_CONSIS_LIM = 0.06

def BETA_BINNING(rebin=True):
    if rebin:
        return Binning(fbinning_beta_rebin())
    else:
        return Binning(fbinning_beta())

def IsWithin_RICHNaF(events):
    
    x_l8 = np.array(events.tk_pos[:, 7, 0])
    y_l8 = np.array(events.tk_pos[:, 7, 1])
    z_l8 = np.array(events.tk_pos[:, 7, 2])
    
    is_rich = z_l8 > -1000
    
    v_dir = ak.to_numpy((events.tk_dir)[:, 7, :])
    r_dir = np.linalg.norm(v_dir, axis=1)
    theta = np.arccos(v_dir[:, 2]/r_dir)   #azimuth angle
    phi = np.arctan2(v_dir[:, 1], v_dir[:, 0])  #polar angle
 
    alpha = np.sin(theta)* np.cos(phi)
    beta = np.sin(theta)* np.sin(phi)
    gamma = v_dir[:, 2]/r_dir
    
    t_naf = (-75.5-z_l8)/gamma;
    x_naf = x_l8 + t_naf*alpha;   
    y_naf = y_l8 + t_naf*beta;
    
    xpos = ak.to_numpy((events.rich_pos)[:, 0])                                                                                                                                                         
    ypos = ak.to_numpy((events.rich_pos)[:, 1])
    
    RichInnerEdge = 17.0
    is_within_naf = is_rich & (abs(x_naf) < RichInnerEdge) & (abs(y_naf) < RichInnerEdge)
    
    return events[is_within_naf]



def IsWithin_RICHAgl(events):
    
    x_l8 = events.tk_pos[:, 7, 0]
    y_l8 = events.tk_pos[:, 7, 1]
    z_l8 = events.tk_pos[:, 7, 2]
    
    is_rich = z_l8 > -1000
    
    v_dir = ak.to_numpy((events.tk_dir)[:, 7, :])
    r_dir = np.linalg.norm(v_dir, axis=1)

    
    theta = np.arccos(v_dir[:, 2]/r_dir)   #azimuth angle
    phi = np.arctan2(v_dir[:, 1], v_dir[:, 0])  #polar angle
 
    alpha = np.sin(theta)* np.cos(phi)
    beta = np.sin(theta)* np.sin(phi)
    gamma = v_dir[:, 2]/r_dir
    
    t_agl = (-74.6-z_l8)/gamma; 
    x_agl = x_l8 + t_agl*alpha;      
    y_agl = y_l8 + t_agl*beta;   

    xpos = ak.to_numpy((events.rich_pos)[:, 0])                                                                                                                                                         
    RichOuterRadius = 58.5
    RichInnerEdge = 19.0
    R2 = x_agl**2 + y_agl**2
    #is_bad = np.minimum(abs(x_agl), abs(y_agl)) >= 40.5
    #is_bad2 =  np.logical_and(np.maximum(abs(x_agl), abs(y_agl)) > 28.5, np.maximum(abs(x_agl), abs(y_agl)) < 29.5)
    selection = is_rich & (R2 < RichOuterRadius * RichOuterRadius) & ((abs(x_agl) > RichInnerEdge) | (abs(y_agl) > RichInnerEdge))
    return events[selection]

def SelectCleanEvent(events):          
    is_1trk = (events.ntrack == 1)       
    nhit2i = ak.to_numpy(np.zeros((len(events), 2), dtype=int))
    for ilay in range(1, 8):     
        for ixy in range(2):  
            mask = 1 << ilay   
            nhit2i[:, ixy] += ak.to_numpy((events.betah2hb[:, ixy] & mask) > 0)   
    is_clean2 = is_1trk | (nhit2i[:, 0] < 3) | (nhit2i[:, 1] < 5)       
    is_clean3 = is_clean2 | (events.betah2r/events.tof_betah < 0)
    is_clean4 = is_clean3 | (abs(events.betah2r) < 0.5)
    return events[is_clean4]

def SelectUnbiasL1(events):
    selection = events.is_ub_l1 == 1
    return events[selection]

def SelectUnbiasL1LowerCut(events, zsel):
    selection = events.tk_exqln[:, 0, 0, 2] > zsel - 0.4
    return events[selection]

def get_richbeta(events, is_cmat=True):  
    if (is_cmat): 
        beta = events.rich_beta_cor 
    else:  
        beta = events.richp_beta_cor 
    return beta  

def Select_HasLIPRing(events):
    selection = events.irichb == 0
    return events[selection]

def Select_HasCIEMATRing(events):
    selection = events.irich >= 0
    return events[selection]

def TofVelocity(events, upperlim = 1.245, lowerlim = 0.1):
    selection = (events.TofBeta <= upperlim) & (events.TofBeta >= lowerlim)
    return events[selection]

def CutRichbetaLimit_LIP(events, upperlim = 1.0, lowerlim = 0.1):
    selection = (events.rich_betap < upperlim) & (events.rich_betap > lowerlim)
    return events[selection]

def CutRichbetaLimit_CMAT(events, upperlim = 1.0, lowerlim = 0.1):
    beta =  get_richbeta(events, is_cmat=True) 
    selection = (beta < upperlim) & (beta > lowerlim)
    return events[selection]

def rich_beta_abovezero(events, lowerlim = 0.1):
    beta = events.rich_beta2[:, 0]
    selection = beta > lowerlim
    return events[selection]

def rich_beta_abovezero_LIP(events, lowerlim = 0.1):
    selection = events.rich_betap > lowerlim
    return events[selection]

def geomagnetic_IGRF_cutoff(events, factor=1.2):
    rigiditybinning = Binning(Rigidity_Analysis_Binning_FullRange())
    rigidity = (events.tk_rigidity1)[:, 0, 2, 1]
    binIndex = rigiditybinning.get_indices(rigidity)
    binlowedge = rigiditybinning.edges[binIndex]
    selection = binlowedge  > factor * (events.mcutoffi)[:, 1, 1]
    return events[selection]

def geomagnetic_cutoff_richlip(events, nuclei, ana_isotope):
    beta = events.richp_beta_cor 
    betabinning = Binning(fbinning_beta())       
    binIndex = betabinning.get_indices(beta)  
    binlowedge = betabinning.edges[binIndex] 
    bin_low_rig = calc_rig(binlowedge, MC_PARTICLE_MASSES[MC_PARTICLE_IDS[ISOTOPES[nuclei][0]]], NUCLEI_CHARGE[nuclei]) 
    factor = CUTOFF_FACTOR[ana_isotope]    
    rigiditycutoff = (events.mcutoffi)[:, 1, 1]      
    selection = bin_low_rig > factor*rigiditycutoff         
    return events[selection]

def geomagnetic_cutoff_richcmat(events, nuclei, ana_isotope, rebin):
    beta = events.rich_beta_cor
    betabinning = BETA_BINNING(rebin)
    binIndex = betabinning.get_indices(beta) 
    binlowedge = betabinning.edges[binIndex] 
    bin_low_rig = calc_rig(binlowedge, ISOTOPES_MASS["Be7"], NUCLEI_CHARGE[nuclei]) 
    factor = CUTOFF_FACTOR[ana_isotope]    
    rigiditycutoff = (events.mcutoffi)[:, 1, 1]      
    selection = bin_low_rig > factor*rigiditycutoff         
    return events[selection]


def geomagnetic_cutoff_tof(events, nuclei, ana_isotope, rebin):
    betabinning = BETA_BINNING(rebin)
    beta = events.tof_betah  
    binIndex = betabinning.get_indices(beta)  
    binlowedge = betabinning.edges[binIndex] 
    bin_low_rig = calc_rig(binlowedge, ISOTOPES_MASS["Be7"], NUCLEI_CHARGE[nuclei]) 
    factor = CUTOFF_FACTOR[ana_isotope]    
    rigiditycutoff = (events.mcutoffi)[:, 1, 1]      
    selection = bin_low_rig > factor*rigiditycutoff         
    return events[selection]

#begin defination of charge selections
def cut_charge_innertrk(events, qsel):
    rightside = qsel + 0.5
    leftside = qsel - 0.5
    selection = (events.tk_qin[:,1] > leftside) & (events.tk_qin[:, 1] < rightside)
    return events[selection]


def cut_charge_uppertof(events, qsel):
    rightside = qsel + 0.7
    leftside = qsel - 0.5
    if (events.tof_qs)[0] == 1:
        tof_q = events.tof_ql[0]
    else:
        tof_q = events.tof_ql[1]
    selection = (tof_q > leftside) & (tof_q < rightside)
    return events[selection]

def cut_charge_lowertof(events, qsel):
    lq = np.zeros(len(events))
    lowq1 = ak.to_numpy(events.tof_ql[:, 2])
    lowq2 = ak.to_numpy(events.tof_ql[:, 3])
    lq = (lowq1+ lowq2)/2
    selection =  (lowq1 > 0) & (lowq2 > 0) & (lq > (qsel -0.6))
    return events[selection]

def cut_charge_trkL1(events, qsel):
    leftside = q_sel - 0.46-(q_sel - 3)*0.16
    rightside = qsel + 0.65
    selection = (events.tk_l1qvs > leftside) & (events.tk_l1qvs < rightside)
    
    return events[selection]
        
def selector_charges(events, qsel):
#    events = cut_charge_innertrk(events, qsel)
#    events = cut_charge_uppertof(events, qsel)
    events = cut_charge_trkL1(events, qsel)
    return events

# end of charge selections, use the selector_charges

#begin defination of RICHAGL selection using LIP variables
def CutHighRig(events, lowerlim): 
    selection = (events.tk_rigidity1)[:, 0, 0, 1] > lowerlim 
                                  #[0]:choutko, [0] no refit [1] inner pattern
    return events[selection] 


def has_none_or_nan(events):
    
    return any(val is None or (isinstance(val, float) and math.isnan(val)) for val in arr)

###########################################################################
#To be modified
def CutTofEdgePaddles(events):
    isedge = (events['tof_barid'][:, 0] == 0) | (events['tof_barid'][:, 0] == 7)
    isedge = isedge | (events['tof_barid'][:, 1] == 0) | (events['tof_barid'][:, 1] == 7)
    isedge = isedge | (events['tof_barid'][:, 2] == 0) | (events['tof_barid'][:, 2] == 7) | (events['tof_barid'][:, 2] == 8) | (events['tof_barid'][:, 2] == 9)
    isedge = isedge | (events['tof_barid'][:, 3] == 0) | (events['tof_barid'][:, 3] == 7)
    return events[isedge==0]
############################################################################

def CutTrackInRichAglAcceptance(events):#constants of geomatric should be put in tools:constants
    xpos = ak.to_numpy((events.rich_pos)[:, 0])
    ypos = ak.to_numpy((events.rich_pos)[:, 1])
    RichInnerRadius = 24.04 
    RichOuterRadius = 58.5
    RichInnerEdge = 19.0
    R2 = xpos * xpos + ypos * ypos
    is_bad = np.minimum(abs(xpos), abs(ypos)) >= 40.5
    is_bad2 =  np.logical_and(np.maximum(abs(xpos), abs(ypos)) > 28.5, np.maximum(abs(xpos), abs(ypos)) < 29.5)
    selection = (R2 < RichOuterRadius * RichOuterRadius) & ((abs(xpos) > RichInnerEdge) | (abs(ypos) > RichInnerEdge)) & (np.logical_not(is_bad)) & (np.logical_not(is_bad2))
    return events[selection]

def CutTrackInRichNaFAcceptance(events):#constants of geomatric should be put in tools:constants
    xpos = (events.rich_pos)[:, 0] 
    ypos = (events.rich_pos)[:, 1]
    RichInnerEdge = 17.0
    selection = (abs(xpos) < RichInnerEdge) & (abs(ypos) < RichInnerEdge)
    return events[selection]

def CutTrackInRichAglAcceptance_LIP(events):#constants of geomatric should be put in tools:constants
    xpos = (events.richp_trackrec)[:, 0] 
    ypos = (events.richp_trackrec)[:, 2]
    RichInnerRadius = 24.04
    RichOuterRadius = 61.0
    RichInnerEdge = 19.0
    R2 = xpos * xpos + ypos * ypos
    selection = (R2 < RichOuterRadius * RichOuterRadius) & ((abs(xpos) > RichInnerEdge) | (abs(ypos) > RichInnerEdge))
    return events[selection]

def CutTrackInRichNaFAcceptance_LIP(events):#constants of geomatric should be put in tools:constants
    xpos = (events.richp_trackrec)[:, 0] 
    ypos = (events.richp_trackrec)[:, 2]
    RichInnerEdge = 17.0
    selection =  (abs(xpos) < RichInnerEdge) & (abs(ypos) < RichInnerEdge)
    return events[selection]

def CutRichProbobility_LIP(events):
    selection = events.richp_prob >= 0.05
    return events[selection]

def CutRichProbobility(events):
    selection = events.rich_pb >= 0.02
    return events[selection]

def CutIsNaF_LIP(events):
    selection = events.richp_isNaF == 1
    return events[selection]

def CutIsNaF(events):
    selection = events.rich_NaF == 1
    return events[selection]

def CutIsAgl_LIP(events):
    selection = events.richp_isNaF == 0
    return events[selection]

def CutIsAgl(events):
    selection = events.rich_NaF == 0
    return events[selection]

#change from lower - 1.5 to -1.0
def CutRichCharge_LIP(events, qsel):
    lowerlim = qsel - 1.5
    upperlim = qsel + 2.0
    selection =(events.rich_qp >lowerlim) & (events.rich_qp < upperlim)
    return events[selection]

def correct_charge(qraw, betaraw, betacorr, index, factor_pe):   
    qcorr = qraw * np.sqrt(((betaraw * index)**2 - 1) *betacorr**2)/np.sqrt(betaraw**2 * (betacorr**2 * index**2 - 1)) * np.sqrt(factor_pe)  
    return qcorr


def CutRichCharge(events, qsel):
    lowerlim = qsel - 1.5
    upperlim = qsel + 2.0
    charge = np.sqrt(events.rich_q[:, 0])
    selection =(charge > lowerlim)  & (charge < upperlim)
    return events[selection]

def CutRichBetaConsistency(events,  upperlim):           
    selection = (abs(events.rich_BetaConsistency) <= upperlim)             
    return events[selection] 


def CutRichNaFBetaConsistency(events, upperlim):
    beta_diff = abs(get_richbeta(events, 1) - get_richbeta(events, 0))
    selection =(beta_diff < upperlim)
    return events[selection]

def CutRichPmts(events, lowerlim):
    selection = events.rich_pmt > lowerlim
    return events[selection]

def CutRichUsedhits_LIP(events):
    selection = events.rich_usedp > 3
    return events[selection]


def cut_richacc_lip(events, lowerlim):
    selection = events.richp_accvis > lowerlim
    return events[selection]

def cut_richangleerr_lip(events, highlim):
    selection = events.richp_angleRecErr < highlim
    return events[selection]

def rich_expected_photoelectron(events, lowerlim):
    selection = (events.RichNExpectedPhotoElectrons > lowerlim)
    return events[selection]

def CutRichBadtiles(events):
    valueTileIndex = events.rich_tile
    selection = (valueTileIndex != 3) & (valueTileIndex != 7) & (valueTileIndex != 87) & (valueTileIndex !=  100) & (valueTileIndex != 108) & (valueTileIndex != 12)  & (valueTileIndex != 20)  
    return events[selection]

def CutRichFractionPE(events, lim):
    selection = events.rich_npe[:, 0]/events.rich_npe[:, 2] > lim
    return events[selection]

def CutRichFractionPE_LIP(events, lim):
    selection = events.richp_npe/events.rich_npe[:, 2] > lim
    return events[selection]

def CutRichEfficiency_LIP(events):
    selection = events["richp_eff"][:, 0] > 0.05
    return events[selection]

def CutRichIsGoodClean(events):
    selection = (events.rich_good) & (events.rich_clean)
    return events[selection]

def CutTofRichBetaConsistency(events, lim, datatype):
    #richbeta = events.rich_beta_cor
    richbeta = events.rich_beta[:, 0]
    if datatype == "ISS":
        tofbeta = events.tof_betah
    else:
        tofbeta = events.tof_betahmc
    selection = abs(tofbeta - richbeta)/richbeta < lim
    return events[selection]

def CutTofRichBetaConsistency_LIP(events, lim, datatype):
    richbeta = events.richp_beta_cor
    if datatype == "ISS":
        tofbeta = events.tof_betah
    else:
        tofbeta = events.tof_betahmc
    selection = abs(tofbeta - richbeta)/richbeta < lim
    return events[selection]

def CutRichPmts_LIP(events, lim=3):
    selection = events.richp_pmts > lim
    return events[selection]


def selector_tof(events, nuclei, isotope, datatype, cutoff=True, rebin=True):
    qsel = NUCLEI_CHARGE[nuclei]
    events = SelectCleanEvent(events)
    events = events[events.is_ub_l1 == True]
    events = CutTofEdgePaddles(events)
    if datatype == "ISS":
        events = events[(events.tof_chisc_n < 5) & (events.tof_chist_n < 10)]
    if cutoff:    
        events = geomagnetic_cutoff_tof(events, nuclei, isotope, rebin)
    return events

def selector_agl_lipvar(events, nuclei, isotope, datatype, cutoff=True, rebin=True):
    qsel = NUCLEI_CHARGE[nuclei]
    events = events[events.is_ub_l1 == True]
    events = SelectCleanEvent(events)
    events = CutIsAgl_LIP(events)
    #events = CutRichbetaLimit_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events, AGL_FractionPE_LIM) 
    events = CutRichBadtiles(events)
    events = CutTrackInRichAglAcceptance_LIP(events)
    events = CutTofRichBetaConsistency_LIP(events, TOFAGL_CONSIS_LIM, datatype)
    if cutoff:
        events = geomagnetic_cutoff_richcmat(events, nuclei, isotope, rebin)
    return events



def selector_naf_lipvar(events, nuclei, isotope, datatype, cutoff=True, rebin=True):
    qsel = NUCLEI_CHARGE[nuclei]
    events = events[events.is_ub_l1 == True]
    events = SelectCleanEvent(events)
    events = CutIsNaF_LIP(events)
    #events = CutRichbetaLimit_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events, NAF_FractionPE_LIM) 
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.01)
    events = CutTrackInRichNaFAcceptance_LIP(events)
    events = CutTofRichBetaConsistency_LIP(events, TOFNAF_CONSIS_LIM, datatype)
    if cutoff:
        events = geomagnetic_cutoff_richcmat(events, nuclei, isotope, rebin)
    return events



def selector_naf_ciematvar(events, nuclei, isotope, datatype, cutoff=True, rebin=True):
    qsel = NUCLEI_CHARGE[nuclei]
    events = events[events.is_ub_l1 == True]
    events = SelectCleanEvent(events)
    events = cut_charge_lowertof(events, qsel)
    events = Select_HasCIEMATRing(events)
    events = CutRichBadtiles(events)    
    events = CutIsNaF(events)
    events = CutRichIsGoodClean(events)
    events = CutRichPmts(events, NAF_NPMT_LIM)
    events = CutRichFractionPE(events, NAF_FractionPE_LIM)
    events = CutRichCharge(events, qsel)
    events = CutTofRichBetaConsistency(events, TOFNAF_CONSIS_LIM, datatype)
    events = CutRichProbobility(events)
    events = CutRichBetaConsistency(events, NAF_BETACONSIS_LIM)
    if cutoff:
        events = geomagnetic_cutoff_richcmat(events, nuclei, isotope, rebin)
    return events

def selector_agl_ciematvar(events, nuclei, isotope, datatype, cutoff=True, rebin=True):
    qsel = NUCLEI_CHARGE[nuclei]
    events = events[events.is_ub_l1 == True]
    events = Select_HasCIEMATRing(events)
    events = cut_charge_lowertof(events, qsel)
    events = CutRichBadtiles(events)
    events = SelectCleanEvent(events)
    events = CutIsAgl(events)
    events = CutRichIsGoodClean(events)
    events = CutRichPmts(events, AGL_NPMT_LIM)
    events = CutRichFractionPE(events, AGL_FractionPE_LIM)
    events = CutRichCharge(events, qsel)
    events = CutTrackInRichAglAcceptance(events)
    events = CutTofRichBetaConsistency(events, TOFAGL_CONSIS_LIM, datatype)
    if cutoff: 
        events = geomagnetic_cutoff_richcmat(events, nuclei, isotope, rebin)
    return events



def selector_tofeffcor(events, nuclei, isotope, isdata=True):
    qsel = NUCLEI_CHARGE[nuclei]
    events = events[(events.tof_chisc_n < 5) & (events.tof_chist_n < 10)]
    return events


def remove_badrun_indst(events):
    with np.load("/home/manbing/Documents/lithiumanalysis/scripts/tools/corrections/badrunlist_P8.npz") as badrunfile:
        badrunlist = badrunfile["badrunnum"]
        
    all_in_arr2 = np.in1d(np.array(events.run), np.array(badrunlist))
    not_in_arr2 = np.logical_not(all_in_arr2)
    return events[not_in_arr2]    



def selector_naf_lipvar_betastudy(events, nuclei):
    qsel = NUCLEI_CHARGE[nuclei]
    events = events[events.is_ub_l1 == True]
    #events = SelectCleanEvent(events)
    events = CutIsNaF_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events, NAF_FractionPE_LIM) 
    events = CutRichBadtiles(events)
    events = CutTrackInRichNaFAcceptance_LIP(events)
    return events


def selector_naf_ciematvar_betastudy(events, nuclei):
    qsel = NUCLEI_CHARGE[nuclei]
    events = events[events.is_ub_l1 == True]
    #events = SelectCleanEvent(events)
    events = CutIsNaF(events)
    events = CutRichIsGoodClean(events)
    selector_isagl_ciematevents = CutRichPmts(events, NAF_NPMT_LIM)
    events = CutRichFractionPE(events, NAF_FractionPE_LIM)
    events = CutRichCharge(events, qsel)
    events = CutRichProbobility(events)
    events = CutTrackInRichNaFAcceptance(events)
    return events


def selector_agl_lipvar_betastudy(events, nuclei):
    qsel = NUCLEI_CHARGE[nuclei]
    events = events[events.is_ub_l1 == True]
    #events = SelectCleanEvent(events)
    events = CutIsAgl_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    #events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events, AGL_FractionPE_LIM) 
    events = CutRichBadtiles(events)
    events = CutTrackInRichAglAcceptance_LIP(events)
    return events

def selector_agl_ciematvar_betastudy(events, nuclei):
    qsel = NUCLEI_CHARGE[nuclei]
    #events = events[events.is_ub_l1 == True]
    #events = SelectCleanEvent(events)
    events = CutIsAgl(events)
    events = CutRichIsGoodClean(events)
    events = CutRichPmts(events, AGL_NPMT_LIM)
    events = CutRichFractionPE(events, AGL_FractionPE_LIM)
    events = CutRichCharge(events, qsel)
    events = CutRichProbobility(events)
    events = CutTrackInRichAglAcceptance(events)
    return events


def selector_istof(events):
    events = SelectCleanEvent(events)
    events = events[events.ntrack == 1]
    events = events[events.is_ub_l1 == True]
    return events

def selector_isnaf_ciemat(events):
    events = events[events.is_ub_l1 == True]
    #events = events[events.is_l9 == True]
    events = SelectCleanEvent(events)
    events = IsWithin_RICHNaF(events)
    #events = Select_HasCIEMATRing(events)
    #events = CutIsNaF(events)
    #events = CutTrackInRichNaFAcceptance(events)
    return events

def selector_isagl_ciemat(events):
    events = SelectCleanEvent(events)
    events = events[events.is_ub_l1 == True]
    events = IsWithin_RICHAgl(events)
    #events = CutIsAgl(events)
    #events = CutTrackInRichAglAcceptance(events)
    events = CutRichBadtiles(events)
    return events

def selector_tofevents(events, nuclei, datatype):
    events = SelectCleanEvent(events)
    events = events[events.ntrack == 1]
    events = events[events.is_ub_l1 == True]
    events = events[(events.tof_chisc_n < 5) & (events.tof_chist_n < 10)]
    return events


def selector_nafevents_ciematvar(events, nuclei, datatype):
    events = IsWithin_RICHNaF(events)
    qsel = NUCLEI_CHARGE[nuclei]
    events = SelectCleanEvent(events)
    events = Select_HasCIEMATRing(events)
    events = CutIsNaF(events)
    events = events[events.is_ub_l1 == True]
    events = cut_charge_lowertof(events, qsel)
    #events = events[events.is_l9 == True]
    #events = CutTrackInRichNaFAcceptance(events)
    events = CutRichIsGoodClean(events)
    events = CutRichPmts(events, NAF_NPMT_LIM)
    events = CutRichFractionPE(events, NAF_FractionPE_LIM)
    events = CutRichCharge(events, qsel)
    events = CutTofRichBetaConsistency(events, TOFNAF_CONSIS_LIM, datatype)
    events = CutRichProbobility(events)
    events = CutRichBetaConsistency(events, NAF_BETACONSIS_LIM)
    return events

def selector_aglevents_ciematvar(events, nuclei, datatype):
    events = IsWithin_RICHAgl(events)
    qsel = NUCLEI_CHARGE[nuclei]
    events = Select_HasCIEMATRing(events)
    events = cut_charge_lowertof(events, qsel)
    events = SelectCleanEvent(events)
    events = events[events.is_ub_l1 == True]
    events = CutIsAgl(events)
    events = CutTrackInRichAglAcceptance(events)
    events = CutRichBadtiles(events)
    events = CutRichIsGoodClean(events)
    events = CutRichPmts(events, AGL_NPMT_LIM)
    events = CutRichFractionPE(events, AGL_FractionPE_LIM)
    events = CutRichCharge(events, qsel)
    events = CutTofRichBetaConsistency(events, TOFAGL_CONSIS_LIM, datatype)
    return events


def select85events(events):
    events = events[events.run <= 1580131140]
    return events


def SelectEventsCharge(events, qsel):
    inntrkz = events.tk_qin[:, 0, 2]
    selection = (inntrkz > (qsel - 0.35)) & (inntrkz < (qsel + 0.45))
    return events[selection]
