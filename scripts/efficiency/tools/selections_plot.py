import json
import os

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from .calculator import calc_rig, calc_kinetic_eng_nucleon
from .constants import MC_PARTICLE_MASSES, MC_PARTICLE_CHARGES, MC_PARTICLE_IDS

def TofVelocity(events, upperlim = 1.245, lowerlim = 0.1):
    selection = (events.TofBeta <= upperlim) & (events.TofBeta >= lowerlim)
    return events[selection]

def rich_beta_limit(events, upperlim = 1.0, lowerlim = 0.1):
    selection = (events.richp_beta < upperlim) & (events.richp_beta > lowerlim)
    return events[selection]

def tof_beta_limit(events, upperlim = 1.0, lowerlim = 0.1):
    selection = (events.TofBeta < upperlim) & (events.TofBeta> lowerlim) & (events.TofBetaMC < upperlim)
    return events[selection]

def tof_beta_iss_limit(events, upperlim = 1.0, lowerlim = 0.1):
    selection = (events.TofBeta < upperlim) & (events.TofBeta> lowerlim) 
    return events[selection]

def geomagnetic_IGRF_cutoff(events, factor):
    selection = events.TrackerTrackChoutkoInnerPlusL1Rigidity > factor * events.IGRFCutOff30PN
    return events[selection]

def geomagnetic_cutoff_tof(events, factor, particle_id):
    rigidity_from_beta = calc_rig(events.TofBeta, MC_PARTICLE_MASSES[particle_id], MC_PARTICLE_CHARGES[particle_id])
    selection = rigidity_from_beta > factor * events.IGRFCutOff30PN
    return events[selection]

def energy_range(events, lowerlim,  upperlim):
    energy = calc_kinetic_eng_nucleon(events.RichBeta)
    selection = (energy < upperlim) & (energy >= lowerlim)
    return events[selection]

def lower_tof_charge_cut(events, lowerlim,  upperlim):
    selection = (events.TofLowerCharge > lowerlim) & (events.TofLowerCharge < upperlim)
    return events[selection]

def upper_tof_charge_cut(events, lowerlim,  upperlim):
    selection = (events.TofLowerCharge > lowerlim) & (events.TofLowerCharge < upperlim)
    return events[selection]

def rigidity_cutoff_cut(events, factor):
    selection = (events.TrackerTrackChoutkoInnerPlusL1Rigidity > factor * events.IGRFCutOff40PN)
    return events[selection]

def selector_tof(events):    
    events = tof_beta_limit(events)
    events = lower_tof_charge_cut(events, 2.5, 3.7)
    events = geomagnetic_cutoff_tof(events, 1.0, MC_PARTICLE_IDS["Li6"])
    return events

def selector_tof_iss(events):    
    events = tof_beta_iss_limit(events)
    events = lower_tof_charge_cut(events, 2.5, 3.7)
    events = geomagnetic_cutoff_tof(events, 1.0, MC_PARTICLE_IDS["Li6"])
    return events

def selector_naf(events):
    events = rich_beta_limit(events)
    events = lower_tof_charge_cut(events, 2.5, 3.7)
    events = geomagnetic_cutoff_rich(events, 1.0, MC_PARTICLE_IDS["Li6"])
    events = rich_charge_cut(events, 2.5, 5.0)
    return events
    
#begin defination of charge selections
def cut_charge_innertrk(events, qsel):
    rightside = qsel + 0.5
    leftside = qsel - 0.5
    selection = (events.tk_qin[1] > leftside) & (events.tk_qin[1] < rightside)
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

def cut_charge_trkL1(events, qsel):
    rightside = qsel + 0.5
    leftside = qsel - 0.6
    selection = (events.tk_l1qvs > leftside) & (events.tk_l1qvs < rightside)
    
    return events[selection]
        
def selector_charges(events, qsel):
#    events = cut_charge_innertrk(events, qsel)
#    events = cut_charge_uppertof(events, qsel)
    events = cut_charge_trkL1(events, qsel)
    return events

# end of charge selections, use the selector_charges

#begin defination of RICHAGL selection using LIP variables
def cut_high_rig(events, lowerlim): 
    selection = (events.tk_rigidity1)[:, 0, 0, 1] > lowerlim 
                                  #[0]:choutko, [0] no refit [1] inner pattern
    return events[selection] 

def cut_background(events):
    nhit2i = np.zeros(2)
    for ilay in range(-1, 9):
        for ixy in range(2):
            if (events.betah2hb[ixy]) : nhit2i[ixy]+=1
    selection = (events.ntrack == 1) | (events.betah2r < 0.5) | (nhit2i[0]<3 or nhit2i[1]<5) 
    return events[selection]

def cut_background_reduce(events):
    selection = (events.ntrack == 1) | (events.betah2r < 1.0) 
    return events[selection]


def cut_geomagneticcutoff_richbeta(events, factor, particle_id):
    rigidity_from_beta = calc_rig(events.richp_beta, MC_PARTICLE_MASSES[particle_id], MC_PARTICLE_CHARGES[particle_id])
    selection = rigidity_from_beta > factor * events.IGRFCutOff30PN
    return events[selection]

def CutTrackInRichAglAcceptance(events):#constants of geomatric should be put in tools:constants
    xpos = (events.rich_pos)[:, 0] 
    ypos = (events.rich_pos)[:, 1]
    RichInnerRadius = 24.04
    RichOuterRadius = 61.0
    RichInnerEdge = 19.0
    R2 = xpos * xpos + ypos * ypos
    selection = (R2 < RichOuterRadius * RichOuterRadius) & ((abs(xpos) > RichInnerEdge) | (abs(ypos) > RichInnerEdge))
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

def CutRichProbobility_LIP(events):
    selection = events.richp_prob >= 0.01
    return events[selection]

def CutRichProbobility(events):
    selection = events.rich_pb >= 0.01
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

def CutRichCharge_LIP(events, qsel):
    lowerlim = qsel - 1.5
    upperlim = qsel + 1.5
    selection =(events.rich_qp >lowerlim)
    return events[selection]

def CutRichCharge(events, qsel):
    lowerlim = qsel - 1.5
    upperlim = qsel + 1.5
    charge = np.sqrt(events.rich_q[:, 0])
    selection =(charge > lowerlim) 
    return events[selection]

def CutRichBetaConsistency(events, lowerlim,  upperlim):
    selection =(events.rich_BetaConsistency >= lowerlim) & (events.rich_BetaConsistency <= upperlim)
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
    selection = (valueTileIndex != 3) & (valueTileIndex != 7) & (valueTileIndex != 87) & (valueTileIndex !=  100) & (valueTileIndex != 108)
    return events[selection]

def CutRichFractionPE(events):
    selection = events.rich_npe[:, 0]/events.rich_npe[:, 2] > 0.35
    return events[selection]

def CutRichFractionPE_LIP(events):
    selection = events.richp_npe/events.rich_npe[:, 2] > 0.35
    return events[selection]

def CutRichEfficiency_LIP(events):
    selection = events["richp_eff"][:, 0] > 0.05
    return events[selection]

def CutRichIsGoodClean(events):
    selection = (events.rich_good) & (events.rich_clean)
    return events[selection]

def CutTofRichBetaConsistency(events, lim):
    richbeta = events.rich_beta2[:, 0]
    selection = abs(events.tof_betah - richbeta)/events.tof_betah < lim
    return events[selection]

def CutTofRichBetaConsistency_LIP(events, lim):
    richbeta = events.richp_beta
    selection = abs(events.tof_betah - richbeta)/events.tof_betah < lim
    return events[selection]

def selector_agl_lipvar(events, qsel):
    events = CutIsAgl_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
 #   events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events) 
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance_LIP(events)
    events = CutTofRichBetaConsistency_LIP(events, 0.1)
    return events

def selector_agl_ciematvar(events, qsel):
    events = CutIsAgl(events)
#    events = CutRichIsGoodClean(events)
    events = CutRichCharge(events, qsel)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
#    events = CutRichPmts(events, 3)
    events = CutTrackInRichAglAcceptance(events)
    events = CutTofRichBetaConsistency(events, 0.1)  
    return events

def TagRichIsAgl_LIP(events, qsel):
    events = CutTrackInRichAglAcceptance_LIP(events)
    return events

def TagRichIsAgl(events, qsel):
#    events = CutRichIsGoodClean(events)
#    events = CutRichPmts(events, 3)
    events = CutTrackInRichAglAcceptance(events)
    return events


def TagRichCharge_LIP(events, qsel):
    events = CutIsAgl_LIP(events)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance_LIP(events)
    events = CutTofRichBetaConsistency_LIP(events, 0.1)
    return events

def TagRichProbobility_LIP(events, qsel):
    events = CutIsAgl_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance_LIP(events)
    events = CutTofRichBetaConsistency_LIP(events, 0.1)
    return events


def TagRichUsedhits_LIP(events, qsel):
    events = CutIsAgl_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichFractionPE_LIP(events) 
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance_LIP(events)
    events = CutTofRichBetaConsistency_LIP(events, 0.1)
    return events


def TagRichFractionPE_LIP(events, qsel):
    events = CutIsAgl_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance_LIP(events)
    events = CutTofRichBetaConsistency_LIP(events, 0.1)
    return events

def TagRichBadtiles_LIP(events, qsel):
    events = CutIsAgl_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events) 
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTofRichBetaConsistency_LIP(events, 0.1)
    events = CutTrackInRichAglAcceptance_LIP(events)
    return events


def TagRichBetaConsistency_LIP(events, qsel):
    events = CutIsAgl_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events) 
    events = CutRichBadtiles(events)
    events = CutTofRichBetaConsistency_LIP(events, 0.1)
    events = CutTrackInRichAglAcceptance_LIP(events)
    return events

def TagTofRichBetaConsistency_LIP(events, qsel):
    events = CutIsAgl_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events) 
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance_LIP(events)
    return events


def TagTrackInRichAglAcceptance_LIP(events, qsel):
    events = CutIsAgl_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events) 
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTofRichBetaConsistency_LIP(events, 0.1)
    return events


def TagRichCharge(events, qsel):
    events = CutRichIsGoodClean(events)
    events = CutIsAgl(events)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance(events)
    events = CutTofRichBetaConsistency(events, 0.1)
    events = CutRichPmts(events, 3)
    return events

def TagRichPmts(events, qsel):
    events = CutRichIsGoodClean(events)
    events = CutIsAgl(events)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance(events)
    events = CutTofRichBetaConsistency(events, 0.1)
    events = CutRichCharge(events, qsel)
    return events

def TagRichIsGoodClean(events, qsel):
    events = CutIsAgl(events)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance(events)
    events = CutTofRichBetaConsistency(events, 0.1)
    events = CutRichCharge(events, qsel)
    events = CutRichPmts(events, 3)
    return events

def TagRichProbobility(events, qsel):
    events = CutIsAgl(events)
    events = CutRichIsGoodClean(events)
    events = CutRichPmts(events, 3)
    events = CutRichCharge(events, qsel)
    events = CutRichFractionPE(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTofRichBetaConsistency(events, 0.1)
    events = CutTrackInRichAglAcceptance(events)
    return events



def TagRichFractionPE(events, qsel):
    events = CutIsAgl(events)
    events = CutRichIsGoodClean(events)
    events = CutRichPmts(events, 3)
    events = CutRichCharge(events, qsel)
    events = CutRichProbobility(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance(events)
    events = CutTofRichBetaConsistency(events, 0.1)    
    return events

def TagRichBadtiles(events, qsel):
    events = CutIsAgl(events)
    events = CutRichPmts(events, 3)
    events = CutRichIsGoodClean(events)
    events = CutRichCharge(events, qsel)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events) 
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTofRichBetaConsistency(events, 0.1)
    events = CutTrackInRichAglAcceptance(events)
    return events

def TagRichBetaConsistency(events, qsel):
    events = CutIsAgl(events)
    events = CutRichIsGoodClean(events)
    events = CutRichPmts(events, 3)
    events = CutRichCharge(events, qsel)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events) 
    events = CutRichBadtiles(events)
    events = CutTofRichBetaConsistency(events, 0.1)
    events = CutTrackInRichAglAcceptance(events)
    return events

def TagTofRichBetaConsistency(events, qsel):
    events = CutIsAgl(events)
    events = CutRichIsGoodClean(events)
    events = CutRichPmts(events, 3)
    events = CutRichCharge(events, qsel)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events) 
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance(events)
    return events


def TagTrackInRichAglAcceptance(events, qsel):
    events = CutIsAgl(events)
    events = CutRichIsGoodClean(events)
    events = CutRichPmts(events, 3)
    events = CutRichCharge(events, qsel)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events) 
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTofRichBetaConsistency(events, 0.1)
    return events

def selector_agl_betabias(events, riglim, qsel):
    events = cut_high_rig(events, riglim)
    events = cut_isAgl(events)
    events = cut_trkInRichAglAcceptance(events)
    events = cut_richcharge(events, qsel)
    events = cut_richprobobility(events)
#    events = cut_richacc(events, 0.1)
#    events = cut_richangleerr(events, 0.01)
  
 #   events = cut_richbeta_consistency(events, 0.0, 0.01)
 #   events = cut_richpmts(events, 4)
    return events

def selector_naf_betabias(events, riglim, qsel):
    events = cut_high_rig(events, riglim)
    events = cut_isNaF(events)
#    events = cut_trkInRichAglAcceptance(events)
    events = cut_richcharge(events, qsel)
    events = cut_richprobobility(events)
#    events = cut_richacc(events, 0.1)
#    events = cut_richangleerr(events, 0.01)
  
 #   events = cut_richbeta_consistency(events, 0.0, 0.01)
 #   events = cut_richpmts(events, 4)
    return events

def selector_agl_ciemat_betabias(events, qsel):
    events = cut_high_rig(events, 200)
    events = events[events.rich_NaF == 0]
    events = cut_trkInRichAglAcceptance(events)
    events = events[np.sqrt(events.rich_q[:, 0]) > (qsel - 0.8)]
    events = events[np.sqrt(events.rich_q[:, 0]) < (qsel + 2.0)]
    events = events[events.rich_pb > 0.01]
    events = events[events.rich_pmt > 3]
    return events



# end of the selector for rich-agl 


# selector for tag and probe
def tagselector_probility(events, qsel):
    tagselector = (events.richp_isNaF == 0) & (events.rich_qp > (qsel - 0.8)) & (events.richp_accvis > 0.01)
    return tagselector

def tagselector_charge(events):
    tagselector = (events.richp_isNaF == 0) & (events.richp_prob > 0.01) & (events.richp_accvis > 0.01)
    return tagselector

def tagselector_accvis(events, qsel):
    tagselector = (events.richp_isNaF == 0) & (events.richp_prob > 0.01) &  (events.rich_qp > (qsel - 0.8))
    return tagselector

def tagselector_flatness(events, qsel):
    tagselector = (events.richp_isNaF == 0) & (events.richp_prob > 0.01) &  (events.rich_qp > (qsel - 0.8)) & (events.richp_accvis > 0.01)
    return tagselector

def tagselector_index(events, qsel):
    tagselector = (events.richp_isNaF == 0) & (events.richp_prob > 0.01) &  (events.rich_qp > (qsel - 0.8)) & (events.richp_accvis > 0.01)
    return tagselector

def tagselector_angle(events, qsel):
    tagselector = (events.richp_isNaF == 0) & (events.richp_prob > 0.01) &  (events.rich_qp > (qsel - 0.8)) & (events.richp_accvis > 0.01)
    return tagselector

def tagselector_angleerr(events, qsel):
    tagselector = (events.richp_isNaF == 0) & (events.richp_prob > 0.01) &  (events.rich_qp > (qsel - 0.8)) & (events.richp_accvis > 0.01)
    return tagselector

def tagselector_richlik(events, qsel):
    tagselector = (events.richp_isNaF == 0) & (events.richp_prob > 0.01) &  (events.rich_qp > (qsel - 0.8)) & (events.richp_accvis > 0.01)
    return tagselector

def tagselector_richnpe(events, qsel):
    tagselector = (events.richp_isNaF == 0) & (events.richp_prob > 0.01) &  (events.rich_qp > (qsel - 0.8)) & (events.richp_accvis > 0.01)
    return tagselector

def probecut_probility(events):
    selection = events.richp_prob > 0.01
    return selection

def probecut_richcharge(events, qsel):
    selection = (events.rich_qp > (qsel - 0.8)) & (events.rich_qp < (qsel + 2.0))
    return selection

