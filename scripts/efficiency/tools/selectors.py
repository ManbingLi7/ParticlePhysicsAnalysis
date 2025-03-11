import json
import os

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from .calculator import calc_rig, calc_ekin_from_beta
from .constants import MC_PARTICLE_MASSES, MC_PARTICLE_CHARGES, MC_PARTICLE_IDS, CUTOFF_FACTOR, ISOTOPES, NUCLEI_CHARGE
from .binnings_collection import fbinning_beta
from .binnings import Binning

NAF_FractionPE_LIM = 0.45
AGL_FractionPE_LIM = 0.4
NAF_NPMT_LIM = 10
AGL_NPMT_LIM = 2
NAF_BETACONSIS_LIM = 0.01
TOFNAF_CONSIS_LIM = 0.06
TOFAGL_CONSIS_LIM = 0.06

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


def get_richbeta(events, is_cmat=True):
    if (is_cmat):
        beta = events.rich_beta_cor
    else:
        beta = events.richp_beta_cor
    return beta
            
def Select_HasLIPRing(events):
    selection = events.irichb == 0
    return events[selection]



def CutRichbetaLimit_LIP(events, upperlim = 1.0, lowerlim = 0.1):
    selection = (events.rich_betap < upperlim) & (events.rich_betap > lowerlim)
    return events[selection]

def CutRichbetaLimit_CMAT(events, isIss,  upperlim = 1.0, lowerlim = 0.1):
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


def geomagnetic_IGRF_cutoff(events, factor):
    selection = (events.tk_rigidity1)[:, 0, 0, 1]  > factor * (events.mcutoffi)[:, 3, 1]
    return events[selection]

def geomagnetic_cutoff_richlip(events, factor, particle_id):
    rigidity_from_beta = calc_rig(events.rich_betap, MC_PARTICLE_MASSES[particle_id], MC_PARTICLE_CHARGES[particle_id])
    rigiditycutoff = (events.mcutoffi)[:, 0, 1] 
    selection = rigidity_from_beta > factor * rigiditycutoff
    return events[selection]

def geomagnetic_cutoff_richcmat(events, nuclei, ana_isotope):
    beta = get_richbeta(events, 1)
    factor = CUTOFF_FACTOR[ana_isotope]
    rigidity_from_beta = calc_rig(beta, MC_PARTICLE_MASSES[MC_PARTICLE_IDS[ISOTOPES[nuclei][0]]], NUCLEI_CHARGE[nuclei])
    rigiditycutoff = (events.mcutoffi)[:, 1, 1] 
    selection = rigidity_from_beta > factor * rigiditycutoff
    return events[selection]


def geomagnetic_cutoff_tof(events, nuclei, ana_isotope):
    betabinning = Binning(fbinning_beta())
    ekinbinning = Binning(fbinning_energy())
    beta = events.tof_betah  
    binIndex = betabinning.get_indices(beta)
    binlowedge = betabinning.edges[binIndex]
    bin_low_rig = calc_rig(binlowedge, MC_PARTICLE_MASSES[MC_PARTICLE_IDS[ISOTOPES[nuclei][0]]], NUCLEI_CHARGE[nuclei])
    factor = CUTOFF_FACTOR[ana_isotope]
    rigiditycutoff = (events.mcutoffi)[:, 1, 1]
    selection = bin_low_rig > factor*rigiditycutoff
    return events[selection]

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
    selection = (events.tk_rigidity1)[:, 0, 2, 1] > lowerlim 
                                  #[0]:choutko, [0] no refit [1] inner pattern
    return events[selection] 

def cut_background(events):
    nhit2i = np.zeros([len(events), 2])
    for ilay in range(1, 8):
        for ixy in range(2):
            if ((events.betah2hb[:, ixy] &(1<<ilay))>0) : nhit2i[:, ixy]+=1
    selection = (events.ntrack == 1) | (abs(events.betah2r) < 0.5) | (nhit2i[:, 0]<3 or nhit2i[:, 1]<5) | (events.betah2r/events.tof_betah) < 0  
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


def CutRichNaFBetaConsistency(events, upperlim, isIss=True):
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

def CutTofRichBetaConsistency(events, lim, isIss=True):
    richbeta = events.rich_beta_cor
    if isIss:
        tofbeta = events.tof_betah
    else:
        tofbeta = events.tof_betahmc
    selection = abs(tofbeta - richbeta)/richbeta < lim
    return events[selection]

def CutTofRichBetaConsistency_LIP(events, lim, isIss=True):
    richbeta = events.richp_beta_cor
    if isIss:
        tofbeta = events.tof_betah
    else:
        tofbeta = events.tof_betahmc
    selection = abs(tofbeta - richbeta)/richbeta < lim
    return events[selection]

def CutRichPmts_LIP(events, lim=3):
    selection = events.richp_pmts > lim
    return events[selection]


def selector_tof(events, nuclei, isotope, datatype):
    qsel = NUCLEI_CHARGE[nuclei]
    events = SelectCleanEvent(events)
    events = events[events.is_ub_l1 == True]
    events = events[(events.tof_chisc_n < 5) & (events.tof_chist_n < 10)]
    if datatype == "ISS":
        events = geomagnetic_cutoff_tof(events, nuclei, isotope)
    return events

def selector_agl_lipvar(events, nuclei, isotope, datatype):
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
    events = CutTofRichBetaConsistency_LIP(events, TOFAGL_CONSIS_LIM, isIss)
    if datatype == "ISS":
        events = geomagnetic_cutoff_richcmat(events, nuclei, isotope)
    return events


def selector_naf_lipvar(events, nuclei, isotope, datatype):
    qsel = NUCLEI_CHARGE[nuclei]
    events = events[events.is_ub_l1 == True]
    events = SelectCleanEvent(events)
    events = CutIsNaF_LIP(events)
    #events = CutRichbetaLimit_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events, NAF_FRACTIONPE_LIM) 
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.01)
    events = CutTrackInRichNaFAcceptance_LIP(events)
    events = CutTofRichBetaConsistency_LIP(events, TOFNAF_CONSIS_LIM, isIss)
    if datatype == "ISS":
        events = geomagnetic_cutoff_richcmat(events, nuclei, isotope)
    return events


def selector_naf_ciematvar(events, nuclei, isotope, datatype):
    qsel = NUCLEI_CHARGE[nuclei]
    events = events[events.is_ub_l1 == True]
    #events = CutRichBadtiles(events)
    events = SelectCleanEvent(events)
    events = CutIsNaF(events)
    events = CutRichIsGoodClean(events)
    events = CutRichPmts(events, NAF_NPMT_LIM)
    events = CutRichFractionPE(events, NAF_FractionPE_LIM)
    events = CutRichCharge(events, qsel)
    events = CutTofRichBetaConsistency(events, TOFNAF_CONSIS_LIM, isdata)
    events = CutRichProbobility(events)
    events = CutRichBetaConsistency(events, NAF_BETACONSIS_LIM)
    events = CutTrackInRichNaFAcceptance(events)
    if datatype == "ISS":
        events = geomagnetic_cutoff_richcmat(events, nuclei, isotope)
    return events

def selector_agl_ciematvar(events, nuclei, isotope, datatype):
    qsel = NUCLEI_CHARGE[nuclei]
    events = events[events.is_ub_l1 == True]
    events = CutRichBadtiles(events)
    events = SelectCleanEvent(events)
    events = CutIsAgl(events)
    events = CutRichIsGoodClean(events)
    events = CutRichPmts(events, AGL_NPMT_LIM)
    events = CutRichFractionPE(events, AGL_FractionPE_LIM)
    events = CutRichCharge(events, qsel)
    #events = CutRichProbobility(events)    
    events = CutTrackInRichAglAcceptance(events)
    events = CutTofRichBetaConsistency(events, TOFAGL_CONSIS_LIM, isdata)
    if datatype == "ISS": 
        events = geomagnetic_cutoff_richcmat(events, nuclei, isotope)
    return events


def selector_agl_event(events):
    events = CutIsAgl(events)
    events = CutTrackInRichAglAcceptance(events)
    return events


def selector_agl_event(events):
    events = CutIsAgl(events)
    events = CutTrackInRichAglAcceptance(events)
    return events

def selector_naf_event(events):
    #events = cut_background_reduce(events)
    #qsel = MC_PARTICLE_CHARGES[MC_PARTICLE_IDS[isotope]]
    events = CutIsNaF(events)
    #events = geomagnetic_cutoff_richcmat(events, isotope)  
    events = CutTrackInRichNaFAcceptance(events)
    return events


def selector_nafeffcor_ciematvar(events, nuclei, isotope, isdata=True):
    qsel = NUCLEI_CHARGE[nuclei]
    #events = CutRichBadtiles(events)
    events = CutIsNaF(events)
    events = CutRichIsGoodClean(events)
    events = CutRichPmts(events, NAF_NPMT_LIM)
    events = CutRichFractionPE(events, NAF_FractionPE_LIM)
    events = CutRichCharge(events, qsel)
    events = CutTofRichBetaConsistency(events, TOFNAF_CONSIS_LIM, isdata)
    events = CutRichProbobility(events)
    events = CutRichBetaConsistency(events, NAF_BETACONSIS_LIM)
    events = CutTrackInRichNaFAcceptance(events)
    return events

def selector_agl_ciematvar(events, nuclei, isotope, isdata=True):
    qsel = NUCLEI_CHARGE[nuclei]
    events = CutRichBadtiles(events)
    events = SelectCleanEvent(events)
    events = CutIsAgl(events)
    events = CutRichIsGoodClean(events)
    events = CutRichPmts(events, AGL_NPMT_LIM)
    events = CutRichFractionPE(events, AGL_FractionPE_LIM)
    events = CutRichCharge(events, qsel)
    #events = CutRichProbobility(events)    
    events = CutTrackInRichAglAcceptance(events)
    events = CutTofRichBetaConsistency(events, TOFAGL_CONSIS_LIM, isdata)
    return events

def selector_tofeffcor(events, nuclei, isotope, isdata=True):
    qsel = NUCLEI_CHARGE[nuclei]
    events = events[(events.tof_chisc_n < 5) & (events.tof_chist_n < 10)]
    return events
