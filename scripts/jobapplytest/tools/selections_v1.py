import json
import os

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from .calculator import calc_rig, calc_ekin_from_beta
from .constants import MC_PARTICLE_MASSES, MC_PARTICLE_CHARGES, MC_PARTICLE_IDS, CUTOFF_FACTOR, ISOTOPES, NUCLEI_CHARGE

def get_richbeta(events, is_cmat=True, is_data=True, is_corr=True):
    if (is_cmat):
        if (is_data):
            if is_corr:
                beta = events.rich_beta_cor
            else:
                beta = events.rich_beta[:, 0]
        else:
            beta = events.rich_beta[:, 0]
    else:
        if is_data:
            if is_corr:
                beta = events.richp_beta_cor
            else:
                beta = events.rich_betap
        else:
            beta = events.richp_beta
    return beta
            
def Select_HasLIPRing(events):
    selection = events.irichb == 0
    return events[selection]

def TofVelocity(events, upperlim = 1.245, lowerlim = 0.1):
    selection = (events.TofBeta <= upperlim) & (events.TofBeta >= lowerlim)
    return events[selection]

def CutRichbetaLimit_LIP(events, upperlim = 1.0, lowerlim = 0.1):
    selection = (events.rich_betap < upperlim) & (events.rich_betap > lowerlim)
    return events[selection]

def CutRichbetaLimit_CMAT(events, upperlim = 1.0, lowerlim = 0.1, isIss=True):
    beta =  get_richbeta(events, is_cmat=True, is_data=isIss, is_corr=True) 
    selection = (beta < upperlim) & (beta > lowerlim)
    return events[selection]

def rich_beta_abovezero(events, lowerlim = 0.1):
    beta = events.rich_beta2[:, 0]
    selection = beta > lowerlim
    return events[selection]

def rich_beta_abovezero_LIP(events, lowerlim = 0.1):
    selection = events.rich_betap > lowerlim
    return events[selection]

def tof_beta_limit(events, upperlim = 1.0, lowerlim = 0.1):
    selection = (events.TofBeta < upperlim) & (events.TofBeta> lowerlim) & (events.TofBetaMC < upperlim)
    return events[selection]

def tof_beta_iss_limit(events, upperlim = 1.0, lowerlim = 0.1):
    selection = (events.TofBeta < upperlim) & (events.TofBeta> lowerlim) 
    return events[selection]

def geomagnetic_IGRF_cutoff(events, factor):
    selection = (events.tk_rigidity1)[:, 0, 0, 1]  > factor * (events.mcutoffi)[:, 3, 1]
    return events[selection]

def geomagnetic_cutoff_tof(events, factor, particle_id):
    rigidity_from_beta = calc_rig(events.TofBeta, MC_PARTICLE_MASSES[particle_id], MC_PARTICLE_CHARGES[particle_id])
    selection = rigidity_from_beta > factor * events.IGRFCutOff30PN
    return events[selection]

def geomagnetic_cutoff_richlip(events, factor, particle_id):
    rigidity_from_beta = calc_rig(events.rich_betap, MC_PARTICLE_MASSES[particle_id], MC_PARTICLE_CHARGES[particle_id])
    rigiditycutoff = (events.mcutoffi)[:, 3, 1] 
    selection = rigidity_from_beta > factor * rigiditycutoff
    return events[selection]

def geomagnetic_cutoff_richcmat(events, nuclei, ana_isotope):
    beta = get_richbeta(events, 1, 1, 1)
    factor = CUTOFF_FACTOR[ana_isotope]
    rigidity_from_beta = calc_rig(beta, MC_PARTICLE_MASSES[MC_PARTICLE_IDS[ISOTOPES[nuclei][0]]], NUCLEI_CHARGE[nuclei])
    rigiditycutoff = (events.mcutoffi)[:, 3, 1] 
    selection = rigidity_from_beta > factor * rigiditycutoff
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
    selection = (events.tk_rigidity1)[:, 0, 0, 1] > lowerlim 
                                  #[0]:choutko, [0] no refit [1] inner pattern
    return events[selection] 

def cut_background(events):
    nhit2i = np.zeros([len(events), 2])
    for ilay in range(1, 8):
        for ixy in range(2):
            if ((events.betah2hb[:, ixy] & (1<<ilay))>0) : nhit2i[:, ixy]+=1
    selection = (events.ntrack == 1) | (abs(events.betah2r) < 0.5) | (nhit2i[:, 0]<3 or nhit2i[:, 1]<5) | (events.betah2r/events.tof_betah) < 0  
    return events[selection]



def SelectCleanEvent(events):
    is_1trk = (events.ntrack == 1)
    nhit2i = np.zeros((len(events), 2), dtype=int)
    for ilay in range(1, 8):
        for ixy in range(2):
            mask = 1 << ilay
            nhit2i[:, ixy] += (events.betah2hb[:, ixy] & mask) > 0
    is_clean2 = is_1trk | (nhit2i[:, 0] < 3) | (nhit2i[:, 1] < 5)
    is_clean3 = is_clean2 | (events.betah2r/events.tof_betah < 0)
    is_clean4 = is_clean3 | (abs(events.betah2r) < 0.5)
    return events[is_clean4]


def cut_background_reduce(events):
    selection = (events.ntrack == 1) | (events.betah2r < 0.5) 
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

def CutTrackInRichNaFAcceptance(events):#constants of geomatric should be put in tools:constants
    xpos = (events.rich_pos)[:, 0] 
    ypos = (events.rich_pos)[:, 1]
    RichInnerEdge = 17.0
    R2 = xpos * xpos + ypos * ypos
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

#change from lower - 1.5 to -1.0
def CutRichCharge_LIP(events, qsel):
    lowerlim = qsel - 1.0
    upperlim = qsel + 1.5
    selection =(events.rich_qp >lowerlim)
    return events[selection]

def correct_charge(qraw, betaraw, betacorr, index, factor_pe):                                                                                                                                             
    qcorr = qraw * np.sqrt(((betaraw * index)**2 - 1) *betacorr**2)/np.sqrt(betaraw**2 * (betacorr**2 * index**2 - 1)) * np.sqrt(factor_pe)                                                                
    return qcorr

def CutRichChargeCorr_LIP(events, qsel):
    lowerlim = qsel - 1.0
    cmat_index = ak.to_numpy(events["rich_index"])
    cmat_rawindex = ak.to_numpy(events["rich_rawindex"])
    correction_index = cmat_rawindex/cmat_index
    betacmat_raw = ak.to_numpy(events["rich_beta"][:,0])/correction_index
    betalip_raw = ak.to_numpy(events["rich_betap"])
    betalip_corr = betalip_raw * correction_index
    lip_index = ak.to_numpy(events["richp_index"])
    usednpe_lip = ak.to_numpy(events["richp_npe"])
    usednpe_ciemat = ak.to_numpy(events["rich_npe"][:, 0])
    factor_pe = usednpe_ciemat/usednpe_lip
    beta_lip_morecorr = 1.0 + 0.0002029502124347422
    betalip_corr_new = betalip_corr/beta_lip_morecorr
    richcharge_lip_raw = ak.to_numpy(events["rich_qp"])                                       
    richcharge_lip_corr = correct_charge(richcharge_lip_raw, betalip_raw, betalip_corr_new, lip_index, factor_pe)  
    selection = richcharge_lip_corr > lowerlim
    return events[selection]

def CutRichCharge(events, qsel):
    lowerlim = qsel - 1.0
    upperlim = qsel + 1.5
    charge = np.sqrt(events.rich_q[:, 0])
    selection =(charge > lowerlim) 
    return events[selection]

def CutRichBetaConsistency(events, lowerlim,  upperlim):
    selection =(abs(events.rich_BetaConsistency) >= lowerlim) & (abs(events.rich_BetaConsistency) <= upperlim)
    return events[selection]

def CutRichBetaConsistency_Corr(events, lowerlim,  upperlim, isIss=True):
    #if isIss:
    #    beta_diff = abs(get_richbeta(events, 1, 1, 1) - get_richbeta(events, 0, 1, 1))/get_richbeta(events, 1, 1, 1)
    #else:
    beta_diff = abs(events.rich_BetaConsistency)
    selection =(beta_diff >= lowerlim) & (beta_diff <= upperlim)
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
    selection = events.rich_npe[:, 0]/events.rich_npe[:, 2] > 0.4
    return events[selection]

def CutRichFractionPE_LIP(events):
    selection = events.richp_npe/events.rich_npe[:, 2] > 0.45
    return events[selection]

def CutRichEfficiency_LIP(events):
    selection = events["richp_eff"][:, 0] > 0.05
    return events[selection]

def CutRichIsGoodClean(events):
    selection = (events.rich_good) & (events.rich_clean)
    return events[selection]

def CutTofRichBetaConsistency(events, lim=0.05):
    richbeta = events.rich_beta2[:, 0]
    selection = abs(events.tof_betah - richbeta)/richbeta < lim
    return events[selection]

def CutTofRichBetaConsistency_LIP(events, lim=0.05):
    richbeta = events.richp_beta
    selection = abs(events.tof_betah - richbeta)/richbeta < lim
    return events[selection]

def CutRichPmts_LIP(events, lim=3):
    selection = events.richp_pmts > lim
    return events[selection]

def selector_agl_lipvar(events, nuclei, isotope, isIss=True):
    qsel = NUCLEI_CHARGE[nuclei] 
    events = cut_background_reduce(events)
    events = CutIsAgl_LIP(events)
    #events = CutRichbetaLimit_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events) 
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance_LIP(events)
    events = CutTofRichBetaConsistency_LIP(events, 0.05)
    return events

def selector_agl_lipvar_iss(events, qsel, isotope):
    events = cut_background_reduce(events)
    events = CutIsAgl_LIP(events)
    events = geomagnetic_cutoff_richlip(events, 1.0, MC_PARTICLE_IDS[isotope])
    events = CutRichbetaLimit_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events) 
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance_LIP(events)
    events = CutTofRichBetaConsistency_LIP(events, 0.05)
    return events

def selector_naf_lipvar_iss(events, nuclei, isotope, isIss=True):
    qsel = NUCLEI_CHARGE[nuclei] 
    events = cut_background_reduce(events)
    events = CutIsNaF_LIP(events)
    events = geomagnetic_cutoff_richlip(events, 1.0, MC_PARTICLE_IDS[isotope])
    #events = CutRichbetaLimit_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events) 
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.01)
    events = CutTrackInRichNaFAcceptance_LIP(events)
    #events = CutTofRichBetaConsistency_LIP(events, 0.1)
    return events

def selector_agl_lipvar_highR(events, qsel, riglimit):
    events = cut_background_reduce(events)
    events = CutIsAgl_LIP(events)
    events = CutHighRig(events, riglimit)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events) 
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance_LIP(events)
    events = CutTofRichBetaConsistency_LIP(events, 0.05)
    return events


def selector_agl_lipvar_forHe(events, qsel):
    events = cut_background_reduce(events)
    events = CutIsAgl_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events)
    events = CutRichChargeCorr_LIP(events, qsel)
    events = CutRichBadtiles(events)
    #events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance_LIP(events)
    events = CutRichPmts_LIP(events)
    events = events[events.rich_usedp > 3]
    events = events[events.rich_pbp > 0.01]
    return events

def selector_naf_lipvar_forHe(events, qsel):
    events = cut_background_reduce(events)
    events = CutIsNaF_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events)
    events = CutRichChargeCorr_LIP(events, qsel)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.01)
    events = CutTrackInRichNaFAcceptance_LIP(events)
    events = CutRichPmts_LIP(events)
    events = events[events.rich_usedp > 3]
    events = events[events.rich_pbp > 0.01]
    return events



def selector_naf_ciematvar_iss(events, qsel, isotope):
    events = cut_background_reduce(events)
    events = CutIsNaF(events)
    events = CutRichIsGoodClean(events)
    events = geomagnetic_cutoff_richlip(events, 1.0, MC_PARTICLE_IDS[isotope])
    #events = CutRichbetaLimit_CMAT(events)
    events = CutRichCharge(events, qsel)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.01)
    events = CutRichPmts(events, 3)
    events = CutTrackInRichNaFAcceptance(events)
    events = CutTofRichBetaConsistency(events, 0.1)  
    return events

def selector_naf_ciematvar(events,  isotope, isdata=False):
    qsel = NUCLEI_CHARGE[nuclei] 
    events = cut_background_reduce(events)
    events = CutIsNaF(events)
    events = CutRichIsGoodClean(events)
    #events = CutRichbetaLimit_CMAT(events)
    events = CutRichCharge(events, qsel)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.01)
    events = CutRichPmts(events, 3)
    events = CutTrackInRichNaFAcceptance(events)
    events = CutTofRichBetaConsistency(events, 0.1)  
    return events


def selector_agl_ciematvar_efficiency(events, qsel):
    events = CutIsAgl(events)
    events = CutRichIsGoodClean(events)
    #events = CutRichbetaLimit_CMAT(events)
    events = CutRichCharge(events, qsel)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events)
    events = CutRichBadtiles(events)
    #events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutRichPmts(events, 3)
    events = CutTrackInRichAglAcceptance(events)
    #events = CutTofRichBetaConsistency(events, 0.05)
    return events


def selector_agl_ciematvar(events, nuclei, isotope, isdata=False):
    #events = cut_background(events)
    qsel = NUCLEI_CHARGE[nuclei]
    events = CutIsAgl(events)
    events = CutRichIsGoodClean(events)
    events = CutRichbetaLimit_CMAT(events, isdata)
    events = CutRichCharge(events, qsel)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency_Corr(events, 0.0, 0.005, isdata)
    events = CutRichPmts(events, 3)
    events = CutTrackInRichAglAcceptance(events)
    #events = CutTofRichBetaConsistency(events, 0.05)
    if isdata:
        events = geomagnetic_cutoff_richcmat(events, nuclei, isotope)
    return events


    

def selector_agl_event(events):
    #events = cut_background_reduce(events)
    #qsel = MC_PARTICLE_CHARGES[MC_PARTICLE_IDS[isotope]]
    events = CutIsAgl(events)
    #events = geomagnetic_cutoff_richcmat(events, isotope)  
    events = CutTrackInRichAglAcceptance(events)
    return events

def selector_agl_ciematvar_forHe(events, qsel):
    events = cut_background_reduce(events)
    events = CutIsAgl(events)
    events = CutRichIsGoodClean(events)
    events = CutRichCharge(events, qsel)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events)
    events = CutRichBadtiles(events)
    #events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutRichPmts(events, 3)
    events = CutTrackInRichAglAcceptance(events)
    return events

def selector_naf_ciematvar_forHe(events, qsel):
    events = cut_background_reduce(events)
    events = CutIsNaF(events)
    events = CutRichIsGoodClean(events)
    events = CutRichCharge(events, qsel)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events)
    events = CutRichBadtiles(events)
    #events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutRichPmts(events, 3)
    events = CutTrackInRichNaFAcceptance(events)
    return events

def TagRichCharge_LIP(events, qsel):
    events = CutIsAgl_LIP(events)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance_LIP(events)
    events = CutTofRichBetaConsistency_LIP(events, 0.05)
    return events

def TagRichProbobility_LIP(events, qsel):
    events = CutIsAgl_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance_LIP(events)
    events = CutTofRichBetaConsistency_LIP(events, 0.05)
    return events


def TagRichUsedhits_LIP(events, qsel):
    events = CutIsAgl_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichFractionPE_LIP(events) 
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance_LIP(events)
    events = CutTofRichBetaConsistency_LIP(events, 0.05)
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
    events = CutTofRichBetaConsistency_LIP(events, 0.05)
    events = CutTrackInRichAglAcceptance_LIP(events)
    return events


def TagRichBetaConsistency_LIP(events, qsel):
    events = CutIsAgl_LIP(events)
    events = CutRichCharge_LIP(events, qsel)
    events = CutRichProbobility_LIP(events)
    events = CutRichUsedhits_LIP(events)
    events = CutRichFractionPE_LIP(events) 
    events = CutRichBadtiles(events)
    events = CutTofRichBetaConsistency_LIP(events, 0.05)
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
    events = CutTofRichBetaConsistency_LIP(events, 0.05)
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
    events = CutTofRichBetaConsistency(events, 0.05)
    events = CutRichCharge(events, qsel)
    return events

def TagRichIsGoodClean(events, qsel):
    events = CutIsAgl(events)
    events = CutRichProbobility(events)
    events = CutRichFractionPE(events)
    events = CutRichBadtiles(events)
    events = CutRichBetaConsistency(events, 0.0, 0.005)
    events = CutTrackInRichAglAcceptance(events)
    events = CutTofRichBetaConsistency(events, 0.05)
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
    events = CutTofRichBetaConsistency(events, 0.05)
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
    events = CutTofRichBetaConsistency(events, 0.05)    
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
    events = CutTofRichBetaConsistency(events, 0.05)
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
    events = CutTofRichBetaConsistency(events, 0.05)
    return events

def selector_agl_betabias(events, riglim, qsel):
    events = CutHighRig(events, riglim)
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
    events = CutHighRig(events, riglim)
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
    events = CutHighRig(events, 200)
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

