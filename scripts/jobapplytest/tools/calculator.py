import numpy as np
from .constants import MC_PARTICLE_MASSES, MC_PARTICLE_IDS, MC_PARTICLE_CHARGES
from .constants import ISOTOPES_CHARGE, ISOTOPES_MASS, MASS_NUCLEON_GEV

mass_nucleon_gev = 0.9314941 #in GeV
#mass_nucleon_gev = 0.938 #in GeV

def calc_ratio_and_err(a, b, delta_a, delta_b):
    ratio = a/b
    err = a/b * np.sqrt((delta_a / a)**2 + (delta_b / b)**2)
    return ratio, err

def calc_ratio_err(a, b, delta_a, delta_b):        
  return a/b * np.sqrt((delta_a / a)**2 + (delta_b / b)**2)   


def calc_mass(beta, rigidity, charge):
    return np.sqrt(charge**2 * rigidity**2 * (1 - beta**2) /beta**2)

def calc_inverse_mass(beta, rigidity, charge):
    return np.sqrt(beta**2/(charge**2 * rigidity**2 * (1 - beta**2)))

def calc_rig(beta, mass, charge):
    return mass * beta /(charge * np.sqrt((1 - beta**2)))

def calc_rig_from_ekin(ekin, mass, charge):
    gamma = ekin/mass_nucleon_gev + 1
    beta = np.sqrt(1 - 1/gamma**2)
    rig = mass * beta * gamma / charge
    return rig
    
def calc_beta(rigidity, mass, charge):
    rqm = (rigidity * charge / mass)**2
    return np.sqrt(rqm / (rqm + 1))

def calc_gamma(beta):
    gamma = 1/np.sqrt(1- beta**2)
    return gamma

def calc_gamma_from_ekin(ekin):
    gamma = ekin/MASS_NUCLEON_GEV + 1.0
    return gamma

def calc_gamma_from_momentum(momentum, mass):
    rqm = (momentum/mass) **2
    return np.sqrt(rqm + 1)
    
def calc_ekin_from_beta(beta):
    ekin = mass_nucleon_gev * (1/ np.sqrt(1 - beta**2) -1)
    ekin = np.nan_to_num(ekin, posinf=np.inf)
    return ekin

def calc_beta_from_ekin(ekin):
    beta = np.sqrt(1- (1/(ekin/mass_nucleon_gev + 1 )) **2)
    beta = np.nan_to_num(beta, posinf=np.inf) 
    return beta

def gaussian(x, A, mu, sigma):
    return A * 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def calc_ekin_from_rigidity(rigidity, particleId):
    mass = MC_PARTICLE_MASSES[particleId]
    charge = MC_PARTICLE_CHARGES[particleId]
    rqm = (rigidity * charge / mass)**2
    return mass_nucleon_gev * (np.sqrt(1 + rqm) - 1)

def calc_ekin_from_rigidity_iso(rigidity, iso):
    mass = ISOTOPES_MASS[iso]
    charge = ISOTOPES_CHARGE[iso]
    rqm = (rigidity * charge / mass)**2
    return mass_nucleon_gev * (np.sqrt(1 + rqm) - 1)

def calc_betafrommomentom(mmom, particleId):
    mass = MC_PARTICLE_MASSES[particleId]
    charge = MC_PARTICLE_CHARGES[particleId]
    rigidity = mmom/charge
    rqm = (rigidity * charge / mass)**2
    return np.sqrt(rqm / (rqm + 1))


def calculate_efficiency(passed, all):
    return (passed + 1) / (all + 2)

def calculate_efficiency_error(passed, all):
    k = passed
    n = all
    return np.sqrt(((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - (k + 1)**2 / (n + 2)**2)

def calculate_efficiency_and_error(passed, all, datatype):
    if datatype == "ISS":
        eff, err = calculate_efficiency(passed, all), calculate_efficiency_error(passed, all)
    else:
        eff = calculate_efficiency_weighted(passed, all)
        err = eff * (1.0 - eff)/passed
    return eff, err


def calculate_efficiency_weighted(passed_values, total_values):                                                                                                                                        
    return passed_values / total_values

def calculate_efficiency_error_weighted(passed_values, total_values, passed_squared_values, total_squared_values):
    return np.sqrt(2*passed_squared_values**2 * (total_squared_values + passed_squared_values - 2*passed_values*total_values)) / (total_squared_values)**2

def calculate_efficiency_and_error_weighted(passed_values, total_values, passed_squared_values, total_squared_values): 
    return calculate_efficiency_weighted(passed_values, total_values), calculate_efficiency_error_weighted(passed_values, total_values, passed_squared_values, total_squared_values)     

def get_acceptance(file_tree, treename, file_pgen, trigger, isotope, variable):
    root_pgen = TFile.Open(file_pgen[isotope], "READ")
    hist_pgen = root_pgen.Get("PGen_px")
    print("trigger from pgen: ", hist_pgen.Integral())
    nbins_pgen = hist_pgen.GetNbinsX()
    minLogMom = None
    maxLogMom = None
    charge = MC_PARTICLE_CHARGES[MC_PARTICLE_IDS[isotope]]
    for i in range(nbins_pgen):
        if hist_pgen.GetBinContent(i+1) > 0 :
            if minLogMom is None:
                minLogMom = hist_pgen.GetXaxis().GetBinLowEdge(i+1)
            maxLogMom = hist_pgen.GetXaxis().GetBinUpEdge(i+1)
            
    
    binning = xbinning[variable]
    minMom = 10**(minLogMom)
    maxMom = 10**(maxLogMom)
    minLogMom_v1 = np.log10(4)
    maxLogMom_v1 = np.log10(8000)

    #print(minMom, maxMom)
    minRigGen = 10**(minLogMom_v1)/charge
    maxRigGen = 10**(maxLogMom_v1)/charge
    minEkinGen = calc_ekin_from_rigidity(minRigGen, MC_PARTICLE_IDS[isotope])
    maxEkinGen = calc_ekin_from_rigidity(maxRigGen, MC_PARTICLE_IDS[isotope])
    #print("minLogMom:", minLogMom_v1, "maxLogMom:", maxLogMom_v1)
    #print("minLogMom:", minLogMom, "maxLogMom:", maxLogMom)
    #print("minRigGen:", minRigGen, "maxRigGen:", maxRigGen)
        
    hist_total = ROOT.TH1F("hist_total", "hist_total",  len(binning)-1, binning)                                          
    rootfile = TFile.Open(file_tree, "READ")
    nucleitree = rootfile.Get(treename)
    hist_pass = TH1F("hist_pass", "hist_pass", len(binning)-1, binning)
    tot_trigger = trigger[isotope]
    
    if variable == "Rigidity":
        for ibin in range(1, len(binning)):
            #print(ibin, binning[ibin], binning[ibin - 1])
            frac = (np.log(binning[ibin]) - np.log(binning[ibin - 1]))/(np.log(maxRigGen) - np.log(minRigGen))                                                                                  
            num = tot_trigger * frac                                                                                                                                     
            hist_total.SetBinContent(ibin, num)
            
        for entry in nucleitree:
            arr_evmom = entry.mmom
            rig_gen = arr_evmom/charge
            hist_pass.Fill(rig_gen)
            ekin_gen = calc_ekin_from_rigidity(rig_gen, MC_PARTICLE_IDS[isotope])
            
    elif (variable == "Ekin"):
        for ibin in range(1, len(binning)):
            upbinrig = calc_rig_from_ekin(binning[ibin], ISOTOPES_MASSES[isotope], charge)
            lowbinrig = calc_rig_from_ekin(binning[ibin-1], ISOTOPES_MASSES[isotope], charge)
            frac = (np.log(upbinrig) - np.log(lowbinrig))/(np.log(maxRigGen) - np.log(minRigGen))                                                                                  
            num = tot_trigger * frac                                                                                                                                     
            hist_total.SetBinContent(ibin, num)
        
        for entry in nucleitree:
            arr_evmom = entry.mmom
            weight = entry.ww
            rig_gen = arr_evmom/charge
            ekin_gen = calc_ekin_from_rigidity(rig_gen, MC_PARTICLE_IDS[isotope])
            hist_pass.Fill(ekin_gen)
    else:
        raise ValueError("Wrong variable is given")

    print("the total trigger: ", tot_trigger)
    print("the number of passed events: ",  hist_pass.Integral())
    arr_tot = np.zeros(len(binning) -1)
    arr_pass = np.zeros(len(binning) -1) 
    for i in range(len(binning) - 1):
        arr_tot[i] = hist_total.GetBinContent(i+1)
        arr_pass[i] = hist_pass.GetBinContent(i+1)

    eff, efferr = calculate_efficiency_and_error(arr_pass, arr_tot, "MC")
    acc = eff * 3.9 * 3.9 * np.pi
    accerr = efferr * 3.9 * 3.9 * np.pi
    xbincenter = get_bin_center(binning)
    for i in range(len(binning) - 1): 
        print(arr_pass[i],  arr_tot[i] , arr_pass[i]/arr_tot[i] * 3.9 * 3.9 * np.pi, acc[i])
        
    return acc, accerr, xbincenter             
