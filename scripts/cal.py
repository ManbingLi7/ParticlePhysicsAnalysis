import numpy as np
from tools.calculator import calc_rig_from_ekin, calc_ekin_from_rigidity_iso, calc_beta_from_ekin
from tools.binnings_collection import Rigidity_Analysis_Binning_FullRange
from tools.constants import ISOTOPES_MASS

def calc_rig(beta, mass, charge):                                                                                                                                              
    return np.sqrt(mass**2 / charge**2 * beta**2 /(1 - beta**2))

def calc_beta(rigidity, mass, charge):                                                                                         
    rqm = (rigidity * charge / mass)**2                                                                                                                                                                    
    return np.sqrt(rqm / (rqm + 1))

mass_nucleon_gev = 0.9134
def calc_ekin_from_beta(beta):                                                                                            
    return mass_nucleon_gev * (1/ np.sqrt(1 - beta**2) -1)                                                                     


def calc_ekin_from_rigidity(rigidity, mass,charge):                                 
    rqm = (rigidity * charge / mass)**2                                              
    return mass_nucleon_gev * (np.sqrt(1 + rqm) - 1)   


mass = 10.0
mass2 = 11.0
#print(calc_rig_from_ekin(np.array([0.4, 1.0, 0.9, 4.4, 3.6, 11.4]), mass * np.ones(6), np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])))
#print(calc_rig_from_ekin(np.array([0.4, 1.0, 0.9, 4.4, 3.6, 11.4]), mass2 * np.ones(6), np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])))

#print(Rigidity_Analysis_Binning_FullRange())
#print(calc_ekin_from_rigidity_iso(Rigidity_Analysis_Binning_FullRange(), 'Be7', 4.0))

#print(calc_beta_from_ekin(0.269))

print(calc_ekin_from_rigidity_iso(10,  'Be7'))








