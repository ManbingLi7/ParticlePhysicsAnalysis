
import numpy as np

from .constants import NAF_INDEX, AGL_INDEX, NUCLEON_MASS

def calc_mass(beta, rigidity, charge):
    return np.sqrt(charge**2 * rigidity**2 * (1 - beta**2) / beta**2)

def calc_rig(beta, mass, charge):
    return np.sqrt(mass**2 / charge**2 * beta**2 / (1 - beta**2))

def calc_beta(rigidity, mass, charge):
    rqm = (rigidity * charge / mass)**2
    return np.sqrt(rqm / (rqm + 1))
