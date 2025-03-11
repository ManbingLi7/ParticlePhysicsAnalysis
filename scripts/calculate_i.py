import numpy as np
mass_nucleon_gev = 0.938
def calc_rig_from_ekin(ekin, mass, charge):
    gamma = ekin/mass_nucleon_gev + 1
    beta = np.sqrt(1 - 1/gamma**2)
    rig = mass * beta * gamma / charge
    return rig

a = calc_rig_from_ekin(12, 12, 6.0)
print(a)
