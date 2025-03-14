import numpy as np
import awkward as ak
from datetime import datetime
from .binnings import Binning
from .calculator import calc_beta_from_ekin

def compute_dayfromtime(timeinsec):                                                                                                                                                                        
    starttime = 1305853512                                                                                                                                                                              
    day = (timeinsec -starttime)/(60*60*24)                                                                                                                                                            
    return day 

def make_year_binning(insec):
    min_year = datetime.strptime("2011-01-01", "%Y-%m-%d").year
    max_year = datetime.strptime("2022-01-01", "%Y-%m-%d").year
    years = np.arange(min_year, max_year + 1)
    timestamps = np.array([datetime(year, 1, 1).timestamp() for year in years])
    return timestamps if insec else years

def make_days_binnning():
    binning = np.linspace(0, 3600, 60)
    return binning

def get_nbins_in_range(range_energy, binning):
    range_energy_bin = np.digitize(range_energy, binning)
    n_bins = range_energy_bin[1] - range_energy_bin[0]
    return n_bins

def get_sub_binning(range_energy, binning):
    range_energy_bin = np.digitize(range_energy, binning)
    sub_energy_binning = binning[range_energy_bin[0]:range_energy_bin[1]+1]
    return sub_energy_binning

def get_bin_center(binning):
    bin_center = (binning[1:] + binning[:-1])/2
    return bin_center

def get_binindices(values, binning_edges): 
    return np.digitize(ak.to_numpy(values), binning_edges) - 1

def Rigidity_Analysis_Binning_FullRange():
 RigidityBinningFullRange = np.array([0.8, 1.0, 1.16, 1.33, 1.51, 1.71, 1.92, 2.15, 2.4, 2.67, 2.97, 3.29, 3.64, 4.02, 4.43, 4.88, 5.37, 5.9, 
                                             6.47, 7.09, 7.76, 8.48, 9.26, 10.1, 11, 12, 13, 14.1, 15.3, 16.6, 18, 19.5, 21.1, 22.8, 
                                             24.7, 26.7, 28.8, 31.1, 33.5, 36.1, 38.9, 41.9, 45.1, 48.5, 52.2, 56.1, 60.3, 64.8, 69.7, 
                                             74.9, 80.5, 86.5, 93, 100, 108, 116, 125, 135, 147, 160, 175, 192, 211, 233, 259, 291, 330, 
                                             379, 441, 525, 660, 880, 1300, 3300])
 return RigidityBinningFullRange

def Rigidity_Analysis_Binning():
 RigidityBinningFullRange = np.array([1.92, 2.15, 2.4, 2.67, 2.97, 3.29, 3.64, 4.02, 4.43, 4.88, 5.37, 5.9, 
                                             6.47, 7.09, 7.76, 8.48, 9.26, 10.1, 11, 12, 13, 14.1, 15.3, 16.6, 18, 19.5, 21.1, 22.8, 
                                             24.7, 26.7, 28.8, 31.1, 33.5, 36.1, 38.9, 41.9, 45.1, 48.5, 52.2, 56.1, 60.3, 64.8, 69.7, 
                                             74.9, 80.5, 86.5, 93, 100, 108, 116, 125, 135, 147, 160, 175, 192, 211, 233, 259, 291, 330, 
                                             379, 441, 525, 660, 880, 1300, 3300])
 return RigidityBinningFullRange

def BeRigidityBinningRICHRange():
 RigidityBinningRICHRange = np.array([1.92, 2.15, 2.4, 2.67, 2.97, 3.29, 3.64, 4.02, 4.43, 4.88, 5.37, 5.9, 
                                      6.47, 7.09, 7.76, 8.48, 9.26, 10.1, 11, 12, 13, 14.1, 15.3, 16.6, 18, 19.5, 21.1, 22.8, 
                                      24.7, 26.7, 28.8, 31.1, 33.5, 36.1, 38.9, 41.9, 45.1, 48.5, 52.2, 56.1, 60.3, 64.8, 69.7, 
                                      74.9, 80.5, 86.5, 93, 100])
 return RigidityBinningRICHRange

def BeRigidityBinningUnfold():
 RigidityBinningRICHRange = np.array([0.8, 1.0, 1.16, 1.33, 1.51, 1.71,  1.92, 2.15, 2.4, 2.67, 2.97, 3.29, 3.64, 4.02, 4.43, 4.88, 5.37, 5.9, 
                                      6.47, 7.09, 7.76, 8.48, 9.26, 10.1, 11, 12, 13, 14.1, 15.3, 16.6, 18, 19.5, 21.1, 22.8, 
                                      24.7, 26.7, 28.8, 31.1, 33.5, 36.1, 38.9, 41.9, 45.1, 48.5, 52.2, 56.1, 60.3, 64.8, 69.7, 
                                      74.9, 80.5, 86.5, 93, 100])
 return RigidityBinningRICHRange


def LithiumRigidityBinningFullRange():
 LithiumRigidityBinningFullRange = np.array([1.33, 1.51, 1.71, 1.92, 2.15, 2.4, 2.67, 2.97, 3.29, 3.64, 4.02, 4.43, 4.88, 5.37, 5.9, 
                                             6.47, 7.09, 7.76, 8.48, 9.26, 10.1, 11, 12, 13, 14.1, 15.3, 16.6, 18, 19.5, 21.1, 22.8, 
                                             24.7, 26.7, 28.8, 31.1, 33.5, 36.1, 38.9, 41.9, 45.1, 48.5, 52.2, 56.1, 60.3, 64.8, 69.7, 
                                             74.9, 80.5, 86.5, 93, 100, 108, 116, 125, 135, 147, 160, 175, 192, 211, 233, 259, 291, 330, 
                                             379, 441, 525, 660, 880, 1300, 1800])
 return LithiumRigidityBinningFullRange

def LithiumRigidityBinning():
 LithiumRigidityBinning = np.array([1.92, 2.15, 2.4, 2.67, 2.97, 3.29, 3.64, 4.02, 4.43, 4.88, 5.37, 5.9,
                                    6.47, 7.09, 7.76, 8.48, 9.26, 10.1, 11, 12, 13, 14.1, 15.3, 16.6, 18, 19.5, 21.1, 22.8,
                                    24.7, 26.7, 28.8, 31.1, 33.5, 36.1, 38.9, 41.9, 45.1, 48.5, 52.2, 56.1, 60.3, 64.8, 69.7,
                                    74.9, 80.5, 86.5, 93, 100, 108, 116, 125, 135, 147, 160, 175, 192, 211, 233, 259, 291, 330,
                                    379, 441, 525, 660, 880, 1300])
 return LithiumRigidityBinning

def LithiumRigidityBinningRICH():
 LithiumRigidityBinning = np.array([5.37, 5.9,
                                    6.47, 7.09, 7.76, 8.48, 9.26, 10.1, 11, 12, 13, 14.1, 15.3, 16.6, 18, 19.5, 21.1, 22.8,
                                    24.7, 26.7, 28.8, 31.1, 33.5, 36.1, 38.9, 41.9, 45.1, 48.5, 52.2, 56.1, 60.3, 64.8, 69.7,
                                    74.9, 80.5, 86.5, 93, 100, 108, 116, 125, 135, 147, 160, 175, 192, 211, 233, 259, 291, 330,
                                    379, 441, 525, 660, 880, 1300])
 return LithiumRigidityBinning

def delta_beta_binning():
    return np.linspace(-0.005, 0.005, 1000)


def LithiumRichAglBetaResolutionBinning():
    LithiumRichAglBetaResolutionBinning = np.array([0.952, 0.95786, 0.960437, 0.962209, 0.96361, 0.96483, 0.965949, 0.967, 0.967993,
                                                    0.968977, 0.969987, 0.97104, 0.972153, 0.97334, 0.974614, 0.975972, 0.977412, 
                                                    0.978944, 0.980588, 0.982383, 0.984354, 0.986508, 0.988829, 0.991324, 0.994032, 0.99692, 0.999741, 
                                                    1.00233, 1.00472, 1.0069, 1.00891, 1.01075, 1.01245, 1.014, 1.01542, 1.01672, 1.01791, 1.01901, 
                                                    1.02001, 1.02092, 1.02176, 1.02253, 1.02323, 1.02388, 1.02447, 1.02501, 1.0255, 1.02595, 1.02636, 
                                                    1.02674, 1.02708, 1.0274, 1.02769, 1.02795, 1.02819, 1.02841, 1.02861, 1.0288, 1.02897, 1.02912, 
                                                    1.02926, 1.02939, 1.02951, 1.02962, 1.02971, 1.0298, 1.02989, 1.02996, 1.03])
    return LithiumRichAglBetaResolutionBinning


def LithiumRichAglBetaBinning():
    LithiumRichAglBetaResolutionBinning = np.array([0.952, 0.95786, 0.960437, 0.962209, 0.96361, 0.96483, 0.965949, 0.967, 0.967993,
                                                    0.968977, 0.969987, 0.97104, 0.972153, 0.97334, 0.974614, 0.975972, 0.977412, 
                                                    0.978944, 0.980588, 0.982383, 0.984354, 0.986508, 0.988829, 0.991324, 0.994032, 0.99692, 0.999741, 
                                                    1.0])
    return LithiumRichAglBetaResolutionBinning

def LithiumBetaResolutionBinning():
    LithiumRichAglBetaResolutionBinning = np.array([0.95,  0.95786,  0.962209,  0.96483,  0.967, 
                                                    0.968977,  0.97104,  0.97334,  0.975972,  
                                                    0.978944,  0.982383,  0.986508,  0.991324,  0.99692, 
                                                    1.00233,  1.0069,  1.01075,  1.014,  1.01672,  1.01901, 
                                                    1.02092,  1.02253,  1.02388,  1.02501,  1.02595,  
                                                    1.02674,  1.0274,  1.02795,  1.02841,  1.0288, 1.02912, 
                                                     1.02939,  1.02962, 1.0298,  1.02996])
    return LithiumRichAglBetaResolutionBinning



def LithiumBetaBinning():
    LithiumRichAglBetaBinning = np.array([0.95,  0.95786,  0.962209,  0.96483,  0.967, 
                                                    0.968977,  0.97104,  0.97334,  0.975972,  
                                                    0.978944,  0.982383,  0.986508,  0.991324,  0.99692, 1.00001])
    return LithiumRichAglBetaBinning


def fbinning_energy():
    binning = np.array([0.338, 0.4185, 0.5077,0.6103,0.7264,0.8561,1.0045,1.1666,1.3476,1.5473,1.7659,2.0085,2.2753,2.5662,2.8812,
                        3.2256,3.5996,4.0029,4.4413,4.9146,5.4229,5.9886,6.5553,7.1793,7.8608,8.5998,9.3963,10.2502,11.1616,12.1303])
    return binning

def fbinning_energy_agl():
    binning = np.array([2.5662,2.8812, 3.2256,3.5996,4.0029,4.4413,4.9146,5.4229,5.9886,6.5553,7.1793,7.8608,8.5998,9.3963,10.2502,11.1616,12.1303])
    return binning

def fbinning_energy_extend():
    binning = np.array([2.61181695e-01, 0.338, 0.4185, 0.5077,0.6103,0.7264,0.8561,1.0045,1.1666,1.3476,1.5473,1.7659,2.0085,2.2753,2.5662,2.8812,
                        3.2256,3.5996,4.0029,4.4413,4.9146,5.4229,5.9886,6.5553,7.1793,7.8608,8.5998,9.3963,10.2502,11.1616,12.1303, 1.32009719e+01, 1.43236147e+01, 1.54953869e+01, 1.67651002e+01])
    return binning

def fbinning_energy_rebin_extend():
    binning = np.array([2.61181695e-01, 0.4185, 0.6103, 0.8561, 1.1666, 1.5473, 2.0085, 2.5662, 
                        3.2256, 4.0029, 4.9146, 5.9886, 7.1793, 8.5998, 10.2502, 12.1303, 1.43236147e+01, 1.67651002e+01])
    return binning

def fbinning_energy_rebin():
    binning = np.array([0.4185, 0.6103, 0.8561, 1.1666, 1.5473, 2.0085, 2.5662, 
                        3.2256, 4.0029, 4.9146, 5.9886, 7.1793, 8.5998, 10.2502, 12.1303])

    return binning

def fbinning_beta():
    beta_bins = np.array([0.67956, 0.72381, 0.7622929739593237, 0.7968619746197845, 0.8272372846220045, 0.853503536110387, 0.8766415673608506,
                          0.8960414182435318, 0.9126636919966328, 0.9267072194282836, 0.9384812791912239, 0.9484808674957008, 0.9568828014038655,
                          0.963885839452842,  0.9696964079962642, 0.9745724529902703, 0.978640846026234, 0.9820203455095603, 0.9848564548217835,
                          0.9872244834005945, 0.9891973378900487, 0.9908991144078576, 0.992229917447369, 0.9933833333253138, 0.9943721022952889,
                          0.9952129963184398, 0.9959243439958353, 0.9965241026630385, 0.9970290356232191, 0.9974539071746928])
    return beta_bins

def fbinning_beta_rebin():
    beta_bins = np.array([0.630824, 0.72381066, 0.79686118, 0.85350287, 0.89604088, 0.9267068,  0.94848055,
                          0.9638856,  0.97457228, 0.98202022, 0.98722439,  0.99089904,  0.99338328,
                          0.99521296, 0.99652407, 0.99745389])
    return beta_bins


def fbinning_energy_Tof():
    energy_per_neculeon_binning = np.array([0.405434, 0.490126, 0.586704, 0.695356, 0.820378, 0.957741, 1.11175, 
                                            1.2825, 1.46996, 1.67875, 1.90892, 2.16044, 2.43325, 2.73209, 3.05696, 3.40778, 3.78939, 4.20178, 
                                            4.64487, 5.1384, 5.6329, 6.17774, 6.77294, 7.41853, 8.11448, 8.86078, 9.65741, 10.5043, 11.4514, 12.4488, 13.4964, 14.6442, 15.8422, 17.1403, 18.5386, 20.037])
    return energy_per_neculeon_binning



def kinetic_energy_neculeon_binning():
    energy_per_neculeon_binning = np.array([0.267027, 0.332288, 0.405434, 0.490126, 0.586704, 0.695356, 0.820378, 0.957741, 1.11175, 
                                            1.2825, 1.46996, 1.67875, 1.90892, 2.16044, 2.43325, 2.73209, 3.05696, 3.40778, 3.78939, 4.20178, 
                                            4.64487, 5.1384, 5.6329, 6.17774, 6.77294, 7.41853, 8.11448, 8.86078, 9.65741, 10.5043, 11.4514, 12.4488, 
                                            13.4964, 14.6442, 15.8422, 17.1403, 18.5386, 20.037, 21.6355, 23.3341, 25.1829, 27.1317, 29.2306, 31.4796])
    return energy_per_neculeon_binning


def mass_binning():
    mass_binning = np.linspace(3, 13, 200)
    return mass_binning

def fbinning_inversemass(nuclei):
    inverse_mass_binning = {'Li': np.linspace(0.03, 0.35, 220), 'Be':  np.linspace(0.03, 0.3, 200), 'B': np.linspace(0.03, 0.2, 200)}
    return inverse_mass_binning[nuclei]

def inverse_massbinning_from_mass():
    mass_binning = np.linspace(3, 13, 200)
    inverse_mass_binning = 1.0/mass_binning
    return inverse_mass_binning

def fbinning_charge(qsel):
    return np.linspace(qsel - 5, qsel + 4, 80)

def expectedPhotoElectrons_binning():
    return np.linspace(0, 100, 100)


def fbinning_BetaConsistency():
    return np.linspace(0.0, 0.01, 500)

def fbinning_tofRichBetaMatch():
    return np.linspace(0.0, 0.07, 300)

def ring_position_binning():
    return np.linspace(-62, 62, 500)

def richNCollectedPhotoElectrons_binning():
    return np.linspace(0, 200, 100)

def richNPhotoElectrons_binning():
    return np.linspace(0, 200, 200)

def rich_fraction_usedPhotoElectrons_binning():
    return np.linspace(0, 2, 100)

def richNExpectedPhotoElectrons_binning():
    return np.linspace(0, 12, 50)

def fbinning_RichProb():
    return np.linspace(-0.1, 1.1, 120)

def richDistanceToTileBorder_binning():
    return np.linspace(0.0, 7.5, 100)

def fbinning_RichHitStatus():
    return np.linspace(-5, 5, 10)

def fbinning_RichNumberHits():
    return np.linspace(0, 300, 150)

def fbinning_fraction():
    return np.linspace(0, 1.0, 25)

def fbinning_RICHnpe():
    return np.linspace(0, 200, 100)
 
def fbinning_RichAcceptance():
    return np.linspace(0, 1, 50)

def fbinning_Flatness():
    return np.linspace(-2, 2, 100)

def fbinning_RICHindex():
    return np.linspace(1.045, 1.06, 150)

def fbinning_RICHAngle():
    return np.linspace(0, 0.45, 150)

def fbinning_RICHlik():
    return np.linspace(0, 10000, 1000)


def fbinning_RICHPosition():
    return np.linspace(-61, 61, 60)

def fbinning_resolution():
    return np.linspace(-0.004, 0.004, 100)

def InverseMassBinning():
    return np.linspace(0.03, 0.3, 200)

def InverseMassBinning_boron():
    return np.linspace(0.03, 0.2, 250)
