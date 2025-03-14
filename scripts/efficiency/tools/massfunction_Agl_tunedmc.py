from .MassFunction import InverseMassFunctionFit
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import tools.roottree as read_tree
from tools.calculator import calc_mass, calc_ekin_from_beta, calc_betafrommomentom

import uproot
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, LeastSquares, NormalConstraint
from iminuit.util import describe, make_func_code
from tools.constants import ISOTOPES, NUCLEI_NUMBER, ISOTOPES_COLOR, ISO_LABELS, ISOTOPES_MASS
from tools.histograms import Histogram
from tools.functions import gaussian, asy_gaussian, poly

from scipy import interpolate
from tools.graphs import MGraph, plot_graph
import uncertainties
from uncertainties import unumpy
from uncertainties import ufloat
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, plot1d_step, FONTSIZE, set_plot_defaultstyle


def expo_func(x, pa, pb, pc):    
    pdf = pa* (1 - np.exp((x-pb)/pc)) 
    return pdf

class AglInverseMassFunctionFit(InverseMassFunctionFit):
    def __init__(self, isotopes, hist_data, fit_energy_range, fit_mass_range, detector, is_constraint, component="All"):
        InverseMassFunctionFit.__init__(self, isotopes, hist_data, fit_energy_range, fit_mass_range, detector, is_constraint, component="All")
        
    #print initial values and print fit results    
    def make_mass_function_simultaneous(self, drawiso="All"):
        mass_binwidth = self.mass_binning.bin_widths[self.fit_mass_binrange[0]: self.fit_mass_binrange[1] +1]
        #num_energybin = self.fit_energy_binrange[1] - self.fit_energy_binrange[0] + 1
        #print("num_energybin:", num_energybin)
        num_energybin = self.num_energybin
        isotopes_atom_num = [NUCLEI_NUMBER[iso] for iso in self.isotopes] 
        def mass_function(x, *pars):
            energy = x[0,:].reshape((num_energybin, -1))
            mass = x[1,:].reshape((num_energybin, -1))
            mua, mub, muc = pars[0:3]
            siga, sigb, sigc, sigd = pars[3:7]
            fraccore = pars[7] * np.ones_like(energy)
            sigma_ratio_a, sigma_ratio_b = pars[8:10]   
            asy_factor_a, asy_factor_b, asy_factor_c = pars[10:13]

            num_iso = len(self.isotopes)
            #3 for mean, 4 for sigma, 1:fraccore, 1 sigma_ratio, 3 asy_factor,  3 * (num_iso -1) for scaling factor of sigma
            numpvalues = 13
            num_common_pars = numpvalues + 3 * (num_iso -1)
    
            if self.is_constraint:
                norm = [np.array(pars[num_common_pars + (i * num_energybin) : num_common_pars + (i+1) * num_energybin]) for i in range(len(self.isotopes)-1)]
                norm_last_iso = self.get_data_infitrange().sum(axis=1) - np.sum(norm, axis=0) 
                norm.append(norm_last_iso)
            else:
                norm = [np.array(pars[num_common_pars + (i * num_energybin) : num_common_pars + (i+1) * num_energybin]) for i in range(len(self.isotopes))]
                
            mean = poly(np.log(energy), mua, mub, muc)
            sigma = poly(np.log(energy), siga, sigb, sigc, sigd)
            asy_factor = poly(np.log(energy), asy_factor_a, asy_factor_b, asy_factor_c)
            sigma_ratio = poly(np.log(energy), sigma_ratio_a, sigma_ratio_b)
            pdf = np.zeros(energy.shape)
            pdf_iso = {iso: np.zeros(energy.shape) for iso in self.isotopes}
        
            niso_factor = len(self.isotopes)
            scale_asyfactor = np.array([1.0, 1.0, 1.0])
            #scale_factors_mean = np.array([1.0, ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be9'], ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be10']])
            scale_factors_mean = {"Be9": np.array([0.71421182,  0.11769659, -0.07044384,  0.01372702]),
                                  "Be10": np.array([0.63930723,  0.11484813, -0.07090506,  0.01428052])}
                                  #"Be10": np.array([0.718554,   -0.02194944,  0.00624895])}

            scale_factors_sigma = {"Be9": np.array([0.78669512, -0.01406936,  0.00871327]),
                                  "Be10": np.array([0.6661744,   0.03194309, -0.00202644])}

            
            for i, n in enumerate(norm):
                isofactor = ISOTOPES_MASS[self.isotopes[0]]/ISOTOPES_MASS[self.isotopes[i]] 
                isotope = self.isotopes[i]
                
                if i == 0:
                    rigsigma_factor = 1.0
                    mean_scale = 1.0
                    sigma_scale = 1.0
                else:
                    rigsigma_factor = expo_func(energy, *pars[numpvalues + (i-1)*3: numpvalues + i*3])
                    #mean_scale = poly(np.log(energy), *scale_factors_mean[self.isotopes[i]])
                    #sigma_scale = poly(np.log(energy), *scale_factors_sigma[self.isotopes[i]])
                    mean_scale = isofactor
                    sigma_scale = isofactor / rigsigma_factor
                    
                coregaus = gaussian(mass, mean * mean_scale, sigma * sigma_scale)
                asygaus = asy_gaussian(mass, mean * mean_scale,  sigma_ratio * sigma * sigma_scale, asy_factor * scale_asyfactor[i])
                pdf += n[:, None] * (fraccore * coregaus + (1 - fraccore) * asygaus)  * mass_binwidth[None, :]
                pdf_iso[isotope] = n[:, None] * (fraccore * coregaus + (1 - fraccore) * asygaus)  * mass_binwidth[None, :]  

            if drawiso == "All":
                return pdf.reshape((-1, ))
            else:
                return pdf_iso[f'{drawiso}'].reshape((-1, ))

        if self.is_constraint:
            parnames = ['x', 'mua', 'mub', 'muc', 'siga', 'sigb', 'sigc', 'sigd', 'fraccore', 'sigma_ratio_a', 'sigma_ratio_b', 'asy_factor_a', 'asy_factor_b', 'asy_factor_c'] + [f'ex{a}_{iso}' for iso in self.isotopes[1:] for a in ["a", "b", "c"]] +  [f"n{isonum}_{ibin}" for isonum in isotopes_atom_num[:-1]  for ibin in range(num_energybin)]
        else:
            parnames = ['x', 'mua', 'mub', 'muc', 'siga', 'sigb', 'sigc', 'sigd', 'fraccore', 'sigma_ratio_a', 'sigma_ratio_b',  'asy_factor_a', 'asy_factor_b', 'asy_factor_c'] + [f'ex{a}_{iso}' for iso in self.isotopes[1:] for a in ["a", "b", "c"]] +  [f"n{isonum}_{ibin}" for isonum in isotopes_atom_num  for ibin in range(num_energybin)]
       
        mass_function.func_code = make_func_code(parnames)
        return mass_function
    
    def get_polypars_index(self):
        parindex = {'mua': 0, 'mub':1, 'muc':2, 'siga':3, 'sigb':4, 'sigc':5, 'sigd': 6, 'fraccore':7, 'sigma_ratio_a':8,  'sigma_ratio_b':9, 'asy_factor_a':10, 'asy_factor_b':11, 'asy_factor_c':12}
        return parindex

    def make_mass_function_binbybin(self, component="All"):
        mass_binwidth = self.mass_binning.bin_widths[self.fit_mass_binrange[0]: self.fit_mass_binrange[1] +1]
        #num_energybin = self.fit_energy_binrange[1] - self.fit_energy_binrange[0] + 1
        #print("num_energybin:", num_energybin)
        num_energybin = self.num_energybin
        num_massbin = self.num_massbin
        isotopes_atom_num = [NUCLEI_NUMBER[iso] for iso in self.isotopes] 
        def mass_function(x, *pars):
            energy = x[0,:].reshape((num_energybin, -1))
            mass = x[1,:].reshape((num_energybin, -1))
            pars = np.array(pars)
            mean = np.hstack([pars[0:num_energybin,None]] * num_massbin)
            #print("mean", mean.shape)
            sigma =  np.hstack([pars[num_energybin: 2*num_energybin, None]] * num_massbin) 
            fraccore =  np.hstack([pars[2*num_energybin: 3*num_energybin, None]] * num_massbin) 
            sigma_ratio =np.hstack([pars[3*num_energybin: 4*num_energybin, None]] * num_massbin)
            asy_factor = np.hstack([pars[4*num_energybin: 5*num_energybin, None]] * num_massbin) 
            norm = [pars[(5+i)*num_energybin : (6+i)*num_energybin, None]  for i in range(len(self.isotopes))]
            pdf = np.zeros(energy.shape)
            pdfgaus = np.zeros(energy.shape)
            pdfasygus = np.zeros(energy.shape)
            
            #print initial values and print fit results                                                    
            for isonum, n in zip(isotopes_atom_num, norm):
                isofactor = isonum/isotopes_atom_num[0]
                coregaus = gaussian(mass, mean/isofactor, sigma/isofactor)
                #print("mass.shape:", mass.shape, (mean/isofactor).shape, (sigma_ratio * sigma/isofactor).shape, asy_factor.shape)
                asygaus = asy_gaussian(mass, mean/isofactor,  (sigma_ratio * sigma)/isofactor, asy_factor)
                pdfgaus += n * fraccore * coregaus  * mass_binwidth[None, :]
                pdfasygus += n * (1 - fraccore) * asygaus * mass_binwidth[None, :]
                pdf += n * (fraccore * coregaus + (1 - fraccore) * asygaus)  * mass_binwidth[None, :]
            if component == "All":
                return pdf.reshape((-1, ))
            elif component == "gaus":
                return pdfgaus.reshape((-1, ))
            else:
                return pdfasygus.reshape((-1, ))
            
        parnames = (['x'] + [f'mean_{ibin}' for ibin in range(num_energybin)] + [f'sigma_{ibin}' for ibin in range(num_energybin)] +
                    [f'fraccore_{ibin}' for ibin in range(num_energybin)] +
                    [f'sigma_ratio_{ibin}' for ibin in range(num_energybin)] +
                    [f'asy_factor_{ibin}' for ibin in range(num_energybin)] +
                    [f"n{isonum}_{ibin}" for isonum in isotopes_atom_num  for ibin in range(num_energybin)])
                   
        #print("length parnames:", len(parnames))
        mass_function.func_code = make_func_code(parnames)
        return mass_function


        
