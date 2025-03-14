from .MassFunction_V2 import InverseMassFunctionFit
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
from scipy.optimize import NonlinearConstraint 
from tools.graphs import MGraph, plot_graph
import uncertainties
from uncertainties import unumpy
from uncertainties import ufloat
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, plot1d_step, FONTSIZE, set_plot_defaultstyle


class LeastSquareMassFit:

    errordef = Minuit.LEAST_SQUARES  # for Minuit to compute errors correctly

    def __init__(self, model, x, y, yerr, guess,  energybins):
        self.model = model
        self.xvalues = x
        self.yvalues = y
        self.yvalueserr = yerr                                                        
        pars = describe(model)
        lsq_parameters = pars[1:]
        self.pmc = [par for key, par in guess.items()]
        print('pmc:', self.pmc)
        print('lsq_parameters:', lsq_parameters)
        self.func_code = make_func_code([par for par in lsq_parameters])
        #self._parameters = list(lsq_parameters)
        #self._parameters = ['mua', 'mub', 'muc']
        self.num_energybin = energybins
        
    def __call__(self, *pars):
        #print(self._parameters)
        yfit = self.model(self.xvalues , *pars)
        energy = self.xvalues[0,:].reshape((self.num_energybin, -1))
        mass = self.xvalues[1,:].reshape((self.num_energybin, -1))    
 
        siga, sigb, sigc, sigd = pars[3:7]
        sigma_ratio_a, sigma_ratio_b, sigma_ratio_c = pars[8:11]                                                                                                                                             
        asy_factor_a, asy_factor_b, asy_factor_c = pars[11:14]
        sigma = poly(np.log(energy), siga, sigb, sigc, sigd)
        fraccore = pars[7] * np.ones_like(energy)                                                                                                                                                              
        asy_factor = poly(np.log(energy), asy_factor_a, asy_factor_b, asy_factor_c)
        sigma_ratio = poly(np.log(energy), sigma_ratio_a, sigma_ratio_b, sigma_ratio_c)
        
        siga_mc, sigb_mc, sigc_mc, sigd_mc = self.pmc[3:7]
        sigma_ratio_amc, sigma_ratio_bmc, sigma_ratio_cmc = self.pmc[8:11]
        asy_factor_amc, asy_factor_bmc, asy_factor_cmc = self.pmc[11:14]
        
        sigma_guess = poly(np.log(energy), siga_mc, sigb_mc, sigc_mc, sigd_mc)
        sigma_err = 0.01 * sigma_guess
        fraccore_guess = self.pmc[7] * np.ones_like(energy)
        fraccore_guess_err = fraccore_guess * 0.05
        
        asy_factor_guess = poly(np.log(energy), asy_factor_amc, asy_factor_bmc, asy_factor_cmc)
        asy_factor_guess_err = asy_factor_guess * 0.05
        #sigma_ratio = poly(np.log(energy), sigma_ratio_a, sigma_ratio_b, sigma_ratio_c)
        sigma_ratio_guess = poly(np.log(energy), sigma_ratio_amc, sigma_ratio_bmc, sigma_ratio_cmc)
        sigma_ratio_guess_err = sigma_ratio_guess * 0.05
        
        constraint_sigma = np.sum((sigma - sigma_guess)**2/sigma_err**2)
        constraint_fraccore = np.sum((fraccore - fraccore_guess)**2/fraccore_guess_err**2)
        constraint_sigmaratio = np.sum((sigma_ratio - sigma_ratio_guess)**2/sigma_ratio_guess_err**2)
        constraint_asyfactor = np.sum((asy_factor - asy_factor_guess)**2/asy_factor_guess_err**2)
        
        chisquare = np.sum((self.yvalues - yfit) ** 2 / self.yvalueserr ** 2) + (constraint_sigma + constraint_fraccore + constraint_sigmaratio + constraint_asyfactor)/(len(mass)) 
        return chisquare


def expo_func(x, pa, pb, pc):    
    pdf = pa* (1 - np.exp((x-pb)/pc)) 
    return pdf

class AglInverseMassFunctionFit(InverseMassFunctionFit):
    def __init__(self, isotopes, hist_data, fit_energy_range, fit_mass_range, detector, is_constraint, is_nonlinearconstraint=False, component="All"):
        InverseMassFunctionFit.__init__(self, isotopes, hist_data, fit_energy_range, fit_mass_range, detector, is_constraint, is_nonlinearconstraint=is_nonlinearconstraint, component="All")
        
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
            sigma_ratio_a, sigma_ratio_b, sigma_ratio_c = pars[8:11]   
            asy_factor_a, asy_factor_b, asy_factor_c = pars[11:14]

            num_iso = len(self.isotopes)
            #3 for mean, 4 for sigma, 1:fraccore, 1 sigma_ratio, 3 asy_factor,  3 * (num_iso -1) for scaling factor of sigma
            numpvalues = 14
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
            sigma_ratio = poly(np.log(energy), sigma_ratio_a, sigma_ratio_b, sigma_ratio_c)
            pdf = np.zeros(energy.shape)
            pdf_iso = {iso: np.zeros(energy.shape) for iso in self.isotopes}
        
            niso_factor = len(self.isotopes)
            scale_asyfactor = np.array([1.0, 1.0, 1.0])
            #scale_factors_mean = np.array([1.0, ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be9'], ISOTOPES_MASS['Be7']/ISOTOPES_MASS['Be10']])
            #scale_factors_mean = {"Be9": np.array([0.78201887, -0.00356474,  0.00108403]),
            #                      "Be10": np.array([0.70600864, -0.00429441,  0.00122645])}
                                  #"Be10": np.array([0.718554,   -0.02194944,  0.00624895])}

            #scale_factors_sigma = {"Be9": np.array([0.78628677, -0.00880776,  0.00260464]),
            #                      "Be10": np.array([0.70650515, -0.00471891,  0.00125659])}

            #####################3
            #scale mean from tuned mc
            scale_factors_mean_tunedmc = {"Be9": np.array([0.71421186,  0.11769652, -0.0704438,   0.01372702]),
                                  "Be10": np.array([0.73089961, -0.03623535,  0.01037224])}
            ###########################################

            #################################
            #scale mean from energy loss
            #################################
            scale_factors_mean_enloss = {"Be9": np.array([7.80273314e-01, -9.69071723e-04,  1.55623772e-04]),
                                  "Be10": np.array([7.02578614e-01, -8.33162517e-04,  6.49571751e-05])}

            scale_factors_mean = np.array([1.0, 0.77957032, 0.70275123])
            
            for i, n in enumerate(norm):
                isofactor = ISOTOPES_MASS[self.isotopes[0]]/ISOTOPES_MASS[self.isotopes[i]] 
                isotope = self.isotopes[i]
                
                if i == 0:
                    rigsigma_factor = 1.0
                    mean_scale = 1.0
                    sigma_scale = 1.0
                else:
                    rigsigma_factor = expo_func(energy, *pars[numpvalues + (i-1)*3: numpvalues + i*3])
                    #mean_scale = poly(np.log(energy), *scale_factors_mean_tunedmc[self.isotopes[i]])
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
            parnames = ['x', 'mua', 'mub', 'muc', 'siga', 'sigb', 'sigc', 'sigd', 'fraccore', 'sigma_ratio_a', 'sigma_ratio_b', 'sigma_ratio_c', 'asy_factor_a', 'asy_factor_b', 'asy_factor_c'] + [f'ex{a}_{iso}' for iso in self.isotopes[1:] for a in ["a", "b", "c"]] +  [f"n{isonum}_{ibin}" for isonum in isotopes_atom_num[:-1]  for ibin in range(num_energybin)]
        else:
            parnames = ['x', 'mua', 'mub', 'muc', 'siga', 'sigb', 'sigc', 'sigd', 'fraccore', 'sigma_ratio_a', 'sigma_ratio_b',  'sigma_ratio_c', 'asy_factor_a', 'asy_factor_b', 'asy_factor_c'] + [f'ex{a}_{iso}' for iso in self.isotopes[1:] for a in ["a", "b", "c"]] +  [f"n{isonum}_{ibin}" for isonum in isotopes_atom_num  for ibin in range(num_energybin)]
       
        mass_function.func_code = make_func_code(parnames)
        return mass_function
    
    def get_polypars_index(self):
        parindex = {'mua': 0, 'mub':1, 'muc':2, 'siga':3, 'sigb':4, 'sigc':5, 'sigd': 6, 'fraccore':7, 'sigma_ratio_a':8,  'sigma_ratio_b':9, 'sigma_ratio_c':10, 'asy_factor_a':11, 'asy_factor_b':12, 'asy_factor_c':13}
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
    '''
    #def chisquarefunc(self, *params):
    #    chi_square = 0.0
        counts = self.hist.values[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
        countserr = self.hist.get_errors()[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
        countserr[countserr == 0.0] = 1   
        yvalues = counts.reshape(-1)
        yvalueserr = countserr.reshape(-1)
        predicted_y = self.mass_function()
        for o, e, err in zip(yvalues, predicted_y, yvalueserr):
            chi_square += ((o - e) ** 2) / (err ** 2)

        param_constraint = ['siga', 'sigb', 'sigc', 'sigd']
        indexpar = self.get_polypars_index()        
        for par in param_constraint:
            chi_square += ((param[indexpar[par]] - guess[par])**2)/(guess_err[par] **2)

        return chi_square
    '''
    def perform_fit(self, guess, fit_simultaneous=True, verbose=False, fixed_pars=None, lim_pars=None, parlim=None):
        counts = self.hist.values[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
        countserr = self.hist.get_errors()[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
        countserr[countserr == 0.0] = 1   
        yvalues = counts.reshape(-1)
        yvalueserr = countserr.reshape(-1)
       
        xvalues = self.get_fit_xvalues()
        
        #pmc = **guess
        if fit_simultaneous:
            cost = LeastSquareMassFit(self.make_mass_function_simultaneous(), xvalues, yvalues, yvalueserr, guess, self.num_energybin)  
        else:
            cost = LeastSquares(xvalues, yvalues, yvalueserr, self.mass_function_binbybin)
            
        m = Minuit(cost, **guess)        
        if fixed_pars is not None:
            for name in fixed_pars:
                if fit_simultaneous:
                    m.fixed[name] = True
                else:
                    for i in range(self.num_energybin):
                        m.fixed[f'{name}_{i}'] = True
        if lim_pars is not None:
            for name, lim in lim_pars.items():
                m.limits[name] = lim
                

        if self.nonlinearconstraint and (parlim is not None):
            ##############################################################
            #replace m.migrad with m.scipy with nonlinear constraint
            ##############################################################
            print('#########################################')
            print('fiting with constraint:')
            
            x_energy = self.energy_binning.bin_centers[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1]
            #constraint_mu = NonlinearConstraint(lambda *par: poly(np.log(x_energy), par[3], par[4], par[5]), parlim['mu'][0, :], parlim['mu'][1, :])
            
            constraint_sig = NonlinearConstraint(lambda *par: poly(np.log(x_energy), par[3], par[4], par[5], par[6]), parlim['sigma'][0, :], parlim['sigma'][1, :])
            print('sigma upper-init: ', parlim['sigma'][1, :]  - poly(np.log(x_energy), guess['siga'], guess['sigb'], guess['sigc'], guess['sigd']))
            print('sigma init-lower: ', poly(np.log(x_energy), guess['siga'], guess['sigb'], guess['sigc'], guess['sigd']) - parlim['sigma'][0, :])
            constraint_sigmaratio = NonlinearConstraint(lambda *par: poly(np.log(x_energy), par[8], par[9], par[10]), parlim['sigma_ratio'][0, :], parlim['sigma_ratio'][1, :])
            constraint_asy_factor = NonlinearConstraint(lambda *par: poly(np.log(x_energy), par[11], par[12], par[13]), parlim['asy_factor'][0, :], parlim['asy_factor'][1, :])
            constraint_fraccore = NonlinearConstraint(lambda *par: poly(np.log(x_energy), par[7]), parlim['fraccore'][0, :], parlim['fraccore'][1, :])
            all_constraints = [constraint_sig, constraint_sigmaratio, constraint_asy_factor, constraint_fraccore]
            #m.precision = 10e-10
            m.scipy(constraints=all_constraints, method="trust-constr")

            
            ##############################################################
        else:
            m.migrad()
        
        if verbose:
            print("N par:", len(m.values)) 
            print(m)

        m_covariance = np.array(m.covariance)
        fit_parameters = uncertainties.correlated_values(m.values, np.array(m.covariance))
        #fit_parameters = unumpy.uarray(m.values, m.errors)
        par_dict = {param: {'value': val, 'error': err} for param, val, err in zip(m.parameters, m.values, m.errors)}
        
        return fit_parameters, par_dict, m_covariance





















    
