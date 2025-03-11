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
from tools.graphs import MGraph, plot_graph
import uncertainties
from uncertainties import unumpy
from uncertainties import ufloat
from scipy.optimize import NonlinearConstraint 
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, plot1d_step, FONTSIZE, set_plot_defaultstyle
import pickle

def expo_func(x, pa, pb, pc):    
    pdf = pa* (1 - np.exp((x-pb)/pc)) 
    return pdf


def get_shape_params_from_poly(pars, energy_bincenter):
    mua, mub, muc = pars[0:3]    
    siga, sigb, sigc = pars[3:6]
    fraccore = pars[6] * np.ones_like(energy_bincenter)
    sigma_ratio = pars[7] * np.ones_like(energy_bincenter)   
    asy_factor = pars[8] * np.ones_like(energy_bincenter)   

    mean = poly(np.log(energy_bincenter), mua, mub, muc)
    sigma = poly(np.log(energy_bincenter), siga, sigb, sigc)
    fraccore = poly(np.log(energy_bincenter), fraccore)
    sigma_ratio = poly(np.log(energy_bincenter), sigma_ratio)    
    asy_factor = poly(np.log(energy_bincenter), asy_factor)
    dictpar = {'mean': mean, 'sigma': sigma, 'fraccore': fraccore, 'sigma_ratio': sigma_ratio, 'asy_factor': asy_factor}
    return dictpar

def get_constraint_function(parsfit, parsmc, parstart, npolypar, spline_err, energy_bincenter):
    mu_kuerr = spline_err(np.log(energy_bincenter))
    muscale_a, muscale_b, muscale_c = parsfit[parstart: parstart+npolypar]
    muscalefactor = poly(np.log(energy_bincenter), muscale_a, muscale_b, muscale_c)    
    muscale_amc, muscale_bmc, muscale_cmc = parsmc[parstart: parstart+npolypar]
    muscalefactor_mc = poly(np.log(energy_bincenter), muscale_amc, muscale_bmc, muscale_cmc)
    constraint_muscale = np.sum((muscalefactor - muscalefactor_mc)**2/mu_kuerr**2)
    return constraint_muscale 


def get_constraint_function_withfixerr(parsfit, parsmc, parstart, npolypar, constrainterr, energy_bincenter):
    muscale_a, muscale_b, muscale_c = parsfit[parstart: parstart+npolypar]
    muscalefactor = poly(np.log(energy_bincenter), muscale_a, muscale_b, muscale_c)
    
    muscale_amc, muscale_bmc, muscale_cmc = parsmc[parstart: parstart+npolypar]
    muscalefactor_mc = poly(np.log(energy_bincenter), muscale_amc, muscale_bmc, muscale_cmc)
    constraint_muscale = np.sum((muscalefactor - muscalefactor_mc)**2/constrainterr**2)
    return constraint_muscale 

class LeastSquareMassFit:
    errordef = Minuit.LEAST_SQUARES  # for Minuit to compute errors correctly
    def __init__(self, nuclei, model, x, y, yerr, guess, energybins, x_fit_energy, isotopes, weight= 1.0, numpvalues = 9, guesserr=None, fitFreeP=False):
        self.nuclei = nuclei
        self.model = model
        self.xvalues = x
        self.yvalues = y
        self.yvalueserr = yerr                                                        
        pars = describe(model)
        lsq_parameters = pars[1:]
        self.pmc = [par for key, par in guess.items()]
        self.func_code = make_func_code([par for par in lsq_parameters])
        self.num_energybin = energybins
        self.x_fit_energy = x_fit_energy
        self.guesserr = guesserr
        self.weight = weight
        self.fitFreeP = fitFreeP
        self.isotopes = isotopes
        self.numpvalues = numpvalues
        
    def __call__(self, *pars):
        #print(self._parameters)
        yfit = self.model(self.xvalues , *pars)
        energy = self.xvalues[0,:].reshape((self.num_energybin, -1))
        mass = self.xvalues[1,:].reshape((self.num_energybin, -1))    

        energy_bincenter = self.x_fit_energy
        dictpars = get_shape_params_from_poly(pars, energy_bincenter)
        dictpars_mc = get_shape_params_from_poly(self.pmc, energy_bincenter)

        dict_err = {}
        constraint = {}
        NaFSpline_err = {}
        errorbar_precent = {'mean': 0.005, 'sigma': 0.03, 'sigma_ratio': 0.03, 'fraccore': 0.03, 'asy_factor':0.03}

        if self.guesserr is None:
            for key, value in errorbar_precent.items():
                dict_err[key] = dictpars_mc[key]* value
                #print('dict_err[key]', key, dict_err[key])  
                constraint[key] = np.sum((dictpars[key] - dictpars_mc[key])**2/dict_err[key]**2)
                
        else:
            #with open('/home/manbing/Documents/Data/data_BeP8/splines_pars_uncertainty.pkl', 'rb') as file:
            with open(f'{self.guesserr}', 'rb') as file:
                dictspline_err = pickle.load(file)
                NaFSpline_err = dictspline_err['NaF']
            for key, value in NaFSpline_err.items():
                if key == 'mean':
                    dict_err[key] = dictpars_mc['mean']* errorbar_precent['mean']
                    constraint[key] = np.sum((dictpars['mean'] - dictpars_mc['mean'])**2/dict_err['mean']**2)
                else:
                    dict_err[key] = NaFSpline_err[key](np.log(energy_bincenter)) * dictpars_mc[key]
                    constraint[key] = np.sum((dictpars[key] - dictpars_mc[key])**2/dict_err[key]**2)  

        with open('/home/manbing/Documents/Data/data_BeP8/FitParsRange/spline_ku_uncertainty.pickle', 'rb') as file:
            spline_ku_err = pickle.load(file)
            NaF_spline_ku_err = spline_ku_err['NaF']

        with open('/home/manbing/Documents/Data/data_BeP8/FitParsRange/spline_ksig_uncertainty.pickle', 'rb') as file:
            dictspline_ksig_err = pickle.load(file)
            spline_ksig_err = dictspline_ksig_err['NaF']

            
        err_muscale = 0.001
        err_sigscale = 0.005
        constraint_muscale_all = 0
        constraint_sigscale_all = 0
        num_iso = len(self.isotopes)
        nump = self.numpvalues
        constraint_muscale = {}
        constraint_sigscale = {}
        for i, iso in enumerate(self.isotopes[1:]):
            constraint_muscale[iso] = get_constraint_function(pars, self.pmc, nump+ i * 3 , 3, NaF_spline_ku_err, energy_bincenter)
            #constraint_muscale[iso] = get_constraint_function_withfixerr(pars, self.pmc, nump+ i * 3 , 3, err_muscale, energy_bincenter)
            #constraint_sigscale[iso] = get_constraint_function_withfixerr(pars, self.pmc, nump + (num_iso-1) * 3 + i * 3, 3, err_sigscale, energy_bincenter)
            constraint_sigscale[iso] = get_constraint_function(pars, self.pmc, nump + (num_iso-1) * 3 + i * 3, 3, spline_ksig_err, energy_bincenter)
            constraint_muscale_all = constraint_muscale_all + constraint_muscale[iso]
            constraint_sigscale_all = constraint_sigscale_all + constraint_sigscale[iso]

        if self.fitFreeP:
            chisquare = np.sum((self.yvalues - yfit) ** 2 / self.yvalueserr ** 2)
        else:
            if self.nuclei == 'Be':
                chisquare = np.sum((self.yvalues - yfit) ** 2 / self.yvalueserr ** 2)   +  constraint['sigma']  + constraint['sigma_ratio'] + constraint['asy_factor']  + constraint['fraccore'] + constraint_muscale_all +  constraint_sigscale_all
            else:
                chisquare = np.sum((self.yvalues - yfit) ** 2 / self.yvalueserr ** 2)   +  constraint['mean'] + 10 * constraint['sigma']  + constraint['sigma_ratio'] + constraint['asy_factor']  + constraint['fraccore'] +  constraint_muscale_all +  constraint_sigscale_all
                
        return chisquare

class NaFInverseMassFunctionFit(InverseMassFunctionFit):
    __ENERGY_RANGE = [1.0, 4.0]
    __INVERSEMASS_RANGE = [0.05, 0.25]
    
    def __init__(self, nuclei, isotopes, hist_data, fit_energy_range, fit_mass_range, detector, is_constraint, is_nonlinearconstraint=False, component="All", numpvalues=9):
        InverseMassFunctionFit.__init__(self, nuclei, isotopes, hist_data, fit_energy_range, fit_mass_range, detector, is_constraint, numpvalues, is_nonlinearconstraint=is_nonlinearconstraint, component="All")
        

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
            siga, sigb, sigc = pars[3:6]
            fraccore = pars[6] * np.ones_like(energy)
            sigma_ratio = pars[7] * np.ones_like(energy)   
            asy_factor = pars[8] * np.ones_like(energy)   

            num_iso = len(self.isotopes)
            numpvalues = self.numpvalues
            #3 for mean, 3 for sigma, 1:fraccore, 1 sigma_ratio, 1 asy_factor,  3 * (num_iso -1) for scaling factor of sigma
            num_common_pars = numpvalues + 2 * 3  * (num_iso -1)
             
            if self.is_constraint:
                norm = [np.array(pars[num_common_pars + (i * num_energybin) : num_common_pars + (i+1) * num_energybin]) for i in range(len(self.isotopes)-1)]
                norm_last_iso = self.get_data_infitrange().sum(axis=1) - np.sum(norm, axis=0) 
                norm.append(norm_last_iso)
            else:
                norm = [np.array(pars[num_common_pars + (i * num_energybin) : num_common_pars + (i+1) * num_energybin]) for i in range(len(self.isotopes))]
                
            mean = poly(np.log(energy), mua, mub, muc)
            sigma = poly(np.log(energy), siga, sigb, sigc)
            
            pdf = np.zeros(energy.shape)
            pdf_iso = {iso: np.zeros(energy.shape) for iso in self.isotopes}
        
            niso_factor = len(self.isotopes)

            
            mean_scale = {iso: np.zeros(energy.shape) for iso in self.isotopes}                                                                                                                                                                 
            sigma_scale = {iso: np.zeros(energy.shape) for iso in self.isotopes}
            
            for i, n in enumerate(norm):
                isofactor = ISOTOPES_MASS[self.isotopes[0]]/ISOTOPES_MASS[self.isotopes[i]]
                isotope = self.isotopes[i]
                if i == 0:
                    mean_scale[isotope] = 1.0
                    sigma_scale[isotope] = 1.0
                else:
                    mean_scale[isotope] = poly(np.log(energy), *pars[numpvalues + (i-1)*3 : numpvalues + i * 3]) * isofactor                                                                                                                   
                    sigma_scale[isotope] = poly(np.log(energy), *pars[numpvalues + (num_iso-1) * 3 + (i-1)*3 : numpvalues +   (num_iso-1) * 3 + i * 3]) * isofactor
                    #mean_scale[isotope] = isofactor
                    #sigma_scale[isotope] = isofactor
               
                #coregaus = gaussian(mass, mean * mean_scale[isotope], sigma * sigma_scale[isotope])
                #asygaus = asy_gaussian(mass, mean * mean_scale[isotope],  sigma_ratio * sigma * sigma_scale[isotope], asy_factor)
                coregaus = gaussian(mass, mean * mean_scale[isotope], sigma * sigma_scale[isotope])
                asygaus = asy_gaussian(mass, mean * mean_scale[isotope],  sigma_ratio * sigma * sigma_scale[isotope], asy_factor)
              
                pdf += n[:, None] * (fraccore * coregaus + (1 - fraccore) * asygaus)  * mass_binwidth[None, :]
                pdf_iso[isotope] = n[:, None] * (fraccore * coregaus + (1 - fraccore) * asygaus)  * mass_binwidth[None, :]  

            if drawiso == "All":
                return pdf.reshape((-1, ))
            else:
                return pdf_iso[f'{drawiso}'].reshape((-1, ))

        if self.is_constraint:
            parnames = ['x', 'mua', 'mub', 'muc', 'siga', 'sigb', 'sigc', 'fraccore', 'sigma_ratio', 'asy_factor' ] + [f'muscale_{iso}_{a}' for iso in self.isotopes[1:] for a in ["a", "b", "c"]] +  [f'sigscale_{iso}_{a}' for iso in self.isotopes[1:] for a in ["a", "b", "c"]] +[f"n{isonum}_{ibin}" for isonum in isotopes_atom_num[:-1] for ibin in range(num_energybin)]
        else:
            parnames = ['x', 'mua', 'mub', 'muc', 'siga', 'sigb', 'sigc', 'fraccore', 'sigma_ratio', 'asy_factor' ] + [f'muscale_{iso}_{a}' for iso in self.isotopes[1:] for a in ["a", "b", "c"]] + [f'sigscale_{iso}_{a}' for iso in self.isotopes[1:] for a in ["a", "b", "c"]] +  [f"n{isonum}_{ibin}" for isonum in isotopes_atom_num  for ibin in range(num_energybin)]
        mass_function.func_code = make_func_code(parnames)
        return mass_function

    def get_polypars_index(self):
        parindex = {'mua': 0, 'mub':1, 'muc':2, 'siga':3, 'sigb':4, 'sigc':5, 'fraccore':6, 'sigma_ratio':7, 'asy_factor':8 }
        return parindex

    def get_scale_index(self):
        numpvalues = 9
        if self.nuclei == 'Be':
            parindex = {'muscale_Be9_a': 10, 'muscale_Be9_b': 11, 'muscale_Be9_c':12,
                        'muscale_Be10_a': 13, 'muscale_Be10_b': 14, 'muscale_Be10_c':15,
                        'sigscale_Be9_a': 16, 'sigscale_Be9_b': 17, 'sigscale_Be9_c':18,
                        'sigscale_Be10_a': 19, 'sigscale_Be10_b': 20, 'sigscale_Be10_c':21}
        elif self.nuclei == 'Li':
            parindex = {'muscale_Li7_a': 10, 'muscale_Li7_b': 11, 'muscale_Li7_c':12,
                        'sigscale_Li7_a': 13, 'sigscale_Li7_b': 14, 'sigscale_Li7_c':15}
        else:
            return print("The nuclei is not setup")
        
        return parindex
  
     
    def make_mass_function_binbybin(self, drawiso=None, component="All"):
        mass_binwidth = self.mass_binning.bin_widths[self.fit_mass_binrange[0]: self.fit_mass_binrange[1] +1]
        #num_energybin = self.fit_energy_binrange[1] - self.fit_energy_binrange[0] + 1
        #print("num_energybin:", num_energybin)
        num_energybin = self.num_energybin
        num_massbin = self.num_massbin
        isotopes_atom_num = [NUCLEI_NUMBER[iso] for iso in self.isotopes] 
        def mass_function(x, *pars):
            num_templateshape_pars = 5 
            energy = x[0,:].reshape((num_energybin, -1))
            mass = x[1,:].reshape((num_energybin, -1))
            pars = np.array(pars)
            mean = np.hstack([pars[0:num_energybin,None]] * num_massbin)
            #print("mean", mean.shape)
            sigma =  np.hstack([pars[num_energybin: 2*num_energybin, None]] * num_massbin) 
            fraccore =  np.hstack([pars[2*num_energybin: 3*num_energybin, None]] * num_massbin) 
            sigma_ratio =np.hstack([pars[3*num_energybin: 4*num_energybin, None]] * num_massbin)
            asy_factor = np.hstack([pars[4*num_energybin: 5*num_energybin, None]] * num_massbin) 
            norm = [pars[(num_templateshape_pars +i)*num_energybin : (num_templateshape_pars + 1 +i)*num_energybin, None]  for i in range(len(self.isotopes))]
            pdf = np.zeros(energy.shape)
            pdfgaus = np.zeros(energy.shape)
            pdfasygus = np.zeros(energy.shape)
            pdf_iso = {iso: np.zeros(energy.shape) for iso in self.isotopes} 
            #print initial values and print fit results                                                    
            
            for i,  n in enumerate(norm):                                                                                                                                               
                isonum = isotopes_atom_num[i]      
                isofactor = isonum/isotopes_atom_num[0]
                coregaus = gaussian(mass, mean/isofactor, sigma/isofactor)
                #print("mass.shape:", mass.shape, (mean/isofactor).shape, (sigma_ratio * sigma/isofactor).shape, asy_factor.shape)
                asygaus = asy_gaussian(mass, mean/isofactor,  (sigma_ratio * sigma)/isofactor, asy_factor)
                pdfgaus += n * fraccore * coregaus  * mass_binwidth[None, :]
                pdfasygus += n * (1 - fraccore) * asygaus * mass_binwidth[None, :]
                pdf += n * (fraccore * coregaus + (1 - fraccore) * asygaus)  * mass_binwidth[None, :]
                pdf_iso[self.isotopes[i]] = n * (fraccore * coregaus + (1 - fraccore) * asygaus)  * mass_binwidth[None, :]


            if drawiso is not None:
                return pdf_iso[drawiso].reshape((-1, ))  
            else:
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
    
    
    def perform_fit(self, guess, guesserr=None, fit_simultaneous=True, verbose=False, fixed_pars=None, lim_pars=None, parlim=None, fitFreeP=False, study_syserr=False):
        counts = self.hist.values[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
        countserr = self.hist.get_errors()[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
        countserr[countserr == 0.0] = 1   
        yvalues = counts.reshape(-1)
        yvalueserr = countserr.reshape(-1)
       
        xvalues = self.get_fit_xvalues()
        if fit_simultaneous:
            cost = LeastSquareMassFit(self.nuclei, self.make_mass_function_simultaneous(), xvalues, yvalues, yvalueserr, guess, self.num_energybin, self.x_fit_energy, self.isotopes, guesserr=guesserr, fitFreeP=fitFreeP) 
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
                
        m.migrad()
        
        if verbose:
            print("N par:", len(m.values)) 
            print(m)

        m_covariance = np.array(m.covariance)
        fit_parameters = uncertainties.correlated_values(m.values, np.array(m.covariance))
        #fit_parameters = unumpy.uarray(m.values, m.errors)
        par_dict = {param: {'value': val, 'error': err} for param, val, err in zip(m.parameters, m.values, m.errors)}
        
        return fit_parameters, par_dict, m_covariance
