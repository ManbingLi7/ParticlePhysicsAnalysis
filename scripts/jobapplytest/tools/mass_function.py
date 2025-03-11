import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import tools.roottree as read_tree
from tools.calculator import calc_mass, calc_ekin_from_beta, calc_betafrommomentom

import uproot
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, LeastSquares, NormalConstraint
from iminuit.util import describe, make_func_code
from tools.constants import ISOTOPES, NUCLEI_NUMBER, ISOTOPES_COLOR, ISO_LABELS
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

class InverseMassFunctionFit:
    def __init__(self, isotopes, hist_data, fit_energy_range, fit_mass_range, detector, is_constraint=True, component="All"):
        self.isotopes = isotopes
        self.hist = hist_data
        self.energy_binning = hist_data.binnings[0]
        self.mass_binning = hist_data.binnings[1]
        self.data = hist_data.values
        self.dateerr = np.sqrt(self.data)
        self.fit_energy_range = fit_energy_range
        self.fit_mass_range = fit_mass_range
        self.fit_mass_binrange = self.mass_binning.get_indices(self.fit_mass_range)
        self.fit_energy_binrange = self.energy_binning.get_indices(self.fit_energy_range)
        self.num_energybin = self.fit_energy_binrange[1] - self.fit_energy_binrange[0] + 1
        self.num_massbin = self.fit_mass_binrange[1] - self.fit_mass_binrange[0] + 1
        self.detector = detector
        self.mass_function = self.make_mass_function_simultaneous(is_constraint)
        self.mass_function_binbybin = self.make_mass_function_binbybin(component)
        self.x_fit_energy = self.energy_binning.bin_centers[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1]
        self.x_fit_mass = self.mass_binning.bin_centers[self.fit_mass_binrange[0] : self.fit_mass_binrange[1] + 1]
        
    #print initial values and print fit results
    def __str__(self):
        print()

    def number_events(self):
        n_counts = np.sum(self.data, axis=1)
        n_counts_err = np.sqrt(n_counts)
        return n_counts


    def get_polypars_index(self):
        parindex = {'mua': 0, 'mub':1, 'muc':2, 'siga':3, 'sigb':4, 'sigc':5, 'fraccore':6, 'sigma_ratio':7, 'asy_factor':8}
        return parindex

    def make_mass_function_simultaneous(self, is_constraint=True, drawiso="All"):
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
            numpvalues = 9
            #3 for mean, 3 for sigma, 1:fraccore, 1 sigma_ratio, 1 asy_factor,  3 * (num_iso -1) for scaling factor of sigma
            num_common_pars = numpvalues + 3 * (num_iso -1)
            
            if is_constraint:
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
            scale_factors_mean = np.array([1.0, 7.0/9.0, 7./10.])
            
            for i, n in enumerate(norm):
                isofactor = isotopes_atom_num[0]/isotopes_atom_num[i]
                isotope = self.isotopes[i]
                
                if i == 0:
                    rigsigma_factor = 1.0
                else:
                    rigsigma_factor = expo_func(energy, *pars[numpvalues + (i-1)*3: numpvalues + i*3])
                    
                coregaus = gaussian(mass, mean * isofactor, sigma * isofactor / rigsigma_factor)
                asygaus = asy_gaussian(mass, mean * isofactor,  sigma_ratio * sigma * isofactor /rigsigma_factor, asy_factor)
                pdf += n[:, None] * (fraccore * coregaus + (1 - fraccore) * asygaus)  * mass_binwidth[None, :]
                pdf_iso[isotope] = n[:, None] * (fraccore * coregaus + (1 - fraccore) * asygaus)  * mass_binwidth[None, :]  

            if drawiso == "All":
                return pdf.reshape((-1, ))
            else:
                return pdf_iso[f'{drawiso}'].reshape((-1, ))
        
        parnames = ['x', 'mua', 'mub', 'muc', 'siga', 'sigb', 'sigc', 'fraccore', 'sigma_ratio', 'asy_factor' ] + [f'ex{a}_{iso}' for iso in self.isotopes[1:] for a in ["a", "b", "c"]] +  [f"n{isonum}_{ibin}" for isonum in (isotopes_atom_num[:-1] if is_constraint else isotopes_atom_num)  for ibin in range(num_energybin)]

        mass_function.func_code = make_func_code(parnames)
        return mass_function

    def get_polypars_index(self):
        parindex = {'mua': 0, 'mub':1, 'muc':2, 'siga':3, 'sigb':4, 'sigc':5, 'fraccore':6, 'sigma_ratio':7, 'asy_factor':8 }
        return parindex
    
    def make_mass_function_binbybin(self, component="All"):
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
    
    def get_fit_xvalues(self):
        x_energy = self.energy_binning.bin_centers[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1]
        x_mass = self.mass_binning.bin_centers[self.fit_mass_binrange[0] : self.fit_mass_binrange[1] + 1]        
        xgrid_energy, xgrid_mass = np.meshgrid(x_energy, x_mass, indexing = "ij")
        xvalues = np.stack((xgrid_energy.reshape(-1), xgrid_mass.reshape(-1)))
        return xvalues

    def get_data_infitrange(self):
        counts = self.hist.values[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
        return counts
    
    def perform_fit(self, guess, fit_simultaneous=True, verbose=False, fixed_pars=None, lim_pars=None):
        counts = self.hist.values[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
        countserr = self.hist.get_errors()[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
        countserr[countserr == 0.0] = 1   
        yvalues = counts.reshape(-1)
        yvalueserr = countserr.reshape(-1)
        
        xvalues = self.get_fit_xvalues()
        if fit_simultaneous:
            cost = LeastSquares(xvalues, yvalues, yvalueserr, self.mass_function)
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

        fit_parameters = uncertainties.correlated_values(m.values, np.array(m.covariance))
        #fit_parameters = unumpy.uarray(m.values, m.errors)
        par_dict = {param: {'value': val, 'error': err} for param, val, err in zip(m.parameters, m.values, m.errors)}
        
        return fit_parameters, par_dict

    def get_fit_values(self, fit_parameters, fit_simultaneous=True):
        xvalues = self.get_fit_xvalues()
        if fit_simultaneous:
            return self.mass_function(xvalues, *unumpy.nominal_values(fit_parameters)).reshape(self.num_energybin, self.num_massbin)
        else:
            return self.mass_function_binbybin(xvalues, *unumpy.nominal_values(fit_parameters)).reshape(self.num_energybin, self.num_massbin)

    def get_fit_values_iso(self, iso, fit_parameters, fit_simultaneous=True):
        xvalues = self.get_fit_xvalues()
        return self.make_mass_function_simultaneous(drawiso= iso)(xvalues, *unumpy.nominal_values(fit_parameters)).reshape(self.num_energybin, self.num_massbin)

    def get_fitmc_component_gaus(self, fit_parameters, fit_simultaneous=False):
        xvalues = self.get_fit_xvalues()
        return self.make_mass_function_binbybin(component="gaus")(xvalues, *unumpy.nominal_values(fit_parameters)).reshape(self.num_energybin, self.num_massbin)

    def get_fitmc_component_asygaus(self, fit_parameters, fit_simultaneous=False):
        xvalues = self.get_fit_xvalues()
        return self.make_mass_function_binbybin(component="asygaus")(xvalues, *unumpy.nominal_values(fit_parameters)).reshape(self.num_energybin, self.num_massbin)
        


    def draw_fit_results(self, fit_parameters, plotdir, fit_simultaneous=True, guess=None):
        fit_values = self.get_fit_values(fit_parameters, fit_simultaneous)
        x_mass_bincenter =  self.mass_binning.bin_centers[self.fit_mass_binrange[0]:  self.fit_mass_binrange[1]+1]
        x_mass_binedges = self.mass_binning.edges[self.fit_mass_binrange[0]:  self.fit_mass_binrange[1]+2]
        fit_values_iso = dict()
        for iso in self.isotopes:
            fit_values_iso[iso] = self.get_fit_values_iso(iso, fit_parameters, fit_simultaneous)
            
        for i, ibin in enumerate(range(self.fit_energy_binrange[0], self.fit_energy_binrange[1] + 1)):
            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14)) 
            ax1.text(0.03, 0.98, f"{self.detector}: [{self.energy_binning.edges[ibin]:.2f}, {self.energy_binning.edges[ibin+1]:.2f}] GeV/n", fontsize=FONTSIZE+1, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black")
            counts = self.hist.values[ibin, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
            countserr = self.hist.get_errors()[ibin, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
            countserr[countserr == 0.0]  = 1
            plot1d_errorbar(figure, ax1, x_mass_binedges, counts, err=countserr, label_x="1/mass (1/GeV)", label_y="counts", col='black', legend='data')

            
            #ax1.plot(get_bin_center(massbinedges[min_massbin: max_massbin + 2]), mass_function_guess[i], color='blue', label='guess')
            ax1.plot(x_mass_bincenter, fit_values[i], color='red', label='fit')
            for iso in self.isotopes:
                ax1.plot(x_mass_bincenter, fit_values_iso[iso][i], color=ISOTOPES_COLOR[iso], label=f'{ISO_LABELS[iso]}')
                
            handles = []
            labels = []
            #for iso in self.isotopes:
            #    ax1.plot(x_mass_bincenter, mass_function_iso[iso][i], color=ISOTOPES_COLOR[iso], label=f'{ISO_LABELS[iso]}')
            #    ax1.fill_between(get_bin_center(massbinedges[min_massbin: max_massbin + 2]), np.zeros_like(mass_function_iso[iso][i]), mass_function_iso[iso][i], interpolate=True, color=ISOTOPES_COLOR[iso], alpha=0.4)
            #    handles.append(mpatches.Patch(color=ISOTOPES_COLOR[iso]))
            #    labels.append(f'{ISO_LABELS[iso]}')
            pull =(counts - fit_values[i])/countserr
            chisquare = np.sum(pull**2)/(len(pull))
            plot1d_errorbar(figure, ax2, x_mass_binedges, counts=pull, err=np.zeros(len(pull)),  label_x="1/mass (1/GeV)", label_y="pull", legend=None,  col="black", setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
            plt.subplots_adjust(hspace=.0)

            fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {chisquare:.1f}",]
            fit_info_formatted = [f"\\fontsize{FONTSIZE}\\selectfont {info}" for info in fit_info]
            ax1.legend(handles, labels, title="\n".join(fit_info), frameon=False, fontsize=FONTSIZE)
            ax1.set_xticklabels([])
            ax1.get_yticklabels()[0].set_visible(False)
            ax2.set_ylim([-3.8, 3.8])
            ax1.set_ylim([0, 1.2 *np.max(counts)])
            if fit_simultaneous:
                savefig_tofile(figure, plotdir, f"fit_mass_bin_{ibin}_{self.detector}_fitsimul", show=False)
            else:
                savefig_tofile(figure, plotdir, f"fit_mass_bin_{ibin}_{self.detector}_binbybin", show=False)



    def draw_fit_results_mc(self, fit_parameters, plotdir, fit_simultaneous=False, guess=None):
        fit_values = self.get_fit_values(fit_parameters, fit_simultaneous)
        x_mass_bincenter =  self.mass_binning.bin_centers[self.fit_mass_binrange[0]:  self.fit_mass_binrange[1]+1]
        x_mass_binedges = self.mass_binning.edges[self.fit_mass_binrange[0]:  self.fit_mass_binrange[1]+2]

        fit_part1 = self.get_fitmc_component_gaus(fit_parameters)
        fit_part2 = self.get_fitmc_component_asygaus(fit_parameters)
        
        for i, ibin in enumerate(range(self.fit_energy_binrange[0], self.fit_energy_binrange[1] + 1)):
            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14)) 
            ax1.text(0.03, 0.98, f"{self.detector}: [{self.energy_binning.edges[ibin]:.2f}, {self.energy_binning.edges[ibin+1]:.2f}] GeV/n", fontsize=FONTSIZE+1, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black")
            counts = self.hist.values[ibin, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
            countserr = self.hist.get_errors()[ibin, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
            countserr[countserr == 0.0]  = 1
            plot1d_errorbar(figure, ax1, x_mass_binedges, counts, err=countserr, label_x="1/mass (1/GeV)", label_y="counts", col='black', legend='data')
            
            #ax1.plot(get_bin_center(massbinedges[min_massbin: max_massbin + 2]), mass_function_guess[i], color='blue', label='guess')
            ax1.plot(x_mass_bincenter, fit_values[i], color='red', label='fit')
            ax1.plot(x_mass_bincenter, fit_part1[i], color='blue', label='part gaus')
            ax1.plot(x_mass_bincenter, fit_part2[i], color='blue', label='part asygaus')
            
            handles = []
            labels = []
            #for iso in self.isotopes:
            #    ax1.plot(x_mass_bincenter, mass_function_iso[iso][i], color=ISOTOPES_COLOR[iso], label=f'{ISO_LABELS[iso]}')
            #    ax1.fill_between(get_bin_center(massbinedges[min_massbin: max_massbin + 2]), np.zeros_like(mass_function_iso[iso][i]), mass_function_iso[iso][i], interpolate=True, color=ISOTOPES_COLOR[iso], alpha=0.4)
            #    handles.append(mpatches.Patch(color=ISOTOPES_COLOR[iso]))
            #    labels.append(f'{ISO_LABELS[iso]}')
            pull =(counts - fit_values[i])/countserr
            chisquare = np.sum(pull**2)/(len(pull))
            plot1d_errorbar(figure, ax2, x_mass_binedges, counts=pull, err=np.zeros(len(pull)),  label_x="1/mass (1/GeV)", label_y="pull", legend=None,  col="black", setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
            plt.subplots_adjust(hspace=.0)

            fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {chisquare:.1f}",]
            fit_info_formatted = [f"\\fontsize{FONTSIZE}\\selectfont {info}" for info in fit_info]
            ax1.legend(handles, labels, title="\n".join(fit_info), frameon=False, fontsize=FONTSIZE)
            ax1.set_xticklabels([])
            ax1.get_yticklabels()[0].set_visible(False)
            ax2.set_ylim([-3.8, 3.8])
            ax1.set_ylim([0.1, 10 *np.max(counts)])
        
            ax1.set_yscale('log')
            if fit_simultaneous:
                savefig_tofile(figure, plotdir, f"fit_mass_bin_{ibin}_{self.detector}_fitsimul", show=False)
            else:
                iso = self.isotopes[0]
                savefig_tofile(figure, plotdir, f"fit_mass_bin_{ibin}_{self.detector}_binbybin_{iso}", show=False)



