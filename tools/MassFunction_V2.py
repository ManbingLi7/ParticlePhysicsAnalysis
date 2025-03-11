import abc
import os 
import math
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import tools.roottree as read_tree
from tools.calculator import calc_mass, calc_ekin_from_beta, calc_betafrommomentom, calc_ratio_and_err

import uproot
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, LeastSquares, NormalConstraint
from iminuit.util import describe, make_func_code
from tools.constants import ISOTOPES, NUCLEI_NUMBER, ISOTOPES_COLOR, ISO_LABELS
from tools.histograms import Histogram
from tools.functions import gaussian, asy_gaussian, poly

from scipy import interpolate
from scipy.optimize import NonlinearConstraint
from tools.graphs import MGraph, plot_graph
import uncertainties
from uncertainties import unumpy
from uncertainties import ufloat
from tools.plottools import plot1dhist, plot2dhist, plot1d_errorbar, savefig_tofile, setplot_defaultstyle, FIGSIZE_BIG, FIGSIZE_SQUARE, FIGSIZE_MID, FIGSIZE_WID, plot1d_step, FONTSIZE, set_plot_defaultstyle
import matplotlib.patches as mpatches
from typing import Annotated

def expo_func(x, pa, pb, pc):    
    pdf = pa* (1 - np.exp((x-pb)/pc)) 
    return pdf

class InverseMassFunctionFit(metaclass=abc.ABCMeta):
    def __init__(self, nuclei, isotopes, hist_data, fit_energy_range, fit_mass_range, detector, is_constraint, numpvalues, is_nonlinearconstraint=False, component="All"):
        self.nuclei = nuclei
        self.is_constraint = is_constraint
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
        self.mass_function = self.make_mass_function_simultaneous()
        self.mass_function_binbybin = self.make_mass_function_binbybin(component=component)
        self.x_fit_energy = self.energy_binning.bin_centers[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1]
        self.x_fit_mass = self.mass_binning.bin_centers[self.fit_mass_binrange[0] : self.fit_mass_binrange[1] + 1]
        self.nonlinearconstraint = is_nonlinearconstraint
        self.numpvalues = numpvalues
        
    #print initial values and print fit results
    def __str__(self):
        print()

    def number_events(self):
        n_counts = np.sum(self.data, axis=1)
        n_counts_err = np.sqrt(n_counts)
        return n_counts

    @abc.abstractmethod 
    def make_mass_function_simultaneous(self, drawiso="All"):
        raise NotImplementedError("make_mass_function_simultaneous is abstract in this MassFunction")

    @abc.abstractmethod
    def get_polypars_index(self):
        raise NotImplementedError("get_polypars_index is abstract in this MassFunction")

    @abc.abstractmethod 
    def make_mass_function_binbybin(self, component="All"):
        raise NotImplementedError("make_mass_function_binbybin is abstract in this MassFunction")
    
    def get_fit_xvalues(self):
        x_energy = self.energy_binning.bin_centers[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1]
        x_mass = self.mass_binning.bin_centers[self.fit_mass_binrange[0] : self.fit_mass_binrange[1] + 1]        
        xgrid_energy, xgrid_mass = np.meshgrid(x_energy, x_mass, indexing = "ij")
        xvalues = np.stack((xgrid_energy.reshape(-1), xgrid_mass.reshape(-1)))
        return xvalues

    def get_data_infitrange(self):
        counts = self.hist.values[self.fit_energy_binrange[0] : self.fit_energy_binrange[1] + 1, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
        return counts

    @abc.abstractmethod 
    def perform_fit(self, guess, fit_simultaneous=True, verbose=False, fixed_pars=None, lim_pars=None, parlim=None):
        raise NotImplementedError("perform_fit is abstract in this MassFunction")
 
    def get_fit_values(self, fit_parameters, fit_simultaneous=True):
        xvalues = self.get_fit_xvalues()
        if fit_simultaneous:
            return self.mass_function(xvalues, *unumpy.nominal_values(fit_parameters)).reshape(self.num_energybin, self.num_massbin)
        else:
            return self.mass_function_binbybin(xvalues, *unumpy.nominal_values(fit_parameters)).reshape(self.num_energybin, self.num_massbin)

    def get_fit_values_iso(self, iso, fit_parameters, fit_simultaneous=True):
        xvalues = self.get_fit_xvalues()
        if fit_simultaneous:     
            return self.make_mass_function_simultaneous(drawiso=iso)(xvalues, *unumpy.nominal_values(fit_parameters)).reshape(self.num_energybin, self.num_massbin)
        else:
            return self.make_mass_function_binbybin(drawiso=iso)(xvalues, *unumpy.nominal_values(fit_parameters)).reshape(self.num_energybin, self.num_massbin)
            
    def get_fitmc_component_gaus(self, fit_parameters, fit_simultaneous=False):
        xvalues = self.get_fit_xvalues()
        return self.make_mass_function_binbybin(component="gaus")(xvalues, *unumpy.nominal_values(fit_parameters)).reshape(self.num_energybin, self.num_massbin)

    def get_fitmc_component_asygaus(self, fit_parameters, fit_simultaneous=False):
        xvalues = self.get_fit_xvalues()
        return self.make_mass_function_binbybin(component="asygaus")(xvalues, *unumpy.nominal_values(fit_parameters)).reshape(self.num_energybin, self.num_massbin)
        
    def draw_fit_results(self, fit_parameters, par_dict, plotdir, fit_simultaneous=True, guess=None, figname=None, setylog=False):
        fit_values = self.get_fit_values(fit_parameters, fit_simultaneous)
        x_mass_bincenter =  self.mass_binning.bin_centers[self.fit_mass_binrange[0]:  self.fit_mass_binrange[1]+1]
        x_mass_binedges = self.mass_binning.edges[self.fit_mass_binrange[0]:  self.fit_mass_binrange[1]+2]
        fit_values_iso = dict()
        for iso in self.isotopes:    
            fit_values_iso[iso] = self.get_fit_values_iso(iso, fit_parameters, fit_simultaneous=fit_simultaneous)

        chisquare = np.zeros_like(self.x_fit_energy)
        for i, ibin in enumerate(range(self.fit_energy_binrange[0], self.fit_energy_binrange[1] + 1)):
            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))
            print('test ibin:', ibin, self.detector, self.energy_binning.edges[ibin])
            ax1.text(0.03, 0.98, f"{self.detector}: [{self.energy_binning.edges[ibin]:.2f}, {self.energy_binning.edges[ibin+1]:.2f}] GeV/n", fontsize=FONTSIZE+1, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black")
            counts = self.hist.values[ibin, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
            countserr = self.hist.get_errors()[ibin, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
            countserr[countserr == 0.0]  = 1
            plot1d_errorbar(figure, ax1, x_mass_binedges, counts, err=countserr, label_x="1/mass (1/GeV)", label_y="counts", col='black', legend='data')

            #ax1.plot(get_bin_center(massbinedges[min_massbin: max_massbin + 2]), mass_function_guess[i], color='blue', label='guess')
            ax1.plot(x_mass_bincenter, fit_values[i], color='red', label='fit')
            for iso in self.isotopes:
                ax1.plot(x_mass_bincenter, fit_values_iso[iso][i], color=ISOTOPES_COLOR[iso], label=f'{ISO_LABELS[iso]}')
                ax1.fill_between(x_mass_bincenter, np.zeros_like(fit_values_iso[iso][i]), fit_values_iso[iso][i], interpolate=True, color=ISOTOPES_COLOR[iso], alpha=0.2)

            handles = []
            labels = []

            totalN = self.get_data_infitrange().sum(axis=1)
            '''
            if self.is_constraint == False:
                for iso in self.isotopes:
                    isonum = NUCLEI_NUMBER[iso] 
                    N_iso, N_isoerr = par_dict[f"n{isonum}_{i}"]['value'], par_dict[f"n{isonum}_{i}"]['error']
                    print("type:", type(N_iso), type(N_isoerr))
                    handles.append(mpatches.Patch(color=ISOTOPES_COLOR[iso]))
                    relativeN_iso, relativeN_isoerr  = calc_ratio_and_err(N_iso, totalN, N_iso, N_isoerr)
                    print("type:", type(relativeN_iso), type(relativeN_isoerr))  
                    print(relativeN_iso, relativeN_isoerr)
                    #labels.append(f'{ISO_LABELS[iso]}{relativeN_iso:.2f} $\\pm$ {relativeN_isoerr:.2f}')
                    labels.append(f'{ISO_LABELS[iso]}')
                    #N_last_iso, N_last_isoerr = 1 - 
            else:
            '''
            for iso in self.isotopes:
                isonum = NUCLEI_NUMBER[iso] 
                handles.append(mpatches.Patch(color=ISOTOPES_COLOR[iso]))
                labels.append(f'{ISO_LABELS[iso]}')

            
            pull =(counts - fit_values[i])/countserr
            chisquare[i] = np.sum(pull**2)/(len(pull))
            plot1d_errorbar(figure, ax2, x_mass_binedges, counts=pull, err=np.zeros(len(pull)),  label_x="1/mass (1/GeV)", label_y="pull", legend=None,  col="black", setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
            plt.subplots_adjust(hspace=.0)

            fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {chisquare[i]:.1f}",]
            fit_info_formatted = [f"\\fontsize{FONTSIZE}\\selectfont {info}" for info in fit_info]
            ax1.legend(handles, labels, title="\n".join(fit_info), frameon=False, fontsize=FONTSIZE)
            ax1.set_xticklabels([])
            ax1.get_yticklabels()[0].set_visible(False)
            ax2.set_ylim([-3.8, 3.8])
            if setylog:
                ax1.set_ylim([0.01, 10 *np.max(counts)])
                ax1.set_yscale('log')
            else:
                ax1.set_ylim([0.01, 1.3 *np.max(counts)])
                
            if fit_simultaneous:
                savefig_tofile(figure, plotdir, f"fitbin{ibin}_{self.detector}_fitsimul_{figname}" if figname is not None else f"fitbin{ibin}_{self.detector}_fitsimul", show=False)
            else:
                savefig_tofile(figure, plotdir, f"fitbin{ibin}_{self.detector}_binbybin_{figname}" if figname is not None else f"fitbin{ibin}_{self.detector}_binbybin", show=False)
                
        graph_chisquare = MGraph(self.x_fit_energy, chisquare, np.zeros_like(chisquare))
        df_chisquare = dict()
        graph_chisquare.add_to_file(df_chisquare, f'graph_chisquare_{self.detector}')
        np.savez(os.path.join(plotdir, f'df_chisquare_{self.detector}_{figname}.npz'), **df_chisquare)


    def draw_fit_results_compare_datamc(self, fit_parameters, hist2d_mc, par_dict, plotdir, fit_simultaneous=True, guess=None, figname=None, setylog=False):
        fit_values = self.get_fit_values(fit_parameters, fit_simultaneous)
        x_mass_bincenter =  self.mass_binning.bin_centers[self.fit_mass_binrange[0]:  self.fit_mass_binrange[1]+1]
        x_mass_binedges = self.mass_binning.edges[self.fit_mass_binrange[0]:  self.fit_mass_binrange[1]+2]
        fit_values_iso = dict()
        for iso in self.isotopes:    
            fit_values_iso[iso] = self.get_fit_values_iso(iso, fit_parameters, fit_simultaneous=fit_simultaneous)

        chisquare = np.zeros_like(self.x_fit_energy)
        print(self.detector, 'self.fit_energy_binrange:', self.fit_energy_binrange)
        print(hist2d_mc[iso].binnings[0].edges)
        print(self.energy_binning.edges)
        assert(np.all(hist2d_mc[iso].binnings[0].edges == self.energy_binning.edges))
        for iso in self.isotopes:
            for i, ibin in enumerate(range(self.fit_energy_binrange[0], self.fit_energy_binrange[1] + 1)):
                figure, ax1 = plt.subplots(figsize=(16, 14))
                ax1.text(0.03, 0.98, f"{self.detector}: [{self.energy_binning.edges[ibin]:.2f}, {self.energy_binning.edges[ibin+1]:.2f}] GeV/n", fontsize=FONTSIZE+1, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black")


                
                counts_mc = hist2d_mc[iso].values[ibin, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
                countserr_mc = hist2d_mc[iso].get_errors()[ibin, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
                countserr_mc[countserr_mc == 0.0]  = 1
                normalization = np.sum(counts_mc)
                counts_mc_normalized = counts_mc/normalization
                countserr_mc_normalized = countserr_mc/normalization
                fit_values_iso_normalized = fit_values_iso[iso][i]/np.sum(fit_values_iso[iso][i])
                plot1d_errorbar(figure, ax1, x_mass_binedges, counts_mc_normalized, err=countserr_mc_normalized, label_x="1/mass (1/GeV)", label_y="counts", col='black', legend='MC')
                ax1.plot(x_mass_bincenter, fit_values_iso_normalized, color=ISOTOPES_COLOR[iso], label=f'{ISO_LABELS[iso]}')
                #ax1.fill_between(x_mass_bincenter, np.zeros_like(fit_values_iso[iso][i]), fit_values_iso[iso][i], interpolate=True, color=ISOTOPES_COLOR[iso], alpha=0.2)
                handles = []
                labels = []

                totalN = self.get_data_infitrange().sum(axis=1)
                '''
                if self.is_constraint == False:
                for iso in self.isotopes:
                    isonum = NUCLEI_NUMBER[iso] 
                    N_iso, N_isoerr = par_dict[f"n{isonum}_{i}"]['value'], par_dict[f"n{isonum}_{i}"]['error']
                    print("type:", type(N_iso), type(N_isoerr))
                    handles.append(mpatches.Patch(color=ISOTOPES_COLOR[iso]))
                    relativeN_iso, relativeN_isoerr  = calc_ratio_and_err(N_iso, totalN, N_iso, N_isoerr)
                    print("type:", type(relativeN_iso), type(relativeN_isoerr))  
                    print(relativeN_iso, relativeN_isoerr)
                    #labels.append(f'{ISO_LABELS[iso]}{relativeN_iso:.2f} $\\pm$ {relativeN_isoerr:.2f}')
                    labels.append(f'{ISO_LABELS[iso]}')
                    #N_last_iso, N_last_isoerr = 1 - 
                else:
                '''
                
                isonum = NUCLEI_NUMBER[iso] 
                handles.append(mpatches.Patch(color=ISOTOPES_COLOR[iso]))
                labels.append(f'{ISO_LABELS[iso]}')

            
                #ratio = (counts_mc_normalized-fit_values_iso_normalized)/counts_mc_normalized
                ratio = counts_mc_normalized/fit_values_iso_normalized
                #chisquare[i] = np.sum(pull**2)/(len(pull))
                #plot1d_errorbar(figure, ax2, x_mass_binedges, counts=ratio, err=np.zeros(len(ratio)),  label_x="1/mass (1/GeV)", label_y="ratio", legend=None,  col="black", setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
                plt.subplots_adjust(hspace=.0)

                fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {chisquare[i]:.1f}",]
                fit_info_formatted = [f"\\fontsize{FONTSIZE}\\selectfont {info}" for info in fit_info]
                ax1.legend(handles, labels, title="\n".join(fit_info), frameon=False, fontsize=FONTSIZE)
                ax1.set_xlabel('1/m')
                #if setylog:
                #ax1.set_ylim([0.01, 1)
                #xax1.set_yscale('log')
                #else:
                #ax1.set_ylim([0.01, 1.3 *np.max(counts_mc_normalized)])
            
                savefig_tofile(figure, plotdir, f"CompareDataMCBin{ibin}_{iso}{self.detector}", show=False)
                                


    def draw_fit_results_mcmix_iso(self, fit_parameters, par_dict, plotdir, hist2d_input, fit_simultaneous=True, guess=None, figname=None, setylog=False):
        fit_values = self.get_fit_values(fit_parameters, fit_simultaneous)
        x_mass_bincenter =  self.mass_binning.bin_centers[self.fit_mass_binrange[0]:  self.fit_mass_binrange[1]+1]
        x_mass_binedges = self.mass_binning.edges[self.fit_mass_binrange[0]:  self.fit_mass_binrange[1]+2]
        fit_values_iso = dict()
        for iso in self.isotopes:
            fit_values_iso[iso] = self.get_fit_values_iso(iso, fit_parameters, fit_simultaneous)

        for iso in self.isotopes:
            chisquare = np.zeros_like(self.x_fit_energy)
            for i, ibin in enumerate(range(self.fit_energy_binrange[0], self.fit_energy_binrange[1] + 1)):
                figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14)) 
                ax1.text(0.03, 0.98, f"{self.detector}: [{self.energy_binning.edges[ibin]:.2f}, {self.energy_binning.edges[ibin+1]:.2f}] GeV/n", fontsize=FONTSIZE+1, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black")
                counts = hist2d_input[iso].values[ibin, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
                countserr = hist2d_input[iso].get_errors()[ibin, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
                countserr[countserr == 0.0]  = 1
                plot1d_errorbar(figure, ax1, x_mass_binedges, counts, err=countserr, label_x="1/mass (1/GeV)", label_y="counts", col='black', legend='data')
                #ax1.plot(get_bin_center(massbinedges[min_massbin: max_massbin + 2]), mass_function_guess[i], color='blue', label='guess')
                normalization = np.sum(counts) /np.sum(fit_values_iso[iso][i])
                normalize_fitvalue =  fit_values_iso[iso][i] * normalization
                ax1.plot(x_mass_bincenter, fit_values_iso[iso][i] * normalization, color='red', label=f'{ISO_LABELS[iso]}')
                
                handles = []
                labels = []

                totalN = self.get_data_infitrange().sum(axis=1)
                if self.is_constraint == False:
                    for iso in self.isotopes:
                        isonum = NUCLEI_NUMBER[iso] 
                        N_iso, N_isoerr = par_dict[f"n{isonum}_{i}"]['value'], par_dict[f"n{isonum}_{i}"]['error']
                        print("type:", type(N_iso), type(N_isoerr))
                        handles.append(mpatches.Patch(color=ISOTOPES_COLOR[iso]))
                        relativeN_iso, relativeN_isoerr  = calc_ratio_and_err(N_iso, totalN, N_iso, N_isoerr)
                        print("type:", type(relativeN_iso), type(relativeN_isoerr))  
                        print(relativeN_iso, relativeN_isoerr)
                        #labels.append(f'{ISO_LABELS[iso]}{relativeN_iso:.2f} $\\pm$ {relativeN_isoerr:.2f}')
                        labels.append(f'{ISO_LABELS[iso]}')
                        #N_last_iso, N_last_isoerr = 1 - 
                
                pull =(counts - normalize_fitvalue)/countserr
                chisquare[i] = np.sum(pull**2)/(len(pull))
                plot1d_errorbar(figure, ax2, x_mass_binedges, counts=pull, err=np.zeros(len(pull)),  label_x="1/mass (1/GeV)", label_y="pull", legend=None,  col="black", setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
                plt.subplots_adjust(hspace=.0)

                fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {chisquare[i]:.1f}",]
                fit_info_formatted = [f"\\fontsize{FONTSIZE}\\selectfont {info}" for info in fit_info]
                ax1.legend(handles, labels, title="\n".join(fit_info), frameon=False, fontsize=FONTSIZE)
                ax1.set_xticklabels([])
                ax1.get_yticklabels()[0].set_visible(False)
                ax2.set_ylim([-3.8, 3.8])
                if setylog:
                    ax1.set_ylim([0.01, 10 *np.max(counts)])
                    ax1.set_yscale('log')
                else:
                    ax1.set_ylim([0.01, 1.3 *np.max(counts)])
                
                if fit_simultaneous:
                    savefig_tofile(figure, plotdir, f"fitbin{ibin}_{self.detector}{iso}_fitsimul_{figname}" if figname is not None else f"fitbin{ibin}_{self.detector}{iso}_fitsimul", show=False)
                else:
                    savefig_tofile(figure, plotdir, f"fitbin{ibin}_{self.detector}{iso}_binbybin_{figname}" if figname is not None else f"fitbin{ibin}_{self.detector}{iso}_binbybin", show=False)
                



    def draw_fit_results_mc(self, fit_parameters, par_dicts, plotdir, fit_simultaneous=False, guess=None, x_label=None, figname=None, drawlog=False):
        fit_values = self.get_fit_values(fit_parameters, fit_simultaneous)
        x_mass_bincenter =  self.mass_binning.bin_centers[self.fit_mass_binrange[0]:  self.fit_mass_binrange[1]+1]
        x_mass_binedges = self.mass_binning.edges[self.fit_mass_binrange[0]:  self.fit_mass_binrange[1]+2]

        fit_part1 = self.get_fitmc_component_gaus(fit_parameters)
        fit_part2 = self.get_fitmc_component_asygaus(fit_parameters)
        chisquare = np.zeros_like(self.x_fit_energy)
        for i, ibin in enumerate(range(self.fit_energy_binrange[0], self.fit_energy_binrange[1] + 1)):
            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14)) 
            #ax1.text(0.03, 0.98, f"{self.detector}: [{self.energy_binning.edges[ibin]:.2f}, {self.energy_binning.edges[ibin+1]:.2f}] GeV/n", fontsize=24, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black")
            ax1.text(0.03, 0.98, f"{self.detector}: [{self.energy_binning.edges[ibin]:.2f}, {self.energy_binning.edges[ibin+1]:.2f}]", fontsize=24, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black")
            counts = self.hist.values[ibin, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
            countserr = self.hist.get_errors()[ibin, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
            countserr[countserr == 0.0]  = 1
            plot1d_errorbar(figure, ax1, x_mass_binedges, counts, err=countserr, label_x="1/mass (1/GeV)", label_y="counts", col='black')
            #ax1.errorbar(x_mass_bincenter, counts, countserr, fmt='.', markersize=10)
            
            #ax1.plot(get_bin_center(massbinedges[min_massbin: max_massbin + 2]), mass_function_guess[i], color='blue', label='guess')
            ax1.plot(x_mass_bincenter, fit_values[i], color='red', label='fit')
            ax1.plot(x_mass_bincenter, fit_part1[i], "--", color='blue', label='part gaus')
            ax1.plot(x_mass_bincenter, fit_part2[i], "--", color='green', label='part asygaus')
            
            handles = []
            labels = []
            pull =(counts - fit_values[i])/countserr
            chisquare[i] = np.sum(pull**2)/(len(pull))
            plot1d_errorbar(figure, ax2, x_mass_binedges, counts=pull, err=np.zeros(len(pull)), label_y="pull", col="black", setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
            plt.subplots_adjust(hspace=.0)

            #fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {chisquare:.1f}",]
            fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {chisquare[i]:.1f}",
                        f"$\\mu$ ={par_dicts[f'mean_{i}']['value']:.4f}$\\pm$ {par_dicts[f'mean_{i}']['error']:.4f}",
                        f"$\\sigma$ ={par_dicts[f'sigma_{i}']['value']:.4f}$\\pm$ {par_dicts[f'sigma_{i}']['error']:.4f}",
                        f"$f_{{c}}$ ={par_dicts[f'fraccore_{i}']['value']:.3f}$\\pm$ {par_dicts[f'fraccore_{i}']['error']:.3f}",
                        f"$\\epsilon$ ={par_dicts[f'sigma_ratio_{i}']['value']:.3f}$\\pm$ {par_dicts[f'sigma_ratio_{i}']['error']:.3f}",
                        f"$\\alpha$ ={par_dicts[f'asy_factor_{i}']['value']:.3f}$\\pm$ {par_dicts[f'asy_factor_{i}']['error']:.3f}",]
                        
            fit_info_formatted = [f"\\fontsize{FONTSIZE}\\selectfont {info}" for info in fit_info]
            ax1.legend(handles, labels, title="\n".join(fit_info), frameon=False, fontsize=12, loc='upper right')
            ax1.set_xticklabels([])
            ax1.get_yticklabels()[0].set_visible(False)
            ax2.set_ylim([-3.8, 3.8])
            #ax1.set_xlim([0.08, 0.23])
            ax2.set_xlim([0.08, 0.23])
            if drawlog:
                ax1.set_ylim([1, 10 *np.max(counts)])
                ax1.set_yscale('log')
            else:
                ax1.set_ylim([0.1, 1.3 *np.max(counts)])
                
            ax2.set_xlabel("1/mass (1/GeV)" if x_label is None else f"{x_label}")
            #ax1.set_yscale('log')
            if fit_simultaneous:
                savefig_tofile(figure, plotdir, f"fitbin{ibin}_{self.detector}_fitsimul_{figname}" if figname is not None else f"fitbin{ibin}_{self.detector}_simufit", show=False)
            else:
                iso = self.isotopes[0]
                savefig_tofile(figure, plotdir, f"fitbin{ibin}_{self.detector}_binbybin_{figname}" if figname is not None else f"fitbin{ibin}_{self.detector}_binbybin", show=False)
      
        graph_chisquare = MGraph(self.x_fit_energy, chisquare, np.zeros_like(chisquare))
        df_chisquare = dict()
        graph_chisquare.add_to_file(df_chisquare, f'graph_chisquare_{self.detector}')
        np.savez(os.path.join(plotdir, f'df_chisquare_{self.detector}_{figname}.npz'), **df_chisquare)



    def draw_fit_results_mc_betareso(self, fit_parameters, par_dicts, plotdir, fit_simultaneous=False, guess=None, x_label=None, figname=None):
        fit_values = self.get_fit_values(fit_parameters, fit_simultaneous)
        x_mass_bincenter =  self.mass_binning.bin_centers[self.fit_mass_binrange[0]:  self.fit_mass_binrange[1]+1]
        x_mass_binedges = self.mass_binning.edges[self.fit_mass_binrange[0]:  self.fit_mass_binrange[1]+2]

        fit_part1 = self.get_fitmc_component_gaus(fit_parameters)
        fit_part2 = self.get_fitmc_component_asygaus(fit_parameters)
        
        for i, ibin in enumerate(range(self.fit_energy_binrange[0], self.fit_energy_binrange[1] + 1)):
            figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14)) 
            ax1.text(0.03, 0.98, f"{self.detector}: [{self.energy_binning.edges[ibin]:.2f}, {self.energy_binning.edges[ibin+1]:.2f}]", fontsize=24, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black")
            counts = self.hist.values[ibin, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
            countserr = self.hist.get_errors()[ibin, self.fit_mass_binrange[0]: self.fit_mass_binrange[1] + 1]
            countserr[countserr == 0.0]  = 1
            plot1d_errorbar(figure, ax1, x_mass_binedges, counts, err=countserr, label_x="1/mass (1/GeV)", label_y="counts", col='black')
            #ax1.errorbar(x_mass_bincenter, counts, countserr, fmt='.', markersize=10)
            
            #ax1.plot(get_bin_center(massbinedges[min_massbin: max_massbin + 2]), mass_function_guess[i], color='blue', label='guess')
            ax1.plot(x_mass_bincenter, fit_values[i], color='red', label='fit')
            ax1.plot(x_mass_bincenter, fit_part1[i], "--", color='blue', label='part gaus')
            ax1.plot(x_mass_bincenter, fit_part2[i], "--", color='green', label='part asygaus')
            
            handles = []
            labels = []
            pull =(counts - fit_values[i])/countserr
            chisquare = np.sum(pull**2)/(len(pull))
            plot1d_errorbar(figure, ax2, x_mass_binedges, counts=pull, err=np.zeros(len(pull)), label_y="pull", col="black", setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
            plt.subplots_adjust(hspace=.0)

            #fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {chisquare:.1f}",]
            fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {chisquare:.1f}",
                        f"$\\mu$ ={par_dicts[f'mean_{i}']['value']:.4f}$\\pm$ {par_dicts[f'mean_{i}']['error']:.4f}",
                        f"$\\sigma$ ={par_dicts[f'sigma_{i}']['value']:.5f}$\\pm$ {par_dicts[f'sigma_{i}']['error']:.5f}",
                        f"$f_{{c}}$ ={par_dicts[f'fraccore_{i}']['value']:.3f}$\\pm$ {par_dicts[f'fraccore_{i}']['error']:.3f}",
                        f"$\\epsilon$ ={par_dicts[f'sigma_ratio_{i}']['value']:.3f}$\\pm$ {par_dicts[f'sigma_ratio_{i}']['error']:.3f}",
                        f"$\\alpha$ ={par_dicts[f'asy_factor_{i}']['value']:.3f}$\\pm$ {par_dicts[f'asy_factor_{i}']['error']:.3f}",]
                        
            fit_info_formatted = [f"\\fontsize{FONTSIZE}\\selectfont {info}" for info in fit_info]
            ax1.legend(handles, labels, title="\n".join(fit_info), frameon=False)
            ax1.set_xticklabels([])
            ax1.get_yticklabels()[0].set_visible(False)
            ax2.set_ylim([-3.8, 3.8])
            ax1.set_ylim([0.1, 1.3 *np.max(counts)])
            ax2.set_xlabel("1/mass (1/GeV)" if x_label is None else f"{x_label}")
            #ax1.set_yscale('log')
            if fit_simultaneous:
                savefig_tofile(figure, plotdir, f"fitbin{ibin}_{self.detector}_fitsimul_{figname}" if figname is not None else f"fitbin{ibin}_{self.detector}_simufit", show=False)
            else:
                iso = self.isotopes[0]
                savefig_tofile(figure, plotdir, f"fitbin{ibin}_{self.detector}_binbybin_{iso}_{figname}" if figname is not None else f"fitbin{ibin}_{self.detector}_binbybin_{iso}", show=False)



