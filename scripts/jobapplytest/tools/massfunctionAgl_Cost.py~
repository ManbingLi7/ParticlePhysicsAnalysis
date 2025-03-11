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

    def __init__(self, model):
        self.model = model
        counts = model.hist.values[model.fit_energy_binrange[0] : model.fit_energy_binrange[1] + 1, model.fit_mass_binrange[0]: model.fit_mass_binrange[1] + 1]   
        countserr = model.hist.get_errors()[model.fit_energy_binrange[0] : model.fit_energy_binrange[1] + 1, model.fit_mass_binrange[0]: model.fit_mass_binrange[1] + 1] 
        countserr[countserr == 0.0] = 1       
        self.yvalues = counts.reshape(-1)                              
        self.yvalueserr = countserr.reshape(-1)                                                                       
        self.function = model.mass_function()
        pars = describe(model.make_mass_function_simultaneous())
        lsq_parameters = pars[1:]

    def __call__(self, *par):
        yfit = model.mass_function(model.get_fit_xvalues() , *par)
        chisquare = np.sum((self.yvalues - yfit) ** 2 / self.yvalueserr ** 2)
        return chisquare

    
        
        


















    
