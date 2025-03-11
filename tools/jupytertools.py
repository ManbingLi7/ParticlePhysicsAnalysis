import numpy as np
from .graphs import MGraph, slice_graph, plot_graph, slice_graph_by_value, concatenate_graphs, scale_graph
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines 
import matplotlib
import os
from .binnings_collection import fbinning_energy_rebin, get_bin_center
from .calculator import calc_rig_from_ekin
from .constants import ISOTOPES_MASS, ISOTOPES_CHARGE,DETECTOR_LABEL 
from .plottools import xaxistitle, xaxis_binning
from .histograms import Histogram, WeightedHistogram, plot_histogram_1d, plot_histogram_2d 
import os
from .binnings import Binning 
from tools.studybeta import minuitfit_LL, cdf_gaussian, calc_signal_fraction, cdf_double_gaus, double_gaus, minuitfit_Chi
from tools.functions import cumulative_norm_gaus, normalized_gaussian, poly, upoly
from scipy.interpolate import UnivariateSpline
import pickle
from tools.plottools import savefig_tofile

xlabel_ekin = r'$\mathrm{E_{k/n} \ (GeV/n)}$'
COL = {'iss': 'black', 'mc': 'tab:orange'}
Marker_com = {'iss': 'o', 'mc': 's'} 

def write_points_totxt_with_binedge(graph1, xbinning, datadir, filename, header=None, fmt=None, writeHeader=True):
    xbin_indices = xbinning.get_indices([graph1.xvalues[0], graph1.xvalues[-1]])  
    bin_edges = xbinning.edges[xbin_indices[0]: xbin_indices[1]+2]   
    # Define header
    if header is None:
        header = "Ekn_low\tEkn_up\tflux\tfluxerr"
    if fmt is None:
        fmt = ('%.4f', '%.4f', '%.7f', '%.7f')
        
    combined_array = np.column_stack((bin_edges[0:-1], bin_edges[1:], graph1.yvalues, graph1.yerrs))
    delimiter = ', '
    if writeHeader:
        np.savetxt(os.path.join(datadir, f'{filename}.txt'), combined_array, fmt=fmt, delimiter=delimiter, header=header, comments='')
    else:
        np.savetxt(os.path.join(datadir, f'{filename}.txt'), combined_array, fmt=fmt, delimiter=delimiter)


        

def write_data_totxt(graph1, xbinning, combinedarray, datadir, filename, header, fmt, writeHeader=True):
    xbin_indices = xbinning.get_indices([graph1.xvalues[0], graph1.xvalues[-1]])  
    bin_edges = xbinning.edges[xbin_indices[0]: xbin_indices[1]+2]   
    if writeHeader:
        np.savetxt(os.path.join(datadir, f'{filename}.txt'), combined_array, fmt=fmt, delimiter='\t', header=header, comments='')
    else:
        np.savetxt(os.path.join(datadir, f'{filename}.txt'), combined_array, fmt=fmt, delimiter='\t')


#default settings for plots
tick_length = 14                                                                                         
tick_width=1.5                                                                                           
tick_labelsize = 40                                                                                      
legendfontsize = 45 
def set_plot_style(plot):                                                                         
    plt.rcParams["font.weight"] = "bold"                                                                 
    plt.rcParams["axes.labelweight"] = "bold"                                                            
    plt.rcParams['font.size']= 45                                                                 
    plt.rcParams['xtick.top'] = True                                                                    
    plt.rcParams['ytick.right'] = True                                                                  
    plot.tick_params(axis='both', which="major",direction='in', length=tick_length, width=tick_width, labelsize=tick_labelsize, pad=9)                                       
    plot.tick_params(axis='both', which="minor",direction='in', length=tick_length/2.0, width=tick_width, labelsize=tick_labelsize, pad=9)                                        
    for axis in ['top','bottom','left','right']:                                                    
        plot.spines[axis].set_linewidth(3)                                                               
    plt.minorticks_on() 



TEXTSIZE = 40
FIGSIZE2 = (20, 15)
FIGSIZE3 = (22, 16)
FIGSIZE_X12 = (26, 22)
FIGSIZE_X12_V2 = (25, 20)
MKSIZE = 28
XLABEL_EKN = r'$\mathrm{E_{k/n} \ GeV/n}$'
FIGSIZE_Y123 = (30, 22)
def SetAx1Axis(ax1, xlabelname, ylabelname, labelfontsize, xlimrange=None, ylimrange=None, custom_ticks=None, custom_tickslabels=None, gridx=False, gridy=False, setylog=False, setxlog=False):
    ax1.set_xlabel(xlabelname, fontsize=labelfontsize, labelpad=24)
    ax1.set_ylabel(ylabelname, fontsize=labelfontsize, labelpad=24)
    if xlimrange is not None:
        ax1.set_xlim(xlimrange)
    if ylimrange is not None:
        ax1.set_ylim(ylimrange)
    if custom_ticks is not None:
        ax1.set_xticks(custom_ticks)
    if custom_tickslabels is not None:
        ax1.set_xticklabels(custom_tickslabels)        
    if gridx == True:
        ax1.grid(axis='x')
    if gridy == True:
        ax1.grid(axis='y')
    if setylog:
        ax1.set_yscale('log')
    if setxlog:
        ax1.set_xscale('log')


def SetAx1Ax2Xaxis(ax1, ax2, xlabelname, labelfontsize, xlimrange=None,  custom_ticks=None, custom_tickslabels=None, gridx1=False,  gridx2=False, xscale=None, rmAx1=True):
    ax2.set_xlabel(xlabelname, fontsize=labelfontsize, labelpad=24)
    ax1.set_xticklabels([])
    #ax1.get_yticklabels()[0].set_visible(False)
    if xscale == 'log':
        ax1.set_xscale('log')
        ax2.set_xscale('log')                                                                                                                       
    if xlimrange is not None:
        ax1.set_xlim(xlimrange)
        ax2.set_xlim(xlimrange)
    if custom_ticks is not None:
        ax2.set_xticks(custom_ticks)
    if custom_tickslabels is not None:
        ax2.set_xticklabels(custom_tickslabels)        
    if gridx1 == True:
        ax1.grid(axis='x')
    if gridx2 == True:
        ax2.grid(axis='x')

    if rmAx1 == True:
        #ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()) 
        ax1.set_xticklabels([])                                                                                                                                                         
        plt.subplots_adjust(hspace=.0)
   

def SetAx1Ax2Yaxis(ax1, ax2, ylabelname1, ylabelname2, labelfontsize, ylimrange1=None,  ylimrange2=None, gridy1=False,  gridy2=False, yscale1=None, yscale2=None):
    ax1.set_ylabel(ylabelname1, fontsize=labelfontsize, labelpad=24)
    ax2.set_ylabel(ylabelname2, fontsize=labelfontsize, labelpad=24)
    if yscale1 == 'log':
        ax1.set_yscale('log')
    if yscale2 == 'log':
        ax2.set_yscale('log')                                                                                                                       
    if ylimrange1 is not None:
        ax1.set_ylim(ylimrange1)
    if ylimrange2 is not None:
        ax2.set_ylim(ylimrange2)
    if gridy1 == True:
        ax1.grid(axis='y')
    if gridy2 == True:
        ax2.grid(axis='y')


def weighted_avg_and_std(values, weights):                                                                                                                                                                         
    """                                                                                                                                                                                                          
    Return the weighted average and standard deviation.                                                                                                                                                          
    values, weights -- Numpy ndarrays with the same shape.                                                                                                                                                        
    """                                                                                                                                                                                                            
    average = np.average(values, weights=weights)                                                                                                                                                                 
    # Fast and numerically precise:                                                                                                                                                                              
    variance = np.average((values-average)**2, weights=weights)                                                                                                                                                   
                                                                                                                                                                                                                  
    return average, variance

def get_hist2d_rawmeanstd(hist2d):
    var_xcenter = hist2d.binnings[0].bin_centers[1:-1]
    var_ycenter = hist2d.binnings[1].bin_centers[1:-1]
    avg = np.zeros(len(var_xcenter))                                                                                                                                                                               
    std = np.zeros(len(var_xcenter))  
    
    for binx in range(1, len(var_xcenter)):                                                                                                                                                                      
        ibin_fitdata = hist2d.values[binx, 1:-1]
        if sum(ibin_fitdata) != 0:                                                                                                                                                                                
            avg[binx], std[binx] = weighted_avg_and_std(var_ycenter, ibin_fitdata)                                                                                                                                 
    return avg, std


def get_musigma_gausfit(hist2d_mc_tofTrue, minbin, maxbin, guess0, xfitrange, fitNsig = 3.0, FigName=None, plotfile=None):
    energy_binvalues = hist2d_mc_tofTrue.binnings[0].bin_centers[minbin:maxbin]
    graph_mean_mcTofTrueReso = MGraph(energy_binvalues, np.zeros_like(energy_binvalues), yerrs=np.zeros_like(energy_binvalues))
    graph_sigma_mcTofTrueReso = MGraph(energy_binvalues, np.zeros_like(energy_binvalues), yerrs=np.zeros_like(energy_binvalues))
    avg, std = get_hist2d_rawmeanstd(hist2d_mc_tofTrue)
    for ip, ibin in enumerate(range(minbin, maxbin)):
   
        lowbinedge = hist2d_mc_tofTrue.binnings[0].edges[ibin]
        upbinedge = hist2d_mc_tofTrue.binnings[0].edges[ibin + 1]
        hist1d_mc = hist2d_mc_tofTrue.project(ibin) 
        hist1d_mc = hist1d_mc * (1/np.sum(hist1d_mc.values))
    
        xbinrange = hist1d_mc.binnings[0].get_indices(xfitrange)
    
        xedges_mc = hist1d_mc.binnings[0].edges[xbinrange[0]:xbinrange[1]+1]
        xvalue_mc = hist1d_mc.binnings[0].bin_centers[xbinrange[0]:xbinrange[1]]
        yvalue_mc = hist1d_mc.values[xbinrange[0]:xbinrange[1]]
        yvalueserr_mc = np.sqrt(hist1d_mc.squared_values[xbinrange[0]:xbinrange[1]])
        yvalueserr_mc[yvalueserr_mc==0] = 0.0001
       
        guess_gaus_mc = guess0
        guess_gaus_mc['mu'] = avg[ibin]
        
        par_mc, parerr_mc = minuitfit_Chi(xvalue_mc, yvalue_mc, yvalueserr_mc,normalized_gaussian, guess_gaus_mc)
        for key in guess_gaus_mc.keys():
            guess_gaus_mc[key] = par_mc[key]
    
        xrange_iter2mc = [guess_gaus_mc['mu'] - fitNsig * guess_gaus_mc['sigma'], guess_gaus_mc['mu'] + fitNsig * guess_gaus_mc['sigma']]
        xbinrange2mc = hist1d_mc.binnings[0].get_indices(xrange_iter2mc)
    
        xedges_mc2 = hist1d_mc.binnings[0].edges[xbinrange2mc[0]:xbinrange2mc[1]+1]
        xvalue_mc2 = hist1d_mc.binnings[0].bin_centers[xbinrange2mc[0]:xbinrange2mc[1]]
        yvalue_mc2 = hist1d_mc.values[xbinrange2mc[0]:xbinrange2mc[1]]
        yvalueserr_mc2 = np.sqrt(hist1d_mc.squared_values[xbinrange2mc[0]:xbinrange2mc[1]])
        yvalueserr_mc2[yvalueserr_mc2==0] = 0.0001
    
        par_mc, parerr_mc = minuitfit_Chi(xvalue_mc2, yvalue_mc2, yvalueserr_mc2,normalized_gaussian, guess_gaus_mc)

        graph_mean_mcTofTrueReso.yvalues[ip] = par_mc['mu']
        graph_mean_mcTofTrueReso.yerrs[ip] = parerr_mc['mu']
        graph_sigma_mcTofTrueReso.yvalues[ip] = par_mc['sigma']
        graph_sigma_mcTofTrueReso.yerrs[ip] = parerr_mc['sigma']
                                                                                                                                                            
        fit_y_mc = normalized_gaussian(xvalue_mc2, *par_mc) 
        draw = False
        plotp = [1]
        if ip in plotp:
            figure, ax1 = plt.subplots(1, 1, figsize=(17, 14))
            plot_histogram_1d(ax1, hist1d_mc, style="mc", color='black', label=None, scale=None, gamma=None, xlog=False, ylog=False, shade_errors=False, setscilabely=True, show_overflow=False) 
            #ax1.legend()
            ax1.plot(xvalue_mc2, fit_y_mc, '-', linewidth=3, color='blue')
            ax1.text(0.6, 0.98, f"[{lowbinedge:.2f}, {upbinedge:.2f}] GeV/n", fontsize=TEXTSIZE-5, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color="black", fontweight="bold") 
            ax1.text(0.03, 0.93, f'{FigName}', fontsize=30, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='black', weight='normal')  
            ax1.text(0.03, 0.85, f"$\\mu:$ {par_mc['mu']:.4f}$\\pm$ {parerr_mc['mu']:.4f}", fontsize=30, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='black', weight='normal')  
            ax1.text(0.03, 0.8, f"$\\sigma:$ {par_mc['sigma']:.4f}$\\pm$ {parerr_mc['sigma']:.4f}", fontsize=30, verticalalignment='top', horizontalalignment='left', transform=ax1.transAxes, color='black', weight='normal')  
            ax1.set_ylabel('Normalized events')
            ax1.set_xlabel(r'm (GeV)')
            ax1.set_xlim([0.9*xfitrange[0], 1.1*xfitrange[1]]) 
            #ax1.set_xlim([0.11, 0.16])
            #ax1.set_yscale('log')
            if plotfile is not None:
                savefig_tofile(figure, plotfile, f"hist1d_{FigName}_{ibin}", show=True) 
    return graph_mean_mcTofTrueReso, graph_sigma_mcTofTrueReso 



def plot_parswitherr(fig, ax1, graph1, col, p0_mean, labelname=None):
    plot_graph(fig, ax1, graph1, color=col, style="EP", xlog=False, ylog=False, scale=None, markersize=mksize, label=labelname)
    popt, pcov = curve_fit(poly, np.log(graph1.xvalues), graph1.yvalues, p0 = p0_mean)  
    ax1.plot(graph1.xvalues, poly(np.log(graph1.xvalues), *popt), "-", color=col)
    polypars = uncertainties.correlated_values(popt, np.array(pcov)) 
    mufit_lower, mufit_upper = get_fitpdferrorband(np.log(graph1.xvalues), polypars, upoly)  
    ax1.fill_between(graph1.xvalues, mufit_lower, mufit_upper, color=col, alpha=0.2)




def get_spline(graph1, weight=False):
    xvector = graph1.xvalues
    yvector = graph1.yvalues
    if weight is False:
        spline_fit = UnivariateSpline(np.log(xvector), yvector, k=3, s=5)  
    else:
        spline_fit = UnivariateSpline(np.log(xvector), yvector, w=1/graph1.yerrs, k=3, s=100)
    return spline_fit
