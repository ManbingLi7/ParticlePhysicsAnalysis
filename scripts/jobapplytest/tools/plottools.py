import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from tools.binnings_collection import get_nbins_in_range, get_sub_binning, get_bin_center
from tools.binnings import Binning
from tools.binnings_collection import BeRigidityBinningRICHRange, fbinning_energy

FIGSIZE_BIG = (16, 11)
FIGSIZE_WID = (16, 10)
FIGSIZE_SQUARE = (10, 10)
FIGSIZE_MID = (14, 11)
MARKER_SIZE = 20
LEGEND_FONTSIZE = 25
figsize = (12, 9)
tick_length = 14
tick_width=1.5
tick_labelsize = 30
legendfontsize = 38
textfontsize = 35
marker_size = 20

LABEL_FONTSIZE = 20
FONTSIZE = 30
FONTSIZE_BIG = 36
FONTSIZE_MID = 30                                                                             
FONTSIZE_SMALL = 20

xbinning = np.linspace(0, 1, 100)
ybinning= np.linspace(0, 1, 100)

def setplot_defaultstyle():
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.size']= FONTSIZE_BIG 
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    plt.minorticks_on()


def set_plot_defaultstyle(plot):
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.size']= FONTSIZE
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    plot.tick_params(axis='both', which="major",direction='in', length=tick_length, width=tick_width, labelsize=tick_labelsize)
    plot.tick_params(axis='both', which="minor",direction='in', length=tick_length/2.0, width=tick_width, labelsize=tick_labelsize)
    for axis in ['top','bottom','left','right']:                                                                                                                                                         
        plot.spines[axis].set_linewidth(2)    
    plt.minorticks_on()

    
def plot2dhist(figure=None, plot=None, xbinning=None, ybinning=None, counts=None, xlabel=None, ylabel=None, zlabel="counts", zmin=None, zmax=None, setlogx=False, setlogy=False, setscilabelx=True, setscilabely=True,  setlogz=True, figsize=FIGSIZE_BIG, tick_length=14, tick_width=1.5):
    setplot_defaultstyle()

    xbincenter = get_bin_center(xbinning)
    ybincenter = get_bin_center(ybinning)
    plot.set_xlabel(f"{xlabel}",fontsize = FONTSIZE)
    plot.set_ylabel(f"{ylabel}", fontsize = FONTSIZE)
    plot.tick_params(axis='both', which="major",direction='in', length=tick_length, width=tick_width, labelsize=tick_labelsize)
    plot.tick_params(axis='both', which="minor",direction='in', length=tick_length/2.0, width=tick_width, labelsize=tick_labelsize)
    setplot_defaultstyle()
    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)

    figure.subplots_adjust(left= 0.13, right=0.95, bottom=0.1, top=0.95)
    counts = np.ma.masked_where(counts == 0, counts).transpose()
    
    scale=1/counts.sum(axis=0)
    if (zmin == None) or (zmax == None):
        if (setlogz == True):
            mesh = plot.pcolormesh(xbincenter, ybincenter, counts, norm=colors.LogNorm(), cmap =plt.cm.get_cmap('jet'))
        else:
            mesh = plot.pcolormesh(xbincenter, ybincenter, counts, cmap =plt.cm.get_cmap('jet'))
    else:
        mesh = plot.pcolormesh(xbincenter, ybincenter, counts,  cmap =plt.cm.get_cmap('jet'), vmin=zmin, vmax=zmax)
    cbar = plt.colorbar(mesh, ax=plot, aspect=10, fraction=0.1)
    cbar.set_label(f"{zlabel}",fontsize=FONTSIZE_SMALL)
    cbar.ax.tick_params(which="major",direction='in', length=tick_length, width=tick_width, labelsize=FONTSIZE_SMALL)
    cbar.ax.tick_params(which="minor",direction='in', length=tick_length/2.0, width=tick_width, labelsize=FONTSIZE_SMALL)
    if setlogy:
        plot.set_yscale("log")
    if setscilabely:
        plot.ticklabel_format(axis='y', scilimits = (-3, 3), style="sci", useMathText=True)
    
    if setlogx:
        plot.set_xscale("log")
    if setscilabelx:
        plot.ticklabel_format(axis='x', scilimits = (-3, 3), style="sci", useMathText=True)


def plot1dhist(figure, plot, xbinning=None, counts=None, err=None, label_x="var_name", label_y="counts",  legend=None, col=None, legendfontsize=LEGEND_FONTSIZE, setlogx=False, setlogy=False, setscilabelx=False, setscilabely=False, figsize=FIGSIZE_BIG, tick_length=14, tick_width=1.5, **kwargs):
    plot.set_xlabel(f"{label_x}",fontsize = FONTSIZE)
    plot.set_ylabel(f"{label_y}", fontsize = FONTSIZE)
    plot.tick_params(axis='both', which="major",direction='in', length=tick_length, width=tick_width, labelsize=FONTSIZE)
    plot.tick_params(axis='both', which="minor",direction='in', length=tick_length/2.0, width=tick_width, labelsize=FONTSIZE)
    setplot_defaultstyle()
    figure.subplots_adjust(left= 0.12, right=0.96, bottom=0.1, top=0.95)
    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)

    xdata = get_bin_center(xbinning)
    ydata = counts
    if err is None:
        ydataerr = np.sqrt(counts)
    else:
        ydataerr = err
       
    plt.rcParams["font.weight"] = "bold"
    #ylim = [min(counts), max(counts) * 1.08]
    #plot.set_ylim(ylim)
    if col is not None:
        plot.errorbar(xdata, ydata, ydataerr, fmt=".", markersize=MARKER_SIZE, label=legend, color= col, **kwargs)
        plot.step(np.concatenate(([xbinning[0]], xbinning)), np.concatenate(([0], ydata, [0])), where="post", color=col, **kwargs)
    else:
        plot.errorbar(xdata, ydata, ydataerr, fmt=".", markersize=MARKER_SIZE, label=legend, **kwargs)
        plot.step(np.concatenate(([xbinning[0]], xbinning)), np.concatenate(([0], ydata, [0])), where="post", **kwargs)

    if legend is not None:
        plot.legend() 
    
    if setlogy:
        plot.set_yscale("log")
    if setscilabely:
        plot.ticklabel_format(axis='y', scilimits = (-3, 3), style="sci", useMathText=True)
    
    if setlogx:
        plot.set_xscale("log")
    if setscilabelx:
        plot.ticklabel_format(axis='x', scilimits = (-3, 3), style="sci", useMathText=True)


def plot1d_errorbar(figure, plot, binning_x, counts, err=None, label_x="var_name", label_y="counts",  legend=None, col=None, style=".", legendfontsize=FONTSIZE, setlogx=False, setlogy=False, setscilabelx=False, setscilabely=False, drawlegend=True, MARKER_SIZE=MARKER_SIZE,  tick_length=14, tick_width=1.5):
    plot.set_xlabel(f"{label_x}",fontsize = FONTSIZE_BIG)
    plot.set_ylabel(f"{label_y}", fontsize = FONTSIZE_BIG)
    plot.tick_params(axis='both', which="major",direction='in', length=tick_length, width=tick_width, labelsize=FONTSIZE_BIG)
    plot.tick_params(axis='both', which="minor",direction='in', length=tick_length/2.0, width=tick_width, labelsize=FONTSIZE_BIG)
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    figure.subplots_adjust(left= 0.12, right=0.96, bottom=0.1, top=0.95)
    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)

    xdata = get_bin_center(binning_x)
    ydata = counts
    if err is None:
        ydataerr = np.sqrt(counts)
    else:
        ydataerr = err
       

    ylim = [min(counts), max(counts) * 1.05]
#    plot.set_ylim(ylim)
    if col is not None:
        plot.errorbar(xdata, ydata, ydataerr, fmt=style, markersize=MARKER_SIZE, label=legend, color= col)
    else:
        plot.errorbar(xdata, ydata, ydataerr, fmt=style, markersize=MARKER_SIZE, label=legend)
    
    if setlogy:
        plot.set_yscale("log")
    if setscilabely:
        plot.ticklabel_format(axis='y', scilimits = (-3, 3), style="sci", useMathText=True)
    
    if setlogx:
        plot.set_xscale("log")
    if setscilabelx:
        plot.ticklabel_format(axis='x', scilimits = (-3, 3), style="sci", useMathText=True)

def plot1d_errorbar_v2(figure, plot, xdata, counts, err=None, label_x="var_name", label_y="counts",  style=".",  legendfontsize=None, setlogx=False, setlogy=False, setscilabelx=False, setscilabely=False, drawlegend=False,  tick_length=14, tick_width=1.5, **kwargs):
    plot.set_xlabel(f"{label_x}",fontsize = FONTSIZE)
    plot.set_ylabel(f"{label_y}", fontsize = FONTSIZE)
    plot.tick_params(axis='both', which="major",direction='in', length=tick_length, width=tick_width, labelsize=FONTSIZE)
    plot.tick_params(axis='both', which="minor",direction='in', length=tick_length/2.0, width=tick_width, labelsize=FONTSIZE)
    setplot_defaultstyle()
    figure.subplots_adjust(left= 0.12, right=0.96, bottom=0.1, top=0.95)
    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)

    ydata = counts
    if err is None:
        ydataerr = np.sqrt(counts)
    else:
        ydataerr = err
       
    plt.rcParams["font.weight"] = "bold"
    ylim = [min(counts), max(counts) * 1.05]

    plot.errorbar(xdata, ydata, ydataerr, fmt=style, **kwargs)

    if drawlegend:
        if legendfontsize:
            plot.legend(fontsize=legendfontsize)

    
    if setlogy:
        plot.set_yscale("log")
    if setscilabely:
        plot.ticklabel_format(axis='y', scilimits = (-3, 3), style="sci", useMathText=True)
    
    if setlogx:
        plot.set_xscale("log")
    if setscilabelx:
        plot.ticklabel_format(axis='x', scilimits = (-3, 3), style="sci", useMathText=True)


def plot1d_step(figure, plot, xbinning, counts, err=None, label_x="var_name", label_y="counts",  legend=None, col=None, legendfontsize=18, setlogx=False, setlogy=False, setscilabelx=False, setscilabely=False, figsize=(12, 9),  tick_length=14, tick_width=1.5, **kwargs):
    plot.set_xlabel(f"{label_x}",fontsize = FONTSIZE)
    plot.set_ylabel(f"{label_y}", fontsize = FONTSIZE)
    plot.tick_params(axis='both', which="major",direction='in', length=tick_length, width=tick_width, labelsize=FONTSIZE)
    plot.tick_params(axis='both', which="minor",direction='in', length=tick_length/2.0, width=tick_width, labelsize=FONTSIZE)
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.minorticks_on()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    for axis in ['top','bottom','left','right']:
        plot.spines[axis].set_linewidth(2)

    xdata = get_bin_center(xbinning)
    ydata = counts
    if err is None:
        ydataerr = np.sqrt(counts)
    else:
        ydataerr = err
       
    plt.rcParams["font.weight"] = "bold"
    ylim = [min(counts), max(counts) * 1.05]
#    plot.set_ylim(ylim)
    if col is not None:
        plot.step(np.concatenate(([xbinning[0]], xbinning)), np.concatenate(([0], ydata, [0])), where="post", color=col, label=legend, **kwargs)
    else:
        plot.step(np.concatenate(([xbinning[0]], xbinning)), np.concatenate(([0], ydata, [0])), where="post", label=legend, **kwargs)
    plot.legend()
    
    if setlogy:
        plot.set_yscale("log")
    if setscilabely:
        plot.ticklabel_format(axis='y', scilimits = (-3, 3), style="sci", useMathText=True)
    
    if setlogx:
        plot.set_xscale("log")
    if setscilabelx:
        plot.ticklabel_format(axis='x', scilimits = (-3, 3), style="sci", useMathText=True)

    
def scale_hist(values, values_target, binning):
    values = values / (np.sum(values * np.diff(binning))) * (np.sum(values_target * np.diff(binning)))
    return values


def savefig_tofile(figure, resultdir, figname, show=False):
    figure.savefig(os.path.join(resultdir, f"{figname}.png"))
    if show==False:
        plt.close()


def superscript(value):
    digits = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻", ".": "·"}
    s = str(value)
    return "".join((digits.get(c, c) for c in s))

def format_order_of_magnitude(value, digits=2, use_tex=False):
    if value == 0:
        return "0"
    elif abs(value) >= 1 and abs(value) < 1000:
        if value == int(value):
            return str(int(value))
        if abs(value) < 10:
            return f"{value:.2f}"
        if abs(value) < 100:
            return f"{value:.1f}"
        return f"{value:.0f}"
    elif abs(value) < 1 and abs(value) >= 0.1:
        return f"{value:.2f}"
    elif abs(value) < 0.1 and abs(value) >= 0.01:
        return f"{value:.3f}"
    sign = np.sign(value)
    value = value * sign
    magnitude = int(np.floor(np.log10(value)))
    mantisse = value / 10**magnitude
    sign_str = "" if sign > 0 else "-"
    if use_tex:
        return f"${sign_str}{mantisse:.{digits}f}\\times 10^{{{magnitude}}}$"
    return f"{sign_str}{mantisse:.{digits}f}×10{superscript(str(magnitude))}"


def flatten(d, parent_key='', sep='_'):  
    items = []
    for k, v in d.items():   
        new_key = parent_key + sep + k if parent_key else k  
        if isinstance(v, MutableMapping):        
            items.extend(flatten(v, new_key, sep=sep).items())    
        else:  
            items.append((new_key, v))  
    return dict(items) 


def plot_comparison_nphist(figure=None, ax1=None, ax2=None, x_binning=None, com=None, com_err=None, ref=None, ref_err=None, xlabel=None, ylabel=None, legendA=None, legendB=None, color=None):
    if figure == None: 
        figure, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[0.6, 0.4]}, figsize=(16, 14))
        
    plot1d_errorbar(figure, ax1, x_binning, com, err=com_err, label_x=xlabel, label_y=ylabel, col=color, legend=legendA)
    plot1d_step(figure, ax1,  x_binning, ref, err=ref_err, label_x=xlabel, label_y=ylabel, col=color, legend=legendB)
    pull = np.array(com/ref)

    #pull_err = ratioerr(pull, com, ref, com_err, ref_err)
    pull_err = np.zeros(len(pull))   
    plot1d_errorbar(figure, ax2, x_binning, counts=pull, err=pull_err,  label_x=xlabel, label_y=r"$\mathrm{this/ref}$", legend=None,  col=color, setlogx=False, setlogy=False, setscilabelx=False,  setscilabely=False)
    plt.subplots_adjust(hspace=.0)                             
    ax1.legend()                                         
    ax1.sharex(ax2)



########################################################
#Here are some constant names and definations for plots
########################################################

xaxistitle = {"Rigidity": "Rigidity (GV)", "Ekinn": "Ekin/n (GeV)"}
xaxis_binning = {"Rigidity": Binning(BeRigidityBinningRICHRange()), "Ekin": Binning(fbinning_energy())} 
