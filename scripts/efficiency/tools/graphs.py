#!/usr/bin/env python3

import numpy as np
import awkward as ak

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

from .binnings import Binning
from .statistics import lafferty_whyatt, row_mean_and_std, approximate_upper_poisson_error, approximate_lower_poisson_error, calculate_ratioerrs
from .utilities import plot_steps, shaded_steps, set_plot_lim, plot_2d, transform_overflow_edges
from .plottools import setplot_defaultstyle
from .statistics import poly_func
from .calculator import calc_ratio_err

def _np(array):
    if array is None:
        return None
    array = ak.to_numpy(array)
    if array.dtype == np.bool_:
        array = array.astype(np.uint8)
    return array


def rebin_indices(old_edges, new_edges):
    assert set(new_edges) <= set(old_edges) or (len(new_edges) == len(old_edges) and np.all(np.abs(new_edges[1:-1] - old_edges[1:-1]) < 1e-10))
    old_centers = (old_edges[1:] + old_edges[:-1]) / 2
    indices = np.digitize(old_centers, new_edges) - 1
    if old_edges[-1] == np.inf and indices[-1] == len(new_edges) - 1:
        indices[-1] -= 1
    return indices


class MGraph:
    def __init__(self, xvalues=None, yvalues=None, yerrs=None, binning=None, labels=None):
        
        if xvalues is None:
            if binning is not None:
                self.xvalues = binning.bin_centers[1:-1]
                self.yvalues = np.zeros_like(self.xvalues)
                self.yerrs = np.zeros_like(self.xvalues)
                self.binning = binning
        else:
            self.xvalues = np.array(xvalues)
            self.yvalues = np.array(yvalues)
            self.yerrs = np.array(yerrs)
            self.binning= binning
            assert(len(xvalues) == len(yvalues) == len(yerrs))
        self.labels = labels


    def __str__(self):
        # Customize the string representation of the object here
        result = "\n" +"xvalues\t\t\tyvalues\t\t\tyerrs\n"
        for x, y, yerr in zip(self.xvalues, self.yvalues, self.yerrs):
            result += str(x) + "\t\t\t" + str(y) + "\t\t\t" + str(yerr) + "\n"
        return result
    
    def getx(self):
        return self.xvalues

    def set_yvalue(self, y_values):
        self.yvalues = y_values

    def gety(self):
        return self.yvalues

    def get_yerrors(self):
        return self.yerrs

    def get_index(self, x_value):
        diff = np.abs(x_value - self.xvalues)
        index = np.argmin(diff)
        return index

    def get_points(self, x_values):
        diff = np.abs(x_value - self.xvalues)
        
    def add(self, other):
        assert np.array_equal(np.array(self.xvalues), np.array(other.xvalues))
        self.yvalues += other.yvalues
        self.yerrs += other.yerrs

    def truediv(self, other):
        assert np.array_equal(np.array(self.xvalues), np.array(other.xvalues))
        self.yvalues = self.yvalues/other.yvalues
        self.yerrs = self.yvalues/other.yvalues * np.sqrt(1/self.yvalues**2  * self.yerrs**2 + 1./other.yvalues**2 * other.yerrs**2)
        
        
    def __truediv__(self, other):
        return MGraph(xvalues=self.xvalues, yvalues=self.yvalues/other.yvalues, yerrs=calculate_ratioerrs(self.yvalues, other.yvalues, self.yerrs, other.yerrs), labels=self.labels) 
    
    def __iadd__(self, other):
        self.add(other)
        return self

    def __add__(self, other):
        return MGraph(xvalues=self.xvalues, yvalues=self.yvalues + other.yvalues, yerrs=self.yerrs+other.yerrs, labels=self.labels)

    def __eq__(self, other):
        return np.all(self.xvalues == other.xvalues) and np.all(self.yvalues == other.yvalues)  and nop.all(self.yerrs == other.yerrs)

    def add_to_file(self, file_dict, name):
        file_dict[f"{name}_xvalues"] = self.xvalues
        file_dict[f"{name}_yvalues"] = self.yvalues
        file_dict[f"{name}_yerrs"] = self.yerrs
        if self.labels is not None:
            file_dict[f"{name}_labels"] = True
            for index, label in enumerate(self.labels):
                file_dict[f"{name}_label_{index}"] = label
        else:
            file_dict[f"{name}_labels"] = False

    
    @staticmethod
    def from_file(file_dict, name):
        xvalues = file_dict[f"{name}_xvalues"]
        yvalues = file_dict[f"{name}_yvalues"]
        yerrs = file_dict[f"{name}_yerrs"]
        labels = None
        if f"{name}_labels" in file_dict and file_dict[f"{name}_labels"]:
            labels = [file_dict[f"{name}_label_{index}"].item() for index in range(2)]
        return MGraph(xvalues, yvalues, yerrs, labels)



def slice_bin_graph(agraph, slice_bin_indice):
    xslice = agraph.xvalues[slice_bin_indices[0] -1: slice_bin_indices[1] -1]
    yslice = agraph.yvalues[slice_bin_indices[0] -1: slice_bin_indices[1] -1]
    y_err = agraph.get_yerrors()[slice_bin_indices[0] -1: slice_bin_indices[1] -1]
    return MGraph(xslice, yslice, y_err, agraph.labels)

def slice_graph(agraph, ipoint, jpoint=None):
    if jpoint is None:
         xslice = agraph.getx()[ipoint:]
         yslice = agraph.gety()[ipoint:]
         y_err =  agraph.get_yerrors()[ipoint:]
    else:
        xslice = agraph.getx()[ipoint:jpoint+1]
        yslice = agraph.gety()[ipoint:jpoint+1]
        y_err =  agraph.get_yerrors()[ipoint:jpoint+1]
    return MGraph(xslice, yslice, y_err, agraph.labels)


def slice_graph_by_value(agraph, lim):
    lowlim = lim[0]
    uplim = lim[1]
    ipoint = agraph.get_index(lowlim)
    jpoint = agraph.get_index(uplim)  
    xslice = agraph.getx()[ipoint:jpoint+1]
    yslice = agraph.gety()[ipoint:jpoint+1]
    y_err =  agraph.get_yerrors()[ipoint:jpoint+1]
    return MGraph(xslice, yslice, y_err, agraph.labels)



def compute_pull_graphs(graph_a, graph_b, slice_range=None, is_ratio=True, show_error=True, allow_diff=0.01):
    if graph_a.xvalues == graph_b.xvalues:
        assert (len(graph_a.yvalues) == len(graph_b.yvalues))
        pull = (graph_a.yvalues - graph_b.yvalues)/graph_a.yvalues
        error = calc_ratio_err(graph_b.yvalues, graph_a.yvalues , graph_b.yerrs, graph_a.yerrs)
        return MGraph(graph_a.xvalues, pull, error)
    else:
        if graph_a.binning is not None and graph_b.binning is not None:
            slice_bin_a = graph_a.binning.get_indices(slice_range)
            slice_bin_b = graph_b.binning.get_indices(slice_range)
            slice_graph_a = slice_bin_graph(graph_a, slice_bin_a)
            slice_graph_b = slice_bin_graph(graph_b, slice_bin_b)
        else:
            slice_graph_a = slice_graph(graph_a, graph_a.get_index(slice_range[0]), graph_a.get_index(slice_range[1]))
            slice_graph_b = slice_graph(graph_b, graph_b.get_index(slice_range[0]), graph_b.get_index(slice_range[1]))
            assert(len(slice_graph_a.xvalues) == len(slice_graph_b.xvalues))
            x_diff = np.abs(slice_graph_a.xvalues - slice_graph_b.xvalues)
            check_ok = x_diff < allow_diff
            if (np.any(~check_ok)):
                print("Warning: The difference of the xvalues in the result graphs is too large, please check your graph consistency")
        if is_ratio:
            pull = slice_graph_a.yvalues/slice_graph_b.yvalues
            error = calc_ratio_err(slice_graph_a.yvalues, slice_graph_b.yvalues , slice_graph_a.yerrs, slice_graph_b.yerrs)
        else:
            pull = (slice_graph_a.yvalues - slice_graph_b.yvalues)/ slice_graph_a.yvalues
            error = calc_ratio_err(slice_graph_b.yvalues, slice_graph_a.yvalues , slice_graph_b.yerrs, slice_graph_a.yerrs)
        if show_error:
            result_graph = MGraph(slice_graph_a.xvalues, pull, error)
        else:
            result_graph = MGraph(slice_graph_a.xvalues, pull, np.zeros_like(slice_graph_a.xvalues))
        return result_graph


def scale_graph(graph, scaler):
    if graph.yvalues is not None:
        new_yvalues = graph.yvalues * scaler
        new_yerrs =  graph.yerrs * scaler  
        return MGraph(xvalues=graph.xvalues, yvalues=new_yvalues, yerrs=new_yerrs, labels=graph.labels)
    else:
        print("The graph is empty")



def concatenate_graphs(agraph, bgraph):
    xpoints = np.concatenate((agraph.getx(), bgraph.getx()))
    ypoints = np.concatenate((agraph.gety(), bgraph.gety()))
    y_errs = np.concatenate((agraph.get_yerrors(), bgraph.get_yerrors()))
    return MGraph(xpoints, ypoints, y_errs, agraph.labels)

#fit graph with poly function
def get_nppolyfit(datagraph, deg):
    coeffs1 = np.polyfit(datagraph.getx(), datagraph.gety(), deg=deg)
    fit1 = np.poly1d(coeffs1)
    return fit1
def getpars_curvefit_poly(datagraph, deg):
    initial_guess = np.zeros(deg) # Initial guess for the polynomial coefficients
    fit_coeffs, _ = curve_fit(poly_func, datagraph.getx()[3:20], datagraph.gety()[3:20], sigma=datagraph.get_yerrors()[3:20], p0=initial_guess)
    return fit_coeffs

def plot_curvefit(figure, plot, graph, func, pars, col=None, label=None):
    plot.plot(graph.getx()[:], func(graph.getx()[:], *pars), '-', label=label, color=col)
    
def plot_graph(figure, plot, graph, color=None, label=None, style="EP", xlog=False, ylog=False, scale=None, shade_errors=False, adjust_limits=None, adjust_limits_x=None, adjust_limits_y=None, flip_axes=False, override_limits=False, draw_zeros=True, adjust_figure=False,  setscilabelx=False, setscilabely=False, tick_length=14, tick_width=1.5, tick_labelsize = 35 , **kwargs):     
    setplot_defaultstyle()
    plot.tick_params(axis='both', which="major",direction='in', length=tick_length, width=tick_width, labelsize=tick_labelsize)
    plot.tick_params(axis='both', which="minor",direction='in', length=tick_length/2.0, width=tick_width, labelsize=tick_labelsize)
    
    values_x = graph.xvalues
    values_y = graph.yvalues
    errors_y_low = graph.yerrs
    errors_y_high = graph.yerrs

    if scale is not None:
        values_y = values_y * scale
        errors_y_low = errors_y_low * scale
        errors_y_high = errors_y_high * scale

    if xlog:            
        if flip_axes:       
            plot.set_yscale("log")        
        else:       
            plot.set_xscale("log") 
    if ylog:     
        if flip_axes:   
            plot.set_xscale("log") 
        else:   
            plot.set_yscale("log")        


    if setscilabelx:
        if flip_axes:
            plot.ticklabel_format(axis='y', scilimits = (-3, 3), style="sci", useMathText=True)
        else:
            plot.ticklabel_format(axis='x', scilimits = (-3, 3), style="sci", useMathText=True)

    if setscilabely:
        if flip_axes:
            plot.ticklabel_format(axis='x', scilimits = (-3, 3), style="sci", useMathText=True)
        else:
            plot.ticklabel_format(axis='y', scilimits = (-3, 3), style="sci", useMathText=True)

    

    if graph.labels is not None:
        if flip_axes:
            plot.set_xlabel(graph.labels[1])
            plot.set_ylabel(graph.labels[0])
        else:
            plot.set_xlabel(graph.labels[0])
            plot.set_ylabel(graph.labels[1])

    result = []
    if style == "EP":
        if not draw_zeros:
            nonzero = values_y > 0
            values_x = values_x[nonzero]
            values_y = values_y[nonzero]
            errors_y_low = errors_y_low[nonzero]
            errors_y_high = errors_y_high[nonzero]
        result = plot.errorbar(values_x, values_y, (errors_y_low, errors_y_high), fmt='.', color=color, label=label, **kwargs)
    elif style == "hist":
        plot.plot(values_x, values_y, "-", color=color, label=label, **kwargs)
        if flip_axes:
            for line in result:
                xdata, ydata = line.get_data()
                line.set_data(ydata, xdata)
        if shade_errors:
            shaded_steps(plot, bin_edges, values_y - errors_y_low, values_y + errors_y_high, color=color, alpha=0.5)
    else:
        raise ValueError(f"Unknown draw style {style}")
    if adjust_limits is not None:
        if adjust_limits_x is None:
            adjust_limits_x = adjust_limits
        if adjust_limits_y is None:
            adjust_limits_y = adjust_limits
    else:
        if adjust_limits_x is None:
            adjust_limits_x = False
        if adjust_limits_y is None:
            adjust_limits_y =True 
    if adjust_limits:
        if adjust_limits_y:
            axis = "x" if flip_axes else "y"
            set_plot_lim(plot, values_y + errors_y_high, log=ylog, axis=axis, override=override_limits)
            set_plot_lim(plot, values_y - errors_y_low, log=ylog, axis=axis, override=override_limits)
        if adjust_limits_x:
            axis = "y" if flip_axes else "x"
            set_plot_lim(plot, bin_edges[0], log=xlog, axis=axis, override=override_limits)
            set_plot_lim(plot, bin_edges[-1], log=xlog, axis=axis, override=override_limits)

    return result

'''
def plot_histogram_2d(plot, histogram, scale=None, transpose=False, show_overflow=True, show_overflow_x=None, show_overflow_y=None, label=None, **kwargs):
    assert histogram.dimensions == 2
    values = histogram.values
    if show_overflow_x is None:
        show_overflow_x = show_overflow
    if show_overflow_y is None:
        show_overflow_y = show_overflow
    if not show_overflow_x:
        values = values[1:-1,:]
    if not show_overflow_y:
        values = values[:,1:-1]
    if transpose:
        values = values.transpose()
    if scale is not None:
        if isinstance(scale, np.ndarray) and not show_overflow_x:
            scale = scale[1:-1]
    if histogram.labels is not None:
        label_x, label_y = histogram.labels
        if transpose:
            label_x, label_y = label_y, label_x
        plot.set_xlabel(label_x)
        plot.set_ylabel(label_y)
    index_x, index_y = 0, 1
    if transpose:
        index_x, index_y = index_y, index_x
    if histogram.binnings[index_x].log:
        plot.set_xscale("log")
    if histogram.binnings[index_y].log:
        plot.set_yscale("log")
    edges_x = histogram.binnings[index_x].edges
    edges_y = histogram.binnings[index_y].edges
    if show_overflow_x:
        edges_x = transform_overflow_edges(edges_x)
    else:
        edges_x = edges_x[1:-1]
    if show_overflow_y:
        edges_y = transform_overflow_edges(edges_y)
    else:
        edges_y = edges_y[1:-1]
    plot_2d(plot, values, edges_x, edges_y, scale=scale, colorbar_label=label, **kwargs)

'''
