
from collections import defaultdict
import os
import pickle
import re

import numpy as np
from numpy.lib import recfunctions
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import curve_fit
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap
from iminuit import Minuit
from iminuit.cost import LeastSquares

from tools.constants import MC_PARTICLE_IDS, MC_PARTICLE_CHARGES, MC_PARTICLE_MASSES, MEASURING_TIME_CORRECTION, RICH_RADIATOR_BETA, RICH_RADIATOR_RESOLUTION
from tools.conversions import calc_rig, calc_beta
from tools.statistics import fermi_function, inverse_fermi_function






def load_flux(path):
    flux_data = np.genfromtxt(path)
    rigidity = flux_data[:,0]
    flux = flux_data[:,3]
    flux_err_low = flux_data[:,4]
    flux_err_high = flux_data[:,5]
    title = None
    with open(path) as flux_file:
        for line in flux_file:
            if not line.startswith("#"):
                break
            if line.startswith("# GraphTitle:"):
                title = line.split(":", 1)[1].split(";")[0].strip()
    array = np.core.records.fromarrays((rigidity, flux, flux_err_low, flux_err_high, flux, flux_err_low, flux_err_high), names=("rigidity", "flux", "flux_error_low", "flux_error_high", "raw_flux", "raw_flux_error_low", "raw_flux_error_high"))
    return array, title


def power_law(rigidity, phi, c, rigidity_scale, gamma):
    effective_rigidity = rigidity + phi
    return (rigidity / effective_rigidity)**2 * (c * (effective_rigidity / rigidity_scale)**gamma)


def fit_flux(flux_array, flux_title, create_plot=False):
    flux_error = (flux_array.flux_error_low + flux_array.flux_error_high) / 2
    guess = dict(phi=1, c=10, rigidity_scale=10, gamma=-2.7)
    loss = LeastSquares(flux_array.rigidity, flux_array.flux, flux_error, model=power_law)
    m = Minuit(loss, **guess)
    m.migrad()

    fit_param_dict = dict(zip(m.parameters, m.values))

    def _parametrized_power_law(rigidity):
        return power_law(rigidity, **fit_param_dict)

    if create_plot:
        dense_rigidity = np.logspace(np.log10(flux_array.rigidity[0]), np.log10(flux_array.rigidity[-1]), 100)
        flux_figure = plt.figure(figsize=(12, 6.15))
        flux_figure.suptitle(flux_title)
        flux_plot = flux_figure.subplots(1, 1)
        flux_plot.errorbar(flux_array.rigidity, flux_array.flux, yerr=np.stack((flux_array.flux_error_low, flux_array.flux_error_high)), fmt=".", label="Flux")
        flux_plot.plot(dense_rigidity, power_law(dense_rigidity, **guess), "-", label="Guess")
        flux_plot.plot(dense_rigidity, power_law(dense_rigidity, **fit_param_dict), "-", label="Fit")
        flux_plot.set_xscale("log")
        flux_plot.set_yscale("log")
        flux_plot.set_xlabel("Rigidity / GV")
        flux_plot.set_ylabel("Flux")
        flux_plot.legend()
        flux_figure.savefig("flux-fit.png", dpi=250)
        plt.close(flux_figure)

    return _parametrized_power_law, fit_param_dict


def load_mc_trigger_density(mc_triggers_filename):
    with open(mc_triggers_filename) as mc_triggers_file:
        components = []
        for line in mc_triggers_file:
            rig_min, rig_max, triggers = map(float, line.split(" "))
            components.append((abs(rig_min), abs(rig_max), triggers))

    def _trigger_density(rigidity):
        cumulative_density = 0
        return np.sum([
            ((rig_min <= rigidity) & (rig_max >= rigidity)) * triggers / (np.log(rig_max) - np.log(rig_min)) / rigidity
            for rig_min, rig_max, triggers in components
        ], axis=0)

    return _trigger_density


def load_mc_trigger_count(mc_triggers_filename, binning):
    from tools.histograms import Histogram
    mc_triggers_hist = Histogram(binning)
    with open(mc_triggers_filename) as mc_triggers_file:
        for line in mc_triggers_file:
            rig_min, rig_max, triggers = map(float, line.split(" "))
            rig_min, rig_max = map(abs, (rig_min, rig_max))
            min_index = np.digitize(rig_min, binning.edges) - 1
            max_index = np.digitize(rig_max, binning.edges)
            rig_in_range = binning.edges[min_index:max_index + 1]
            for index, (bin_min, bin_max) in enumerate(zip(rig_in_range[:-1], rig_in_range[1:])):
                fraction = (np.log10(min(bin_max, rig_max)) - np.log10(max(bin_min, rig_min))) / (np.log10(rig_max) - np.log10(rig_min))
                mc_triggers_hist.values[min_index + index] += fraction * triggers
    return mc_triggers_hist


def load_weighted_mc_trigger_count(mc_triggers_filename, binning, mc_weighting):
    from tools.histograms import Histogram
    from tools.binnings import Binning, increase_bin_number
    fine_binning = increase_bin_number(binning, 10)
    mc_triggers_hist = Histogram(fine_binning)
    with open(mc_triggers_filename) as mc_triggers_file:
        for line in mc_triggers_file:
            rig_min, rig_max, triggers = map(float, line.split(" "))
            rig_min, rig_max = map(abs, (rig_min, rig_max))
            min_index = np.digitize(rig_min, fine_binning.edges) - 1
            max_index = np.digitize(rig_max, fine_binning.edges)
            rig_in_range = fine_binning.edges[min_index:max_index + 1]
            for index, (bin_min, bin_max) in enumerate(zip(rig_in_range[:-1], rig_in_range[1:])):
                eff_min = max(bin_min, rig_min)
                eff_max = min(bin_max, rig_max)
                fraction = (np.log10(eff_max) - np.log10(eff_min)) / (np.log10(rig_max) - np.log10(rig_min))
                weight = mc_weighting.get_weights(None, (eff_min + eff_max) / 2)
                mc_triggers_hist.values[min_index + index] += fraction * triggers * weight
    return mc_triggers_hist.rebin(binning)



def interpolate_measuring_time(rig_values, mt_values, max_f_value=10, apply_correction=False):
    max_value = mt_values[-1]
    rel_values = mt_values / (max_value + 1e-7)
    max_rig = rig_values[-1]
    min_rig = rig_values[0]
    inv_rel_values = inverse_fermi_function(rel_values)
    inv_rel_values[inv_rel_values > max_f_value] = max_f_value
    inv_rel_values[inv_rel_values < -max_f_value] = -max_f_value
    for index in range(1, len(inv_rel_values)):
        if inv_rel_values[index] < inv_rel_values[index - 1]:
            print("Warning: T({index}) = {rel_values[index]} < T({index-1}) = {rel_values[index-1]}")
            inv_rel_values[index] = inv_rel_values[index - 1]
    #spline = UnivariateSpline(np.log(rig_values), inv_rel_values, ext="extrapolate", s=5)
    sel = np.isfinite(inv_rel_values)
    spline = PchipInterpolator(np.log(rig_values[sel]), inv_rel_values[sel], extrapolate=True)
    #spline = interp1d(np.log(rig_values)[sel], inv_rel_values[sel], kind="cubic", bounds_error=False, fill_value=(-1e7, 1e7))
    corr_rig, corr_factor = MEASURING_TIME_CORRECTION
    corr_func = interp1d(corr_rig, corr_factor, kind="linear", bounds_error=False, fill_value=(1, 1))
    
    def _interpolate_measuring_time(rigidity):
        norm_y_values = spline(np.log(rigidity))
        y_values = fermi_function(norm_y_values)
        y_values[rigidity < min_rig] = 0
        y_values[rigidity >= max_rig] = 1
        corr = np.ones_like(rigidity)
        if apply_correction:
            corr = corr_func(rigidity)
        return y_values * max_value * corr
    return _interpolate_measuring_time
 


def plot_steps(plot, edges, values, **kwargs):
    x = np.concatenate(([edges[0]], edges))
    y = np.concatenate(([0], values, [0]))
    return plot.step(x, y, where="post", **kwargs)

def shaded_steps(plot, edges, values_low=None, values_high=None, values=None, errors=None, **kwargs):
    if values is not None and errors is not None:
        values_low = values - errors
        values_high = values + errors
    if values_low is None and values_high is None:
        raise ValueError("Either values_low and values_high or values and errors is required!")
    x = np.concatenate(([edges[0]], edges))
    y_low = np.concatenate(([0], values_low, [0]))
    y_high = np.concatenate(([0], values_high, [0]))
    return plot.fill_between(x, y_low, y_high, step="post", **kwargs)


def round_up(value, digits=1, log=False):
    if log:
        return 10**(round_up(np.log10(value), digits=digits))
    if value == 0:
        return 0
    if value < 0:
        return -round_down(-value)
    factor = 10**digits
    return np.ceil(value * factor) / factor

def round_down(value, digits=1, log=False):
    if log:
        return 10**(round_down(np.log10(value), digits=digits))
    if value == 0:
        return 0
    if value < 0:
        return -round_up(-value)
    factor = 10**digits
    return np.floor(value * factor) / factor


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
        return f"\\({sign_str}{mantisse:.{digits}f}\\times 10^{{{magnitude}}}\\)"
    return f"{sign_str}{mantisse:.{digits}f}×10{superscript(str(magnitude))}"


def format_human(value):
    order_of_magnitude = int(np.log10(value))
    prefix = np.round(value / 10**order_of_magnitude)
    rounded_value = prefix * 10**order_of_magnitude
    if order_of_magnitude < 3:
        return f"{rounded_value:.0f}"
    if order_of_magnitude < 6:
        return f"{rounded_value / int(1e3):.0f} Thousand"
    if order_of_magnitude < 9:
        return f"{rounded_value / int(1e6):.0f} Million"
    if order_of_magnitude < 12:
        return f"{rounded_value / int(1e9):.0f} Billion"
    return f"{rounded_value:.0g}"


def set_plot_lim_x(plot, values, log=False, override=False):
    xmin_old, xmax_old = plot.get_xlim()
    xmax_new = xmax_old
    if np.any(np.isfinite(values)):
        xmax_new = round_up(np.max(values[np.isfinite(values)]))
    if override:
        xmax = xmax_new
    else:
        xmax = max(xmax_old, xmax_new)
    xmin_new = 0
    if log:
        xmin_new = xmin_old
        if np.any(np.isfinite(values) & (values > 0)):
            xmin_new = round_down(np.min(values[np.isfinite(values) & (values > 0)]))
    if override:
        xmin = xmin_new
    else:
        xmin = min(xmin_old, xmin_new)
    if not log and xmin < 0 and xmin_new == 0:
        xmin = 0
    plot.set_xlim(left=xmin, right=xmax)

def set_plot_lim_y(plot, values, log=False, override=False):
    ymin_old, ymax_old = plot.get_ylim()
    ymax_new = ymax_old
    if np.any(np.isfinite(values)):
        ymax_new = round_up(np.max(values[np.isfinite(values)]))
    if override:
        ymax = ymax_new
    else:
        ymax = max(ymax_old, ymax_new)
    ymin_new = 0
    if log:
        ymin_new = ymin_old
        if np.any(np.isfinite(values) & (values > 0)):
            ymin_new = round_down(np.min(values[np.isfinite(values) & (values > 0)]))
    if override:
        ymin = ymin_new
    else:
        ymin = min(ymin_old, ymin_new)
    if not log and ymin < 0 and ymin_new == 0:
        ymin = 0
    plot.set_ylim(bottom=ymin, top=ymax)

def set_plot_lim(plot, values, log=False, axis="y", override=False):
    if axis == "x":
        set_plot_lim_x(plot, values, log, override=override)
    elif axis == "y":
        set_plot_lim_y(plot, values, log, override=override)


def set_rigidity_ticks(plot, axis="x"):
    axis = plot.xaxis if axis == "x" else plot.yaxis
    xmin, xmax = axis.get_view_interval()
    xticks = [t for t in (0.1, 1, 10, 100, 1000, 10000, 100000) if xmin <= t <= xmax]
    labels = [str(t) for t in xticks]
    axis.set_ticks(xticks)
    axis.set_ticklabels(labels)
    def _label(tick):
        if tick / 10**(int(np.log10(tick))) > 5:
            return ""
        if (tick - int(tick)) / tick < 1e-10:
            return str(int(tick))
        if tick < 1:
            return f"{tick:.1f}"
        return str(tick)
    if len(xticks) <= 2:
        minor_xticks = [t * f for t in (1, 10, 100, 1000) for f in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) if xmin <= t * f <= xmax]
        minor_labels = [_label(tick) for tick in minor_xticks]
        axis.set_ticks(minor_xticks, minor=True)
        axis.set_ticklabels(minor_labels, minor=True)


def plot_2d(plot, values, edges_x, edges_y, min_value=None, max_value=None, cmap=None, log=False, colorbar=True, colorbar_ax=None, colorbar_width=0.05, colorbar_label=None, scale=None, mask_zeros=True, **kwargs):
    if mask_zeros:
        values_z = np.ma.masked_where(values == 0, values).transpose()
    else:
        values_z = values.transpose()
    if scale is not None:
        values_z = values_z * scale
    if min_value is not None and max_value is not None and max_value < min_value:
        max_value = min_value
    norm = LogNorm(min_value, max_value) if log else Normalize(min_value, max_value)
    mesh = plot.pcolormesh(edges_x, edges_y, values_z, cmap=cmap, norm=norm, **kwargs)
    if colorbar:
        plt.colorbar(mesh, ax=plot, cax=colorbar_ax, fraction=colorbar_width, label=colorbar_label)


def hist_most_probable_value(histogram, axis=1, with_overflow=False):
    values = histogram.values
    centers = histogram.binnings[axis].bin_centers
    offset = 0
    if not with_overflow:
        values = values[tuple([slice(1, -1) for axis in range(histogram.dimensions)])]
        offset = 1
        centers = centers[1:-1]
    indices = np.argmax(values, axis=axis) + offset
    return centers[indices - 1]


def transform_overflow_edges(edges):
    underflow = 2 * edges[1] - edges[2]
    overflow = 2 * edges[-2] - edges[-3]
    return np.array([underflow] + list(edges[1:-1]) + [overflow])


def make_colormap_passed():
    red = ((0.0, 0.6, 0.6), (1.0, 0.0, 0.0))
    green = ((0.0, 1.0, 1.0), (1.0, 0.6, 0.6))
    blue = ((0.0, 0.6, 0.6), (1.0, 0.0, 0.0))
    return LinearSegmentedColormap("passed", {"red": red, "green": green, "blue": blue})

def make_colormap_failed():
    red = ((0.0, 1.0, 1.0), (1.0, 0.6, 0.6))
    green = ((0.0, 0.6, 0.6), (1.0, 0.0, 0.0))
    blue = ((0.0, 0.6, 0.6), (1.0, 0.0, 0.0))
    return LinearSegmentedColormap("failed", {"red": red, "green": green, "blue": blue})

def make_downwards_colormap_rgb(name, weight_red, weight_green, weight_blue, base=0.6):
    r0 = base + (1 - base) * weight_red
    r1 = base * weight_red
    g0 = base + (1 - base) * weight_green
    g1 = base * weight_green
    b0 = base + (1 - base) * weight_blue
    b1 = base * weight_blue
    red = ((0.0, r0, r0), (1.0, r1, r1))
    green = ((0.0, g0, g0), (1.0, g1, g1))
    blue = ((0.0, b0, b0), (1.0, b1, b1))
    return LinearSegmentedColormap(name, dict(red=red, green=green, blue=blue))


PARTICLE_COLORMAPS = {
    "Default": make_downwards_colormap_rgb("Default", 0.5, 0.5, 0.5),
    "Positron": make_downwards_colormap_rgb("Positron", 1, 0.1, 0),
    "Electron": make_downwards_colormap_rgb("Electron", 1, 0.2, 0),
    "Proton": make_downwards_colormap_rgb("Proton", 0, 0, 1),
    "Antiproton": make_downwards_colormap_rgb("Antiproton", 0, 0.1, 1),
    "Alpha": make_downwards_colormap_rgb("He4", 0, 1, 0),
    "He3": make_downwards_colormap_rgb("He3", 0.25, 1, 0),
    "Li6": make_downwards_colormap_rgb("Li6", 1, 0.75, 0),
    "Li7": make_downwards_colormap_rgb("Li7", 0.75, 1, 0),
    "C12": make_downwards_colormap_rgb("C12", 0, 1, 0.8),
    "N14": make_downwards_colormap_rgb("N14", 0, 0.8, 1),
}


def translate_parameter_name(name):
    return name.replace("mu", "\\mu").replace("sigma", "\\sigma").replace("delta", "\\Delta").replace("tau", "\\tau")


def rec_to_float(recarray):
    result = np.zeros((len(recarray), (len(recarray.dtype.names))), dtype=np.float32)
    for index, column in enumerate(recarray.dtype.names):
        result[:,index] = recarray[column]
    return result


def float_or_int(value):
    if int(value) == value:
        return int(value)
    return float(value)


def filter_branches(array, branches):
    if isinstance(array, ak.Array):
        return np.core.records.fromarrays([ak.to_numpy(array[branch]) for branch in branches], names=branches)
    return recfunctions.require_fields(array, [(name, array.dtype[name]) for name in branches])


def title_to_name(title):
    return re.sub("_+", "_", "".join(c if c.isalnum() else "_" for c in title))


BR_LENGTH = 27 * 86400
BR_2394 = 1230768000

def start_time_of_bartels_rotation(br_number):
    return BR_2394 + BR_LENGTH * (br_number - 2394)


def sort_by_dependencies(elements, dependencies):
    result = []
    done = set()
    remaining = set(elements)
    while remaining:
        size = len(remaining)
        to_remove = []
        for element in remaining:
            if all(dep in done or dep not in elements for dep in dependencies[element]):
                result.append(element)
                done.add(element)
        remaining = remaining - done
        if len(remaining) == size:
            raise ValueError(f"Cannot resolve order of remaining elements: {remaining!r}")
    return result


def resolve_derived_branches(required_branches, dependencies, derivation_functions):
    derived_branches = set()
    while True:
        required_after, newly_derived = resolve_derived_branches_step(required_branches, dependencies, derivation_functions)
        derived_branches = derived_branches | newly_derived
        if required_after == required_branches:
            return required_branches, sort_by_dependencies(derived_branches, dependencies)
        required_branches = required_after


def resolve_derived_branches_step(required_branches, dependencies, derivation_functions):
    required_branches = set(required_branches)
    derived_branches = {branch for branch in required_branches if branch in derivation_functions}
    dependency_branches = {dependency for branch in derived_branches for dependency in dependencies[branch]}
    return (required_branches - derived_branches) | dependency_branches, derived_branches


def recursive_dependencies(variable, dependencies):
    vars = dependencies.get(variable, [])
    dependency_vars = []
    for var in vars:
        dependency_vars.extend(recursive_dependencies(var, dependencies))
    return set(vars) | set(dependency_vars)


def decompose_graph(nodes, edges):
    nodes_in_graph = set()
    edges_in_graph = set()
    nodes_to_check = []
    while nodes:
        node = nodes[0]
        nodes_in_graph.add(node)
        nodes_to_check.append(node)
        while nodes_to_check:
            node = nodes_to_check.pop(0)
            for edge in edges:
                if edge[0] == node:
                    other = edge[1]
                elif edge[1] == node:
                    other = edge[0]
                else:
                    continue
                edges_in_graph.add(edge)
                if other not in nodes_in_graph:
                    nodes_in_graph.add(other)
                    nodes_to_check.append(other)
        yield nodes_in_graph, edges_in_graph
        nodes = [node for node in nodes if node not in nodes_in_graph]
        edges = [edge for edge in edges if edge not in edges_in_graph]
        nodes_in_graph = set()
        edges_in_graph = set()


class Palette:
    def __init__(self, colors, index=None):
        self.colors = colors
        if index is None:
            index = 0
        self.index = index

    def get_color(self):
        color = self.colors[self.index % len(self.colors)]
        self.index += 1
        return color

    def reset(self):
        self.index = 0

def make_tab_palette():
    return Palette(("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"))


def check_binning_compatibility(points_1, points_2, precision=1e-7):
    low_end = max(points_1[0], points_2[0])
    high_end = min(points_1[-1], points_2[-1])
    points_1_in = points_1[(points_1 >= low_end) & (points_1 <= high_end)]
    points_2_in = points_2[(points_2 >= low_end) & (points_2 <= high_end)]
    if len(points_1_in) != len(points_2_in):
        return False
    if np.any(np.abs(points_1 - points_2) > precision): 
        return False
    return True

def get_compatible_binpoints(points_1, points_2, precision=1e-7):
    low_end = max(points_1[0], points_2[0])
    high_end = min(points_1[-1], points_2[-1])
    sel_1 = (points_1 >= low_end) & (points_1 <= high_end)
    sel_2 = (points_2 >= low_end) & (points_2 <= high_end)
    assert np.all(np.abs(points_1[sel_1] - points_2[sel_2]) < precision)
    return points_1[sel_1], sel_1, sel_2


def local_power_law(x, a, gamma):
    return a * x**gamma

def interpolate_power_law(x, y, y_error, target_rigs, window_size=3):
    result = []
    for rig in target_rigs:
        closest = np.argmin(np.abs(x - rig))
        first_bin = max(0, closest - window_size)
        last_bin = min(len(x), closest + window_size)
        fit_params, fit_param_cov = curve_fit(local_power_law, x[first_bin:last_bin], y[first_bin:last_bin], sigma=y_error[first_bin:last_bin], absolute_sigma=True)
        result.append(local_power_law(rig, *fit_params))
    return np.array(result)
    

def compare_fluxes(fluxes, plotdir, prefix, title, rigidity_estimator, max_diff=0.15):
    assert len(fluxes) > 1
    ref_flux = fluxes[0]
    comp_fluxes = fluxes[1:]

    ref_flux_rigidities, ref_flux_values, ref_flux_errors, ref_flux_label = ref_flux
    
    figure = plt.figure(figsize=(12, 6.15))
    figure.suptitle(title)
    plot = figure.subplots(1, 1)
    plot.set_xlabel(f"{rigidity_estimator} / GV")
    plot.set_ylabel(f"Ratio to {ref_flux_label}")
    plot.set_ylim(1 - max_diff, 1 + max_diff)
    plot.set_xscale("log")
    plot.plot(ref_flux_rigidities, np.ones_like(ref_flux_rigidities), "-", color="gray")
    for comp_flux_rigidities, comp_flux_values, comp_flux_errors, comp_flux_label in comp_fluxes:
        if check_binning_compatibility(ref_flux_rigidities, comp_flux_rigidities):
            rig_points, ref_sel, comp_sel = get_compatible_binpoints(ref_flux_rigidities, comp_flux_rigidities)
            ratio = comp_flux_values[comp_sel] / ref_flux_values[ref_sel]
            ratio_error = ratio * np.sqrt((comp_flux_errors[comp_sel] / comp_flux_values[comp_sel])**2 + (ref_flux_errors[ref_sel] / ref_flux_values[ref_sel])**2)
            plot.errorbar(rig_points, ratio, ratio_error, fmt=".", label=f"{comp_flux_label}") 
        else:
            interpolated_flux_values = interpolate_power_law(ref_flux_rigidities, ref_flux_values, ref_flux_errors, comp_flux_rigidities)
            ratio = comp_flux_values / interpolated_flux_values
            ratio_error = ratio * np.sqrt((comp_flux_errors / comp_flux_values)**2)
            plot.errorbar(comp_flux_rigidities, ratio, ratio_error, fmt=".", label=f"{comp_flux_label}")
    plot.legend()
    set_rigidity_ticks(plot)
    save_figure(figure, plotdir, f"{prefix}_comparison")


def rigidity_from_radiator_and_particle(radiator_name, particle_name):
    beta = RICH_RADIATOR_BETA[radiator_name]
    particle_id = MC_PARTICLE_IDS[particle_name]
    charge = MC_PARTICLE_CHARGES[particle_id]
    mass = MC_PARTICLE_MASSES[particle_id]
    return calc_rig(beta, mass, charge)

def rigidity_from_beta_resolution(radiator_name, particle_name_1, particle_name_2, precision=1e-4):
    particle_id_1 = MC_PARTICLE_IDS[particle_name_1]
    particle_id_2 = MC_PARTICLE_IDS[particle_name_2]
    charge_1 = abs(MC_PARTICLE_CHARGES[particle_id_1])
    charge_2 = abs(MC_PARTICLE_CHARGES[particle_id_2])
    mass_1 = MC_PARTICLE_MASSES[particle_id_1] + 1e-7
    mass_2 = MC_PARTICLE_MASSES[particle_id_2] + 1e-7
    if abs(charge_1 / mass_1 - charge_2 / mass_2) < 1e-2:
        return 0
    beta_resolution = RICH_RADIATOR_RESOLUTION[radiator_name]
    rig = 1
    step = 1
    dir = 1
    for _ in range(100):
        delta_beta = abs(calc_beta(rig, mass_1, charge_1) - calc_beta(rig, mass_2, charge_2))
        if abs((delta_beta - beta_resolution) / delta_beta) <= precision:
            #print(f"{particle_name_1} {particle_name_2} {radiator_name} converged to {rig:.2f}")
            return rig
        if delta_beta > beta_resolution:
            if dir < 0:
                step /= 2
            dir = 1
        else:
            if dir > 0:
                step /= 2
            dir = -1
            if step >= rig:
                step /= 2
        rig += dir * step
    raise ValueError("Max iterations reached.")


PATTERNS = {
    r"R\((?P<radiator_name>NaF|AGL),(?P<particle_name>[A-Za-z0-9]+)\)": rigidity_from_radiator_and_particle,
    r"RDeltaBeta\((?P<radiator_name>NaF|AGL),(?P<particle_name_1>[A-Za-z0-9]+),(?P<particle_name_2>[A-Za-z0-9]+)\)": rigidity_from_beta_resolution,
}

def parse_rigidity_value(raw_value):
    if isinstance(raw_value, int) or isinstance(raw_value, float):
        return raw_value
    elif isinstance(raw_value, str):
        for pattern, function in PATTERNS.items():
            match = re.fullmatch(pattern, raw_value)
            if match is not None:
                return function(**match.groupdict())
        raise ValueError(f"Cannot parse rigidity expression {raw_value!r}!")
    raise ValueError(f"Cannot understand rigidity value {raw_value!r}!")


def parse_bool(bool_str):
    return bool_str.lower() in ("true", "yes", "y")


def save_figure(figure, plotdir, prefix, dpi=250, save_png=True, save_pdf=False, save_pickle=True, close_figure=True):
    if save_png:
        figure.savefig(os.path.join(plotdir, f"{prefix}.png"), dpi=dpi)
    if save_pdf:
        figure.savefig(os.path.join(plotdir, f"{prefix}.pdf"), dpi=dpi)
    if save_pickle:
        with open(os.path.join(plotdir, f"{prefix}.pck"), "wb") as pickle_file:
            pickle.dump(figure, pickle_file)
    if close_figure:
        plt.close(figure)


def plot_feature_importance(bdt, variables, title, plotdir, prefix, labelling=None):
    ncuts = bdt.get_score(importance_type="weight")
    gain = bdt.get_score(importance_type="gain")
    ncuts = {var: ncuts.get(var, 0) for var in variables}
    gain = {var: gain.get(var, 0) for var in variables}
    ncuts_arr = np.array([ncuts[var] for var in variables])
    gain_arr = np.array([gain[var] for var in variables])
    total_gain_arr = ncuts_arr * gain_arr
    total_gain = {var: tg for var, tg in zip(variables, total_gain_arr)}
    indices = np.arange(len(variables))

    def _make_labels(vars):
        if labelling is None:
            return vars
        return [labelling.get_label(var) for var in vars]

    gain_figure = plt.figure(figsize=(16, 8.2))
    gain_figure.suptitle(title)
    gain_plot = gain_figure.subplots(1, 1, gridspec_kw=dict(left=0.33))
    gain_plot.set_xlabel("Avg. Gain")
    variables_by_gain = sorted(variables, key=lambda v: gain[v])
    gain_plot.barh(indices, [gain[var] for var in variables_by_gain], tick_label=_make_labels(variables_by_gain))
    save_figure(gain_figure, plotdir, f"{prefix}_gain")

    ncuts_figure = plt.figure(figsize=(16, 8.2))
    ncuts_figure.suptitle(title)
    ncuts_plot = ncuts_figure.subplots(1, 1, gridspec_kw=dict(left=0.33))
    ncuts_plot.set_xlabel("Occurance")
    variables_by_ncuts = sorted(variables, key=lambda v: ncuts[v])
    ncuts_plot.barh(indices, [ncuts[var] for var in variables_by_ncuts], tick_label=_make_labels(variables_by_ncuts))
    save_figure(ncuts_figure, plotdir, f"{prefix}_ncuts")

    total_gain_figure = plt.figure(figsize=(16, 8.2))
    total_gain_figure.suptitle(title)
    total_gain_plot = total_gain_figure.subplots(1, 1)
    total_gain_plot.set_xlabel("Total Gain")
    variables_by_total_gain = sorted(variables, key=lambda v: total_gain[v])
    total_gain_plot.barh(indices, [total_gain[var] for var in variables_by_total_gain], tick_label=_make_labels(variables_by_total_gain))
    total_gain_figure.subplots_adjust(left=0.2, right=0.95)
    save_figure(total_gain_figure, plotdir, f"{prefix}_total_gain")

    ncuts_gain_figure = plt.figure(figsize=(16, 8.2))
    ncuts_gain_figure.suptitle(title)
    ncuts_gain_plot_gain = ncuts_gain_figure.subplots(1, 1, gridspec_kw=dict(left=0.33))
    ncuts_gain_plot_ncuts = ncuts_gain_plot_gain.twiny()
    ncuts_gain_plot_gain.set_xlabel("Avg. Gain")
    ncuts_gain_plot_ncuts.set_xlabel("Occurance")
    ncuts_gain_plot_gain.barh(indices, [gain[var] for var in variables_by_gain], tick_label=_make_labels(variables_by_gain), height=0.4, label="Avg. Gain", color="tab:blue")
    ncuts_gain_plot_ncuts.barh(indices + 0.5, [ncuts[var] for var in variables_by_gain], tick_label=_make_labels(variables_by_gain), height=0.4, label="Occurance", color="tab:orange")
    ncuts_gain_figure.legend()
    save_figure(ncuts_gain_figure, plotdir, f"{prefix}_ncuts_gain")

    ncuts_gain_2d_figure = plt.figure(figsize=(16, 8.2))
    ncuts_gain_2d_figure.suptitle(title)
    ncuts_gain_2d_plot = ncuts_gain_2d_figure.subplots(1, 1)
    ncuts_gain_2d_plot.set_xlabel("Occurance")
    ncuts_gain_2d_plot.set_ylabel("Avg. Gain")
    ncuts_gain_2d_plot.set_yscale("log")
    ncuts_gain_2d_plot.plot(ncuts_arr, gain_arr, ".")
    for variable, ncuts, gain in zip(variables, ncuts_arr, gain_arr):
        ncuts_gain_2d_plot.text(ncuts, gain, variable, ha="center")
    save_figure(ncuts_gain_2d_figure, plotdir, f"{prefix}_ncuts_gain_2d")
    return total_gain


def sort_contour_points(line_segments):
    connections = defaultdict(lambda: [])
    for first, second in line_segments:
        connections[first].append(second)
        connections[second].append(first)
    assert all(len(points) % 2 == 0 for points in connections.values())

    while connections:
        line = []
        start_point = list(connections)[0]
        current_point = start_point
        line.append(current_point)
        while len(line) == 1 or current_point != start_point:
            next_point = connections[current_point][0]
            connections[current_point].remove(next_point)
            connections[next_point].remove(current_point)
            current_point = next_point
            line.append(current_point)
        point_array = np.array(line)
        yield point_array[:,0], point_array[:,1]
        connections = {key: value for key, value in connections.items() if value}


def create_histogram_contour(histogram, target_efficiency=0.9):
    assert histogram.dimensions == 2
    assert 0 <= target_efficiency <= 1

    total_events = histogram.values.sum()
    sorted_counts = np.sort(histogram.values.flatten())[::-1]
    cumulative = np.cumsum(sorted_counts) / total_events
    cutoff_index = np.argmin(np.abs(cumulative - target_efficiency))
    cutoff_value = sorted_counts[cutoff_index]

    edges_x = transform_overflow_edges(histogram.binnings[1].edges)
    edges_y = transform_overflow_edges(histogram.binnings[0].edges)
    indices_x, indices_y = np.meshgrid(np.arange(len(histogram.binnings[1]) + 2), np.arange(len(histogram.binnings[0]) + 2))
    values = np.zeros((len(edges_y) + 1, len(edges_x) + 1))
    values[1:-1,1:-1] = histogram.values
    selection_2d = values >= cutoff_value

    vertical_edge_left_indices_x = indices_x[:,:-1]
    vertical_edge_right_indices_x = indices_x[:,1:]
    vertical_edge_indices_y = indices_y[:,:-1]
    vertical_edge = selection_2d[vertical_edge_indices_y,vertical_edge_left_indices_x] != selection_2d[vertical_edge_indices_y,vertical_edge_right_indices_x]
    vertical_edge_x = vertical_edge_right_indices_x[vertical_edge]
    vertical_edge_y = vertical_edge_indices_y[vertical_edge]
    vertical_coords_x = edges_x[vertical_edge_x - 1]
    vertical_coords_y_start = edges_y[vertical_edge_y - 1]
    vertical_coords_y_stop = edges_y[vertical_edge_y]

    horizontal_edge_left_indices_y = indices_y[:-1,:]
    horizontal_edge_right_indices_y = indices_y[1:,:]
    horizontal_edge_indices_x = indices_x[:-1,:]
    horizontal_edge = selection_2d[horizontal_edge_left_indices_y,horizontal_edge_indices_x] != selection_2d[horizontal_edge_right_indices_y,horizontal_edge_indices_x]
    horizontal_edge_x = horizontal_edge_indices_x[horizontal_edge]
    horizontal_edge_y = horizontal_edge_right_indices_y[horizontal_edge]
    horizontal_coords_x_start = edges_x[horizontal_edge_x - 1]
    horizontal_coords_x_stop = edges_x[horizontal_edge_x]
    horizontal_coords_y = edges_y[horizontal_edge_y - 1]

    for coords_x, coords_y in sort_contour_points([((x, y_start), (x, y_stop)) for x, y_start, y_stop in zip(vertical_coords_x, vertical_coords_y_start, vertical_coords_y_stop)] + [((x_start, y), (x_stop, y)) for x_start, x_stop, y in zip(horizontal_coords_x_start, horizontal_coords_x_stop, horizontal_coords_y)]):
        yield coords_y, coords_x
