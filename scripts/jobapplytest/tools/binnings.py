#!/usr/bin/env python3

from datetime import datetime

import numpy as np
import awkward as ak

from tools.mvas import MVA
from tools.constants import MC_PARTICLE_IDS, MC_PARTICLE_CHARGE_ARRAY
from tools.utilities import parse_rigidity_value

def add_overflow(edges):
    if edges[0] == -np.inf and edges[-1] == np.inf:
        return edges
    return np.concatenate(([-np.inf], edges, [np.inf]))

class Binning:
    def __init__(self, edges, log=False, overflow=True):
        if overflow:
            self.edges = add_overflow(edges)
        else:
            self.edges = edges
        self.bin_centers = (self.edges[1:] + self.edges[:-1]) / 2
        self.bin_widths = (self.edges[1:] - self.edges[:-1])
        self.log = log

    def add_to_file(self, file_dict, name):
        file_dict[f"{name}_edges"] = self.edges
        file_dict[f"{name}_log"] = self.log

    @staticmethod
    def from_file(file_dict, name):
        return Binning(edges=file_dict[f"{name}_edges"], log=file_dict[f"{name}_log"])

    def __eq__(self, other):
        return len(self.edges) == len(other.edges) and np.all(self.edges == other.edges) and self.log == other.log

    def __ne__(self, other):
        return len(self.edges) != len(other.edges) or np.any(self.edges != other.edges) or self.log != other.log

    def __len__(self):
        return len(self.edges) - 1

    def get_indices(self, values, with_overflow=True):
        return np.clip(np.digitize(ak.to_numpy(values), self.edges) - 1, 0 if with_overflow else 1, len(self.edges) - (2 if with_overflow else 3))


def get_rebin_factor(binning):
    for factor in (2, 3, 5, 7):
        if (len(binning) - 2) % factor == 0:
            return factor
    return 1


def reduce_bins(binning, factor):
    bins = len(binning) - 2
    assert bins % factor == 0
    new_edges = binning.edges[1:-1:factor]
    assert new_edges[-1] == binning.edges[-2]
    return Binning(new_edges, log=binning.log)


def increase_bin_number(binning, factor):
    assert int(factor) == factor and factor > 1
    edges = binning.edges[1:-1]
    def _make_edges(min, max, n):
        if binning.log:
            return np.logspace(np.log10(min), np.log10(max), n, endpoint=False)
        return np.linspace(min, max, n, endpoint=False)
    new_edges = np.concatenate([_make_edges(low, high, factor) for (low, high) in zip(edges[:-1], edges[1:])] + [[edges[-1]]])
    return Binning(new_edges, log=binning.log)
    
def combine_binnings(binnings):
    log = any(binning.log for binning in binnings)
    edges = np.array(sorted({edge for binning in binnings for edge in binning.edges if np.isfinite(edge)}))
    distanced = np.concatenate(([True], (edges[1:] - edges[:-1]) > 1e-10))
    return Binning(edges[distanced], log=log)

def make_lin_binning(start, stop, nbins):
    return Binning(np.linspace(start, stop, nbins + 1))

def make_log_binning(start, stop, nbins):
    return Binning(np.logspace(np.log10(start), np.log10(stop), nbins + 1), log=True)

def make_int_binning(n, nmin=0):
    return Binning(np.arange(nmin, n + 1) - 0.5)

def make_bool_binning():
    return make_int_binning(2)

def make_lin_binning_with_known_edge(start, stop, nbins_min, known_edge):
    assert stop >= known_edge >= start
    bin_width_max = (stop - start) / nbins_min
    bins_before_edge = int(np.ceil((known_edge - start) / bin_width_max))
    bin_width = (known_edge - start) / bins_before_edge
    bins_after_edge = int(np.ceil((stop - known_edge) / bin_width))
    nbins = bins_before_edge + bins_after_edge
    new_stop = start + nbins * bin_width
    assert new_stop >= stop
    assert nbins >= nbins_min
    return make_lin_binning(start, new_stop, nbins)

def make_log_binning_width_known_edge(start, stop, nbins_bin, known_edge):
    binning = make_lin_binning_with_known_edge(np.log10(start), np.log10(stop), nbins_min, np.log10(known_edge))
    edges = 10**binning.edges
    return Binning(edges, log=True)

def make_rigidity_binning(rig_min, rig_max, log_rig_resolution):
    log_rig_min = np.log10(rig_min)
    log_rig_max = np.log10(rig_max)
    nbins = int(np.ceil((log_rig_max - log_rig_min) / log_rig_resolution))
    return Binning(np.logspace(log_rig_min, log_rig_max, nbins + 1), log=True)

def make_flux_nn_rigidity_binning(config):
    return Binning(np.array([
        0.80, 1.00, 1.16, 1.33, 1.51, 1.71, 1.92, 2.15, 2.40,
        2.67, 2.97, 3.29, 3.64, 4.02, 4.43, 4.88, 5.37, 5.90,
        6.47, 7.09, 7.76, 8.48, 9.26, 10.1, 11.0, 12.0, 13.0,
        14.1, 15.3, 16.6, 18.0, 19.5, 21.1, 22.8, 24.7, 26.7,
        28.8, 31.1, 33.5, 36.1, 38.9, 41.9, 45.1, 48.5, 52.2,
        56.1, 60.3, 64.8, 69.7, 74.9, 80.5, 86.5, 93.0, 100.0,
        108.0, 116.0, 125.0, 135.0, 147.0, 160.0, 175.0, 192.0, 211.0,
        233.0, 259.0, 291.0, 330.0, 379.0, 441.0, 525.0, 643.0, 822.0,
        1130.0, 1800.0, 3000.0, 6000.00]), log=True)




