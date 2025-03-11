
import os
import json

import numpy as np

from .binnings import Binning
from .constants import MC_PARTICLE_IDS
from .statistics import fermi_function, inverse_fermi_function, scaled_fermi_function
from .utilities import interpolate_measuring_time, fit_flux, load_flux, load_mc_trigger_density, power_law, resolve_derived_branches, save_figure, set_rigidity_ticks

class McWeightingFromBinnedFlux:
    def __init__(self, binning, iss_events, mc_events, scale_factor=None):
        self.binning = binning
        self.iss_events = iss_events
        self.mc_events = mc_events
        self.scale_factor = scale_factor if scale_factor is not None else 1

    @staticmethod
    def load(filename, scale_factor=None):
        with np.load(filename) as file:
            binning = Binning.from_file(file, "rigidity_binning")
            iss_events = file["raw_counts"]
            mc_events = file["mc_true_counts"]
            return McWeightingFromBinnedFlux(binning, iss_events, mc_events, scale_factor)

    def get_weights(self, mc_particle_ids, rigidities):
        indices = self.binning.get_indices(rigidities)
        return self.iss_events[indices] / self.mc_events[indices] * self.scale_factor

    def get_flat_weights(self, mc_particle_ids, rigidities):
        indices = self.binning.get_indices(rigidities)
        return self.scale_factor / self.mc_events[indices]


class McWeightingFromFit:
    def __init__(self, fit_function, trigger_density_function):
        self.fit_function = fit_function
        self.trigger_density_function = trigger_density_function

    @staticmethod
    def load(flux_filename, mc_triggers_filename):
        flux_array, flux_title = load_flux(flux_filename)
        fit_function, flux_parameters = fit_flux(flux_array, flux_title)
        mc_trigger_density_function = load_mc_trigger_density(mc_triggers_filename)
        return McWeightingFromFit(fit_function=fit_function, trigger_density_function=mc_trigger_density_function)

    def get_weights(self, mc_particle_ids, rigidities):
        return self.fit_function(rigidities) / self.trigger_density_function(rigidities)

    def get_flat_weights(self, mc_particle_ids, rigidities):
        return 1 / self.trigger_density_function(rigidities)


def _make_flux_correction_from_ratio(filename):
    with open(filename) as ratio_file:
        parameters = json.load(ratio_file)
    def _correction(rigidity):
        return 1 / scaled_fermi_function(np.log10(rigidity), **parameters)
    return _correction
        

def make_flux_fit_function(parameters, do_correction=True):
    correction = lambda x: np.ones_like(x)
    if do_correction:
        correction = _make_flux_correction_from_ratio(os.path.join(os.environ["ANTIMATTERSEARCH"], "data", "flux_ratio_fullspan_2018_fullspan.json"))
    def _fit_function(rigidity):
        return power_law(rigidity, **parameters) * correction(rigidity)
    return _fit_function

def make_flat_fit_function():
    def _fit_function(rigidity):
        return power_law(rigidity, phi=0, c=1, rigidity_scale=1, gamma=-1)
    return _fit_function


class McEventRatioWeighting:
    def __init__(self, rigidity_edges, weights, flat_weights):
        self.rigidity_binning = Binning(rigidity_edges)
        self.weights = weights
        self.flat_weights = flat_weights

    def get_weights(self, mc_particle_ids, rigidities):
        bins = np.clip(self.rigidity_binning.get_indices(rigidities), 0, len(self.weights) - 1)
        return np.nan_to_num(self.weights[bins])

    def get_flat_weights(self, mc_particle_ids, rigidities):
        bins = np.clip(self.rigidity_binning.get_indices(rigidities), 0, len(self.flat_weights) - 1)
        return np.nan_to_num(self.flat_weights[bins])


class McUnfoldedEventRatioWeighting:
    def __init__(self, rigidity_edges, weights, flat_weights):
        self.rigidity_binning = Binning(rigidity_edges)
        self.weights = weights
        self.flat_weights = flat_weights

    def get_weights(self, mc_particle_ids, rigidities):
        bins = np.clip(self.rigidity_binning.get_indices(rigidities) - 1, 0, len(self.weights) - 1)
        return np.nan_to_num(self.weights[bins])

    def get_flat_weights(self, mc_particle_ids, rigidities):
        bins = np.clip(self.rigidity_binning.get_indices(rigidities) - 1, 0, len(self.flat_weights) - 1)
        return np.nan_to_num(self.flat_weights[bins])



class McWeightingFromConfig:
    def __init__(self, flux_fit_function, flat_fit_function, trigger_density_function, measuring_time_function, total_measuring_time, flux_fraction):
        self.flux_fit_function = flux_fit_function
        self.flat_fit_function = flat_fit_function
        self.trigger_density_function = trigger_density_function
        self.measuring_time_function = measuring_time_function
        self.total_measuring_time = total_measuring_time
        self.flux_fraction = flux_fraction

    @staticmethod
    def load(weight_config_filename):
        with open(weight_config_filename) as weight_file:
            weight_config_data = json.load(weight_file)
        weightings = {}
        weight_type = weight_config_data.get("weight_type", "normal")
        if weight_type == "normal":
            for key, weight_data in weight_config_data.items():
                trigger_density_function = load_mc_trigger_density(weight_data["triggers"])
                flux_fit_function = make_flux_fit_function(weight_data["flux_parameters"])
                flat_fit_function = make_flat_fit_function()
                measuring_time_function = interpolate_measuring_time(np.array(weight_data["measuring_time"]["rigidity"]), np.array(weight_data["measuring_time"]["values"]), apply_correction=True)
                total_measuring_time = weight_data["measuring_time"]["values"][-1]
                flux_fraction = weight_data["flux_fraction"]
                weightings[MC_PARTICLE_IDS[key]] = McWeightingFromConfig(flux_fit_function=flux_fit_function, flat_fit_function=flat_fit_function, trigger_density_function=trigger_density_function, measuring_time_function=measuring_time_function, total_measuring_time=total_measuring_time, flux_fraction=flux_fraction)
        elif weight_type == "event_ratio":
            for species, species_config in weight_config_data["weights"].items():
                weightings[MC_PARTICLE_IDS[species]] = McEventRatioWeighting(np.array(weight_config_data["rigidity_edges"]), np.array(species_config["weights"]), np.array(species_config["flat_weights"]))
        elif weight_type == "unfolded":
            for species, species_config in weight_config_data["weights"].items():
                weightings[MC_PARTICLE_IDS[species]] = McUnfoldedEventRatioWeighting(np.array(species_config["rigidity_edges"]), np.array(species_config["weights"]), np.array(species_config["flat_weights"]))
        return PerParticleIdMcWeighting(weightings)

    def get_weights(self, mc_particle_ids, rigidities):
        return self.flux_fit_function(rigidities) / self.trigger_density_function(rigidities) * self.flux_fraction * (3.9**2 * np.pi) * self.measuring_time_function(rigidities)

    def get_flat_weights(self, mc_particle_ids, rigidities):
        return self.flat_fit_function(rigidities) / self.trigger_density_function(rigidities) * self.flux_fraction * (3.9**2 * np.pi) * self.total_measuring_time


class PerParticleIdMcWeighting:
    def __init__(self, weightings):
        self.weightings = weightings

    def get_weights(self, mc_particle_ids, rigidities):
        if len(mc_particle_ids) == 0:
            return np.ones((0,))
        mc_particle_id = mc_particle_ids[0]
        assert np.all(mc_particle_ids == mc_particle_id)
        return self.weightings[mc_particle_id].get_weights(mc_particle_ids, rigidities)

    def get_flat_weights(self, mc_particle_ids, rigidities):
        if len(mc_particle_ids) == 0:
            return np.ones((0,))
        mc_particle_id = mc_particle_ids[0]
        assert np.all(mc_particle_ids == mc_particle_id)
        return self.weightings[mc_particle_id].get_flat_weights(mc_particle_ids, rigidities)


def load_mc_weighting(filename, scale_factor=None):
    if not ":" in filename:
        if filename.endswith(".npz"):
            return McWeightingFromBinnedFlux.load(filename, scale_factor)
        elif filename.endswith(".json"):
            return McWeightingFromConfig.load(filename)
        raise ValueError(f"Don't know how to load mc weighting information from {filename!r}")
    elif ":" in filename:
        flux_filename, triggers_filename = filename.split(":", 1)
        assert flux_filename.endswith(".txt") and triggers_filename.endswith(".txt")
        return McWeightingFromFit.load(flux_filename, triggers_filename)
    else:
        raise ValueError(f"Don't know how to load mc weighting information from {filename!r}")
 
