
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline

from .binnings import Binning, Binnings
from .constants import ACCEPTANCE_CATEGORIES, MC_PARTICLE_IDS
from .corrections import Corrections
from .histograms import plot_histogram_1d
from .roottree import read_tree, count_total_events
from .selection import Selection, cut_pattern
from .statistics import fermi_function, inverse_fermi_function, scaled_fermi_function
from .variables import DerivedVariables, VariableLabels
from .utilities import interpolate_measuring_time, fit_flux, load_flux, load_mc_trigger_density, power_law, resolve_derived_branches, save_figure, set_rigidity_ticks
from .weighting import load_mc_weighting


def parse_category(category_str):
    parts = [part.strip().lower() for part in category_str.split("&")]
    value = 0
    mask = 0
    for part in parts:
        inverted = False
        if part.startswith("!"):
            inverted = True
            part = part[1:]
        bit = ACCEPTANCE_CATEGORIES[part]
        mask |= bit
        if not inverted:
            value |= bit
    return mask, value


def parse_sample(config, sample_name, workdir):
    sample_config = config["samples"][sample_name]
    category_mask, category_value = parse_category(sample_config["category"])
    rigidity_estimator = sample_config["rigidity_estimator"]
    binnings = Binnings((config, workdir))
    labelling = VariableLabels(config=config, workdir=workdir, rigidity_estimator=rigidity_estimator)
    selections = {selection_name: Selection.load(config["selections"][selection_name], rigidity_estimator, binnings, config=config, workdir=workdir, labelling=labelling)
                  for selection_name in sample_config["selections"]}
    category_cut = cut_pattern(category_mask, category_value)
    label = sample_config.get("label", None)
    tag_cuts = sample_config.get("tag_cuts", None)
    return selections, rigidity_estimator, category_cut, label, tag_cuts


class Sample:
    def __init__(self, sample_name, selections, category_cut, rigidity_estimator, corrections=None, derived_variables=None, rigidity_range=None, time_range=None, cutoff=None, mc_weighting=None, label=None, tag_cuts_name=None):
        self.sample_name = sample_name
        self.selections = selections
        self.category_cut = category_cut
        self.rigidity_estimator = rigidity_estimator
        self.corrections = corrections
        self.derived_variables = derived_variables
        self.rigidity_range = rigidity_range
        self.time_range = time_range
        self.cutoff = cutoff
        self.mc_weighting = mc_weighting
        self.label = label or sample_name
        self.tag_cuts_name = tag_cuts_name

    @staticmethod
    def load(config, sample_name, workdir):
        selections, rigidity_estimator, category_cut, label, tag_cuts_name = parse_sample(config, sample_name, workdir)
        derived_variables = DerivedVariables(config=config, workdir=workdir, rigidity_estimator=rigidity_estimator)
        return Sample(sample_name, selections, category_cut, rigidity_estimator, derived_variables=derived_variables, label=label, tag_cuts_name=tag_cuts_name)

    def set_rigidity_range(self, rig_min, rig_max):
        self.rigidity_range = (rig_min, rig_max)

    def set_time_range(self, time_min, time_max):
        self.time_range = (time_min, time_max)

    def set_cutoff(self, cutoff_type, cutoff_angle, cutoff_factor):
        self.cutoff = (cutoff_type, cutoff_angle, cutoff_factor)
        self.derived_variables.overwrite_backtracing()

    def set_corrections(self, dataset_correction, config, workdir):
        if dataset_correction in config["datasets"]:
            dataset_correction = config["datasets"][dataset_correction].get("corrections", None)
        if dataset_correction is None:
            return
        sample_config = config["samples"][self.sample_name]
        if "corrections" not in sample_config:
            return
        sample_correction = sample_config["corrections"]
        corrections_path = os.path.join(workdir, "corrections", "corrections", "results", f"corrections_{dataset_correction}_{sample_correction}.json")
        self.corrections = Corrections.load(corrections_path)

    def get_cutoff_branch(self):
        if self.cutoff is None:
            return None
        cutoff_type, cutoff_angle, cutoff_factor = self.cutoff
        return f"{self.rigidity_estimator}{cutoff_type}{cutoff_angle}Factor"

    def set_mc_weighting(self, filename, scale_factor=None):
        self.mc_weighting = load_mc_weighting(filename, scale_factor=scale_factor)

    def read_tree(self, filename, treename, branches, rank, nranks, chunk_size=1000000, verbose=True, prefix="selections", resultdir="results", debug=False, apply_selections=True, pass_all=False):
        branches = branches + ["McParticleID", "McRigidity", "TotalWeight", "TotalFlatWeight", "PrescalingWeight", "McToIssWeight", "AcceptanceCategory", "TrkHitPatternXY"]
        if self.cutoff:
            cutoff_branch = self.get_cutoff_branch()
            _, _, cutoff_factor = self.cutoff
            branches = branches + [cutoff_branch]
        if self.time_range is not None:
            branches = branches + ["Time"]
        if self.rigidity_range is not None:
            branches = branches + [self.rigidity_estimator]
        if apply_selections:
            selection_branches = [var for sel in self.selections.values() for var in sel.cuts]
        else:
            selection_branches = []
        all_branches = list(set(branches) | set(selection_branches))
        primary_branches, derived_branches = resolve_derived_branches(all_branches, self.derived_variables.dependencies, self.derived_variables.functions)
        derived_weight_branches = {"McToIssWeight", "McToFlatWeight", "TotalWeight", "TotalFlatWeight"}

        for chunk in read_tree(filename, treename, branches=primary_branches - derived_weight_branches, rank=rank, nranks=nranks, chunk_size=chunk_size, verbose=verbose):
            if apply_selections:
                chunk = chunk[self.category_cut(chunk.AcceptanceCategory)]
            if self.rigidity_range is not None:
                rigidity = np.abs(chunk[self.rigidity_estimator])
                min_rig, max_rig = self.rigidity_range
                chunk = chunk[(rigidity >= min_rig) & (rigidity < max_rig)]
            is_mc = np.any(chunk.McParticleID != 0)
            if not is_mc:
                if self.time_range is not None:
                    time_min, time_max = self.time_range
                    time = chunk.Time
                    chunk = chunk[(time >= time_min) & (time <= time_max)]
            chunk["TotalWeight"] = chunk.PrescalingWeight
            if self.corrections is not None:
                chunk = self.corrections.apply(chunk)
            for branch in derived_branches:
                chunk[branch] = self.derived_variables.functions[branch](chunk)
            if is_mc and self.mc_weighting is not None:
                chunk["McToIssWeight"] = self.mc_weighting.get_weights(chunk.McParticleID, np.abs(chunk.McRigidity))
                chunk["McToFlatWeight"] = self.mc_weighting.get_flat_weights(chunk.McParticleID, np.abs(chunk.McRigidity))
            else:
                chunk["McToIssWeight"] = np.ones(len(chunk))
                chunk["McToFlatWeight"] = np.ones(len(chunk))
            chunk["TotalWeight"] = chunk.PrescalingWeight * chunk.McToIssWeight
            chunk["TotalFlatWeight"] = chunk.PrescalingWeight * chunk.McToFlatWeight
            if not is_mc:
                if self.cutoff is not None:
                    above_cutoff = chunk[cutoff_branch] >= cutoff_factor
                    chunk = chunk[above_cutoff]
            if apply_selections:
                for selection_name, selection in self.selections.items():
                    chunk_after_selection = selection.select(chunk, debug=debug)
                    if not pass_all:
                        chunk = chunk_after_selection
                    if len(chunk) == 0:
                        break
            if len(chunk) > 0:
                yield chunk

        if apply_selections:
            self.save_selections(resultdir, f"{prefix}_{rank}")

    def save_selections(self, resultdir, prefix):
        result_dict = {"selections": list(self.selections.keys())}
        for selection_name, selection in self.selections.items():
            selection.add_to_file(result_dict, f"selection_{selection_name}")
        os.makedirs(resultdir, exist_ok=True)
        np.savez(os.path.join(resultdir, f"{prefix}_selections.npz"), **result_dict)

    def merge_selections(self, nranks, first_rank=0, prefix="selections", resultdir="results"):
        for rank in range(first_rank, first_rank + nranks):
            temp_filename = os.path.join(resultdir, f"{prefix}_{rank}_selections.npz")
            with np.load(temp_filename) as temp_file:
                for selection_name, selection in self.selections.items():
                    selection.add(Selection.from_file(temp_file, f"selection_{selection_name}"))
            os.remove(temp_filename)

    def plot_selections(self, resultdir="plots", outputprefix="sample", style="mc", rigidity_estimator="|R|"):
        for selection_name, selection in self.selections.items():
            selection.plot(resultdir=resultdir, prefix=f"{outputprefix}_selection_{selection_name}", title=f"Selection {selection_name!r}")

        stack_figure = plt.figure(figsize=(12, 6.15))
        stack_plot = stack_figure.subplots(1, 1)
        stack_plot.set_title(self.label)
        stack_plot.set_xlabel(f"{rigidity_estimator} / GV")
        stack_plot.set_ylabel("Events")
        first_selection = self.selections[list(self.selections.keys())[0]]
        total_events = first_selection.total_passed_histogram + first_selection.total_failed_histogram
        plot_histogram_1d(stack_plot, total_events, style=style, label="Before Selection", log=True)
        for selection_name, selection in self.selections.items():
            plot_histogram_1d(stack_plot, selection.total_passed_histogram, style=style, label=selection_name, log=True)
        stack_plot.legend()
        set_rigidity_ticks(stack_plot)
        save_figure(stack_figure, resultdir, f"{outputprefix}_stacked")

        if first_selection.has_mc_data():
            stack_mc_figure = plt.figure(figsize=(12, 6.15))
            stack_mc_plot = stack_mc_figure.subplots(1, 1)
            stack_mc_plot.set_title(self.label)
            stack_mc_plot.set_xlabel(f"MC Rigidity / GV")
            stack_mc_plot.set_ylabel("Events")
            total_events = first_selection.total_passed_mc_histogram + first_selection.total_failed_mc_histogram
            plot_histogram_1d(stack_mc_plot, total_events, style=style, label="Before Selection", log=True)
            for selection_name, selection in self.selections.items():
                plot_histogram_1d(stack_mc_plot, selection.total_passed_mc_histogram, style=style, label=selection_name, log=True)
            stack_mc_plot.legend()
            set_rigidity_ticks(stack_mc_plot)
            save_figure(stack_mc_figure, resultdir, f"{outputprefix}_stacked_mc")
