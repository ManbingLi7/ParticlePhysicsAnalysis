
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from uncertainties import ufloat

from .binnings import Binning
from .histograms import WeightedHistogram, plot_histogram_1d, plot_histogram_2d
from .statistics import calculate_efficiency, calculate_efficiency_error
from .utilities import round_up, set_rigidity_ticks, make_colormap_passed, make_colormap_failed, save_figure


def cut_twosided(min_value, max_value):
    def _cut_twosided(values):
        return np.all((values >= min_value, values <= max_value), axis=0)
    return _cut_twosided

def cut_greater(min_value):
    def _cut_greater(values):
        return values >= min_value
    return _cut_greater

def cut_lesser(max_value):
    def _cut_lesser(values):
        return values <= max_value
    return _cut_lesser

def cut_bool(target):
    def _cut_bool(values):
        return values == target
    return _cut_bool

def cut_pattern(mask, value):
    def _cut_pattern(values):
        return (values & mask) == value
    return _cut_pattern


class Cut:
    def __init__(self, cut_function, passed_histogram, failed_histogram, passed_histogram_per_rig, failed_histogram_per_rig, passed_histogram_per_mc_rig, failed_histogram_per_mc_rig, label, tagger=None, fill_hists=True):
        self.cut_function = cut_function
        self.passed_histogram = passed_histogram
        self.failed_histogram = failed_histogram
        self.passed_histogram_per_rig = passed_histogram_per_rig
        self.failed_histogram_per_rig = failed_histogram_per_rig
        self.passed_histogram_per_mc_rig = passed_histogram_per_mc_rig
        self.failed_histogram_per_mc_rig = failed_histogram_per_mc_rig
        self.label = label
        self.tagger = tagger
        self.fill_hists = fill_hists

    @staticmethod
    def create(cut_function, binning, rigidity_binning, label, variable, rigidity_estimator, tagger=None, fill_hists=True):
        return Cut(
            cut_function=cut_function,
            passed_histogram=WeightedHistogram(binning, labels=(variable,)),
            failed_histogram=WeightedHistogram(binning, labels=(variable,)),
            passed_histogram_per_rig=WeightedHistogram(rigidity_binning, binning, labels=(f"{rigidity_estimator} / GV", variable)),
            failed_histogram_per_rig=WeightedHistogram(rigidity_binning, binning, labels=(f"{rigidity_estimator} / GV", variable)),
            passed_histogram_per_mc_rig=WeightedHistogram(rigidity_binning, binning, labels=(f"MC Rigidity / GV", variable)),
            failed_histogram_per_mc_rig=WeightedHistogram(rigidity_binning, binning, labels=(f"MC Rigidity / GV", variable)),
            label=label,
            tagger=tagger,
            fill_hists=fill_hists)

    @staticmethod
    def load(cut_config, variable, binning, rigidity_binning, rigidity_estimator, config, workdir, labelling, fill_hists=True):
        if "min" in cut_config and "max" in cut_config:
            min_value = cut_config["min"]
            max_value = cut_config["max"]
            cut_func = cut_twosided(min_value=min_value, max_value=max_value)
            unit = labelling.units.get(variable, "")
            label = f"{min_value:.2g}{unit} ≤ {labelling.labels.get(variable, variable)} ≤ {max_value:.2g}{unit}"
        elif "min" in cut_config:
            min_value = cut_config["min"]
            cut_func = cut_greater(min_value=min_value)
            label = f"{labelling.labels.get(variable, variable)} ≥ {min_value:.2g}{labelling.units.get(variable, '')}"
        elif "max" in cut_config:
            max_value = cut_config["max"]
            cut_func = cut_lesser(max_value=max_value)
            label = f"{labelling.labels.get(variable, variable)} ≤ {max_value:.2g}{labelling.units.get(variable, '')}"
        elif "bool" in cut_config:
            value = cut_config["bool"]
            cut_func = cut_bool(target=value)
            label = labelling.labels.get(variable, variable)
            if not value:
                label = f"not {label}"
        elif "mask" in cut_config and "value" in cut_config:
            mask = cut_config["mask"]
            value = cut_config["value"]
            cut_func = cut_pattern(mask=mask, value=value)
            label = f"{labelling.labels.get(variable, variable)} & {labelling.items.get(variable, {}).get(mask, mask)} = {labelling.items.get(variable, {}).get(value, value)}"
        elif "in_range" in cut_config:
            range_filename = os.path.join(workdir, "range", cut_config["in_range"], "range", "results", "range.json")
            with open(range_filename, "r") as range_file:
                range_config = json.load(range_file)
                min_value = range_config[variable]["cut_low"]
                max_value = range_config[variable]["cut_high"]
                unit = labelling.units.get(variable, "")
                if min_value is not None:
                    if max_value is not None:
                        cut_func = cut_twosided(min_value=min_value, max_value=max_value)
                        label = f"{min_value:.2g}{unit} ≤ {labelling.labels.get(variable, variable)} ≤ {max_value:.2g}{unit}"
                    else:
                        cut_func = cut_greater(min_value=min_value)
                        label = f"{labelling.labels.get(variable, variable)} ≥ {min_value:.2g}{unit}"
                else:
                    if max_value is not None:
                        cut_func = cut_lesser(max_value=max_value)
                        label = f"{labelling.labels.get(variable, variable)} ≤ {max_value:.2g}{unit}"
                    else:
                        raise ValueError(f"Invalid range cut: {range_config!r}")
        else:
            raise ValueError(f"Cannot parse cut {cut_config!r}.")
        tagger = cut_config.get("tag_cuts", None)
        return Cut.create(cut_func, binning, rigidity_binning, label, variable, rigidity_estimator, tagger=tagger, fill_hists=fill_hists)

    def select(self, values, rigidity, weights, mc_rigidity=None):
        passed = self.cut_function(values)
        if not self.fill_hists:
            return passed
        failed = np.invert(passed)
        self.passed_histogram.fill(values[passed], weights=weights[passed])
        self.failed_histogram.fill(values[failed], weights=weights[failed])
        self.passed_histogram_per_rig.fill(rigidity[passed], values[passed], weights=weights[passed])
        self.failed_histogram_per_rig.fill(rigidity[failed], values[failed], weights=weights[failed])
        if mc_rigidity is not None:
            self.passed_histogram_per_mc_rig.fill(mc_rigidity[passed], values[passed], weights=weights[passed])
            self.failed_histogram_per_mc_rig.fill(mc_rigidity[failed], values[failed], weights=weights[failed])
        return passed

    def add_to_file(self, file_dict, name):
        self.passed_histogram.add_to_file(file_dict, f"{name}_passed")
        self.failed_histogram.add_to_file(file_dict, f"{name}_failed")
        self.passed_histogram_per_rig.add_to_file(file_dict, f"{name}_passed_per_rig")
        self.failed_histogram_per_rig.add_to_file(file_dict, f"{name}_failed_per_rig")
        self.passed_histogram_per_mc_rig.add_to_file(file_dict, f"{name}_passed_per_mc_rig")
        self.failed_histogram_per_mc_rig.add_to_file(file_dict, f"{name}_failed_per_mc_rig")
        file_dict[f"{name}_label"] = self.label

    @staticmethod
    def from_file(file_dict, name):
        return Cut(
            cut_function=None,
            passed_histogram=WeightedHistogram.from_file(file_dict, f"{name}_passed"),
            failed_histogram=WeightedHistogram.from_file(file_dict, f"{name}_failed"),
            passed_histogram_per_rig=WeightedHistogram.from_file(file_dict, f"{name}_passed_per_rig"),
            failed_histogram_per_rig=WeightedHistogram.from_file(file_dict, f"{name}_failed_per_rig"),
            passed_histogram_per_mc_rig=WeightedHistogram.from_file(file_dict, f"{name}_passed_per_mc_rig"),
            failed_histogram_per_mc_rig=WeightedHistogram.from_file(file_dict, f"{name}_failed_per_mc_rig"),
            label=file_dict[f"{name}_label"].item())

    def add(self, other):
        assert self.label == other.label
        self.passed_histogram.add(other.passed_histogram)
        self.failed_histogram.add(other.failed_histogram)
        self.passed_histogram_per_rig.add(other.passed_histogram_per_rig)
        self.failed_histogram_per_rig.add(other.failed_histogram_per_rig)
        self.passed_histogram_per_mc_rig.add(other.passed_histogram_per_mc_rig)
        self.failed_histogram_per_mc_rig.add(other.failed_histogram_per_mc_rig)

    def __iadd__(self, other):
        self.add(other)
        return self

    def __add__(self, other):
        assert self.label == other.label
        return Cut(
            cut_function=None,
            passed_histogram=self.passed_histogram + other.passed_histogram,
            failed_histogram=self.failed_histogram + other.failed_histogram,
            passed_histogram_per_rig=self.passed_histogram_per_rig + other.passed_histogram_per_rig,
            failed_histogram_per_rig=self.failed_histogram_per_rig + other.failed_histogram_per_rig,
            passed_histogram_per_mc_rig=self.passed_histogram_per_mc_rig + other.passed_histogram_per_mc_rig,
            failed_histogram_per_mc_rig=self.failed_histogram_per_mc_rig + other.failed_histogram_per_mc_rig,
            label=self.label,
            tagger=self.tagger)

    def has_mc_data(self):
        return self.passed_histogram_per_mc_rig.values.sum() + self.failed_histogram_per_mc_rig.values.sum() > 0

    def get_efficiency(self):
        passed = self.passed_histogram.values.sum()
        failed = self.failed_histogram.values.sum()
        all = passed + failed
        return calculate_efficiency(passed, all), calculate_efficiency_error(passed, all)

    def get_efficiency_per_rigidity(self):
        passed = self.passed_histogram_per_rig.values.sum(axis=1)
        failed = self.failed_histogram_per_rig.values.sum(axis=1)
        all = passed + failed
        return calculate_efficiency(passed, all), calculate_efficiency_error(passed, all)

    def get_efficiency_per_mc_rigidity(self):
        passed = self.passed_histogram_per_mc_rig.values.sum(axis=1)
        failed = self.failed_histogram_per_mc_rig.values.sum(axis=1)
        all = passed + failed
        return calculate_efficiency(passed, all), calculate_efficiency_error(passed, all)

    def plot(self, plot, style="mc", **kwargs):
        plot_histogram_1d(plot, self.passed_histogram, style=style, color="green", label="passed", **kwargs)
        plot_histogram_1d(plot, self.failed_histogram, style=style, color="red", label="failed", **kwargs)
        efficiency = ufloat(*self.get_efficiency())
        plot.plot([np.nan], [np.nan], color="white", label=f"$\\epsilon={efficiency:L}$")

    def plot_efficiency_per_rigidity(self, plot, style="mc", **kwargs):
        efficiency, efficiency_error = self.get_efficiency_per_rigidity()
        rigidity_bin_centers = self.passed_histogram_per_rig.binnings[0].bin_centers
        facecolor = "none" if style == "mc" else None
        plot.errorbar(rigidity_bin_centers, efficiency, efficiency_error, marker="o", mfc=facecolor, **kwargs)

    def plot_efficiency_per_mc_rigidity(self, plot, style="mc", **kwargs):
        efficiency, efficiency_error = self.get_efficiency_per_mc_rigidity()
        rigidity_bin_centers = self.passed_histogram_per_rig.binnings[0].bin_centers
        facecolor = "none" if style == "mc" else None
        plot.errorbar(rigidity_bin_centers, efficiency, efficiency_error, marker="o", mfc=facecolor, **kwargs)

    def plot_per_rigidity(self, plot, **kwargs):
        scale = 1 / (self.passed_histogram_per_rig.values.sum(axis=1) + self.failed_histogram_per_rig.values.sum(axis=1))
        positive_passed_values = self.passed_histogram_per_rig.values[1:-1,:] * np.expand_dims(scale[1:-1], 1)
        positive_failed_values = self.failed_histogram_per_rig.values[1:-1,:] * np.expand_dims(scale[1:-1], 1)
        positive_passed_values = positive_passed_values[positive_passed_values > 0]
        positive_failed_values = positive_failed_values[positive_failed_values > 0]
        max_value = 0
        min_value = 1
        if len(positive_passed_values) > 0:
            max_value = max(positive_passed_values.max(), max_value)
            min_value = min(positive_passed_values.min(), min_value)
        if len(positive_failed_values) > 0:
            max_value = max(positive_failed_values.max(), max_value)
            min_value = min(positive_failed_values.min(), min_value)
        plot_histogram_2d(plot, self.passed_histogram_per_rig, show_overflow_x=False, min_value=min_value, max_value=max_value, scale=scale, cmap=make_colormap_passed(), log=True)
        plot_histogram_2d(plot, self.failed_histogram_per_rig, show_overflow_x=False, min_value=min_value, max_value=max_value, scale=scale, cmap=make_colormap_failed(), log=True)
        set_rigidity_ticks(plot)

    def plot_per_mc_rigidity(self, plot, **kwargs):
        scale = 1 / (self.passed_histogram_per_mc_rig.values.sum(axis=1) + self.failed_histogram_per_mc_rig.values.sum(axis=1))
        positive_passed_values = self.passed_histogram_per_mc_rig.values[1:-1,:] * np.expand_dims(scale[1:-1], 1)
        positive_failed_values = self.failed_histogram_per_mc_rig.values[1:-1,:] * np.expand_dims(scale[1:-1], 1)
        positive_passed_values = positive_passed_values[positive_passed_values > 0]
        positive_failed_values = positive_failed_values[positive_failed_values > 0]
        max_value = 0
        min_value = 1
        if len(positive_passed_values) > 0:
            max_value = max(positive_passed_values.max(), max_value)
            min_value = min(positive_passed_values.min(), min_value)
        if len(positive_failed_values) > 0:
            max_value = max(positive_failed_values.max(), max_value)
            min_value = min(positive_failed_values.min(), min_value)
        plot_histogram_2d(plot, self.passed_histogram_per_mc_rig, show_overflow_x=False, min_value=min_value, max_value=max_value, scale=scale, cmap=make_colormap_passed(), log=True)
        plot_histogram_2d(plot, self.failed_histogram_per_mc_rig, show_overflow_x=False, min_value=min_value, max_value=max_value, scale=scale, cmap=make_colormap_failed(), log=True)
        set_rigidity_ticks(plot)


class Selection:
    def __init__(self, cuts, rigidity_estimator, rigidity_binning, total_passed_histogram=None, total_failed_histogram=None, total_passed_mc_histogram=None, total_failed_mc_histogram=None, after_all_other_passed_histograms=None, after_all_other_failed_histograms=None, fill_hists=True):
        self.cuts = cuts
        self.rigidity_estimator = rigidity_estimator
        self.rigidity_binning = rigidity_binning
        if total_passed_histogram is None:
            total_passed_histogram = WeightedHistogram(rigidity_binning)
        self.total_passed_histogram = total_passed_histogram
        if total_failed_histogram is None:
            total_failed_histogram = WeightedHistogram(rigidity_binning)
        self.total_failed_histogram = total_failed_histogram
        if total_passed_mc_histogram is None:
            total_passed_mc_histogram = WeightedHistogram(rigidity_binning)
        self.total_passed_mc_histogram = total_passed_mc_histogram
        if total_failed_mc_histogram is None:
            total_failed_mc_histogram = WeightedHistogram(rigidity_binning)
        self.total_failed_mc_histogram = total_failed_mc_histogram
        if after_all_other_passed_histograms is None:
            after_all_other_passed_histograms = {
                cut_name: WeightedHistogram(cut.passed_histogram.binnings[0])
                for cut_name, cut in cuts.items()
            }
        self.after_all_other_passed_histograms = after_all_other_passed_histograms
        if after_all_other_failed_histograms is None:
            after_all_other_failed_histograms = {
                cut_name: WeightedHistogram(cut.failed_histogram.binnings[0])
                for cut_name, cut in cuts.items()
            }
        self.after_all_other_failed_histograms = after_all_other_failed_histograms
        self.fill_hists = fill_hists


    @staticmethod
    def load(selection_config, rigidity_estimator, binnings, config, workdir, labelling, fill_hists=True):
        cuts = {}
        for variable, cut_config in selection_config["cuts"].items():
            cuts[variable] = Cut.load(cut_config, variable, binnings.variable_binnings[variable], binnings.rigidity_tev_binning, rigidity_estimator=rigidity_estimator, config=config, workdir=workdir, labelling=labelling, fill_hists=fill_hists)
        return Selection(cuts, rigidity_estimator, binnings.rigidity_tev_binning, fill_hists=fill_hists)

    def add_to_file(self, file_dict, name):
        file_dict[f"{name}_cuts"] = list(self.cuts.keys())
        for cut_name, cut in self.cuts.items():
            cut.add_to_file(file_dict, f"{name}_cut_{cut_name}")
            self.after_all_other_passed_histograms[cut_name].add_to_file(file_dict, f"{name}_after_all_other_{cut_name}_passed")
            self.after_all_other_failed_histograms[cut_name].add_to_file(file_dict, f"{name}_after_all_other_{cut_name}_failed")
        self.total_passed_histogram.add_to_file(file_dict, f"{name}_selection_total_passed")
        self.total_failed_histogram.add_to_file(file_dict, f"{name}_selection_total_failed")
        self.total_passed_mc_histogram.add_to_file(file_dict, f"{name}_selection_total_passed_mc")
        self.total_failed_mc_histogram.add_to_file(file_dict, f"{name}_selection_total_failed_mc")
        self.rigidity_binning.add_to_file(file_dict, f"{name}_rigidity_binning")
        file_dict[f"{name}_rigidity_estimator"] = self.rigidity_estimator

    @staticmethod
    def from_file(file_dict, name):
        cut_names = list(file_dict[f"{name}_cuts"])
        rigidity_binning = Binning.from_file(file_dict, f"{name}_rigidity_binning")
        rigidity_estimator = file_dict[f"{name}_rigidity_estimator"].item()
        total_passed = WeightedHistogram.from_file(file_dict, f"{name}_selection_total_passed")
        total_failed = WeightedHistogram.from_file(file_dict, f"{name}_selection_total_failed")
        total_passed_mc = WeightedHistogram.from_file(file_dict, f"{name}_selection_total_passed_mc")
        total_failed_mc = WeightedHistogram.from_file(file_dict, f"{name}_selection_total_failed_mc")
        cuts = {cut_name: Cut.from_file(file_dict, f"{name}_cut_{cut_name}") for cut_name in cut_names}
        after_all_other_passed = {cut_name: WeightedHistogram.from_file(file_dict, f"{name}_after_all_other_{cut_name}_passed") for cut_name in cut_names}
        after_all_other_failed = {cut_name: WeightedHistogram.from_file(file_dict, f"{name}_after_all_other_{cut_name}_failed") for cut_name in cut_names}
        return Selection(cuts, rigidity_estimator, rigidity_binning,
                         total_passed_histogram=total_passed, total_failed_histogram=total_failed,
                         total_passed_mc_histogram=total_passed_mc, total_failed_mc_histogram=total_failed_mc,
                         after_all_other_passed_histograms=after_all_other_passed, after_all_other_failed_histograms=after_all_other_failed)

    def select(self, chunk, debug=False):
        return chunk[self.apply(chunk, debug=debug)]

    def apply(self, chunk, debug=False):
        selections = []
        rig = np.abs(chunk[self.rigidity_estimator])
        mc_rig = None
        if np.any(chunk.McParticleID > 0):
            mc_rig = np.abs(chunk.McRigidity)
        weights = chunk.TotalWeight
        for variable, cut in self.cuts.items():
            selections.append(cut.select(chunk[variable], rig, weights, mc_rigidity=mc_rig))
        if debug:
            cut_labels = np.array([cut.label for cut in self.cuts.values()])
            for index, event in enumerate(chunk):
                if not np.all([s[index] for s in selections]):
                    label = ", ".join(f"{cut_label} ({variable}={event[variable]})" for passed_cut, cut_label, variable in zip(selections, cut_labels, self.cuts.keys()) if not passed_cut[index])
                    event = chunk[index]
                    print(f"Event {event.RunNumber} {event.EventNumber} failed selection: {label}")
        if len(self.cuts) > 1 and self.fill_hists:
            for index, (variable, selection) in enumerate(zip(self.cuts, selections)):
                selection_without = np.all(selections[:index] + selections[index+1:], axis=0)
                passed_after = np.all((selection_without, selection), axis=0)
                failed_after = np.all((selection_without, np.invert(selection)), axis=0)
                self.after_all_other_passed_histograms[variable].fill(chunk[variable][passed_after], weights=weights[passed_after])
                self.after_all_other_failed_histograms[variable].fill(chunk[variable][failed_after], weights=weights[failed_after])

        passed = np.all(selections, axis=0)
        if not self.fill_hists:
            return passed
        failed = np.invert(passed)
        self.total_passed_histogram.fill(rig[passed], weights=weights[passed])
        self.total_failed_histogram.fill(rig[failed], weights=weights[failed])
        if mc_rig is not None:
            self.total_passed_mc_histogram.fill(mc_rig[passed], weights=weights[passed])
            self.total_failed_mc_histogram.fill(mc_rig[failed], weights=weights[failed])
        return passed

    def add(self, other):
        assert set(self.cuts) == set(other.cuts)
        for cut_name in self.cuts:
            self.cuts[cut_name].add(other.cuts[cut_name])
            self.after_all_other_passed_histograms[cut_name].add(other.after_all_other_passed_histograms[cut_name])
            self.after_all_other_failed_histograms[cut_name].add(other.after_all_other_failed_histograms[cut_name])
        self.total_passed_histogram.add(other.total_passed_histogram)
        self.total_failed_histogram.add(other.total_failed_histogram)
        self.total_passed_mc_histogram.add(other.total_passed_mc_histogram)
        self.total_failed_mc_histogram.add(other.total_failed_mc_histogram)

    def __iadd__(self, other):
        self.add(other)
        return self

    def __add__(self, other):
        cuts = {variable: self.cuts[variable] + other.cuts[variable] for variable in self.cuts}
        assert self.rigidity_estimator == other.rigidity_estimator
        assert self.rigidity_binning == other.rigidity_binning
        after_all_other_passed = {variable: self.after_all_other_passed_histograms[variable] + other.after_all_other_passed_histograms[variable] for variable in self.cuts}
        after_all_other_failed = {variable: self.after_all_other_failed_histograms[variable] + other.after_all_other_failed_histograms[variable] for variable in self.cuts}
        return Selection(
            cuts=cuts,
            rigidity_estimator=self.rigidity_estimator,
            rigidity_binning=self.rigidity_binning,
            total_passed_histogram=self.total_passed_histogram + other.total_passed_histogram,
            total_failed_histogram=self.total_failed_histogram + other.total_failed_histogram,
            total_passed_mc_histogram=self.total_passed_mc_histogram + other.total_passed_mc_histogram,
            total_failed_mc_histogram=self.total_failed_mc_histogram + other.total_failed_mc_histogram,
            after_all_other_passed_histograms=after_all_other_passed,
            after_all_other_failed_histograms=after_all_other_failed)


    def has_mc_data(self):
        return self.total_passed_mc_histogram.values.sum() + self.total_failed_mc_histogram.values.sum() > 0

    def plot_cut(self, cut_name, plot, style="mc", **kwargs):
        cut = self.cuts[cut_name]
        plot.set_xlabel(cut_name)
        plot.set_ylabel("Events")
        plot.set_title(cut.label)
        cut.plot(plot, style=style, **kwargs)

    def cut_plot_efficiency(self, cut_name, plot, style="mc", **kwargs):
        cut = self.cuts[cut_name]
        plot.set_xlabel("|R| / GV")
        plot.set_ylabel("$\\epsilon$")
        plot.set_xscale("log")
        plot.set_ylim(0, 1)
        cut.plot_efficiency_per_rigidity(plot, label=cut.label, style=style, **kwargs)

    def cut_plot_mc_efficiency(self, cut_name, plot, style="mc", **kwargs):
        cut = self.cuts[cut_name]
        plot.set_xlabel("MC Rigidity / GV")
        plot.set_ylabel("$\\epsilon$")
        plot.set_xscale("log")
        plot.set_ylim(0, 1)
        cut.plot_efficiency_per_mc_rigidity(plot, label=cut.label, style=style, **kwargs)

    def plot(self, resultdir="plots", prefix="selection", title=None, style="mc"):
        for cut_name in self.cuts:
            cut_figure = plt.figure(figsize=(12, 6.15))
            cut_plot = cut_figure.subplots(1, 1)
            self.plot_cut(cut_name, cut_plot, style=style)
            cut_plot.legend()
            save_figure(cut_figure, resultdir, f"{prefix}_cut_{cut_name}", close_figure=False)
            cut_plot.set_yscale("log")
            ymin, ymax = cut_plot.get_ylim()

            cut_plot.set_ylim(None, ymax)
            save_figure(cut_figure, resultdir, f"{prefix}_cut_{cut_name}_log")

            if len(self.cuts) > 1:
                after_all_other_figure = plt.figure(figsize=(12, 6.15))
                after_all_other_plot = after_all_other_figure.subplots(1, 1)
                after_all_other_plot.set_xlabel(cut_name)
                after_all_other_plot.set_ylabel("Events")
                after_all_other_plot.set_title(f"{self.cuts[cut_name].label}, after all other cuts")
                plot_histogram_1d(after_all_other_plot, self.after_all_other_passed_histograms[cut_name],
                                  style=style, color="green", label="passed")
                plot_histogram_1d(after_all_other_plot, self.after_all_other_failed_histograms[cut_name],
                                  style=style, color="red", label="failed")
                save_figure(after_all_other_figure, resultdir, f"{prefix}_cut_{cut_name}_after_all_other", close_figure=False)
                after_all_other_plot.set_yscale("log")
                after_all_other_plot.set_ylim(0.9, ymax)
                save_figure(after_all_other_figure, resultdir, f"{prefix}_cut_{cut_name}_after_all_other_log")

            per_rig_figure = plt.figure(figsize=(12, 6.15))
            per_rig_plot = per_rig_figure.subplots(1, 1)
            per_rig_plot.set_title(f"{self.cuts[cut_name].label} per rigidity")
            self.cuts[cut_name].plot_per_rigidity(per_rig_plot)
            save_figure(per_rig_figure, resultdir, f"{prefix}_cut_{cut_name}_per_rig")

            if self.cuts[cut_name].has_mc_data():
                per_mc_rig_figure = plt.figure(figsize=(12, 6.15))
                per_mc_rig_plot = per_mc_rig_figure.subplots(1, 1)
                per_mc_rig_plot.set_title(f"{self.cuts[cut_name].label} per MC rigidity")
                self.cuts[cut_name].plot_per_mc_rigidity(per_mc_rig_plot)
                save_figure(per_mc_rig_figure, resultdir, f"{prefix}_cut_{cut_name}_per_mc_rig")

        efficiency_figure = plt.figure(figsize=(16, 8.2))
        efficiency_plot = efficiency_figure.subplots(1, 1)
        efficiency_plot.set_title(title)
        for cut_name in self.cuts:
            self.cut_plot_efficiency(cut_name, efficiency_plot, style=style)
        total_passed = self.total_passed_histogram.values
        total_failed = self.total_failed_histogram.values
        total_all = total_passed + total_failed
        total_efficiency = calculate_efficiency(total_passed, total_all)
        total_efficiency_error = calculate_efficiency_error(total_passed, total_all)
        rigidity_bin_centers = self.rigidity_binning.bin_centers
        facecolor = "none" if style == "mc" else None
        efficiency_plot.errorbar(rigidity_bin_centers, total_efficiency, total_efficiency_error, label="Total", marker="o", mfc=facecolor)
        efficiency_plot.set_ylim(0, 1)
        set_rigidity_ticks(efficiency_plot)
        efficiency_plot.legend()
        save_figure(efficiency_figure, resultdir, f"{prefix}_efficiency")

        efficiency_mc_figure = plt.figure(figsize=(16, 8.2))
        efficiency_mc_plot = efficiency_mc_figure.subplots(1, 1)
        efficiency_mc_plot.set_title(title)
        for cut_name in self.cuts:
            self.cut_plot_mc_efficiency(cut_name, efficiency_mc_plot, style=style)
        total_mc_passed = self.total_passed_mc_histogram.values
        total_mc_failed = self.total_failed_mc_histogram.values
        total_mc_all = total_mc_passed + total_mc_failed
        total_mc_efficiency = calculate_efficiency(total_mc_passed, total_mc_all)
        total_mc_efficiency_mc_error = calculate_efficiency_error(total_mc_passed, total_mc_all)
        rigidity_bin_centers = self.rigidity_binning.bin_centers
        facecolor = "none" if style == "mc" else None
        efficiency_mc_plot.errorbar(rigidity_bin_centers, total_mc_efficiency, total_mc_efficiency_mc_error, label="Total", marker="o", mfc=facecolor)
        efficiency_mc_plot.legend()
        efficiency_mc_plot.set_ylim(0, 1)
        set_rigidity_ticks(efficiency_mc_plot)
        save_figure(efficiency_mc_figure, resultdir, f"{prefix}_efficiency_mc")
