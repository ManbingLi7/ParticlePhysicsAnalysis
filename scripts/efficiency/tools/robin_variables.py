
import json
import os

import numpy as np
import awkward as ak
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize, newton

from tools.binnings import Binning, Binnings
from tools.constants import ACCEPTANCE_CATEGORIES, MC_PARTICLE_CHARGE_ARRAY, MC_PARTICLE_LABELS, MC_PARTICLE_MASS_ARRAY, MC_PARTICLE_IDS, DEFAULT_RIGIDITY, TRK_LAYER_RADIUS, TRK_LAYER_SIZE_X, TRK_LAYER_SIZE_Y, NAF_INDEX, AGL_INDEX, NAF_SIZE, AGL_SIZE, NAF_TILE_SIZE, AGL_TILE_SIZE, RICH_RADIATOR_X, RICH_RADIATOR_Y, RICH_RADIATOR_Z, RICH_RESOLUTION_NAF, RICH_RESOLUTION_AGL, TRK_N_SENSORS, TRK_SENSOR_SIZE_X, TRK_SENSOR_SIZE_Y
from tools.conversions import calc_mass, calc_rig, calc_beta
from tools.likelihoods import Likelihood
from tools.mvas import MVA
from tools.selection import Selection
from tools.statistics import bethe_bloch, bethe_bloch_pm
from tools.utilities import rigidity_from_beta_resolution
from tools.trackerfeet import read_tracker_feet_file


AC_INNER = ACCEPTANCE_CATEGORIES["inner"]
AC_L1 = ACCEPTANCE_CATEGORIES["l1"]
AC_L9 = ACCEPTANCE_CATEGORIES["l9"]
AC_TRD = ACCEPTANCE_CATEGORIES["trd"]
AC_RICH = ACCEPTANCE_CATEGORIES["rich"]
AC_ECAL = ACCEPTANCE_CATEGORIES["ecal"]


def trk_rigidity_fallback(chunk):
    has_all = chunk.TrkRigidityAll != -10000
    return has_all * chunk.TrkRigidityAll + np.invert(has_all) * chunk.TrkRigidityInner

def trk_rigidity_abs(rigidity_estimator):
    def _trk_rigidity_abs(chunk):
        return np.abs(chunk[rigidity_estimator])
    return _trk_rigidity_abs

def trk_rigidity_identity(rigidity_estimator):
    def _trk_rigidity_identity(chunk):
        return chunk[rigidity_estimator]
    return _trk_rigidity_identity

def calculate_mass(rigidity_estimator, beta_estimator, charge_estimator="TrkCharge"):
    if isinstance(charge_estimator, str):
        get_charge = lambda c: np.round(ak.to_numpy(c[charge_estimator]))
    else:
        get_charge = lambda c: charge_estimator
    def _calculate_mass(chunk):
        rigidity = np.abs(chunk[rigidity_estimator])
        beta = np.minimum(np.abs(ak.to_numpy(chunk[beta_estimator])), 1)
        charge = get_charge(chunk)
        return calc_mass(beta, rigidity, charge)
    return _calculate_mass

def beta_distance(rigidity_estimator, beta_estimator, particle_id):
    hypothesis_mass = MC_PARTICLE_MASS_ARRAY[particle_id]
    hypothesis_charge = MC_PARTICLE_CHARGE_ARRAY[particle_id]
    def _beta_distance(chunk):
        expected_beta = calc_beta(np.abs(chunk[rigidity_estimator]), hypothesis_mass, hypothesis_charge)
        return chunk[beta_estimator] - expected_beta
    return _beta_distance


def expected_beta(rigidity_estimator, particle_id):
    hypothesis_mass = MC_PARTICLE_MASS_ARRAY[particle_id]
    hypothesis_charge = MC_PARTICLE_CHARGE_ARRAY[particle_id]
    def _expected_beta(chunk):
        return calc_beta(np.abs(chunk[rigidity_estimator]), hypothesis_mass, hypothesis_charge)
    return _expected_beta

def min_beta_distance(rigidity_estimator, beta_estimator, particle_ids):
    beta_distances = [beta_distance(rigidity_estimator, beta_estimator, particle_id) for particle_id in particle_ids]
    def _min_beta_distance(chunk):
        return np.min(np.abs([beta_distance(chunk) for beta_distance in beta_distances]), axis=0)
    return _min_beta_distance

def min_beta_distance_if(rigidity_estimator, beta_estimator, particle_ids, condition):
    beta_distances = [beta_distance(rigidity_estimator, beta_estimator, particle_id) for particle_id in particle_ids]
    def _min_beta_distance_if(chunk):
        return np.min(np.abs([beta_distance(chunk) for beta_distance in beta_distances]), axis=0) * chunk[condition]
    return _min_beta_distance_if

def default_mass(rigidity_estimator, charge_estimator="TrkCharge"):
    calc_tof_mass = calculate_mass(rigidity_estimator, "TofBeta", charge_estimator)
    calc_rich_mass = calculate_mass(rigidity_estimator, "RichBeta", charge_estimator)
    def _default_mass(chunk):
        has_rich = chunk.HasRich
        tof_mass = calc_tof_mass(chunk)
        rich_mass = calc_rich_mass(chunk)
        return has_rich * rich_mass + np.invert(has_rich) * tof_mass
    return _default_mass

def default_beta(chunk):
    has_rich = chunk.HasRich
    return has_rich * chunk.RichBeta + np.invert(has_rich) * chunk.TofBeta

def rich_is_agl(chunk):
    return chunk.HasRich & np.invert(chunk.RichIsNaF)


def trd_hit_amplitude_per_pathlength(chunk):
    return chunk.TrdHitAmplitude / chunk.TrdHitPathlength


def trd_trk_distance_x(chunk):
    return chunk.TrkTrackFitCoordXAtTrd - chunk.TrdTrackCoordX

def trd_trk_distance_y(chunk):
    return chunk.TrkTrackFitCoordYAtTrd - chunk.TrdTrackCoordY

def trd_trk_distance(chunk):
    return np.sqrt(trd_trk_distance_x(chunk)**2 + trd_trk_distance_y(chunk)**2)

def trd_trk_angle_distance_x(chunk):
    return np.arctan(chunk.TrkTrackFitDirXZAtTrd) - np.arctan(chunk.TrdTrackDirXZ)

def trd_trk_angle_distance_y(ref_name):
    def _trd_trk_angle_distance_y(chunk):
        ref_sign = np.sign(chunk[ref_name])
        return (np.arctan(chunk.TrkTrackFitDirYZAtTrd) - np.arctan(chunk.TrdTrackDirYZ)) * ref_sign
    return _trd_trk_angle_distance_y

def tof_trk_distance_x(layer):
    def _tof_trk_distance_x(chunk):
        return chunk.TrkTrackFitCoordXAtTof[:,layer] - chunk.TofClusterCoordX[:,layer]
    return _tof_trk_distance_x

def tof_trk_distance_y(layer):
    def _tof_trk_distance_y(chunk):
        return chunk.TrkTrackFitCoordYAtTof[:,layer] - chunk.TofClusterCoordY[:,layer]
    return _tof_trk_distance_y

def tof_trk_distance(layer):
    distance_x = tof_trk_distance_x(layer)
    distance_y = tof_trk_distance_y(layer)
    def _tof_trk_distance(chunk):
        return np.sqrt(distance_x(chunk)**2 + distance_y(chunk)**2)
    return _tof_trk_distance


def tof_upper_charge_from_layers(chunk):
    return (np.abs(chunk.TofChargeInLayer[:,0]) + np.abs(chunk.TofChargeInLayer[:,1])) / 2

def tof_lower_charge_from_layers(chunk):
    return (np.abs(chunk.TofChargeInLayer[:,2]) + np.abs(chunk.TofChargeInLayer[:,3])) / 2


def trk_hits_inside_layer(chunk):
    if len(chunk) == 0:
        return np.empty((0, 9), dtype=bool)
    distance_x = np.abs(chunk.TrkTrackHitCoordX)
    distance_y = np.abs(chunk.TrkTrackHitCoordY)
    sq_radius = distance_x**2 + distance_y**2
    return np.all((distance_x <= np.expand_dims(TRK_LAYER_SIZE_X, axis=0), distance_y <= np.expand_dims(TRK_LAYER_SIZE_Y, axis=0), sq_radius <= np.expand_dims(TRK_LAYER_RADIUS, axis=0)**2), axis=0)

def trk_n_hits_inside_layer_inner(chunk):
    inside = trk_hits_inside_layer(chunk)
    return np.sum(inside[:,1:8], axis=1)

def trk_n_hits_inside_layer(chunk):
    inside = trk_hits_inside_layer(chunk)
    return np.sum(inside, axis=1)

def trk_hit_inside_layer(layer):
    def _trk_hit_inside_layer(chunk):
        return trk_hits_inside_layer(chunk)[:,layer - 1]
    return _trk_hit_inside_layer

def trk_hit_coord_in_double_layer(layers, coord):
    first_layer = layers[0] - 1
    second_layer = layers[1] - 1
    hit_coord_name = f"TrkTrackHitCoord{coord}"
    has_hit_name = f"TrkTrackHasHit{coord}"
    def _trk_hit_coord_in_double_layer(chunk):
        first_coord = chunk[hit_coord_name][:,first_layer]
        second_coord = chunk[hit_coord_name][:,second_layer]
        has_first = ak.values_astype(chunk[has_hit_name][:,first_layer], np.int32)
        has_second = ak.values_astype(chunk[has_hit_name][:,second_layer], np.int32)
        return (first_coord * has_first + second_coord * has_second) / (has_first + has_second + 1e-7)
    return _trk_hit_coord_in_double_layer

def trk_external_layer_hit_pattern(chunk):
    has_hit_l1 = ak.values_astype(chunk.TrkTrackHasHitX[:,0] & chunk.TrkTrackHasHitY[:,0], np.int32)
    has_hit_l9 = ak.values_astype(chunk.TrkTrackHasHitX[:,8] & chunk.TrkTrackHasHitY[:,8], np.int32)
    return has_hit_l1 + 2 * has_hit_l9

def trk_has_hit_in_external_layers(chunk):
    return (chunk.TrkTrackHasHitY[:,0] & chunk.TrkTrackHasHitX[:,0]) | (chunk.TrkTrackHasHitX[:,8] & chunk.TrkTrackHasHitY[:,8])

def trk_n_hits_in_external_layers(chunk):
    return ak.values_astype(chunk.TrkTrackHasHitX[:,0] & chunk.TrkTrackHasHitY[:,0], np.int32) + ak.values_astype(chunk.TrkTrackHasHitX[:,8] & chunk.TrkTrackHasHitY[:,8], np.int32)


def trk_hit_radius(coord_x, coord_y):
    def _trk_hit_radius(chunk):
        return np.sqrt(chunk[coord_x]**2 + chunk[coord_y]**2)
    return _trk_hit_radius


def trk_foot_distance():
    tracker_feet_x, tracker_feet_y = read_tracker_feet_file()
    feet_x = ak.Array([np.array(tracker_feet_x[layer]) for layer in range(9)])
    feet_y = ak.Array([np.array(tracker_feet_y[layer]) for layer in range(9)])
    assert ak.all(ak.num(feet_x) == ak.num(feet_y))
    def _trk_foot_distance(chunk):
        if len(chunk) == 0:
            return np.zeros((0, 9))
        min_r = np.zeros((len(chunk), 9))
        for layer in range(9):
            pos_x = ak.to_regular(chunk.TrkTrackFitCoordX[:,layer])
            pos_y = ak.to_regular(chunk.TrkTrackFitCoordY[:,layer])
            delta_x = pos_x - feet_x[layer,np.newaxis]
            delta_y = pos_y - feet_y[layer,np.newaxis]
            delta_r = np.sqrt(delta_x**2 + delta_y**2)
            min_r[:,layer] = ak.min(delta_r, axis=1)
        return min_r
    return _trk_foot_distance

def min_trk_foot_distance():
    _trk_foot_distance = trk_foot_distance()
    def _min_trk_foot_distance(chunk):
        return ak.min(_trk_foot_distance(chunk), axis=1)
    return _min_trk_foot_distance

def number_of_hit_trk_feet(feet_size=0.5):
    _trk_foot_distance = trk_foot_distance()
    def _number_of_hit_trk_feet(chunk):
        return ak.sum(_trk_foot_distance(chunk) <= feet_size, axis=1)
    return _number_of_hit_trk_feet

def trk_distance_to_sensor_edge(layer, coord="xy"):
    n_sensors = TRK_N_SENSORS[layer - 1]
    n_strips = len(n_sensors)
    edges_y = (np.arange(n_strips + 1) - n_strips / 2) * TRK_SENSOR_SIZE_Y
    binning_y = Binning(edges_y)
    offset_x = (np.array(n_sensors) % 2) * TRK_SENSOR_SIZE_X / 2
    offset_y = (n_strips % 2) * TRK_SENSOR_SIZE_Y / 2

    def _trk_distance_to_sensor_edge(chunk):
        pos_x = chunk.TrkTrackFitCoordX[:,layer - 1]
        pos_y = chunk.TrkTrackFitCoordY[:,layer - 1]
        shifted_y = np.abs(pos_y - offset_y)
        rounded_y = _round(shifted_y / TRK_SENSOR_SIZE_Y) * TRK_SENSOR_SIZE_Y
        delta_y = np.abs(shifted_y - rounded_y)
        indices_x = binning_y.get_indices(pos_y, with_overflow=False)
        offsets_x = offset_x[indices_x - 1]
        shifted_x = np.abs(pos_x - offsets_x)
        rounded_x = _round(shifted_x / TRK_SENSOR_SIZE_X) * TRK_SENSOR_SIZE_X
        delta_x = np.abs(shifted_x - rounded_x)
        if coord == "x":
            return delta_x
        elif coord == "y":
            return delta_y
        return np.minimum(delta_x, delta_y)
    return _trk_distance_to_sensor_edge


def trk_track_hits_naf(chunk):
    x = np.abs(chunk.TrkTrackFitCoordXAtRich - RICH_RADIATOR_X)
    y = np.abs(chunk.TrkTrackFitCoordYAtRich - RICH_RADIATOR_Y)
    return (x <= NAF_SIZE) & (y <= NAF_SIZE)

def trk_track_hits_rich(chunk):
    x = np.abs(chunk.TrkTrackFitCoordXAtRich - RICH_RADIATOR_X)
    y = np.abs(chunk.TrkTrackFitCoordYAtRich - RICH_RADIATOR_Y)
    return (x <= AGL_SIZE) & (y <= AGL_SIZE)

def trk_track_hits_agl(chunk):
    x = np.abs(chunk.TrkTrackFitCoordXAtRich - RICH_RADIATOR_X)
    y = np.abs(chunk.TrkTrackFitCoordYAtRich - RICH_RADIATOR_Y)
    return (x <= AGL_SIZE) & (y <= AGL_SIZE) & ((x > NAF_SIZE) | (y > NAF_SIZE))

def trk_track_distance_to_rich_tile_border(chunk):
    coord_x = chunk.TrkTrackFitCoordXAtRich - RICH_RADIATOR_X
    coord_y = chunk.TrkTrackFitCoordYAtRich - RICH_RADIATOR_Y
    naf_rel_x = coord_x / NAF_TILE_SIZE
    naf_rel_y = coord_y / NAF_TILE_SIZE
    naf_closest_x = _round(naf_rel_x)
    naf_closest_y = _round(naf_rel_y)
    naf_distance_x = np.abs((naf_rel_x - naf_closest_x) * NAF_TILE_SIZE)
    naf_distance_y = np.abs((naf_rel_y - naf_closest_y) * NAF_TILE_SIZE)
    naf_distance = np.minimum(naf_distance_x, naf_distance_y)
    agl_rel_x = coord_x / AGL_TILE_SIZE
    agl_rel_y = coord_y / AGL_TILE_SIZE
    agl_closest_x = _round(agl_rel_x - 0.5) + 0.5
    agl_closest_y = _round(agl_rel_y - 0.5) + 0.5
    agl_distance_x = np.abs((agl_rel_x - agl_closest_x) * AGL_TILE_SIZE)
    agl_distance_y = np.abs((agl_rel_y - agl_closest_y) * AGL_TILE_SIZE)
    agl_distance = np.minimum(agl_distance_x, agl_distance_y)
    return chunk.TrkTrackHitsNaF * naf_distance + np.invert(chunk.TrkTrackHitsNaF) * agl_distance

def trk_track_distance_to_naf_corners(chunk):
    abs_coord_x = np.abs(chunk.TrkTrackFitCoordXAtRich - RICH_RADIATOR_X)
    abs_coord_y = np.abs(chunk.TrkTrackFitCoordYAtRich - RICH_RADIATOR_Y)
    distance_x = abs_coord_x - NAF_SIZE
    distance_y = abs_coord_y - NAF_SIZE
    return np.sqrt(distance_x**2 + distance_y**2)


def cherenkov_threshold_ratio_if_no_ring(particles, rigidity_estimator):
    particle_ids = [MC_PARTICLE_IDS[particle] for particle in particles]
    hypotheses = [(MC_PARTICLE_MASS_ARRAY[particle_id], MC_PARTICLE_CHARGE_ARRAY[particle_id]) for particle_id in particle_ids]
    threshold_rig_naf = max((calc_rig(1 / NAF_INDEX, mass, charge) for (mass, charge) in hypotheses))
    threshold_rig_agl = max((calc_rig(1 / AGL_INDEX, mass, charge) for (mass, charge) in hypotheses))
    def _cherenkov_threshold_ratio_if_no_ring(chunk):
        has_ring = chunk.HasRich
        away_from_tile_border = (chunk.TrkTrackDistanceToRichTileBorder >= 1) & (np.invert(chunk.TrkTrackHitsNaF) | (chunk.TrkTrackDistanceToNaFCorners > 0.33))
        ref_rig = threshold_rig_naf * chunk.TrkTrackHitsNaF + threshold_rig_agl * np.invert(chunk.TrkTrackHitsNaF)
        return (np.abs(chunk[rigidity_estimator]) / ref_rig) * np.invert(has_ring) * away_from_tile_border
    return _cherenkov_threshold_ratio_if_no_ring


def cherenkov_threshold_ratio_if_ring(particles, rigidity_estimator):
    particle_ids = [MC_PARTICLE_IDS[particle] for particle in particles]
    hypotheses = [(MC_PARTICLE_MASS_ARRAY[particle_id], MC_PARTICLE_CHARGE_ARRAY[particle_id]) for particle_id in particle_ids]
    threshold_rig_naf = min((calc_rig(1 / NAF_INDEX, mass, charge) for (mass, charge) in hypotheses))
    threshold_rig_agl = min((calc_rig(1 / AGL_INDEX, mass, charge) for (mass, charge) in hypotheses))
    def _cherenkov_threshold_ratio_if_no_ring(chunk):
        has_ring = chunk.HasRich
        ref_rig = threshold_rig_naf * chunk.TrkTrackHitsNaF + threshold_rig_agl * np.invert(chunk.TrkTrackHitsNaF)
        return (np.abs(chunk[rigidity_estimator]) / ref_rig) * has_ring + 10 * np.invert(has_ring)
    return _cherenkov_threshold_ratio_if_no_ring


def rich_ring_as_expected(particle_ids, rigidity_estimator):
    hypotheses = [(MC_PARTICLE_MASS_ARRAY[particle_id], MC_PARTICLE_CHARGE_ARRAY[particle_id]) for particle_id in particle_ids]
    def _rich_ring_as_expected(chunk):
        has_rich_ring = chunk.HasRich
        min_expected_beta = np.min([calc_beta(np.abs(chunk[rigidity_estimator]) * 0.75, mass, charge) for mass, charge in hypotheses], axis=0)
        max_expected_beta = np.max([calc_beta(np.abs(chunk[rigidity_estimator]) * 2, mass, charge) for mass, charge in hypotheses], axis=0)
        threshold_beta = chunk.TrkTrackHitsNaF / NAF_INDEX + chunk.TrkTrackHitsAGL / AGL_INDEX
        return (has_rich_ring & (max_expected_beta >= threshold_beta)) + (np.invert(has_rich_ring) & ((min_expected_beta <= threshold_beta) | np.invert(chunk.TrkTrackHitsRich) | (chunk.TrkTrackDistanceToRichTileBorder <= 1) | (chunk.TrkTrackHitsNaF & (chunk.TrkTrackDistanceToNaFCorners <= 0.5))))
    return _rich_ring_as_expected


def should_have_rich_beta(particle_name):
    def _should_have_rich_beta(chunk):
        expected_beta = chunk[f"ExpectedBeta{particle_name}"]
        return (chunk.TrkTrackHitsAGL & (expected_beta >= 1 / AGL_INDEX)) | (chunk.TrkTrackHitsNaF & (expected_beta >= 1 / NAF_INDEX))
    return _should_have_rich_beta

def rich_beta_could_resolve(rigidity_estimator, particle_name_1, particle_name_2):
    max_rig_naf = rigidity_from_beta_resolution("NaF", particle_name_1, particle_name_2)
    max_rig_agl = rigidity_from_beta_resolution("AGL", particle_name_1, particle_name_2)
    def _rich_beta_could_resolve(chunk):
        return (chunk.TrkTrackHitsAGL & (np.abs(chunk[rigidity_estimator]) <= max_rig_agl)) | (chunk.TrkTrackHitsNaF & (np.abs(chunk[rigidity_estimator]) <= max_rig_naf))
    return _rich_beta_could_resolve


def rich_beta_could_resolve_ratio(rigidity_estimator, particle_name_1, particle_name_2):
    max_rig_naf = rigidity_from_beta_resolution("NaF", particle_name_1, particle_name_2)
    max_rig_agl = rigidity_from_beta_resolution("AGL", particle_name_1, particle_name_2)
    def _rich_beta_could_resolve(chunk):
        return (chunk.TrkTrackHitsAGL * (np.abs(chunk[rigidity_estimator]) / max_rig_agl)) + (chunk.TrkTrackHitsNaF * (np.abs(chunk[rigidity_estimator]) / max_rig_naf))
    return _rich_beta_could_resolve

def rich_beta_could_resolve_ratio_naf_only(rigidity_estimator, particle_name_1, particle_name_2):
    max_rig_naf = rigidity_from_beta_resolution("NaF", particle_name_1, particle_name_2)
    #max_rig_agl = rigidity_from_beta_resolution("AGL", particle_name_1, particle_name_2)
    def _rich_beta_could_resolve_naf_only(chunk):
        return chunk.TrkTrackHitsNaF * (np.abs(chunk[rigidity_estimator]) / max_rig_naf)
    return _rich_beta_could_resolve_naf_only
 


def rich_photo_electron_ratio(numerator, denominator):
    def _rich_photo_electron_ratio(chunk):
        return chunk[numerator] / (chunk[denominator] + 1e-7)
    return _rich_photo_electron_ratio


def calculate_asymmetry(name1, name2, ref_name, abs=False):
    def _calculate_asymmetry(chunk):
        r1 = chunk[f"TrkRigidity{name1}"]
        r2 = chunk[f"TrkRigidity{name2}"]
        s1 = 1 / (r1 / (r1 != DEFAULT_RIGIDITY))
        s2 = 1 / (r2 / (r2 != DEFAULT_RIGIDITY))
        result = (s1 - s2) / (s1 + s2 + ((r1 == DEFAULT_RIGIDITY) & (r2 == DEFAULT_RIGIDITY)))
        if abs:
            result = np.abs(result)
        return result
    return _calculate_asymmetry

def calculate_matching(name1, name2, ref_name, square=False):
    def _calculate_matching(chunk):
        ref_sign = np.sign(chunk[ref_name])
        r1 = chunk[f"TrkRigidity{name1}"]
        r2 = chunk[f"TrkRigidity{name2}"]
        s1 = 1 / (r1 / (r1 != DEFAULT_RIGIDITY))
        s2 = 1 / (r2 / (r2 != DEFAULT_RIGIDITY))
        sig1 = chunk[f"TrkInverseRigidityError{name1}"]
        sig2 = chunk[f"TrkInverseRigidityError{name2}"]
        result = (s1 - s2) / np.sqrt(sig1**2 + sig2**2) * ref_sign
        if square:
            result = result**2
        return result
    return _calculate_matching

def calculate_relative_error(name):
    def _calculate_relative_error(chunk):
        r = np.abs(1 / chunk[f"TrkRigidity{name}"])
        s = chunk[f"TrkInverseRigidityError{name}"]
        return s / r
    return _calculate_relative_error

def calculate_min_charge_xy(charge_names):
    functions = [calculate_min_charge(charge_name) for charge_name in charge_names]
    def _calculate_min_charge_xy(chunk):
        return np.min([function(chunk) for function in functions], axis=0)
    return _calculate_min_charge_xy

def calculate_max_charge_xy(charge_names):
    functions = [calculate_max_charge(charge_name) for charge_name in charge_names]
    def _calculate_max_charge_xy(chunk):
        return np.max([function(chunk) for function in functions], axis=0)
    return _calculate_max_charge_xy

def calculate_min_charge(charge_name):
    def _calculate_min_charge(chunk):
        charges = chunk[charge_name]
        return ak.min(charges[charges > 0], axis=1)
    return _calculate_min_charge

def calculate_max_charge(charge_name):
    def _calculate_max_charge(chunk):
        charges = chunk[charge_name]
        return ak.max(charges[charges > 0], axis=1)
    return _calculate_max_charge

def calculate_charge_rms(charge_name):
    def _calculate_charge_rms(chunk):
        charges = chunk[charge_name]
        return ak.std(charges[charges > 0], axis=1)
    return _calculate_charge_rms

def calculate_inner_tracker_charge_rms(chunk):
    nhits = chunk.TrkNLayersInnerY
    return chunk.TrkChargeError * np.minimum(np.sqrt(nhits - 1), 1)


def calculate_max_relative_delta_sagitta(chunk):
    rigidity_without = chunk.TrkRigiditiesWithoutHit
    rigidity_without = rigidity_without[(rigidity_without > -9999) & (rigidity_without != 0)]
    sagitta_without = 1 / rigidity_without
    sagitta = 1 / chunk.TrkRigidityInner
    max_delta = ak.max(np.abs(sagitta_without - sagitta), axis=1)
    mask = sagitta != 0
    return mask * (max_delta / (np.abs(sagitta) + 1e-7))


def trk_fit_residuals_x(chunk):
    return (chunk.TrkTrackFitCoordX - chunk.TrkTrackHitCoordX) * 1e4

def trk_fit_residuals_y(chunk):
    return (chunk.TrkTrackFitCoordY - chunk.TrkTrackHitCoordY) * 1e4

def trk_has_good_hit_x(chunk):
    return np.abs(chunk.TrkTrackFitCoordX - chunk.TrkTrackHitCoordX) * 1e4 < 75

def trk_has_good_hit_y(chunk):
    return np.abs(chunk.TrkTrackFitCoordY - chunk.TrkTrackHitCoordY) * 1e4 < 30

def trk_has_good_hit_xy(chunk):
    return (np.abs(chunk.TrkTrackFitCoordX - chunk.TrkTrackHitCoordX) * 1e4 < 75) & (np.abs(chunk.TrkTrackFitCoordY - chunk.TrkTrackHitCoordY) * 1e4 < 30)

def trk_nlayers_good_xy(chunk):
    return ak.sum(trk_has_good_hit_xy(chunk), axis=1)

def trk_nlayers_inner_good_xy(chunk):
    return ak.sum(trk_has_good_hit_xy(chunk)[:,1:8], axis=1)

def trk_has_hit_xy(chunk):
    return chunk.TrkTrackHasHitX & chunk.TrkTrackHasHitY

def trk_has_hit_in_layer_x(layer):
    def _trk_has_hit_in_layer_x(chunk):
        return chunk.TrkTrackHasHitX[:,layer - 1]
    return _trk_has_hit_in_layer_x

def trk_has_hit_in_layer_y(layer):
    def _trk_has_hit_in_layer_y(chunk):
        return chunk.TrkTrackHasHitY[:,layer - 1]
    return _trk_has_hit_in_layer_y

def trk_has_hit_in_layer_xy(layer):
    def _trk_has_hit_in_layer_xy(chunk):
        return chunk.TrkTrackHasHitX[:,layer - 1] & chunk.TrkTrackHasHitY[:,layer - 1]
    return _trk_has_hit_in_layer_xy

def trk_hit_coord_in_layer(layer, dimension):
    branch = f"TrkTrackHitCoord{dimension}"
    def _trk_hit_coord_in_layer(chunk):
        return chunk[branch][:,layer - 1]
    return _trk_hit_coord_in_layer

def trk_fit_coord_in_layer(layer, dimension):
    branch = f"TrkTrackFitCoord{dimension}"
    def _trk_hit_coord_in_layer(chunk):
        return chunk[branch][:,layer - 1]
    return _trk_hit_coord_in_layer

def trk_abs_fit_coord_in_layer(layer, dimension):
    branch = f"TrkTrackFitCoord{dimension}"
    def _trk_abs_hit_coord_in_layer(chunk):
        return np.abs(chunk[branch][:,layer - 1])
    return _trk_abs_hit_coord_in_layer



def trk_has_layer_charge_x(layer):
    def _trk_has_layer_charge_x(chunk):
        return chunk["TrkLayerChargesX"][:,layer - 1] > 0
    return _trk_has_layer_charge_x

def trk_has_layer_charge_y(layer):
    def _trk_has_layer_charge_y(chunk):
        return chunk["TrkLayerChargesY"][:,layer - 1] > 0
    return _trk_has_layer_charge_y

def trk_has_layer_charge_x_and_y(layer):
    def _trk_has_layer_charge_x_and_y(chunk):
        return (chunk["TrkLayerChargesX"][:,layer - 1] > 0) & (chunk["TrkLayerChargesY"][:,layer - 1] > 0)
    return _trk_has_layer_charge_x_and_y

def trk_has_layer_charge_x_or_y(layer):
    def _trk_has_layer_charge_x_or_y(chunk):
        return (chunk["TrkLayerChargesX"][:,layer - 1] > 0) | (chunk["TrkLayerChargesY"][:,layer - 1] > 0)
    return _trk_has_layer_charge_x_or_y


def trk_layer_charge_yj(layer):
    def _trk_layer_charge_yj(chunk):
        return chunk.TrkLayerChargesYJ[:,layer - 1]
    return _trk_layer_charge_yj

def trk_layer_charge_x(layer):
    def _trk_layer_charge_x(chunk):
        return chunk["TrkLayerChargesX"][:,layer - 1]
    return _trk_layer_charge_x

def trk_layer_charge_y(layer):
    def _trk_layer_charge_y(chunk):
        return chunk["TrkLayerChargesY"][:,layer - 1]
    return _trk_layer_charge_y

def trk_layer_charge(layer):
    def _trk_layer_charge(chunk):
        charge_x = chunk.TrkLayerChargesX[:,layer - 1]
        charge_y = chunk.TrkLayerChargesY[:,layer - 1]
        return (charge_x + charge_y) / (ak.values_astype(charge_x != 0, np.float32) + ak.values_astype(charge_y != 0, np.float32) + 1e-7)
    return _trk_layer_charge

def trk_layer_charge_difference(layer):
    has_layer_charge = trk_has_layer_charge_x_and_y(layer)
    def _trk_layer_charge_difference(chunk):
        diff = np.zeros_like(chunk, dtype=np.float32)
        x_and_y = has_layer_charge(chunk)
        diff[x_and_y] = chunk["TrkLayerChargesY"][x_and_y, layer - 1] - chunk["TrkLayerChargesX"][x_and_y, layer - 1]
        return diff
    return _trk_layer_charge_difference

def trk_max_layer_charge_difference(chunk):
    qx = chunk["TrkLayerChargesX"]
    qy = chunk["TrkLayerChargesY"]
    diff = ak.mask(qy - qx, (qx > 0) & (qy > 0))
    return ak.max(diff, axis=1)


def trk_layer_deposited_energy_x(layer):
    def _trk_deposited_energy_x(chunk):
        return chunk.TrkTrackHitDepositedEnergyX[:,layer-1]
    return _trk_deposited_energy_x

def trk_layer_deposited_energy_y(layer):
    def _trk_deposited_energy_y(chunk):
        return chunk.TrkTrackHitDepositedEnergyY[:,layer-1]
    return _trk_deposited_energy_y


#BETHE_BLOCH_PARAMETERS_MC = (0.00128667, -0.03131471, 0.06569076)
BETHE_BLOCH_PARAMETERS_MC = (0.00116977, -0.04183937, 0.0657267)
BETHE_BLOCH_PARAMETERS_ISS = (0.00081383, -0.03405449, 0.04474442)

def mass_from_deposited_energy(deposited_energy_variable, rigidity_estimator, charge, bethe_bloch_parameters):
    def _mass_from_deposited_energy(chunk):
        edep = chunk[deposited_energy_variable]
        rigidity = np.abs(chunk[rigidity_estimator])
        momentum = rigidity * charge
        def _mass_objective(args, index):
            mass, = args
            l1 = (bethe_bloch_pm(momentum[index], mass, charge, *bethe_bloch_parameters) - edep[index])**2
            return l1
        for index in range(len(rigidity)):
            if momentum[index] > 10000 or momentum[index] < 2 or momentum[index] > 6:
                continue
            result = minimize(_mass_objective, (10,), (index,), bounds=((0, 100),))
            print(index, momentum[index], edep[index], result.success, result.x)
        guess = np.ones(len(rigidity)) * 4
        dlm_guess = np.zeros(len(rigidity))
        return minimize(_mass_objective, (guess, dlm_guess))
    return _mass_from_deposited_energy

def mass_from_deposited_energy_newton(deposited_energy_variable, rigidity_estimator, charge, bethe_bloch_parameters):
    def _mass_from_deposited_energy(chunk):
        edep = chunk[deposited_energy_variable]
        rigidity = np.abs(chunk[rigidity_estimator])
        momentum = rigidity * charge
        def _mass_objective(mass):
            return bethe_bloch_pm(momentum, mass, charge, *bethe_bloch_parameters) - edep
        guess = np.ones(len(rigidity))
        return newton(_mass_objective, guess, disp=False)
    return _mass_from_deposited_energy


def trk_mass_from_deposited_energy(rigidity_estimator, charge):
    _mass_func_iss = mass_from_deposited_energy("TrkTrackMeanDepositedEnergyPerPathlength", rigidity_estimator, charge, BETHE_BLOCH_PARAMETERS_ISS)
    _mass_func_mc = mass_from_deposited_energy("TrkTrackMeanDepositedEnergyPerPathlength", rigidity_estimator, charge, BETHE_BLOCH_PARAMETERS_MC)
    def _trk_mass(chunk):
        if np.all(chunk.McParticleID != 0):
            return _mass_func_mc(chunk)
        return _mass_func_iss(chunk)
    return _trk_mass
    

def trk_double_layer_pattern(pattern_name):
    def _trk_double_layer_pattern(chunk):
        pattern = chunk[pattern_name]
        l2 = (pattern & 0b100) > 0
        l34 = (pattern & 0b11000) > 0
        l56 = (pattern & 0b1100000) > 0
        l78 = (pattern & 0b110000000) > 0
        return l2 + l34 * 2 + l56 * 4 + l78 * 8
    return _trk_double_layer_pattern


def trk_n_layers_inner_upper(pattern_name):
    def _trk_n_layers_inner_upper(chunk):
        pattern = chunk[pattern_name]
        return (ak.values_astype((pattern & 0b100) > 0, "int32")
            + ak.values_astype((pattern & 0b1000) > 0, "int32")
            + ak.values_astype((pattern & 0b10000) > 0, "int32")
            + ak.values_astype((pattern & 0b100000) > 0, "int32"))
    return _trk_n_layers_inner_upper

def trk_n_layers_inner_lower(pattern_name):
    def _trk_n_layers_inner_lower(chunk):
        pattern = chunk[pattern_name]
        return (ak.values_astype((pattern & 0b1000000) > 0, "int32")
            + ak.values_astype((pattern & 0b10000000) > 0, "int32")
            + ak.values_astype((pattern & 0b100000000) > 0, "int32"))
    return _trk_n_layers_inner_lower


def trk_log_chi2(chi2):
    def _trk_log_chi2(chunk):
        return np.log10(np.maximum(chunk[chi2], 1e-3))
    return _trk_log_chi2
    

def cutoff_factor(variable, binning, cutoff, angle):
    def _above_cutoff(chunk):
        rigidity = np.abs(chunk[variable])
        cutoff_rigidity = chunk[f"{cutoff}Cutoff{angle}"]
        indices = np.digitize(ak.to_numpy(rigidity), binning.edges)
        lower_edge_rigidity = binning.edges[indices - 1]
        return lower_edge_rigidity / cutoff_rigidity
    return _above_cutoff


def mc_rigidity(chunk):
    return chunk.McMomentum / MC_PARTICLE_CHARGE_ARRAY[chunk.McParticleID]

def mc_beta(chunk):
    return calc_beta(mc_rigidity(chunk), MC_PARTICLE_MASS_ARRAY[chunk.McParticleID], MC_PARTICLE_CHARGE_ARRAY[chunk.McParticleID])

def mc_mass(chunk):
    return MC_PARTICLE_MASS_ARRAY[chunk.McParticleID]

def mc_abs_rigidity(chunk):
    return np.abs(mc_rigidity(chunk))


def abs_difference_to_true_value(true_branch, measured_branch):
    def _abs_difference_to_true_value(chunk):
        return np.abs(chunk[measured_branch] - chunk[true_branch])
    return _abs_difference_to_true_value

def difference_to_true_value(true_branch, measured_branch):
    def _difference_to_true_value(chunk):
        return chunk[measured_branch] - chunk[true_branch]
    return _difference_to_true_value


def rich_beta_resolution_is_good(max_delta_naf, max_delta_agl):
    def _rich_beta_resolution_is_good(chunk):
        return (chunk.TrkTrackHitsNaF & (chunk.AbsRichBetaMinusTrueBeta <= max_delta_naf)) | (chunk.TrkTrackHitsAGL & (chunk.AbsRichBetaMinusTrueBeta < max_delta_agl))
    return _rich_beta_resolution_is_good


def track_angle(chunk):
    dir_xz = chunk.TrkTrackFitDirXZ[:,0]
    dir_yz = chunk.TrkTrackFitDirYZ[:,0]
    dir = np.sqrt(dir_xz**2 + dir_yz**2)
    return np.arctan(dir) * 180 / np.pi

def track_angle_xz(chunk):
    dir_xz = chunk.TrkTrackFitDirXZ[:,0]
    return np.arctan(dir_xz) * 180 / np.pi

def track_angle_yz(chunk):
    dir_yz = chunk.TrkTrackFitDirYZ[:,0]
    return np.arctan(dir_yz) * 180 / np.pi

def track_angle_phi(chunk):
    dir_xz = chunk.TrkTrackFitDirXZ[:,0]
    dir_yz = chunk.TrkTrackFitDirYZ[:,0]
    return np.arctan2(dir_yz, dir_xz) * 180 / np.pi

def trk_fit_angle_at_layer(layer):
    def _trk_fit_angle(chunk):
        dir_xz = chunk.TrkTrackFitDirXZ[:,layer - 1]
        dir_yz = chunk.TrkTrackFitDirYZ[:,layer - 1]
        dir = np.sqrt(dir_xz**2 + dir_yz**2)
        return np.arctan(dir)
    return _trk_fit_angle

def trk_fit_pathlength_in_layer(layer):
    angle_name = f"TrkTrackFitAngleL{layer}"
    nominal_depth = 1 # todo
    def _tracker_layer_effective_depth(chunk):
        angle = chunk[angle_name]
        hypothenuse = nominal_depth / np.cos(angle)
        return hypothenuse
    return _tracker_layer_effective_depth

def trk_energy_deposition_per_pathlength(layer, orientation):
    energy_deposition_name = f"TrkTrackHitDepositedEnergy{orientation}L{layer}"
    pathlength_name = f"TrkTrackFitPathlengthL{layer}"
    def _trk_energy_deposition_per_pathlength(chunk):
        return chunk[energy_deposition_name] / chunk[pathlength_name]
    return _trk_energy_deposition_per_pathlength

def trk_mean_energy_deposition_per_pathlength(chunk):
    total_deposited_energy_x = np.sum((chunk[f"TrkTrackHitDepositedEnergyPerPathlengthXL{layer}"] for layer in range(1, 10)), axis=1)
    total_deposited_energy_y = np.sum((chunk[f"TrkTrackHitDepositedEnergyPerPathlengthYL{layer}"] for layer in range(1, 10)), axis=1)
    total_hits_x = np.sum(((chunk[f"TrkTrackHitDepositedEnergyPerPathlengthXL{layer}"] > 0) for layer in range(1, 10)), axis=1)
    total_hits_y = np.sum(((chunk[f"TrkTrackHitDepositedEnergyPerPathlengthYL{layer}"] > 0) for layer in range(1, 10)), axis=1)
    return (total_deposited_energy_x + total_deposited_energy_y) / (total_hits_x + total_hits_y)


def has_trigger(chunk):
    # ignoring gamma trigger
    return chunk.TriggerFlags & 0x5f > 0

def has_physics_trigger(chunk):
    return chunk.TriggerFlags & 0x1e > 0

def has_unbiased_trigger(chunk):
    return chunk.TriggerFlags & 0x41 > 0

def has_unbiased_tof_trigger(chunk):
    return chunk.TriggerFlags & 0x1 > 0

def unbiased_trigger_weight(chunk):
    p = np.zeros(len(chunk))
    p[chunk.TriggerFlags & 0x1 > 0] += 0.01
    p[chunk.TriggerFlags & 0x40 > 0] += 0.001
    return 1 / p


def trd_log_likelihood_ratio(numerator, denominator):
    num_key = f"TrdLlh{numerator}"
    denom_key = f"TrdLlh{denominator}"
    def _trd_log_likelihood_ratio(chunk):
        return np.log(chunk[denom_key]) - np.log(chunk[num_key])
    return _trd_log_likelihood_ratio


TRD_MEDIAN_CHARGE_PARAMETERS = (549.69, 2.126, 1.493, 4094.6, -0.1833)

def charge_to_trd_amplitude(charge, slope, curvature, ref_charge, cutoff, cutoff_slope):
    return np.minimum(slope * np.log(1 + np.exp((charge - ref_charge) * curvature)) / curvature, cutoff * (1 - np.exp(cutoff_slope * charge)))

def trd_amplitude_to_charge(amplitude, slope, curvature, ref_charge, cutoff, cutoff_slope):
    return np.maximum(np.log(np.exp(amplitude * curvature / slope) - 1) / curvature + ref_charge, np.log(1 - amplitude / cutoff) / cutoff_slope)


def _round(array):
    return ak.values_astype(array + 0.5, "int32")

def trd_median_charge():
    calc_median = trd_median("TrdHitAmplitude")
    def _trd_median_charge(chunk):
        return trd_amplitude_to_charge(calc_median(chunk), *TRD_MEDIAN_CHARGE_PARAMETERS)
    return _trd_median_charge


def trd_median_charge_12(rigidity_estimator):
    with open(os.path.join(os.environ["ANTIMATTERSEARCH"], "data", "TrdCharge.json")) as parameter_file:
        parameters = json.load(parameter_file)
    x1 = parameters["1"]["x"]
    y1 = parameters["1"]["y"]
    x2 = parameters["2"]["x"]
    y2 = parameters["2"]["y"]
    
    spline_q1 = UnivariateSpline(np.log(x1), y1, ext="extrapolate", s=5)
    spline_q2 = UnivariateSpline(np.log(x2), y2, ext="extrapolate", s=5)

    def _trd_median_charge(chunk):
        amp = np.sqrt(chunk.TrdMedianHitAmplitudePerPathlength)
        rig = np.abs(chunk[rigidity_estimator])
        amp_q1 = spline_q1(np.log(rig))
        amp_q2 = spline_q2(np.log(rig))
        return (amp - amp_q1) / (amp_q2 - amp_q1) + 1
    return _trd_median_charge


def trd_median(variable):
    def _trd_median(chunk):
        if len(chunk) == 0:
            return np.zeros((0,))
        hit_sel = chunk.TrdHitDistanceToTrack < 0.25
        sorted_hits = ak.sort(chunk[variable][hit_sel], axis=1)
        nhits = ak.to_numpy(ak.num(sorted_hits))
        if np.all(nhits == 0):
            return np.zeros((len(chunk),))
        all_hits = ak.flatten(sorted_hits)
        start_indices = np.minimum(np.cumsum(nhits) - nhits, len(all_hits) - 1)
        index_low = _round((nhits - 1.25) / 2)
        index_high = _round((nhits - 0.25) / 2)
        median = (all_hits[start_indices + index_low] + all_hits[start_indices + index_high]) / 2
        return median * (nhits > 0)
    return _trd_median

def trd_nhits_on_track(chunk):
    return ak.sum(chunk.TrdHitDistanceToTrack < 0.25, axis=1)


def energy_over_rigidity(rigidity_estimator):
    def _energy_over_rigidity(chunk):
        return chunk.EcalEnergyDeposited / np.abs(chunk[rigidity_estimator])
    return _energy_over_rigidity

def log_energy_over_rigidity(rigidity_estimator):
    def _energy_over_rigidity(chunk):
        return np.log(chunk.EcalEnergyDeposited / np.abs(chunk[rigidity_estimator]))
    return _energy_over_rigidity

def total_energy(rigidity_estimator, particle_id):
    mass = MC_PARTICLE_MASS_ARRAY[particle_id]
    charge = MC_PARTICLE_CHARGE_ARRAY[particle_id]
    def _total_energy(chunk):
        momentum = chunk[rigidity_estimator] * charge
        return np.sqrt(momentum**2 + mass**2)
    return _total_energy


def backtracing_from_space_or_mc(chunk):
    return (chunk.BacktracingStatus == 1) | (chunk.McParticleID != 0)


class LazyMVA:
    def __init__(self, load_function):
        self.load_function = load_function
        self.mva = None

    def predict(self, *args, **kwargs):
        if self.mva is None:
            self.mva = self.load_function()
        return self.mva.predict(*args, **kwargs)

    def predict_as_efficiency(self, *args, **kwargs):
        if self.mva is None:
            self.mva = self.load_function()
        return self.mva.predict_as_efficiency(*args, **kwargs)


class LazySelection:
    def __init__(self, selection_config, rigidity_estimator, binnings, config, workdir, labelling):
        self.selection = None
        self.selection_config = selection_config
        self.rigidity_estimator = rigidity_estimator
        self.binnings = binnings
        self.config = config
        self.workdir = workdir
        self.labelling = labelling

    def load(self):
        self.selection = Selection.load(self.selection_config, rigidity_estimator=self.rigidity_estimator, binnings=self.binnings, config=self.config, workdir=self.workdir, labelling=self.labelling)

    def apply(self, chunk):
        if self.selection is None:
            self.load()
        if len(chunk) == 0:
            return np.zeros(0, dtype=np.bool)
        return self.selection.apply(chunk)


class DerivedVariables:
    def __init__(self, config=None, workdir=None, rigidity_estimator=None):
        self.functions = None
        self.dependencies = None
        self.rigidity_estimator = rigidity_estimator
        if config is not None:
            self.initialize(config, workdir, rigidity_estimator)

    def initialize(self, config, workdir, rigidity_estimator):
        binnings = Binnings((config, workdir))
        labelling = VariableLabels(config=config, workdir=workdir, rigidity_estimator=rigidity_estimator)
        self.functions = {
            "TrkRigidityFallback": trk_rigidity_fallback,
            "TrdTrkDistanceX": trd_trk_distance_x,
            "TrdTrkDistanceY": trd_trk_distance_y,
            "TrdTrkDistance": trd_trk_distance,
            "TrdTrkAngleDistanceXZ": trd_trk_angle_distance_x,
            "TrdTrkAngleDistanceYZ": trd_trk_angle_distance_y(rigidity_estimator),
            "TrkMatchingChoutkoVsKalmanAll": calculate_matching("All", "KalmanAll", rigidity_estimator),
            "TrkMatchingChoutkoVsKalmanInner": calculate_matching("Inner", "KalmanInner", rigidity_estimator),
            "TrkMatchingChoutkoVsAlcarazAll": calculate_matching("All", "AlcarazAll", rigidity_estimator),
            "TrkMatchingChoutkoVsAlcarazInner": calculate_matching("Inner", "AlcarazInner", rigidity_estimator),
            "TrkMatchingKalmanVsAlcarazAll": calculate_matching("KalmanAll", "AlcarazAll", rigidity_estimator),
            "TrkMatchingKalmanVsAlcarazInner": calculate_matching("KalmanInner", "AlcarazInner", rigidity_estimator),
            "TrkMatchingAllVsInnerChoutko": calculate_matching("All", "Inner", rigidity_estimator),
            "TrkMatchingAllVsInnerKalman": calculate_matching("KalmanAll", "KalmanInner", rigidity_estimator),
            "TrkMatchingAllVsInnerAlcaraz": calculate_matching("AlcarazAll", "AlcarazInner", rigidity_estimator),
            "TrkMatchingUpperVsLowerHalfChoutko": calculate_matching("UpperHalf", "LowerHalf", rigidity_estimator),
            "TrkMatchingUpperHalfVsInnerChoutko": calculate_matching("UpperHalf", "Inner", rigidity_estimator),
            "TrkMatchingLowerHalfVsInnerChoutko": calculate_matching("LowerHalf", "Inner", rigidity_estimator),
            "TrkMatchingUpperVsLowerHalfKalman": calculate_matching("KalmanUpperHalf", "KalmanLowerHalf", rigidity_estimator),
            "TrkMatchingUpperHalfVsInnerKalman": calculate_matching("KalmanUpperHalf", "KalmanInner", rigidity_estimator),
            "TrkMatchingLowerHalfVsInnerKalman": calculate_matching("KalmanLowerHalf", "KalmanInner", rigidity_estimator),
            "TrkSquareMatchingChoutkoVsKalmanAll": calculate_matching("All", "KalmanAll", rigidity_estimator, square=True),
            "TrkSquareMatchingChoutkoVsKalmanInner": calculate_matching("Inner", "KalmanInner", rigidity_estimator, square=True),
            "TrkSquareMatchingChoutkoVsAlcarazAll": calculate_matching("All", "AlcarazAll", rigidity_estimator, square=True),
            "TrkSquareMatchingChoutkoVsAlcarazInner": calculate_matching("Inner", "AlcarazInner", rigidity_estimator, square=True),
            "TrkSquareMatchingKalmanVsAlcarazAll": calculate_matching("KalmanAll", "AlcarazAll", rigidity_estimator, square=True),
            "TrkSquareMatchingKalmanVsAlcarazInner": calculate_matching("KalmanInner", "AlcarazInner", rigidity_estimator, square=True),
            "TrkSquareMatchingAllVsInnerChoutko": calculate_matching("All", "Inner", rigidity_estimator, square=True),
            "TrkSquareMatchingAllVsInnerKalman": calculate_matching("KalmanAll", "KalmanInner", rigidity_estimator, square=True),
            "TrkSquareMatchingAllVsInnerAlcaraz": calculate_matching("AlcarazAll", "AlcarazInner", rigidity_estimator, square=True),
            "TrkSquareMatchingUpperVsLowerHalfChoutko": calculate_matching("UpperHalf", "LowerHalf", rigidity_estimator, square=True),
            "TrkSquareMatchingUpperHalfVsInnerChoutko": calculate_matching("UpperHalf", "Inner", rigidity_estimator, square=True),
            "TrkSquareMatchingLowerHalfVsInnerChoutko": calculate_matching("LowerHalf", "Inner", rigidity_estimator, square=True),
            "TrkSquareMatchingUpperVsLowerHalfKalman": calculate_matching("KalmanUpperHalf", "KalmanLowerHalf", rigidity_estimator, square=True),
            "TrkSquareMatchingUpperHalfVsInnerKalman": calculate_matching("KalmanUpperHalf", "KalmanInner", rigidity_estimator, square=True),
            "TrkSquareMatchingLowerHalfVsInnerKalman": calculate_matching("KalmanLowerHalf", "KalmanInner", rigidity_estimator, square=True),
            "TrkAsymmetryChoutkoVsKalmanAll": calculate_asymmetry("All", "KalmanAll", rigidity_estimator),
            "TrkAsymmetryChoutkoVsKalmanInner": calculate_asymmetry("Inner", "KalmanInner", rigidity_estimator),
            "TrkAsymmetryChoutkoVsAlcarazAll": calculate_asymmetry("All", "AlcarazAll", rigidity_estimator),
            "TrkAsymmetryChoutkoVsAlcarazInner": calculate_asymmetry("Inner", "AlcarazInner", rigidity_estimator),
            "TrkAsymmetryAllVsInnerChoutko": calculate_asymmetry("All", "Inner", rigidity_estimator),
            "TrkAsymmetryAllVsInnerKalman": calculate_asymmetry("KalmanAll", "KalmanInner", rigidity_estimator),
            "TrkAsymmetryAllVsInnerAlcaraz": calculate_asymmetry("AlcarazAll", "AlcarazInner", rigidity_estimator),
            "TrkAsymmetryUpperVsLowerHalfChoutko": calculate_asymmetry("UpperHalf", "LowerHalf", rigidity_estimator),
            "TrkAsymmetryUpperHalfVsInnerChoutko": calculate_asymmetry("UpperHalf", "Inner", rigidity_estimator),
            "TrkAsymmetryLowerHalfVsInnerChoutko": calculate_asymmetry("LowerHalf", "Inner", rigidity_estimator),
            "TrkAsymmetryUpperVsLowerHalfKalman": calculate_asymmetry("KalmanUpperHalf", "KalmanLowerHalf", rigidity_estimator),
            "TrkAsymmetryUpperHalfVsInnerKalman": calculate_asymmetry("KalmanUpperHalf", "KalmanInner", rigidity_estimator),
            "TrkAsymmetryLowerHalfVsInnerKalman": calculate_asymmetry("KalmanLowerHalf", "KalmanInner", rigidity_estimator),
            "TrkAbsAsymmetryChoutkoVsKalmanAll": calculate_asymmetry("All", "KalmanAll", rigidity_estimator, abs=True),
            "TrkAbsAsymmetryChoutkoVsKalmanInner": calculate_asymmetry("Inner", "KalmanInner", rigidity_estimator, abs=True),
            "TrkAbsAsymmetryChoutkoVsAlcarazAll": calculate_asymmetry("All", "AlcarazAll", rigidity_estimator, abs=True),
            "TrkAbsAsymmetryChoutkoVsAlcarazInner": calculate_asymmetry("Inner", "AlcarazInner", rigidity_estimator, abs=True),
            "TrkAbsAsymmetryAllVsInnerChoutko": calculate_asymmetry("All", "Inner", rigidity_estimator, abs=True),
            "TrkAbsAsymmetryAllVsInnerKalman": calculate_asymmetry("KalmanAll", "KalmanInner", rigidity_estimator, abs=True),
            "TrkAbsAsymmetryAllVsInnerAlcaraz": calculate_asymmetry("AlcarazAll", "AlcarazInner", rigidity_estimator, abs=True),
            "TrkAbsAsymmetryUpperVsLowerHalfChoutko": calculate_asymmetry("UpperHalf", "LowerHalf", rigidity_estimator, abs=True),
            "TrkAbsAsymmetryUpperHalfVsInnerChoutko": calculate_asymmetry("UpperHalf", "Inner", rigidity_estimator, abs=True),
            "TrkAbsAsymmetryLowerHalfVsInnerChoutko": calculate_asymmetry("LowerHalf", "Inner", rigidity_estimator, abs=True),
            "TrkAbsAsymmetryUpperVsLowerHalfKalman": calculate_asymmetry("KalmanUpperHalf", "KalmanLowerHalf", rigidity_estimator, abs=True),
            "TrkAbsAsymmetryUpperHalfVsInnerKalman": calculate_asymmetry("KalmanUpperHalf", "KalmanInner", rigidity_estimator, abs=True),
            "TrkAbsAsymmetryLowerHalfVsInnerKalman": calculate_asymmetry("KalmanLowerHalf", "KalmanInner", rigidity_estimator, abs=True),
            "TrkRelativeErrorChoutkoAll": calculate_relative_error("All"),
            "TrkRelativeErrorKalmanAll": calculate_relative_error("KalmanAll"),
            "TrkRelativeErrorAlcarazAll": calculate_relative_error("AlcarazAll"),
            "TrkRelativeErrorChoutkoInner": calculate_relative_error("Inner"),
            "TrkRelativeErrorKalmanInner": calculate_relative_error("KalmanInner"),
            "TrkRelativeErrorAlcarazInner": calculate_relative_error("AlcarazInner"),
            "TrkFitResidualsX": trk_fit_residuals_x,
            "TrkFitResidualsY": trk_fit_residuals_y,
            "TrkTrackHasHitXY": trk_has_hit_xy,
            "TrkHasGoodHitX": trk_has_good_hit_x,
            "TrkHasGoodHitY": trk_has_good_hit_y,
            "TrkHasGoodHitXY": trk_has_good_hit_xy,
            "TrkNLayersGoodXY": trk_nlayers_good_xy,
            "TrkNLayersInnerGoodXY": trk_nlayers_inner_good_xy,
            "TrkHitInsideLayer": trk_hits_inside_layer,
            "TrkNHitsInsideLayer": trk_n_hits_inside_layer,
            "TrkNHitsInsideLayerInner": trk_n_hits_inside_layer_inner,
            "TrkTrackHitCoordXL34": trk_hit_coord_in_double_layer((3, 4), "X"),
            "TrkTrackHitCoordXL56": trk_hit_coord_in_double_layer((5, 6), "X"),
            "TrkTrackHitCoordXL78": trk_hit_coord_in_double_layer((7, 8), "X"),
            "TrkTrackHitCoordYL34": trk_hit_coord_in_double_layer((3, 4), "Y"),
            "TrkTrackHitCoordYL56": trk_hit_coord_in_double_layer((5, 6), "Y"),
            "TrkTrackHitCoordYL78": trk_hit_coord_in_double_layer((7, 8), "Y"),
            "TrkTrackRadiusL34": trk_hit_radius("TrkTrackHitCoordXL34", "TrkTrackHitCoordYL34"),
            "TrkTrackRadiusL56": trk_hit_radius("TrkTrackHitCoordXL56", "TrkTrackHitCoordYL56"),
            "TrkTrackRadiusL78": trk_hit_radius("TrkTrackHitCoordXL78", "TrkTrackHitCoordYL78"),
            "MinTrkFootDistance": min_trk_foot_distance(),
            "NHitTrkFeet": number_of_hit_trk_feet(),
            "TrkTrackMeanDepositedEnergyPerPathlength": trk_mean_energy_deposition_per_pathlength,
            "TrkTrackMassFromDepositedEnergyQ2": trk_mass_from_deposited_energy(rigidity_estimator, 2),
            "TrkMinLayerChargeX": calculate_min_charge("TrkLayerChargesX"),
            "TrkMinLayerChargeY": calculate_min_charge("TrkLayerChargesY"),
            "TrkMaxLayerChargeX": calculate_max_charge("TrkLayerChargesX"),
            "TrkMaxLayerChargeY": calculate_max_charge("TrkLayerChargesY"),
            "TrkMinLayerCharge": calculate_min_charge_xy(("TrkLayerChargesX", "TrkLayerChargesY")),
            "TrkMaxLayerCharge": calculate_max_charge_xy(("TrkLayerChargesX", "TrkLayerChargesY")),
            "TrkLayerChargeRmsX": calculate_charge_rms("TrkLayerChargesX"),
            "TrkLayerChargeRmsY": calculate_charge_rms("TrkLayerChargesY"),
            "TrkMaxLayerChargeDelta": trk_max_layer_charge_difference,
            "TrkMaxRelativeDeltaSagitta": calculate_max_relative_delta_sagitta,
            "TrkDoubleLayerPatternY": trk_double_layer_pattern("TrkHitPatternY"),
            "TrkDoubleLayerPatternXY": trk_double_layer_pattern("TrkHitPatternXY"),
            "TrkExternalLayerPattern": trk_external_layer_hit_pattern,
            "TrkHasHitInExternalLayers": trk_has_hit_in_external_layers,
            "TrkNHitsInExternalLayers": trk_n_hits_in_external_layers,
            "TrkNHitsInUpperInnerLayersY": trk_n_layers_inner_upper("TrkHitPatternY"),
            "TrkNHitsInUpperInnerLayersXY": trk_n_layers_inner_upper("TrkHitPatternXY"),
            "TrkNHitsInLowerInnerLayersY": trk_n_layers_inner_lower("TrkHitPatternY"),
            "TrkNHitsInLowerInnerLayersXY": trk_n_layers_inner_lower("TrkHitPatternXY"),
            "McRigidity": mc_rigidity,
            "McAbsRigidity": mc_abs_rigidity,
            "McBeta": mc_beta,
            "McMass": mc_mass,
            "TrackAngle": track_angle,
            "TrackAnglePhi": track_angle_phi,
            "TrackAngleXZ": track_angle_xz,
            "TrackAngleYZ": track_angle_yz,
            "HasTrigger": has_trigger,
            "HasPhysicsTrigger": has_physics_trigger,
            "HasUnbiasedTrigger": has_unbiased_trigger,
            "HasUnbiasedTofTrigger": has_unbiased_tof_trigger,
            "UnbiasedTriggerWeight": unbiased_trigger_weight,
            "TofUpperChargeFromLayers": tof_upper_charge_from_layers,
            "TofLowerChargeFromLayers": tof_lower_charge_from_layers,
            "Beta": default_beta,
            "TofMass": calculate_mass(rigidity_estimator, "TofBeta", "TrkCharge"),
            "RichMass": calculate_mass(rigidity_estimator, "RichBeta", "TrkCharge"),
            "Mass": default_mass(rigidity_estimator, "TrkCharge"),
            "TrkTrackHitsRich": trk_track_hits_rich,
            "TrkTrackHitsAGL": trk_track_hits_agl,
            "TrkTrackHitsNaF": trk_track_hits_naf,
            "RichBetaResolutionIsGood": rich_beta_resolution_is_good(max_delta_naf=2 * (config["analysis"].get("resolution", {}).get("NaF", RICH_RESOLUTION_NAF)), max_delta_agl=2 * (config["analysis"].get("resolution", {}).get("AGL", RICH_RESOLUTION_AGL))),
            "RichRingAsExpectedHeAll": rich_ring_as_expected((MC_PARTICLE_IDS["He4"], MC_PARTICLE_IDS["He3"]), "TrkRigidityAll"),
            "RichRingAsExpectedHeInner": rich_ring_as_expected((MC_PARTICLE_IDS["He4"], MC_PARTICLE_IDS["He3"]), "TrkRigidityInner"),
            "CherenkovThresholdRatioIfRingAll": cherenkov_threshold_ratio_if_ring(("He3", "He4"), "TrkRigidityAll"),
            "CherenkovThresholdRatioIfNoRingAll": cherenkov_threshold_ratio_if_no_ring(("He3", "He4"), "TrkRigidityAll"),
            "CherenkovThresholdRatioIfRingInner": cherenkov_threshold_ratio_if_ring(("He3", "He4"), "TrkRigidityInner"),
            "CherenkovThresholdRatioIfNoRingInner": cherenkov_threshold_ratio_if_no_ring(("He3", "He4"), "TrkRigidityInner"),
            "TrkTrackDistanceToRichTileBorder": trk_track_distance_to_rich_tile_border,
            "TrkTrackDistanceToNaFCorners": trk_track_distance_to_naf_corners,
            "RichIsAGL": rich_is_agl,
            "TrkChargeRms": calculate_inner_tracker_charge_rms,
            "TrdHitAmplitudePerPathlength": trd_hit_amplitude_per_pathlength,
            "TrdMedianCharge": trd_median_charge_12(rigidity_estimator),
            "TrdMedianHitAmplitude": trd_median("TrdHitAmplitude"),
            "TrdMedianHitAmplitudePerPathlength": trd_median("TrdHitAmplitudePerPathlength"),
            "TrdNHitsOnTrack": trd_nhits_on_track,
            "TrdLikelihoodRatioHeliumOverElectron": trd_log_likelihood_ratio("Helium", "Electron"),
            "TrdLikelihoodRatioHeliumOverProton": trd_log_likelihood_ratio("Helium", "Proton"),
            "TrdLikelihoodRatioProtonOverElectron": trd_log_likelihood_ratio("Proton", "Electron"),
            "EnergyOverRigidity": energy_over_rigidity(rigidity_estimator),
            "LogEnergyOverRigidity": log_energy_over_rigidity(rigidity_estimator),
            "RichPhotoElectronRatioRingVsEvent": rich_photo_electron_ratio("RichNPhotoElectrons", "RichNCollectedPhotoElectrons"),
            "RichPhotoElectronRatioExpectedVsMeasured": rich_photo_electron_ratio("RichNExpectedPhotoElectrons", "RichNPhotoElectrons"),
            "BacktracingFromSpaceOrMc": backtracing_from_space_or_mc,
        }
        self.dependencies = {
            "TrkRigidityFallback": ("TrkRigidityAll", "TrkRigidityInner"),
            "TrdTrkDistanceX": ("TrkTrackFitCoordXAtTrd", "TrdTrackCoordX"),
            "TrdTrkDistanceY": ("TrkTrackFitCoordYAtTrd", "TrdTrackCoordY"),
            "TrdTrkDistance": ("TrkTrackFitCoordXAtTrd", "TrkTrackFitCoordYAtTrd", "TrdTrackCoordX", "TrdTrackCoordY"),
            "TrdTrkAngleDistanceXZ": ("TrkTrackFitDirXZAtTrd", "TrdTrackDirXZ"),
            "TrdTrkAngleDistanceYZ": ("TrkTrackFitDirYZAtTrd", "TrdTrackDirYZ"),
            "TrkMatchingChoutkoVsKalmanAll": ("TrkRigidityAll", "TrkRigidityKalmanAll", "TrkInverseRigidityErrorAll", "TrkInverseRigidityErrorKalmanAll"),
            "TrkMatchingChoutkoVsKalmanInner": ("TrkRigidityInner", "TrkRigidityKalmanInner", "TrkInverseRigidityErrorInner", "TrkInverseRigidityErrorKalmanInner"),
            "TrkMatchingChoutkoVsAlcarazAll": ("TrkRigidityAll", "TrkRigidityAlcarazAll", "TrkInverseRigidityErrorAll", "TrkInverseRigidityErrorAlcarazAll"),
            "TrkMatchingChoutkoVsAlcarazInner": ("TrkRigidityInner", "TrkRigidityAlcarazInner", "TrkInverseRigidityErrorInner", "TrkInverseRigidityErrorAlcarazInner"),
            "TrkMatchingKalmanVsAlcarazAll": ("TrkRigidityKalmanAll", "TrkRigidityAlcarazAll", "TrkInverseRigidityErrorKalmanAll", "TrkInverseRigidityErrorAlcarazAll"),
            "TrkMatchingKalmanVsAlcarazInner": ("TrkRigidityKalmanInner", "TrkRigidityAlcarazInner", "TrkInverseRigidityErrorKalmanInner", "TrkInverseRigidityErrorAlcarazInner"),
            "TrkMatchingAllVsInnerChoutko": ("TrkRigidityInner", "TrkRigidityAll", "TrkInverseRigidityErrorInner", "TrkInverseRigidityErrorAll"),
            "TrkMatchingAllVsInnerKalman": ("TrkRigidityKalmanInner", "TrkRigidityKalmanAll", "TrkInverseRigidityErrorKalmanInner", "TrkInverseRigidityErrorKalmanAll"),
            "TrkMatchingAllVsInnerAlcaraz": ("TrkRigidityAlcarazInner", "TrkRigidityAlcarazAll", "TrkInverseRigidityErrorAlcarazInner", "TrkInverseRigidityErrorAlcarazAll"),
            "TrkMatchingUpperVsLowerHalfChoutko": ("TrkRigidityUpperHalf", "TrkRigidityLowerHalf", "TrkInverseRigidityErrorUpperHalf", "TrkInverseRigidityErrorLowerHalf"),
            "TrkMatchingUpperVsLowerHalfKalman": ("TrkRigidityKalmanUpperHalf", "TrkRigidityKalmanLowerHalf", "TrkInverseRigidityErrorKalmanUpperHalf", "TrkInverseRigidityErrorKalmanLowerHalf"),
            "TrkMatchingUpperHalfVsInnerChoutko": ("TrkRigidityUpperHalf", "TrkRigidityInner", "TrkInverseRigidityErrorUpperHalf", "TrkInverseRigidityErrorInner"),
            "TrkMatchingUpperHalfVsInnerKalman": ("TrkRigidityKalmanUpperHalf", "TrkRigidityKalmanInner", "TrkInverseRigidityErrorKalmanUpperHalf", "TrkInverseRigidityErrorKalmanInner"),
            "TrkMatchingLowerHalfVsInnerChoutko": ("TrkRigidityLowerHalf", "TrkRigidityInner", "TrkInverseRigidityErrorLowerHalf", "TrkInverseRigidityErrorInner"),
            "TrkMatchingLowerHalfVsInnerKalman": ("TrkRigidityKalmanLowerHalf", "TrkRigidityKalmanInner", "TrkInverseRigidityErrorKalmanLowerHalf", "TrkInverseRigidityErrorKalmanInner"),
            "TrkSquareMatchingChoutkoVsKalmanAll": ("TrkRigidityAll", "TrkRigidityKalmanAll", "TrkInverseRigidityErrorAll", "TrkInverseRigidityErrorKalmanAll"),
            "TrkSquareMatchingChoutkoVsKalmanInner": ("TrkRigidityInner", "TrkRigidityKalmanInner", "TrkInverseRigidityErrorInner", "TrkInverseRigidityErrorKalmanInner"),
            "TrkSquareMatchingChoutkoVsAlcarazAll": ("TrkRigidityAll", "TrkRigidityAlcarazAll", "TrkInverseRigidityErrorAll", "TrkInverseRigidityErrorAlcarazAll"),
            "TrkSquareMatchingChoutkoVsAlcarazInner": ("TrkRigidityInner", "TrkRigidityAlcarazInner", "TrkInverseRigidityErrorInner", "TrkInverseRigidityErrorAlcarazInner"),
            "TrkSquareMatchingKalmanVsAlcarazAll": ("TrkRigidityKalmanAll", "TrkRigidityAlcarazAll", "TrkInverseRigidityErrorKalmanAll", "TrkInverseRigidityErrorAlcarazAll"),
            "TrkSquareMatchingKalmanVsAlcarazInner": ("TrkRigidityKalmanInner", "TrkRigidityAlcarazInner", "TrkInverseRigidityErrorKalmanInner", "TrkInverseRigidityErrorAlcarazInner"),
            "TrkSquareMatchingAllVsInnerChoutko": ("TrkRigidityInner", "TrkRigidityAll", "TrkInverseRigidityErrorInner", "TrkInverseRigidityErrorAll"),
            "TrkSquareMatchingAllVsInnerKalman": ("TrkRigidityKalmanInner", "TrkRigidityKalmanAll", "TrkInverseRigidityErrorKalmanInner", "TrkInverseRigidityErrorKalmanAll"),
            "TrkSquareMatchingAllVsInnerAlcaraz": ("TrkRigidityAlcarazInner", "TrkRigidityAlcarazAll", "TrkInverseRigidityErrorAlcarazInner", "TrkInverseRigidityErrorAlcarazAll"),
            "TrkSquareMatchingUpperVsLowerHalfChoutko": ("TrkRigidityUpperHalf", "TrkRigidityLowerHalf", "TrkInverseRigidityErrorUpperHalf", "TrkInverseRigidityErrorLowerHalf"),
            "TrkSquareMatchingUpperVsLowerHalfKalman": ("TrkRigidityKalmanUpperHalf", "TrkRigidityKalmanLowerHalf", "TrkInverseRigidityErrorKalmanUpperHalf", "TrkInverseRigidityErrorKalmanLowerHalf"),
            "TrkSquareMatchingUpperHalfVsInnerChoutko": ("TrkRigidityUpperHalf", "TrkRigidityInner", "TrkInverseRigidityErrorUpperHalf", "TrkInverseRigidityErrorInner"),
            "TrkSquareMatchingUpperHalfVsInnerKalman": ("TrkRigidityKalmanUpperHalf", "TrkRigidityKalmanInner", "TrkInverseRigidityErrorKalmanUpperHalf", "TrkInverseRigidityErrorKalmanInner"),
            "TrkSquareMatchingLowerHalfVsInnerChoutko": ("TrkRigidityLowerHalf", "TrkRigidityInner", "TrkInverseRigidityErrorLowerHalf", "TrkInverseRigidityErrorInner"),
            "TrkSquareMatchingLowerHalfVsInnerKalman": ("TrkRigidityKalmanLowerHalf", "TrkRigidityKalmanInner", "TrkInverseRigidityErrorKalmanLowerHalf", "TrkInverseRigidityErrorKalmanInner"),
            "TrkAsymmetryChoutkoVsKalmanAll": ("TrkRigidityAll", "TrkRigidityKalmanAll"),
            "TrkAsymmetryChoutkoVsKalmanInner": ("TrkRigidityInner", "TrkRigidityKalmanInner"),
            "TrkAsymmetryChoutkoVsAlcarazAll": ("TrkRigidityAll", "TrkRigidityAlcarazAll"),
            "TrkAsymmetryChoutkoVsAlcarazInner": ("TrkRigidityInner", "TrkRigidityAlcarazInner"),
            "TrkAsymmetryAllVsInnerChoutko": ("TrkRigidityInner", "TrkRigidityAll"),
            "TrkAsymmetryAllVsInnerKalman": ("TrkRigidityKalmanInner", "TrkRigidityKalmanAll"),
            "TrkAsymmetryAllVsInnerAlcaraz": ("TrkRigidityAlcarazInner", "TrkRigidityAlcarazAll"),
            "TrkAsymmetryUpperVsLowerHalfChoutko": ("TrkRigidityUpperHalf", "TrkRigidityLowerHalf"),
            "TrkAsymmetryUpperVsLowerHalfKalman": ("TrkRigidityKalmanUpperHalf", "TrkRigidityKalmanLowerHalf"),
            "TrkAsymmetryUpperHalfVsInnerChoutko": ("TrkRigidityUpperHalf", "TrkRigidityInner"),
            "TrkAsymmetryUpperHalfVsInnerKalman": ("TrkRigidityKalmanUpperHalf", "TrkRigidityKalmanInner"),
            "TrkAsymmetryLowerHalfVsInnerChoutko": ("TrkRigidityLowerHalf", "TrkRigidityInner"),
            "TrkAsymmetryLowerHalfVsInnerKalman": ("TrkRigidityKalmanLowerHalf", "TrkRigidityKalmanInner"),
            "TrkAbsAsymmetryChoutkoVsKalmanAll": ("TrkRigidityAll", "TrkRigidityKalmanAll"),
            "TrkAbsAsymmetryChoutkoVsKalmanInner": ("TrkRigidityInner", "TrkRigidityKalmanInner"),
            "TrkAbsAsymmetryChoutkoVsAlcarazAll": ("TrkRigidityAll", "TrkRigidityAlcarazAll"),
            "TrkAbsAsymmetryChoutkoVsAlcarazInner": ("TrkRigidityInner", "TrkRigidityAlcarazInner"),
            "TrkAbsAsymmetryAllVsInnerChoutko": ("TrkRigidityInner", "TrkRigidityAll"),
            "TrkAbsAsymmetryAllVsInnerKalman": ("TrkRigidityKalmanInner", "TrkRigidityKalmanAll"),
            "TrkAbsAsymmetryAllVsInnerAlcaraz": ("TrkRigidityAlcarazInner", "TrkRigidityAlcarazAll"),
            "TrkAbsAsymmetryUpperVsLowerHalfChoutko": ("TrkRigidityUpperHalf", "TrkRigidityLowerHalf"),
            "TrkAbsAsymmetryUpperVsLowerHalfKalman": ("TrkRigidityKalmanUpperHalf", "TrkRigidityKalmanLowerHalf"),
            "TrkAbsAsymmetryUpperHalfVsInnerChoutko": ("TrkRigidityUpperHalf", "TrkRigidityInner"),
            "TrkAbsAsymmetryUpperHalfVsInnerKalman": ("TrkRigidityKalmanUpperHalf", "TrkRigidityKalmanInner"),
            "TrkAbsAsymmetryLowerHalfVsInnerChoutko": ("TrkRigidityLowerHalf", "TrkRigidityInner"),
            "TrkAbsAsymmetryLowerHalfVsInnerKalman": ("TrkRigidityKalmanLowerHalf", "TrkRigidityKalmanInner"),
            "TrkRelativeErrorChoutkoAll": ("TrkRigidityAll", "TrkInverseRigidityErrorAll"),
            "TrkRelativeErrorChoutkoInner": ("TrkRigidityInner", "TrkInverseRigidityErrorInner"),
            "TrkRelativeErrorKalmanAll": ("TrkRigidityKalmanAll", "TrkInverseRigidityErrorKalmanAll"),
            "TrkRelativeErrorKalmanInner": ("TrkRigidityKalmanInner", "TrkInverseRigidityErrorKalmanInner"),
            "TrkRelativeErrorAlcarazAll": ("TrkRigidityAlcarazAll", "TrkInverseRigidityErrorAlcarazAll"),
            "TrkRelativeErrorAlcarazInner": ("TrkRigidityAlcarazInner", "TrkInverseRigidityErrorAlcarazInner"),
            "TrkFitResidualsX": ("TrkTrackFitCoordX", "TrkTrackHitCoordX"),
            "TrkFitResidualsY": ("TrkTrackFitCoordY", "TrkTrackHitCoordY"),
            "TrkTrackHasHitXY": ("TrkTrackHasHitX", "TrkTrackHasHitY"),
            "TrkHasGoodHitX": ("TrkTrackFitCoordX", "TrkTrackHitCoordX"),
            "TrkHasGoodHitY": ("TrkTrackFitCoordY", "TrkTrackHitCoordY"),
            "TrkHasGoodHitXY": ("TrkTrackFitCoordX", "TrkTrackFitCoordY", "TrkTrackHitCoordX", "TrkTrackHitCoordY"),
            "TrkNLayersInnerGoodXY": ("TrkTrackFitCoordX", "TrkTrackFitCoordY", "TrkTrackHitCoordX", "TrkTrackHitCoordY"),
            "TrkHitsInsideLayer": ("TrkTrackHitCoordX", "TrkTrackHitCoordY"),
            "TrkNHitsInsideLayer": ("TrkTrackHitCoordX", "TrkTrackHitCoordY"),
            "TrkNHitsInsideLayerInner": ("TrkTrackHitCoordX", "TrkTrackHitCoordY"),
            "TrkTrackHitCoordXL34": ("TrkTrackHitCoordX", "TrkTrackHasHitX"),
            "TrkTrackHitCoordXL56": ("TrkTrackHitCoordX", "TrkTrackHasHitX"),
            "TrkTrackHitCoordXL78": ("TrkTrackHitCoordX", "TrkTrackHasHitX"),
            "TrkTrackHitCoordYL34": ("TrkTrackHitCoordY", "TrkTrackHasHitY"),
            "TrkTrackHitCoordYL56": ("TrkTrackHitCoordY", "TrkTrackHasHitY"),
            "TrkTrackHitCoordYL78": ("TrkTrackHitCoordY", "TrkTrackHasHitY"),
            "TrkTrackRadiusL34": ("TrkTrackHitCoordXL34", "TrkTrackHitCoordYL34"),
            "TrkTrackRadiusL56": ("TrkTrackHitCoordXL56", "TrkTrackHitCoordYL56"),
            "TrkTrackRadiusL78": ("TrkTrackHitCoordXL78", "TrkTrackHitCoordYL78"),
            "MinTrkFootDistance": ("TrkTrackFitCoordX", "TrkTrackFitCoordY"),
            "NHitTrkFeet": ("TrkTrackFitCoordX", "TrkTrackFitCoordY"),
            "TrkTrackMeanDepositedEnergyPerPathlength": tuple([f"TrkTrackHitDepositedEnergyPerPathlength{orientation}L{layer}" for orientation in ("X", "Y") for layer in range(1, 10)]),
            "TrkTrackMassFromDepositedEnergyQ2": ("TrkTrackMeanDepositedEnergyPerPathlength", rigidity_estimator, "McParticleID"),
            "TrkMinLayerChargeX": ("TrkLayerChargesX",),
            "TrkMinLayerChargeY": ("TrkLayerChargesY",),
            "TrkMaxLayerChargeX": ("TrkLayerChargesX",),
            "TrkMaxLayerChargeY": ("TrkLayerChargesY",),
            "TrkMinLayerCharge": ("TrkLayerChargesX", "TrkLayerChargesY"),
            "TrkMaxLayerCharge": ("TrkLayerChargesX", "TrkLayerChargesY"),
            "TrkLayerChargeRmsX": ("TrkLayerChargesX",),
            "TrkLayerChargeRmsY": ("TrkLayerChargesY",),
            "TrkMaxLayerChargeDelta": ("TrkLayerChargesX", "TrkLayerChargesY"),
            "TrkMaxRelativeDeltaSagitta": ("TrkRigidityInner", "TrkRigiditiesWithoutHit"),
            "TrkDoubleLayerPatternY": ("TrkHitPatternY",),
            "TrkDoubleLayerPatternXY": ("TrkHitPatternXY",),
            "TrkExternalLayerPattern": ("TrkTrackHasHitX", "TrkTrackHasHitY"),
            "TrkHasHitInExternalLayers": ("TrkTrackHasHitX", "TrkTrackHasHitY"),
            "TrkNHitsInExternalLayers": ("TrkTrackHasHitX", "TrkTrackHasHitY"),
            "TrkNHitsInUpperInnerLayersY": ("TrkHitPatternY",),
            "TrkNHitsInUpperInnerLayersXY": ("TrkHitPatternXY",),
            "TrkNHitsInLowerInnerLayersY": ("TrkHitPatternY",),
            "TrkNHitsInLowerInnerLayersXY": ("TrkHitPatternXY",),
            "McRigidity": ("McMomentum", "McParticleID"),
            "McAbsRigidity": ("McMomentum", "McParticleID"),
            "McBeta": ("McMomentum", "McParticleID"),
            "McMass": ("McParticleID",),
            "TrackAngle": ("TrkTrackFitDirXZ", "TrkTrackFitDirYZ"),
            "TrackAnglePhi": ("TrkTrackFitDirXZ", "TrkTrackFitDirYZ"),
            "TrackAngleXZ": ("TrkTrackFitDirXZ",),
            "TrackAngleYZ": ("TrkTrackFitDirYZ",),
            "HasTrigger": ("TriggerFlags",),
            "HasPhysicsTrigger": ("TriggerFlags",),
            "HasUnbiasedTrigger": ("TriggerFlags",),
            "HasUnbiasedTofTrigger": ("TriggerFlags",),
            "UnbiasedTriggerWeight": ("TriggerFlags",),
            "Beta": ("TofBeta", "RichBeta", "HasRich"),
            "TofUpperChargeFromLayers": ("TofChargeInLayer",),
            "TofLowerChargeFromLayers": ("TofChargeInLayer",),
            "TofMass": (rigidity_estimator, "TofBeta", "TrkCharge"),
            "RichMass": (rigidity_estimator, "RichBeta", "TrkCharge"),
            "Mass": (rigidity_estimator, "TofBeta", "RichBeta", "HasRich", "TrkCharge"),
            "TrkTrackDistanceToRichTileBorder": ("TrkTrackFitCoordXAtRich", "TrkTrackFitCoordYAtRich", "TrkTrackHitsNaF"),
            "TrkTrackDistanceToNaFCorners": ("TrkTrackFitCoordXAtRich", "TrkTrackFitCoordYAtRich"),
            "TrkTrackHitsRich": ("TrkTrackFitCoordXAtRich", "TrkTrackFitCoordYAtRich"),
            "TrkTrackHitsAGL": ("TrkTrackFitCoordXAtRich", "TrkTrackFitCoordYAtRich"),
            "TrkTrackHitsNaF": ("TrkTrackFitCoordXAtRich", "TrkTrackFitCoordYAtRich"),
            "RichBetaResolutionIsGood": ("TrkTrackHitsNaF", "TrkTrackHitsAGL", "AbsRichBetaMinusTrueBeta"),
            "RichRingAsExpectedHeAll": ("TrkTrackHitsNaF", "TrkTrackHitsAGL", "TrkTrackHitsRich", "TrkRigidityAll", "TrkTrackDistanceToRichTileBorder", "TrkTrackDistanceToNaFCorners"),
            "RichRingAsExpectedHeInner": ("TrkTrackHitsNaF", "TrkTrackHitsAGL", "TrkTrackHitsRich", "TrkRigidityInner", "TrkTrackDistanceToRichTileBorder", "TrkTrackDistanceToNaFCorners"),
            "CherenkovThresholdRatioIfNoRingAll": ("TrkTrackHitsNaF", "TrkRigidityAll", "HasRich", "TrkTrackDistanceToRichTileBorder", "TrkTrackDistanceToNaFCorners"),
            "CherenkovThresholdRatioIfRingAll": ("TrkTrackHitsNaF", "TrkRigidityAll", "HasRich"),
            "CherenkovThresholdRatioIfNoRingInner": ("TrkTrackHitsNaF", "TrkRigidityInner", "HasRich", "TrkTrackDistanceToRichTileBorder", "TrkTrackDistanceToNaFCorners"),
            "CherenkovThresholdRatioIfRingInner": ("TrkTrackHitsNaF", "TrkRigidityInner", "HasRich"),
            "RichPhotoElectronRatioRingVsEvent": ("RichNPhotoElectrons", "RichNCollectedPhotoElectrons"),
            "RichPhotoElectronRatioExpectedVsMeasured": ("RichNPhotoElectrons", "RichNExpectedPhotoElectrons"),
            "RichIsAGL": ("HasRich", "RichIsNaF"),
            "TrdHitAmplitudePerPathlength": ("TrdHitAmplitude", "TrdHitPathlength"),
            "TrdMedianCharge": ("TrdMedianHitAmplitudePerPathlength", rigidity_estimator),
            "TrdMedianHitAmplitude": ("TrdNHits", "TrdHitAmplitude"),
            "TrdMedianHitAmplitudePerPathlength": ("TrdNHits", "TrdHitAmplitudePerPathlength", "TrdHitDistanceToTrack"),
            "TrdNHitsOnTrack": ("TrdHitDistanceToTrack",),
            "TrdLikelihoodRatioHeliumOverElectron": ("TrdLlhHelium", "TrdLlhElectron"),
            "TrdLikelihoodRatioHeliumOverProton": ("TrdLlhHelium", "TrdLlhProton"),
            "TrdLikelihoodRatioProtonOverElectron": ("TrdLlhProton", "TrdLlhElectron"),
            "TrkChargeRms": ("TrkChargeError", "TrkNLayersInnerY"),
            "EnergyOverRigidity": ("EcalEnergyDeposited", rigidity_estimator),
            "LogEnergyOverRigidity": ("EcalEnergyDeposited", rigidity_estimator),
            "BacktracingFromSpaceOrMc": ("BacktracingStatus", "McParticleID"),
        }

        for tof_layer in range(1, 5):
            self.functions[f"TofTrkDistanceXT{tof_layer}"] = tof_trk_distance_x(tof_layer - 1)
            self.functions[f"TofTrkDistanceYT{tof_layer}"] = tof_trk_distance_y(tof_layer - 1)
            self.functions[f"TofTrkDistanceT{tof_layer}"] = tof_trk_distance(tof_layer - 1)
            self.dependencies[f"TofTrkDistanceXT{tof_layer}"] = ("TrkTrackFitCoordXAtTof", "TofClusterCoordX")
            self.dependencies[f"TofTrkDistanceYT{tof_layer}"] = ("TrkTrackFitCoordYAtTof", "TofClusterCoordY")
            self.dependencies[f"TofTrkDistanceT{tof_layer}"] = ("TrkTrackFitCoordXAtTof", "TrkTrackFitCoordYAtTof", "TofClusterCoordX", "TofClusterCoordY")

        for trk_layer in range(1, 10):
            self.functions[f"TrkChargeL{trk_layer}"] = trk_layer_charge(trk_layer)
            self.functions[f"TrkChargeXL{trk_layer}"] = trk_layer_charge_x(trk_layer)
            self.functions[f"TrkChargeYL{trk_layer}"] = trk_layer_charge_y(trk_layer)
            self.functions[f"TrkChargeYJL{trk_layer}"] = trk_layer_charge_yj(trk_layer)
            self.functions[f"TrkHasChargeL{trk_layer}"] = trk_has_layer_charge_x_or_y(trk_layer)
            self.functions[f"TrkHasChargeXYL{trk_layer}"] = trk_has_layer_charge_x_and_y(trk_layer)
            self.functions[f"TrkHasChargeXL{trk_layer}"] = trk_has_layer_charge_x(trk_layer)
            self.functions[f"TrkHasChargeYL{trk_layer}"] = trk_has_layer_charge_y(trk_layer)
            self.dependencies[f"TrkChargeL{trk_layer}"] = ("TrkLayerChargesX", "TrkLayerChargesY")
            self.dependencies[f"TrkChargeXL{trk_layer}"] = ("TrkLayerChargesX",)
            self.dependencies[f"TrkChargeYL{trk_layer}"] = ("TrkLayerChargesY",)
            self.dependencies[f"TrkChargeYJL{trk_layer}"] = ("TrkLayerChargesYJ",)
            self.dependencies[f"TrkHasChargeL{trk_layer}"] = ("TrkLayerChargesX", "TrkLayerChargesY")
            self.dependencies[f"TrkHasChargeXYL{trk_layer}"] = ("TrkLayerChargesX", "TrkLayerChargesY")
            self.dependencies[f"TrkHasChargeXL{trk_layer}"] = ("TrkLayerChargesX",)
            self.dependencies[f"TrkHasChargeYL{trk_layer}"] = ("TrkLayerChargesY",)

            self.functions[f"TrkTrackHasHitXL{trk_layer}"] = trk_has_hit_in_layer_x(trk_layer)
            self.functions[f"TrkTrackHasHitYL{trk_layer}"] = trk_has_hit_in_layer_y(trk_layer)
            self.functions[f"TrkTrackHasHitXYL{trk_layer}"] = trk_has_hit_in_layer_xy(trk_layer)
            self.dependencies[f"TrkTrackHasHitXL{trk_layer}"] = ("TrkTrackHasHitX",)
            self.dependencies[f"TrkTrackHasHitYL{trk_layer}"] = ("TrkTrackHasHitY",)
            self.dependencies[f"TrkTrackHasHitXYL{trk_layer}"] = ("TrkTrackHasHitX", "TrkTrackHasHitY")

            self.functions[f"TrkHitInsideLayerL{trk_layer}"] = trk_hit_inside_layer(trk_layer)
            self.dependencies[f"TrkHitInsideLayerL{trk_layer}"] = ("TrkTrackHitCoordX", "TrkTrackHitCoordY")

            self.functions[f"TrkTrackHitCoordXL{trk_layer}"] = trk_hit_coord_in_layer(trk_layer, "X")
            self.functions[f"TrkTrackHitCoordYL{trk_layer}"] = trk_hit_coord_in_layer(trk_layer, "Y")
            self.functions[f"TrkTrackFitCoordXL{trk_layer}"] = trk_fit_coord_in_layer(trk_layer, "X")
            self.functions[f"TrkTrackFitCoordYL{trk_layer}"] = trk_fit_coord_in_layer(trk_layer, "Y")
            self.functions[f"AbsTrkTrackFitCoordXL{trk_layer}"] = trk_abs_fit_coord_in_layer(trk_layer, "X")
            self.functions[f"AbsTrkTrackFitCoordYL{trk_layer}"] = trk_abs_fit_coord_in_layer(trk_layer, "Y")
            self.functions[f"TrkTrackHitDepositedEnergyXL{trk_layer}"] = trk_layer_deposited_energy_x(trk_layer)
            self.functions[f"TrkTrackHitDepositedEnergyYL{trk_layer}"] = trk_layer_deposited_energy_y(trk_layer)
            self.dependencies[f"TrkTrackHitCoordXL{trk_layer}"] = ("TrkTrackHitCoordX",)
            self.dependencies[f"TrkTrackHitCoordYL{trk_layer}"] = ("TrkTrackHitCoordY",)
            self.dependencies[f"TrkTrackFitCoordXL{trk_layer}"] = ("TrkTrackFitCoordX",)
            self.dependencies[f"TrkTrackFitCoordYL{trk_layer}"] = ("TrkTrackFitCoordY",)
            self.dependencies[f"AbsTrkTrackFitCoordXL{trk_layer}"] = ("TrkTrackFitCoordX",)
            self.dependencies[f"AbsTrkTrackFitCoordYL{trk_layer}"] = ("TrkTrackFitCoordY",)
            self.dependencies[f"TrkTrackHitDepositedEnergyXL{trk_layer}"] = ("TrkTrackHitDepositedEnergyX",)
            self.dependencies[f"TrkTrackHitDepositedEnergyYL{trk_layer}"] = ("TrkTrackHitDepositedEnergyY",)
            self.functions[f"TrkTrackFitRadiusL{trk_layer}"] = trk_hit_radius(f"TrkTrackFitCoordXL{trk_layer}", f"TrkTrackFitCoordYL{trk_layer}")
            self.dependencies[f"TrkTrackFitRadiusL{trk_layer}"] = (f"TrkTrackFitCoordXL{trk_layer}", f"TrkTrackFitCoordYL{trk_layer}")
            self.functions[f"TrkTrackFitDistanceToSensorEdgeL{trk_layer}"] = trk_distance_to_sensor_edge(trk_layer, coord="xy")
            self.functions[f"TrkTrackFitDistanceToSensorEdgeXL{trk_layer}"] = trk_distance_to_sensor_edge(trk_layer, coord="x")
            self.functions[f"TrkTrackFitDistanceToSensorEdgeYL{trk_layer}"] = trk_distance_to_sensor_edge(trk_layer, coord="y")
            self.dependencies[f"TrkTrackFitDistanceToSensorEdgeL{trk_layer}"] = ("TrkTrackFitCoordX", "TrkTrackFitCoordY")
            self.dependencies[f"TrkTrackFitDistanceToSensorEdgeXL{trk_layer}"] = ("TrkTrackFitCoordX", "TrkTrackFitCoordY")
            self.dependencies[f"TrkTrackFitDistanceToSensorEdgeYL{trk_layer}"] = ("TrkTrackFitCoordX", "TrkTrackFitCoordY")
            self.functions[f"TrkTrackFitAngleL{trk_layer}"] = trk_fit_angle_at_layer(trk_layer)
            self.dependencies[f"TrkTrackFitAngleL{trk_layer}"] = ("TrkTrackFitDirXZ", "TrkTrackFitDirYZ")
            self.functions[f"TrkTrackFitPathlengthL{trk_layer}"] = trk_fit_pathlength_in_layer(trk_layer)
            self.dependencies[f"TrkTrackFitPathlengthL{trk_layer}"] = (f"TrkTrackFitAngleL{trk_layer}",)
            self.functions[f"TrkTrackHitDepositedEnergyPerPathlengthXL{trk_layer}"] = trk_energy_deposition_per_pathlength(trk_layer, "X")
            self.functions[f"TrkTrackHitDepositedEnergyPerPathlengthYL{trk_layer}"] = trk_energy_deposition_per_pathlength(trk_layer, "Y")
            self.dependencies[f"TrkTrackHitDepositedEnergyPerPathlengthXL{trk_layer}"] = (f"TrkTrackHitDepositedEnergyXL{trk_layer}", f"TrkTrackFitPathlengthL{trk_layer}")
            self.dependencies[f"TrkTrackHitDepositedEnergyPerPathlengthYL{trk_layer}"] = (f"TrkTrackHitDepositedEnergyYL{trk_layer}", f"TrkTrackFitPathlengthL{trk_layer}")

        for rigidity in ("Inner", "All", "KalmanInner", "KalmanAll", "InnerPlusL1", "InnerPlusL9", "KalmanInnerPlusL1", "KalmanInnerPlusL9", "UpperHalf", "LowerHalf", "KalmanUpperHalf", "KalmanLowerHalf", "AlcarazInner", "AlcarazAll", "ChikanianInner", "ChikanianAll", "GBLAll", "GBLInner", "GBLLowerHalf", "GBLUpperHalf"):
            self.functions[f"TrkRigidityAbs{rigidity}"] = trk_rigidity_abs(f"TrkRigidity{rigidity}")
            self.dependencies[f"TrkRigidityAbs{rigidity}"] = (f"TrkRigidity{rigidity}",)
            self.functions[f"TrkRigidityAbs{rigidity}FullRange"] = trk_rigidity_abs(f"TrkRigidity{rigidity}")
            self.dependencies[f"TrkRigidityAbs{rigidity}FullRange"] = (f"TrkRigidity{rigidity}",)

        for orientation in ("X", "Y"):
            for geometry in ("All", "Inner"):
                for fit in ("", "Kalman"):
                    chi2_name = f"TrkChi2{orientation}{fit}{geometry}"
                    self.functions[f"Log{chi2_name}"] = trk_log_chi2(chi2_name)
                    self.dependencies[f"Log{chi2_name}"] = (chi2_name,)

        for rigidity in ("TrkRigidityInner", "TrkRigidityAll", "TrkRigidityFallback"):
            for cutoff in ("IGRF", "Stoermer"):
                for angle_num in (25, 30, 35, 40):
                    for chargesign in ("PN", "P", "N"):
                        angle = f"{angle_num}{chargesign}"
                        name = f"{rigidity}{cutoff}{angle}Factor"
                        self.functions[name] = cutoff_factor(rigidity, binnings.flux_binning, cutoff, angle)
                        self.dependencies[name] = (rigidity, f"{cutoff}Cutoff{angle}")

        for particle_name, particle_id in MC_PARTICLE_IDS.items():
            for beta in ("Beta", "TofBeta", "RichBeta"):
                beta_distance_name = f"{beta}Distance{particle_name}"
                self.functions[beta_distance_name] = beta_distance(rigidity_estimator, beta, particle_id)
                self.dependencies[beta_distance_name] = (rigidity_estimator, beta)
            self.functions[f"ExpectedBeta{particle_name}"] = expected_beta(rigidity_estimator, particle_id)
            self.dependencies[f"ExpectedBeta{particle_name}"] = (rigidity_estimator,)
            self.functions[f"ShouldHaveRichBeta{particle_name}"] = should_have_rich_beta(particle_name)
            self.dependencies[f"ShouldHaveRichBeta{particle_name}"] = (f"ExpectedBeta{particle_name}", "TrkTrackHitsAGL", "TrkTrackHitsNaF")
            for other_particle_name, other_particle_id in MC_PARTICLE_IDS.items():
                if other_particle_id > particle_id and MC_PARTICLE_CHARGE_ARRAY[particle_id] > 0 and MC_PARTICLE_CHARGE_ARRAY[particle_id] == MC_PARTICLE_CHARGE_ARRAY[other_particle_id]:
                    self.functions[f"RichBetaCouldResolve{particle_name}{other_particle_name}"] = rich_beta_could_resolve(rigidity_estimator, particle_name, other_particle_name)
                    self.dependencies[f"RichBetaCouldResolve{particle_name}{other_particle_name}"] = (rigidity_estimator, "TrkTrackHitsAGL", "TrkTrackHitsNaF")
                    self.functions[f"RichBetaCouldResolve{particle_name}{other_particle_name}Ratio"] = rich_beta_could_resolve_ratio(rigidity_estimator, particle_name, other_particle_name)
                    self.dependencies[f"RichBetaCouldResolve{particle_name}{other_particle_name}Ratio"] = (rigidity_estimator, "TrkTrackHitsAGL", "TrkTrackHitsNaF")
                    self.functions[f"RichBetaCouldResolve{particle_name}{other_particle_name}RatioNaFOnly"] = rich_beta_could_resolve_ratio_naf_only(rigidity_estimator, particle_name, other_particle_name)
                    self.dependencies[f"RichBetaCouldResolve{particle_name}{other_particle_name}RatioNaFOnly"] = (rigidity_estimator, "TrkTrackHitsNaF")
        for beta in ("Beta", "TofBeta", "RichBeta"):
            if "signal_ids" in config["analysis"]:
                min_distance_name = f"Min{beta}SignalDistance"
                signal_ids = [MC_PARTICLE_IDS[particle_name] for particle_name in config["analysis"]["signal_ids"]]
                self.functions[min_distance_name] = min_beta_distance(rigidity_estimator, beta, signal_ids)
                self.dependencies[min_distance_name] = (rigidity_estimator, beta)
            self.functions[f"Abs{beta}MinusTrueBeta"] = abs_difference_to_true_value("McBeta", beta)
            self.functions[f"{beta}MinusTrueBeta"] = difference_to_true_value("McBeta", beta)
            self.functions[f"{beta}MinusTrueBetaAGL"] = difference_to_true_value("McBeta", beta)
            self.functions[f"{beta}MinusTrueBetaNaF"] = difference_to_true_value("McBeta", beta)
            self.dependencies[f"Abs{beta}MinusTrueBeta"] = ("McBeta", beta)
            self.dependencies[f"{beta}MinusTrueBeta"] = ("McBeta", beta)
            self.dependencies[f"{beta}MinusTrueBetaAGL"] = ("McBeta", beta)
            self.dependencies[f"{beta}MinusTrueBetaNaF"] = ("McBeta", beta)
        if "signal_ids" in config["analysis"]:
            signal_ids = [MC_PARTICLE_IDS[particle_name] for particle_name in config["analysis"]["signal_ids"]]
            self.functions[f"MinRichBetaSignalDistanceIfRing"] = min_beta_distance_if(rigidity_estimator, "RichBeta", signal_ids, "HasRich")
            self.dependencies[f"MinRichBetaSignalDistanceIfRing"] = (rigidity_estimator, "RichBeta", "HasRich")
            self.functions[f"MinRichBetaSignalDistanceIfAGL"] = min_beta_distance_if(rigidity_estimator, "RichBeta", signal_ids, "RichIsAGL")
            self.dependencies[f"MinRichBetaSignalDistanceIfAGL"] = (rigidity_estimator, "RichBeta", "RichIsAGL")
            self.functions[f"MinRichBetaSignalDistanceIfNaF"] = min_beta_distance_if(rigidity_estimator, "RichBeta", signal_ids, "RichIsNaF")
            self.dependencies[f"MinRichBetaSignalDistanceIfNaF"] = (rigidity_estimator, "RichBeta", "RichIsNaF")

        for charge in range(1, config["analysis"]["max_charge"] + 1):
            self.functions[f"TofMassQ{charge}"] = calculate_mass(rigidity_estimator, "TofBeta", charge)
            self.dependencies[f"TofMassQ{charge}"] = (rigidity_estimator, "TofBeta")
            self.functions[f"RichMassQ{charge}"] = calculate_mass(rigidity_estimator, "RichBeta", charge)
            self.dependencies[f"RichMassQ{charge}"] = (rigidity_estimator, "RichBeta")

        for rigidity_postfix in ("KalmanAll", "KalmanInner"):
            self.functions[f"RichMass{rigidity_postfix}"] = calculate_mass(f"TrkRigidity{rigidity_postfix}", "RichBeta", "TrkCharge")
            self.dependencies[f"RichMass{rigidity_postfix}"] = (f"TrkRigidity{rigidity_postfix}", "RichBeta", "TrkCharge")

        for mass_prefix in ("Mass", "RichMass", "TofMass"):
            for postfix in [""] + [f"Q{charge}" for charge in range(1, config["analysis"]["max_charge"] + 1)]:
                mass = f"{mass_prefix}{postfix}"
                self.functions[f"Abs{mass}MinusTrueMass"] = abs_difference_to_true_value("McMass", mass)
                self.functions[f"{mass}MinusTrueMass"] = difference_to_true_value("McMass", mass)
                self.dependencies[f"Abs{mass}MinusTrueMass"] = ("McMass", mass)
                self.dependencies[f"{mass}MinusTrueMass"] = ("McMass", mass)

        if "signal_ids" in config["analysis"]:
            for particle_name in config["analysis"]["signal_ids"]:
                total_energy_name = f"TotalEnergy{particle_name}"
                self.functions[total_energy_name] = total_energy(rigidity_estimator, MC_PARTICLE_IDS[particle_name])
                self.dependencies[total_energy_name] = (rigidity_estimator,)
                energy_over_energy_name = f"EnergyOverEnergy{particle_name}"
                self.functions[energy_over_energy_name] = energy_over_rigidity(total_energy_name)
                self.dependencies[energy_over_energy_name] = (total_energy_name, "EcalEnergyDeposited")

        if "likelihoods" in config:
            for likelihood_config in config["likelihoods"].values():
                for likelihood_varname in likelihood_config["creates"].values():
                    likelihood = Likelihood.load(config, workdir, likelihood_varname, rigidity_estimator)
                    if likelihood is not None:
                        self.functions[likelihood_varname] = likelihood
                        self.dependencies[likelihood_varname] = (rigidity_estimator, likelihood.input_variable,)
        for mva_name, mva_config in list(config.get("mvas", {}).items()) + list(config.get("regression_mvas", {}).items()):
            if "creates" in mva_config:
                for mva_var_name, load_mva, mva_dependencies in MVA.load_all(mva_config, mva_name, rigidity_estimator, config, workdir, binnings):
                    lazy_mva = LazyMVA(load_mva)
                    self.functions[mva_var_name] = lazy_mva.predict
                    self.dependencies[mva_var_name] = mva_dependencies
                    self.functions[f"{mva_var_name}SignalEfficiency"] = lazy_mva.predict_as_efficiency
                    self.dependencies[f"{mva_var_name}SignalEfficiency"] = mva_dependencies

        for selection_name, selection_config in config["selections"].items():
            var_name = f"PassesSelection{selection_name}"
            self.functions[var_name] = LazySelection(selection_config, rigidity_estimator=rigidity_estimator, binnings=binnings, config=config, workdir=workdir, labelling=labelling).apply
            self.dependencies[var_name] = list(selection_config["cuts"]) + ["McRigidity", rigidity_estimator]

    def overwrite_backtracing(self):
        self.functions["BacktracingFromSpaceOrMc"] = lambda chunk: np.ones(len(chunk), dtype=bool)


class VariableLabels:
    def __init__(self, config=None, workdir=None, rigidity_estimator=None):
        self.labels = None
        self.simple_labels = None
        self.descriptions = None
        self.units = None
        self.items = None
        self.rigidity_estimator = rigidity_estimator
        if config is not None:
            self.initialize(config, workdir, rigidity_estimator)

    def initialize(self, config, workdir, rigidity_estimator):
        self.labels = {
            "Time": "Time",
            "RunNumber": "Run Number",
            "EventNumber": "Event Number",
            "TotalWeight": "Total Weight",
            "Prescaling Weight": "Prescaling Weight",
            "McWeight": "MC Weight",
            "TriggerFlags": "Trigger Flags",
            "AcceptanceCategory": "Acceptance Category",
            "McAcceptanceCategory": "Acceptance Category (MC track)",
            "McParticleID": "MC Particle ID",
            "McMomentum": "MC True Momentum",
            "TrdIsBadRun": "Event in TRD Bad Run",
            "TrdIsCalibrated": "Run has TRD calibration",
            "TrdNSegmentsXZ": "TRD Segments in XZ plane",
            "TrdNSegmentsYZ": "TRD Segments in YZ plane",
            "TrdHasTrack": "Event has TRD track",
            "TrdLrHeliumElectron": "TRD Likelihood ratio He/(He+e⁻)",
            "TrdLrHeliumProton": "TRD Likelihood ratio He/(He+p)",
            "TrdLikelihoodRatioHeliumOverElectron": "TRD Likelihood ratio He/e⁻",
            "TrdLikelihoodRatioHeliumOverProton": "TRD Likelihood ratio He/p",
            "TrdLlhHelium": "TRD Likelihood Helium",
            "TrdLlhProton": "TRD Likelihood Proton",
            "TrdLlhElectron": "TRD Likelihood Electron",
            "TrdNActiveLayers": "TRD number of layers with hit",
            "TrdNHits": "TRD number of hits",
            "TrdTrackCoordX": "TRD Track X at TRD center",
            "TrdTrackCoordY": "TRD Track Y at TRD center",
            "TrdTrackCoordZ": "TRD Track Z at TRD center",
            "TrdNVerticesX": "TRD vertices in XZ plane",
            "TrdNVerticesY": "TRD vertices in YZ plane",
            "TrdNVerticesXY": "matching TRD vertice pairs",
            "TrdNVerticesOnTrackX": "TRD vertices in XZ plane on track",
            "TrdNVerticesOnTrackY": "TRD vertices in YZ plane on track",
            "TrdNVerticesOnTrackXY": "Matching TRD vertice pairs on track",
            "TrdHitCoordX": "TRD Hit X",
            "TrdHitCoordY": "TRD Hit Y",
            "TrdHitCoordZ": "TRD Hit Z",
            "TrdHitPathlength": "TRD Hit pathlength in tube",
            "TrdHitAmplitude": "TRD Hit Amplitude",
            "TrdHitDistanceToTrack": "TRD Hit Distance to Track",
            "TrdTrkDistanceX": "$X_{TRD} - X_{Tracker}$",
            "TrdTrkDistanceY": "$Y_{TRD} - Y_{Tracker}$",
            "TrdTrkDistance": "$|x_{TRD} - x_{Tracker}|$",
            "AccNClusters": "ACC Clusters",
            "AccNClustersTrigger": "ACC Clusters from trigger",
            "HasRich": "Event has RICH ring",
            "RichIsNaF": "RICH ring is from NaF radiator",
            "RichIsAGL": "RICH ring is from Aerogel radiator",
            "RichIsGood": "RICH ring is not contaminated",
            "RichBeta": "RICH β",
            "RichBetaError": "RICH β error",
            "RichBetaConsistency": "RICH β consistency",
            "RichBetaProbability": "RICH β probability",
            "RichCharge": "RICH Charge",
            "RichDistanceToTileBorder": "RICH track distance to radiator tile border",
            "RichNCollectedPhotoElectrons": "Collected photo electrons in RICH ring",
            "RichNExpectedPhotoElectrons": "Expected photo electrons in RICH ring",
            "RichNPhotoElectrons": "Photo electrons in RICH ring",
            "RichNHits": "Hits in RICH ring",
            "RichNUsedHits": "Hits in RICH ring used for reconstruction",
            "RichNPMTs": "Numebr of PMTs in RICH ring",
            "TofCharge": "ToF Charge",
            "TofChargeLower": "Lower ToF Charge",
            "TofChargeUpper": "Upper ToF Charge",
            "TofBeta": "ToF β",
            "TofBetaError": "ToF β error",
            "TofNClusters": "Clusters in ToF",
            "TofNLayers": "ToF layers with cluster",
            "TofClusterCoordX": "ToF cluster X coordinate",
            "TofClusterCoordY": "ToF cluster Y coordinate",
            "TofClusterCoordZ": "ToF cluster Z coordinate",
            "TofClusterEnergy": "Deposited Energy in ToF cluster",
            "TofChargeInLayer": "ToF Layer Charge",
            "TofHasChargeInLayer": "ToF Layer Charge exists",
            "TrkNTracks": "Tracks in tracker",
            "TrkCharge": "Tracker Charge",
            "TrkChargeError": "Trk Charge error",
            "TrkChargeYJ": "Trk Charge (Yi Jia)",
            "TrkNLayersInnerXY": "Inner tracker layers with XY hit",
            "TrkNLayersInnerY": "Inner tracker layers with Y hit",
            "TrkNHitsInLowerInnerLayersY": "Y hits in layer 6–8",
            "TrkNHitsInUpperInnerLayersY": "Y hits in layer 2-5",
            "TrkNClusters": "Clusters in tracker",
            "TrkNHits": "Hits in tracker",
            "TrkNUnassociatedHits": "Unassociated tracker hits",
            "TrkHitPatternXY": "Tracker Layer XY Hit Pattern",
            "TrkHitPatternY": "Tracker Layer Y Hit Pattern",
            "TrkClusterDistances": "Distance to closest unused tracker cluster",
            "TrkFitResiduals": "Tracker fit residuals",
            "TrkRigiditiesWithoutHit": "Rigidity of fit without hit in this layer",
            "TrkClusterSignalRatios": "Tracker hit signal ratio",
            "TrkLayerChargesX": "Tracker charge in layer from X clusters",
            "TrkLayerChargesY": "Tracker charge in layer from Y clusters",
            "TrkLayerChargesYJ": "Tracker charge in layer (Yi Jia method)",
            "TrkHasFitCoords": "Tracker track has fit coordinates stored",
            "TrkFitCoordsAreAll": "Tracker track fit coordinates belong to All geometry fit",
            "TrkTrackHasHitX": "Tracker track has X hit in layer",
            "TrkTrackHasHitY": "Tracker track has Y hit in layer",
            "TrkTrackHitCoordX": "Tracker hit X coordinate", 
            "TrkTrackHitCoordY": "Tracker hit Y coordinate", 
            "TrkTrackHitCoordZ": "Tracker hit Z coordinate", 
            "TrkTrackHitDepositedEnergyX": "Tracker hit X cluster deposited energy", 
            "TrkTrackHitDepositedEnergyY": "Tracker hit Y cluster deposited energy", 
            "TrkTrackFitCoordX": "Track fit interpolated X coordinate", 
            "TrkTrackFitCoordY": "Track fit interpolated Y coordinate", 
            "TrkTrackFitCoordZ": "Track fit interpolated Z coordinate", 
            "TrkTrackFitDirXZ": "Track fit direction X/Z",
            "TrkTrackFitDirYZ": "Track fit direction Y/Z",
            "TrkTrackFitCoordXAtTrd": "Track fit X coordinate at TRD center",
            "TrkTrackFitCoordYAtTrd": "Track fit Y coordinate at TRD center",
            "TrkTrackFitDirXZAtTrd": "Track fit direction X/Z at TRD center",
            "TrkTrackFitDirYZAtTrd": "Track fit direction Y/Z at TRD center",
            "TrkTrackFitCoordXAtRich": "Track fit X coordinate at RICH radiator",
            "TrkTrackFitCoordYAtRich": "Track fit Y coordinate at RICH radiator",
            "TrkTrackFitDirXZAtRich": "Track fit direction X/Z at RICH radiator",
            "TrkTrackFitDirYZAtRich": "Track fit direction Y/Z at RICH radiator",
            "TrkTrackFitCoordXAtEcal": "Track fit X coordinate at Ecal center",
            "TrkTrackFitCoordYAtEcal": "Track fit Y coordinate at Ecal center",
            "TrkTrackFitDirXZAtEcal": "Track fit direction X/Z at Ecal center",
            "TrkTrackFitDirYZAtEcal": "Track fit direction Y/Z at Ecal center",
            "TrkTrackFitCoordXAtTof": "Track fit interpolated X coordinate at ToF layer", 
            "TrkTrackFitCoordYAtTof": "Track fit interpolated Y coordinate at ToF layer", 
            "HasEcal": "ECAL shower exists",
            "EcalDepositedEnergy": "Deposited Energy in ECAL",
            "Longitude": "Longitude",
            "Latitude": "Latitude",
            "BacktracingStatus": "Geomagnetic Backtracing",
            "BacktracingFromSpaceOrMc": "Event backtraced to Space",
            "TrkRigidityFallback": "Track fit rigidity All or Inner",
            "TrdTrkDistanceX": "Distance between TRD and Tracker track X at TRD center",
            "TrdTrkDistanceY": "Distance between TRD and Tracker track Y at TRD center",
            "TrdTrkAngleDistanceXZ": "$\\Theta_X$ Tracker - TRD",
            "TrdTrkAngleDistanceYZ": "$\\Theta_Y$ Tracker - TRD",
            "TrkFitResidualsX": "Track fit residuals X",
            "TrkFitResidualsY": "Track fit residuals Y",
            "TrkTrackHasHitXY": "Track has XY hit in layer",
            "TrkTrackHasHitXY": "Track has XY hit in layer",
            "TrkTrackHasGoodHitX": "Hit residual X < 75um in layer",
            "TrkTrackHasGoodHitY": "Hit residual Y < 30um in layer",
            "TrkTrackHasGoodHitXY": "Hit residual X < 75um and Y < 30um in layer",
            "TrkNLayersGoodXY": "Tracker layers with small fit residual",
            "TrkNLayersInnerGoodXY": "Inner tracker layers with small fit residual",
            "TrkHitInsideLayer": "Tracker hit coord is within layer",
            "TrkNHitsInsideLayer": "Tracker hits within layer",
            "TrkNHitsInsideLayerInner": "Inner tracker hits within layer",
            "TrkTrackHitCoordXL34": "Avg. tracker hit X coordinate in layers 3 and 4",
            "TrkTrackHitCoordXL56": "Avg. tracker hit X coordinate in layers 5 and 6",
            "TrkTrackHitCoordXL78": "Avg. tracker hit X coordinate in layers 7 and 8",
            "TrkTrackRadiusL34": "Track Radius in layer 3 and 4",
            "TrkTrackRadiusL56": "Track Radius in layer 5 and 6",
            "TrkTrackRadiusL78": "Track Radius in layer 7 and 8",
            "MinTrkFootDistance": "Minimum distance from any tracker foot",
            "NHitTrkFeet": "Tracker feet along track",
            "TrkMinLayerChargeX": "Lowest tracker layer charge from X clusters",
            "TrkMinLayerChargeY": "Lowes tracker layer charge from Y clusters",
            "TrkMaxLayerChargeX": "Highest tracker layer charge from X clusters",
            "TrkMaxLayerChargeY": "Highest tracker layer charge from Y clusters",
            "TrkMinLayerCharge": "Lowest tracker layer charge",
            "TrkMaxLayerCharge": "Highest tracker layer charge",
            "TrkLayerChargeRmsX": "RMS($Q_{X,1-9}$)",
            "TrkLayerChargeRmsY": "RMS($Q_{Y,1-9}$)",
            "TrkMaxLayerChargeDelta": "Highest difference between X and Y tracker layer charge",
            "TrkMaxRelativeDeltaSagitta": "Max($|\\Delta S/S|$)",
            "TrkDoubleLayerPatternY": "Hit Y pattern in inner tracker double layers",
            "TrkDoubleLayerPatternXY": "Hit XY pattern in inner tracker double layers",
            "TrkExternalLayerPattern": "Hit pattern in external tracker layers",
            "TrkHasHitInExternalLayers": "Track has at least one hit in tracker layer 1 or 9",
            "TrkNHitsInExternalLayers": "Tracker hits in layer 1 and 9",
            "McRigidity": "MC true Rigidity",
            "McBeta": "MC true β",
            "McMass": "MC true mass",
            "TrackAngle": "Track angle relative to vertical",
            "TrackAnglePhi": "Track angle in plane",
            "TrackAngleXZ": "Track angle XZ",
            "TrackAngleYZ": "Track angle XZ",
            "HasTrigger": "Event is triggered",
            "HasPhysicsTrigger": "Event is triggered by any physics trigger",
            "HasUnbiasedTrigger": "Event is triggered by unbiased trigger",
            "HasUnbiasedTofTrigger": "Event is triggered by unbiased ToF trigger",
            "UnbiasedTriggerWeight": "Unbiased Trigger prescaling weight",
            "TofUpperChargeFromLayers": "Upper ToF Charge from layers",
            "TofLowerChargeFromLayers": "Lower ToF Charge from layers",
            "Beta": "β (RICH or ToF)",
            "TofMass": "$m_{ToF}$",
            "RichMass": "$m_{RICH}$",
            "TrkTrackHitsRich": "Tracker track intersects RICH radiator",
            "TrkTrackHitsAGL": "Tracker track intersects RICH Aerogel radiator",
            "TrkTrackHitsNaF": "Tracker track intersects RICH NaF radiator",
            "RichBetaResolutionIsGood": "RICH β difference to true β is below limit",
            "MinTofBetaSignalDistance": "$\\beta_{ToF}-\\beta_{{}^4He/{}^3He}$",
            "TrkChargeRms": "Tracker charge RMS",
            "TrdHitAmplitudePerPathlength": "TRD Hit Amplitude per Pathlength",
            "TrdMedianCharge": "TRD charge",
            "TrdMedianHitAmplitude": "TRD Median Hit Amplitude",
            "TrdMedianHitAmplitudePerPathlength": "TRD Median Hit Amplitude per Pathlength",
            "TrdNHitsOnTrack": "TRD hits within 2.5mm from track",
            "EnergyOverRigidity": "ECAL deposited Energy over Rigidity",
            "LogEnergyOverRigidity": "log(ECAL deposited Energy over Rigidity)",
            "RichPhotoElectronRatioRingVsEvent": "RICH ratio of photo electrons ring vs. event",
            "RichPhotoElectronRatioExpectedVsMeasured": "RICH ratio of photo electrons expected vs. measured",
        }
        self.units = {
            "Time": "s",
            "McMomentum": "GeV",
            "TrdTrackCoordX": "cm",
            "TrdTrackCoordY": "cm",
            "TrdTrackCoordZ": "cm",
            "TrdHitCoordX": "cm",
            "TrdHitCoordY": "cm",
            "TrdHitCoordZ": "cm",
            "TrdHitPathlength": "cm",
            "TrdHitAmplitude": "ADC",
            "TrdHitDistanceToTrack": "cm",
            "RichDistanceToTileBorder": "cm",
            "TofClusterCoordX": "cm",
            "TofClusterCoordY": "cm",
            "TofClusterCoordZ": "cm",
            "TofClusterEnergy": "MeV",
            "TrkClusterDistances": "um",
            "TrkFitResiduals": "um",
            "TrkRigiditiesWithoutHit": "GV",
            "TrkTrackHitCoordX": "cm", 
            "TrkTrackHitCoordY": "cm", 
            "TrkTrackHitCoordZ": "cm", 
            "TrkTrackHitDepositedEnergyX": "MeV", 
            "TrkTrackHitDepositedEnergyY": "MeV", 
            "TrkTrackFitCoordX": "cm", 
            "TrkTrackFitCoordY": "cm", 
            "TrkTrackFitCoordZ": "cm", 
            "TrkTrackFitCoordXAtTrd": "cm",
            "TrkTrackFitCoordYAtTrd": "cm",
            "TrkTrackFitCoordXAtRich": "cm",
            "TrkTrackFitCoordYAtRich": "cm",
            "TrkTrackFitCoordXAtEcal": "cm",
            "TrkTrackFitCoordYAtEcal": "cm",
            "TrkTrackFitCoordXAtTof": "cm", 
            "TrkTrackFitCoordYAtTof": "cm", 
            "EcalDepositedEnergy": "GeV",
            "Longitude": "°",
            "Latitude": "°",
            "TrkRigidityFallback": "GV",
            "TrdTrkDistanceX": "cm",
            "TrdTrkDistanceY": "cm",
            "TrdTrkDistance": "cm",
            "TrdTrkAngleDistanceXZ": "rad",
            "TrdTrkAngleDistanceYZ": "rad",
            "TrkFitResidualsX": "um",
            "TrkFitResidualsY": "um",
            "TrkTrackHitCoordXL34": "cm",
            "TrkTrackHitCoordXL56": "cm",
            "TrkTrackHitCoordXL78": "cm",
            "TrkTrackRadiusL34": "cm",
            "TrkTrackRadiusL56": "cm",
            "TrkTrackRadiusL78": "cm",
            "MinTrkFootDistance": "cm",
            "McRigidity": "GV",
            "McMass": "GeV",
            "TrackAngle": "°",
            "TrackAnglePhi": "°",
            "TrackAngleXZ": "°",
            "TrackAngleYZ": "°",
            "RichMass": "GeV",
            "TrdHitAmplitudePerPathlength": "ADC/cm",
            "TrdMedianHitAmplitude": "ADC",
            "TrdMedianHitAmplitudePerPathlength": "ADC/cm",
            "EnergyOverRigidity": "GeV/GV",
        }
        self.simple_labels = {
            "TrdMedianCharge": "TRD Charge",
            "TrdMedianHitAmplitude": "TRD Median",
            "TrdMedianHitAmplitudePerPathlength": "TRD Median",
            "TrkMaxRelativeDeltaSagitta": "Max($|\\Delta S/S|$)",
        }

        self.descriptions = {
            "TofCharge": "Charge measurement (average of upper and lower ToF)",
            "TrackerCharge": "Charge measurement (average of inner tracker layers)",
            "TrkMaxRelativeDeltaSagitta": "Largest relative absolute change of Tracker Sagitta from refit without one track hit each",
            "TrkLayerChargeRmsX": "RMS of charge measurements in non-bending plane in all tracker layers",
            "TrkLayerChargeRmsY": "RMS of charge measurements in bending plane in all tracker layers",
            "TrdTrkAngleDistanceXZ": "Angle between TRD track and tracker track in non-bending plane",
            "TrdTrkAngleDistanceYZ": "Angle between TRD track and tracker track in bending plane",
            "NHitTrackerFeet": "Number of tracker feet within 5mm of the tracker track",
            "MinTrkFootDistance": "Smallest distance between track and any tracker foot",
            "TrkTrackRadiusL34": "Track distance from tracker center between layer 3 and 4",
            "TrkTrackRadiusL56": "Track distance from tracker center between layer 5 and 6",
            "TrkTrackRadiusL78": "Track distance from tracker center between layer 7 and 8",
            "TrkDoubleLayerPatternY": "Hit Y pattern in inner tracker double layers",
        }

        self.items = {}
        # "TriggerFlags": {}, # todo
        self.items["AcceptanceCategory"] = {
            AC_INNER | AC_RICH | AC_TRD | AC_L1 | AC_L9: "Fullspan",
            AC_INNER | AC_RICH | AC_TRD | AC_L1: "Inner+L1",
            AC_INNER | AC_RICH | AC_TRD | AC_L9: "Inner+L9",
            AC_INNER | AC_RICH | AC_TRD: "Inner Only",
        }
        self.items["McAcceptanceCategory"] = self.items["AcceptanceCategory"]
        self.items["McParticleID"] = MC_PARTICLE_LABELS
        self.items["BacktracingStatus"] = {0: "Unavailable", 1: "From Space", 2: "From Atmosphere", 3: "Trapped"}
        self.items["TrkDoubleLayerPatternY"] = {0b1111: "L2&(L3|L4)&(L5|L6)&(L7|L8)", 0b1110: "(L3|L4)&(L5|L6)&(L7|L8)"}
        self.items["TrkDoubleLayerPatternXY"] = self.items["TrkDoubleLayerPatternY"]
        self.items["TrkExternalLayerPattern"] = {0b00: "No hit", 0b01: "Hit in L1", 0b10: "Hit in L9", 0b11: "Hit in L1 and L9"}

        fit_algorithms = {
            "": ("C", "Choutko"),
            "Choutko": ("C", "Choutko"),
            "Kalman": ("K", "Kalman"),
            "Alcaraz": ("Al", "Alcaraz"),
            "Chikanian": ("Ch", "Chikanian"),
            "GBL": ("G", "GBL"),
        }
        fit_geometries = {
            "Inner": ("I", "Inner"),
            "All": ("A", "All"),
            "LowerHalf": ("L", "Lower Half"),
            "UpperHalf": ("U", "Upper Half"),
            "Upper": ("U", "Upper Half"),
            "InnerPlusL1": ("Inner+L1", "Inner+L1"),
            "InnerPlusL9": ("Inner+L9", "Inner+L9"),
        }

        for algorithm, (algorithm_label, algorithm_description) in fit_algorithms.items():
            for geometry, (geometry_label, geometry_description) in fit_geometries.items():
                self.labels[f"TrkHas{algorithm}{geometry}"] = f"{algorithm_description} fit in {geometry_description} geometry exists"
                #self.labels[f"TrkRigidity{algorithm}{geometry}"] = f"Rigidity {algorithm_label} {geometry_label}"
                self.labels[f"TrkRigidity{algorithm}{geometry}"] = f"$R_{{{algorithm_label},{geometry_label}}}$"
                self.simple_labels[f"TrkRigidity{algorithm}{geometry}"] = f"$R$"
                #self.labels[f"TrkRigidityAbs{algorithm}{geometry}"] = f"Rigidity {algorithm_label} {geometry_label}"
                self.labels[f"TrkRigidityAbs{algorithm}{geometry}"] = f"$|R_{{{algorithm_label},{geometry_label}}}|$"
                self.simple_labels[f"TrkRigidityAbs{algorithm}{geometry}"] = f"$|R|$"
                self.labels[f"TrkInverseRigidityError{algorithm}{geometry}"] = f"$\\sigma_{{1/R_{{{algorithm_label},{geometry_label}}}}}$"
                #self.labels[f"TrkChi2X{algorithm}{geometry}"] = f"χ²ₓ({algorithm_label}, {geometry_label})"
                #self.labels[f"TrkChi2Y{algorithm}{geometry}"] = f"χ²ᵧ({algorithm_label}, {geometry_label})"
                self.labels[f"TrkChi2X{algorithm}{geometry}"] = f"$\\chi^2_{{X,{algorithm_label},{geometry_label}}}$"
                self.labels[f"TrkChi2Y{algorithm}{geometry}"] = f"$\\chi^2_{{Y,{algorithm_label},{geometry_label}}}$"
                self.simple_labels[f"TrkChi2X{algorithm}{geometry}"] = f"$\\chi^2_X$"
                self.simple_labels[f"TrkChi2Y{algorithm}{geometry}"] = f"$\\chi^2_Y$"
                self.descriptions[f"TrkChi2X{algorithm}{geometry}"] = f"Trackfit $\\chi^2$ in non-bending plane from {algorithm_description} algorithm in {geometry_description} geometry"
                self.descriptions[f"TrkChi2Y{algorithm}{geometry}"] = f"Trackfit $\\chi^2$ in non-bending plane from {algorithm_description} algorithm in {geometry_description} geometry"
                self.labels[f"TrkRelativeError{algorithm}{geometry}"] = f"$\\sigma_R/R_{{{algorithm_label},{geometry_label}}}$"
                self.descriptions[f"TrkRelativeError{algorithm}{geometry}"] = f"Relative rigidity error of trackfit with {algorithm_description} algorithm in {geometry_description} geometry"
                self.units[f"TrkRigidity{algorithm}{geometry}"] = "GV"
                self.units[f"TrkRigidityAbs{algorithm}{geometry}"] = "GV"
                self.units[f"TrkInverseRigidityError{algorithm}{geometry}"] = "1/GV"

        for algorithm, (algorithm_label, algorithm_description) in fit_algorithms.items():
            for geometry1, (geometry1_label, geometry1_description) in fit_geometries.items():
                for geometry2, (geometry2_label, geometry2_description) in fit_geometries.items():
                    r1 = f"$R_{{{algorithm_label},{geometry1_label}}}$"
                    r2 = f"$R_{{{algorithm_label},{geometry2_label}}}$"
                    self.labels[f"TrkMatching{geometry1}Vs{geometry2}{algorithm}"] = f"(1/{r2}-1/{r1})/$\\sigma$"
                    self.labels[f"TrkSquareMatching{geometry1}Vs{geometry2}{algorithm}"] = f"(1/{r2}-1/{r1})²/$\\sigma$²"
                    self.labels[f"TrkAsymmetry{geometry1}Vs{geometry2}{algorithm}"] = f"(1/{r2}-1/{r1})/(1/{r1}+1/{r2})"
                    self.labels[f"TrkAbsAsymmetry{geometry1}Vs{geometry2}{algorithm}"] = f"|(1/{r2}-1/{r1})/(1/{r1}+1/{r2})|"
                    r1 = f"$R_{{{geometry1_label}}}$"
                    r2 = f"$R_{{{geometry2_label}}}$"
                    self.simple_labels[f"TrkMatching{geometry1}Vs{geometry2}{algorithm}"] = f"(1/{r2}-1/{r1})/$\\sigma$"
                    self.simple_labels[f"TrkSquareMatching{geometry1}Vs{geometry2}{algorithm}"] = f"(1/{r2}-1/{r1})²/$\\sigma$²"
                    self.simple_labels[f"TrkAsymmetry{geometry1}Vs{geometry2}{algorithm}"] = f"(1/{r2}-1/{r1})/(1/{r1}+1/{r2})"
                    self.simple_labels[f"TrkAbsAsymmetry{geometry1}Vs{geometry2}{algorithm}"] = f"|(1/{r2}-1/{r1})/(1/{r1}+1/{r2})|"
                    self.descriptions[f"TrkMatching{geometry1}Vs{geometry2}{algorithm}"] = f"Difference between sagittas in {geometry1_description} and {geometry2_description} geometry, both with {algorithm_description} algorithm, divided by sagitta uncertainty, multiplied with rigidity sign"
                    self.descriptions[f"TrkSquareMatching{geometry1}Vs{geometry2}{algorithm}"] = f"Squared difference between sagittas in {geometry1_description} and {geometry2_description} geometry, both with {algorithm_description} algorithm, divided by sagitta variance"
                    self.descriptions[f"TrkAsymmetry{geometry1}Vs{geometry2}{algorithm}"] = f"Difference between sagittas in {geometry1_description} and {geometry2_description} geometry, both with {algorithm_description} algorithm, divided by their sum"
                    self.descriptions[f"TrkAbsAsymmetry{geometry1}Vs{geometry2}{algorithm}"] = f"Absolute difference between sagittas in {geometry1_description} and {geometry2_description}, both with {algorithm_description} algorithm, divided by their sum"

        for geometry, (geometry_label, geometry_description) in fit_geometries.items():
            for algorithm1, (algorithm1_label, algorithm1_description) in fit_algorithms.items():
                for algorithm2, (algorithm2_label, algorithm2_description) in fit_algorithms.items():
                    r1 = f"$R_{{{algorithm1_label},{geometry_label}}}$"
                    r2 = f"$R_{{{algorithm2_label},{geometry_label}}}$"
                    self.labels[f"TrkMatching{algorithm1}Vs{algorithm2}{geometry}"] = f"(1/{r2}-1/{r1})/$\\sigma$"
                    self.labels[f"TrkSquareMatching{algorithm1}Vs{algorithm2}{geometry}"] = f"(1/{r2}-1/{r1})²/$\\sigma$²"
                    self.labels[f"TrkAsymmetry{algorithm1}Vs{algorithm2}{geometry}"] = f"(1/{r2}-1/{r1})/(1/{r1}+1/{r2})"
                    self.labels[f"TrkAbsAsymmetry{algorithm1}Vs{algorithm2}{geometry}"] = f"|(1/{r2}-1/{r1})/(1/{r1}+1/{r2})|"
                    r1 = f"$R_{{{algorithm1_label}}}$"
                    r2 = f"$R_{{{algorithm2_label}}}$"
                    self.simple_labels[f"TrkMatching{algorithm1}Vs{algorithm2}{geometry}"] = f"(1/{r2}-1/{r1})/$\\sigma$"
                    self.simple_labels[f"TrkSquareMatching{algorithm1}Vs{algorithm2}{geometry}"] = f"(1/{r2}-1/{r1})²/$\\sigma$²"
                    self.simple_labels[f"TrkAsymmetry{algorithm1}Vs{algorithm2}{geometry}"] = f"(1/{r2}-1/{r1})/(1/{r1}+1/{r2})"
                    self.simple_labels[f"TrkAbsAsymmetry{algorithm1}Vs{algorithm2}{geometry}"] = f"|(1/{r2}-1/{r1})/(1/{r1}+1/{r2})|"
                    self.descriptions[f"TrkMatching{algorithm1}Vs{algorithm2}{geometry}"] = f"Difference between sagittas with {algorithm1_description} and {algorithm2_description} algorithm, both in {geometry_description} geometry, divided by sagitta uncertainty, multiplied with rigidity sign"
                    self.descriptions[f"TrkSquareMatching{algorithm1}Vs{algorithm2}{geometry}"] = f"Squared difference between sagittas with {algorithm1_description} and {algorithm2_description} algorithm, both in {geometry_description} geometry, divided by sagitta variance"
                    self.descriptions[f"TrkAsymmetry{algorithm1}Vs{algorithm2}{geometry}"] = f"Difference between sagittas with {algorithm1_description} and {algorithm2_description} algorithm, both in {geometry_description} geometry, divided by their sum"
                    self.descriptions[f"TrkAbsAsymmetry{algorithm1}Vs{algorithm2}{geometry}"] = f"Absolute difference between sagittas with {algorithm1_description} and {algorithm2_description} algorithm, both in {geometry_description} geometry, diveded by their sum"

        for layer in range(1, 10):
            self.labels[f"TrkTrackHasHitXL{layer}"] = f"Track has X-Hit in layer {layer}"
            self.labels[f"TrkTrackHasHitYL{layer}"] = f"Track has Y-Hit in layer {layer}"
            self.labels[f"TrkTrackHitCoordXL{layer}"] = f"Tracker Hit X coordinate in layer {layer}"
            self.labels[f"TrkTrackHitCoordYL{layer}"] = f"Tracker Hit Y coordinate in layer {layer}"
            self.labels[f"TrkTrackFitCoordXL{layer}"] = f"Trackfit X coordinate in layer {layer}"
            self.labels[f"TrkTrackFitCoordYL{layer}"] = f"Trackfit Y coordinate in layer {layer}"
            self.labels[f"AbsTrkTrackFitCoordXL{layer}"] = f"Absolute Trackfit X coordinate in layer {layer}"
            self.labels[f"AbsTrkTrackFitCoordYL{layer}"] = f"Absolute Trackfit Y coordinate in layer {layer}"
            self.labels[f"TrkTrackFitRadiusL{layer}"] = f"Trackfit radius in layer {layer}"
            self.labels[f"TrkHasChargeL{layer}"] = f"Tracker has charge in layer {layer}"
            self.labels[f"TrkChargeL{layer}"] = f"Tracker charge in layer {layer}"
            self.labels[f"TrkTrackFitDistanceToSensorEdgeXL{layer}"] = f"Sensor Edge Distance X in layer {layer}"
            self.labels[f"TrkTrackFitDistanceToSensorEdgeYL{layer}"] = f"Sensor Edge Distance Y in layer {layer}"
            self.labels[f"TrkTrackFitDistanceToSensorEdgeL{layer}"] = f"Sensor Edge Distance in layer {layer}"
            self.units[f"TrkTrackHitCoordXL{layer}"] = "cm"
            self.units[f"TrkTrackHitCoordYL{layer}"] = "cm"
            self.units[f"TrkTrackFitCoordXL{layer}"] = "cm"
            self.units[f"TrkTrackFitCoordYL{layer}"] = "cm"
            self.units[f"TrkTrackFitRadiusL{layer}"] = "cm"
            self.units[f"TrkTrackFitDistanceToSensorEdgeXL{layer}"] = "cm"
            self.units[f"TrkTrackFitDistanceToSensorEdgeYL{layer}"] = "cm"
            self.units[f"TrkTrackFitDistanceToSensorEdgeL{layer}"] = "cm"
            self.descriptions[f"TrkTrackFitDistanceToSensorEdgeXL{layer}"] = f"Distance between track position in layer {layer} and tracker sensor edge in non-bending plane"
            self.descriptions[f"TrkTrackFitDistanceToSensorEdgeYL{layer}"] = f"Distance between track position in layer {layer} and tracker sensor edge in bending plane"
            self.descriptions[f"TrkTrackFitDistanceToSensorEdgeL{layer}"] = f"Distance between track position in layer {layer} and tracker sensor edge"


        for cutoff_model in ("Stoermer", "IGRF"):
            for angle in (25, 30, 35, 40):
                for sign in ("P", "N", "PN"):
                    self.labels[f"{cutoff_model}Cutoff{angle}{sign}"] = "{cutoff_model} Cutoff rigidity {angle}° {sign} / GV"
                    self.units[f"{cutoff_model}Cutoff{angle}{sign}"] = "GV"

        for tof_layer in range(1, 5):
            tof_label = "UToF" if tof_layer < 2 else "LToF"
            self.labels[f"TofTrkDistanceXT{tof_layer}"] = f"$X_{{Tracker}}-X_{{{tof_label}}}$"
            self.labels[f"TofTrkDistanceYT{tof_layer}"] = f"$Y_{{Tracker}}-Y_{{{tof_label}}}$"
            self.labels[f"TofTrkDistanceT{tof_layer}"] = f"$\\Delta{{Tracker,{tof_label}}}$"
            self.descriptions[f"TofTrkDistanceXT{tof_layer}"] = f"Track X distance to ToF cluster in ToF plane {tof_layer}"
            self.descriptions[f"TofTrkDistanceYT{tof_layer}"] = f"Track Y distance to ToF cluster in ToF plane {tof_layer}"
            self.descriptions[f"TofTrkDistanceT{tof_layer}"] = f"Track distance to ToF cluster in ToF plane {tof_layer}"
            self.units[f"TofTrkDistanceXT{tof_layer}"] = "cm"
            self.units[f"TofTrkDistanceYT{tof_layer}"] = "cm"
            self.units[f"TofTrkDistanceT{tof_layer}"] = "cm"

    def get_label(self, variable):
        label = self.labels.get(variable, variable)
        if variable in self.units:
            label = f"{label} / {self.units[variable]}"
        return label

    def get_simple_label(self, variable):
        label = self.simple_labels.get(variable, self.labels.get(variable, variable))
        if variable in self.units:
            label = f"{label} / {self.units[variable]}"
        return label

    def get_description(self, variable):
        return self.descriptions.get(variable, self.labels.get(variable))


VARIABLE_REQUIREMENTS = {
    "TrkChi2XInner": ("TrkHasInner",),
    "TrkChi2YInner": ("TrkHasInner",),
    "TrkChi2XAll": ("TrkHasAll",),
    "TrkChi2YAll": ("TrkHasAll",),
    "TrkChi2XKalmanInner": ("TrkHasKalmanInner",),
    "TrkChi2YKalmanInner": ("TrkHasKalmanInner",),
    "TrkChi2XKalmanAll": ("TrkHasKalmanAll",),
    "TrkChi2YKalmanAll": ("TrkHasKalmanAll",),
    "TrkChi2XAlcarazInner": ("TrkHasAlcarazInner",),
    "TrkChi2YAlcarazInner": ("TrkHasAlcarazInner",),
    "TrkChi2XAlcarazAll": ("TrkHasAlcarazAll",),
    "TrkChi2YAlcarazAll": ("TrkHasAlcarazAll",),
    "TrkMatchingChoutkoVsKalmanInner": ("TrkHasInner", "TrkHasKalmanInner"),
    "TrkMatchingChoutkoVsKalmanAll": ("TrkHasAll", "TrkHasKalmanAll"),
    "TrkMatchingChoutkoVsAlcarazInner": ("TrkHasInner", "TrkHasAlcarazInner"),
    "TrkMatchingChoutkoVsAlcarazAll": ("TrkHasAll", "TrkHasAlcarazAll"),
    "TrkMatchingKalmanVsAlcarazInner": ("TrkHasKalmanInner", "TrkHasAlcarazInner"),
    "TrkMatchingKalmanVsAlcarazAll": ("TrkHasKalmanAll", "TrkHasAlcarazAll"),
    "TrkMatchingAllVsInnerChoutko": ("TrkHasInner", "TrkHasAll"),
    "TrkMatchingAllVsInnerKalman": ("TrkHasKalmanInner", "TrkHasKalmanAll"),
    "TrkMatchingAllVsInnerAlcaraz": ("TrkHasAlcarazInner", "TrkHasAlcarazAll"),
    "TrkMatchingUpperVsLowerHalfChoutko": ("TrkHasUpperHalf", "TrkHasLowerHalf"),
    "TrkMatchingUpperVsLowerHalfKalman": ("TrkHasKalmanUpperHalf", "TrkHasKalmanLowerHalf"),
    "TrkMatchingUpperHalfVsInnerChoutko": ("TrkHasUpperHalf", "TrkHasInner"),
    "TrkMatchingUpperHalfVsInnerKalman": ("TrkHasKalmanUpperHalf", "TrkHasKalmanInner"),
    "TrkMatchingLowerHalfVsInnerChoutko": ("TrkHasLowerHalf", "TrkHasInner"),
    "TrkMatchingLowerHalfVsInnerKalman": ("TrkHasKalmanLowerHalf", "TrkHasKalmanInner"),
    "TrkAsymmetryChoutkoVsKalmanInner": ("TrkHasInner", "TrkHasKalmanInner"),
    "TrkAsymmetryChoutkoVsKalmanAll": ("TrkHasAll", "TrkHasKalmanAll"),
    "TrkAsymmetryChoutkoVsAlcarazInner": ("TrkHasInner", "TrkHasAlcarazInner"),
    "TrkAsymmetryChoutkoVsAlcarazAll": ("TrkHasAll", "TrkHasAlcarazAll"),
    "TrkAsymmetryKalmanVsAlcarazInner": ("TrkHasKalmanInner", "TrkHasAlcarazInner"),
    "TrkAsymmetryKalmanVsAlcarazAll": ("TrkHasKalmanAll", "TrkHasAlcarazAll"),
    "TrkAsymmetryAllVsInnerChoutko": ("TrkHasInner", "TrkHasAll"),
    "TrkAsymmetryAllVsInnerKalman": ("TrkHasKalmanInner", "TrkHasKalmanAll"),
    "TrkAsymmetryAllVsInnerAlcaraz": ("TrkHasAlcarazInner", "TrkHasAlcarazAll"),
    "TrkAsymmetryUpperVsLowerHalfChoutko": ("TrkHasUpperHalf", "TrkHasLowerHalf"),
    "TrkAsymmetryUpperVsLowerHalfKalman": ("TrkHasKalmanUpperHalf", "TrkHasKalmanLowerHalf"),
    "TrkAsymmetryUpperHalfVsInnerChoutko": ("TrkHasUpperHalf", "TrkHasInner"),
    "TrkAsymmetryUpperHalfVsInnerKalman": ("TrkHasKalmanUpperHalf", "TrkHasKalmanInner"),
    "TrkAsymmetryLowerHalfVsInnerChoutko": ("TrkHasLowerHalf", "TrkHasInner"),
    "TrkAsymmetryLowerHalfVsInnerKalman": ("TrkHasKalmanLowerHalf", "TrkHasKalmanInner"),
    "TrkRelativeErrorChoutkoInner": ("TrkHasInner",),
    "TrkRelativeErrorChoutkoAll": ("TrkHasAll",),
    "TrkRelativeErrorKalmanInner": ("TrkHasKalmanInner",),
    "TrkRelativeErrorKalmanAll": ("TrkHasKalmanAll",),
    "RichBeta": ("HasRich",),
}
