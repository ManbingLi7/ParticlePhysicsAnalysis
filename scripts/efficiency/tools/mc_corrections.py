import numpy as np
import awkward as ak

def apply_mc_correction(value, true_value, shift, scaler):
    corrected_value = (value - true_value) * scaler + true_value
    corrected_value = corrected_value + shift
    return corrected_value

mc_beta_correction_shift = {"Tof": 0.0016, "RichNaF": 0.00018, "RichAgl": 0.000115}
mc_beta_correction_scale = {"Tof": 1.0, "RichNaF": 1.1, "RichAgl": 1.0}
