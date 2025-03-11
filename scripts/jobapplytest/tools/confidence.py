
import numpy as np
from scipy.stats import norm, chi2, poisson

def loss_at_angle(loss, x, y, dx, dy):
    def _loss_at_r(r):
        return loss(x + r * dx, y + r * dy)
    return _loss_at_r

def loss_in_direction(loss, x0, dx):
    def _loss_at_r(r):
        return loss(x0 + r * dx)
    return _loss_at_r


def fit_linear(loss_func, target_value, precision=1e-6):
    min_r = 0
    max_r = 1
    while loss_func(max_r) < target_value:
        min_r = max_r
        max_r *= 2
    loss = loss_func((min_r + max_r) / 2)
    while np.isfinite(loss) and abs(loss - target_value) > precision:
        mid_r = (min_r + max_r) / 2
        if loss_func(mid_r) < target_value:
            min_r = mid_r
        else:
            max_r = mid_r
        loss = loss_func((min_r + max_r) / 2)
    if not np.isfinite(loss):
        print("Cannot calculate limit, loss not finite")
    best_r = (min_r + max_r) / 2
    best_fit = loss_func(best_r)
    diff = abs(best_fit - target_value)
    return best_r


def calculate_contour(loss_function, best_fit_parameters, loss_delta, sections=120):
    assert len(best_fit_parameters) == 2
    minimum_loss = loss_function(*best_fit_parameters)
    target_loss = minimum_loss + loss_delta
    param_x, param_y = best_fit_parameters

    contour_points = []
    for index in range(sections):
        angle = 2 * np.pi * index / sections
        dx_0 = np.cos(angle)
        dy_0 = np.sin(angle)
        effective_angle = np.arctan2(dx_0, dy_0 / 10)
        dx = np.cos(effective_angle)
        dy = np.sin(effective_angle)

        linear_loss = loss_at_angle(loss_function, param_x, param_y, dx, dy)
        target_r = fit_linear(linear_loss, target_loss)
        contour_points.append((param_x + dx * target_r, param_y + dy * target_r))
    contour_points.append(contour_points[0])
    return contour_points


def calculate_confidence_interval(confidence_level, loss, parameters, adjust_lower=False, compensation=0):
    loss_at_minimum = loss(*parameters)
    p = norm.cdf(confidence_level) - norm.cdf(-confidence_level)
    cl = chi2(1).ppf(p)
    l1 = loss_at_angle(loss, parameters[0], parameters[1], -1, compensation)
    r1 = fit_linear(l1, loss_at_minimum + cl)
    limit1 = parameters[0] - r1
    if adjust_lower:
        if limit1 < 0:
            delta_loss_at_zero = loss(0, parameters[1]) - loss_at_minimum
            p_less = chi2(1).cdf(delta_loss_at_zero) / 2
            cln = chi2(1).ppf(p - p_less)
            clo = cl
            cl = cln
            limit1 = 0
            l2 = loss_at_angle(loss, parameters[0], parameters[1], 1, -compensation)
    l2 = loss_at_angle(loss, parameters[0], parameters[1], 1, -compensation)
    r2 = fit_linear(l2, loss_at_minimum + cl)
    limit2 = parameters[0] + r2
    return limit1, limit2

def calculate_confidence_interval_1d(confidence_level, loss, parameters, parameter_index=0):
    loss_at_minimum = loss(*parameters)
    p = norm.cdf(confidence_level) - norm.cdf(-confidence_level)
    cl = chi2(1).ppf(p)
    param_values = list(parameters)
    def _loss(parameter_value):
        param_values[parameter_index] = parameter_value
        return loss(*param_values)
    l1 = loss_in_direction(_loss, parameters[parameter_index], 1)
    r1 = fit_linear(l1, loss_at_minimum + cl)
    limit1 = parameters[parameter_index] + r1
    l2 = loss_in_direction(_loss, parameters[parameter_index], -1)
    r2 = fit_linear(l2, loss_at_minimum + cl)
    limit2 = parameters[parameter_index] - r2
    return limit2, limit1
