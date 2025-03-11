import numpy as np
import scipy.special
from scipy.stats import chi2 as scp_chi2, norm as scp_gaussian, poisson as scp_poisson
from scipy.interpolate import interp1d, PchipInterpolator
from uncertainties import unumpy
import uncertainties

def gaussian(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def normalized_gaussian(x, norm, mu, sigma):
    return norm * 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def cumulative_norm_gaus(edges, norm, mu, sigma):
    x = (edges[1:] + edges[:-1])/2
    pdf = normalized_gaussian(x, norm, mu, sigma)
    cpdf = np.cumsum(pdf)
    return np.concatenate(([0], cpdf)) 

def double_gaussian(x, counts, mu, sigma, sigma_ratio, fraction_sec):
    return counts * ((1 - fraction_sec) * gaussian(x, mu, sigma) + fraction_sec * gaussian(x, mu, sigma_ratio * sigma))

def asy_gaussian(x, mu, sigma, asyfactor):
    norm_left = 2 / ((1 + asyfactor) * np.sqrt(2.0 * np.pi * sigma**2))
    norm_right = 2 / ((1 + asyfactor) * np.sqrt(2.0 * np.pi * (asyfactor * sigma)**2))
    inleft = np.zeros(x.shape)
    inright = np.zeros(x.shape)
    inleft[sigma != 0] = (x[sigma != 0] - mu[sigma != 0])/(sigma[sigma != 0])
    inright[sigma != 0] = (x[sigma != 0] - mu[sigma != 0])/(asyfactor[sigma != 0] * sigma[sigma != 0])
    pdf = np.zeros(x.shape)
    pdf[x<mu] = norm_left[x<mu] * np.exp(-0.5 * inleft[x<mu]**2)
    pdf[x>=mu] = norm_right[x>=mu] * asyfactor[x>=mu] * np.exp(-0.5 * inright[x>=mu]**2)
    return pdf

def asy_gaussian_1d(x, mu, sigma, asyfactor):
    if isinstance(x, float):
        x = np.array([x])
    norm_left = 2 / ((1 + asyfactor) * np.sqrt(2.0 * np.pi * sigma**2))
    norm_right = 2 / ((1 + asyfactor) * np.sqrt(2.0 * np.pi * (asyfactor * sigma)**2))
    inleft = np.zeros(x.shape)
    inright = np.zeros(x.shape)
    inleft = (x - mu)/(sigma) if sigma != 0 else 0
    inright = (x - mu)/(asyfactor * sigma) if sigma != 0 else 0
    pdf = np.zeros(x.shape)
    pdf[x<mu] = norm_left * np.exp(-0.5 * inleft[x<mu]**2)
    pdf[x>=mu] = norm_right * asyfactor * np.exp(-0.5 * inright[x>=mu]**2)
    return pdf

def upoly(x, *upars):
    result = unumpy.uarray([0.0] * len(x), [0.0] * len(x))
    for order, par in enumerate(upars):
        result += x**order * par
    return result

def poly(x, *pars):
    result = np.zeros(x.shape)
    for order, par in enumerate(pars):
        result += x**order * par
    return result

def expo_func(x, pa, pb, pc):
    pdf = pa* (1 - np.exp((x-pb)/pc))
    return pdf

def exp_modified_gaussian(x, mu, sigma, tau):
    l2 = 1 / (2 * tau)
    ls = sigma**2 / tau
    return l2 * np.exp(l2 * (2 * mu + ls - 2 * x)) * scipy.special.erfc((mu + ls - x) / (2**0.5 * sigma))

def normalize_exp_modified_gaussian(x, mu, sigma, tau, norm):
    l2 = 1 / (2 * tau)
    ls = sigma**2 / tau
    return norm * l2 * np.exp(l2 * (2 * mu + ls - 2 * x)) * scipy.special.erfc((mu + ls - x) / (2**0.5 * sigma))

def cumulative_normalize_exp_modified_gaussian(edges, mu, sigma, tau, norm):
    x = (edges[1:] + edges[:-1])/2  
    l2 = 1 / (2 * tau)
    ls = sigma**2 / tau
    pdf = normalize_exp_modified_gaussian(x, mu, sigma, tau, norm)
    cpdf = np.cumsum(pdf)
    return np.concatenate(([0], cpdf)) 



def exponential_distribution(x, tau):
    return np.exp(-x / tau) / tau

def exp_tailed_gaussian(x, mu, sigma, ncut, tau):
    factor = tau / np.sqrt(2 * np.pi * sigma**2) * np.exp(ncut * sigma / tau - ncut**2 / 2)
    y = gaussian(x, mu, sigma)
    tail_high = x > mu + ncut * sigma
    tail_low = x < mu - ncut * sigma
    y[tail_high] = factor * exponential_distribution(x[tail_high] - mu, 1 / tau)
    y[tail_low] = factor * exponential_distribution(mu - x[tail_low], 1 / tau)
    return y

def asymm_exp_tailed_gaussian(x, mu, sigma, ncut_high, tau_high, ncut_low, tau_low):
    factor_high = tau_high / np.sqrt(2 * np.pi * sigma**2) * np.exp(ncut_high * sigma / tau_high - ncut_high**2 / 2)
    factor_low = tau_low / np.sqrt(2 * np.pi * sigma**2) * np.exp(ncut_low * sigma / tau_low - ncut_low**2 / 2)
    y = gaussian(x, mu, sigma)
    tail_high = x > mu + ncut_high * sigma
    tail_low = x < mu - ncut_low * sigma
    y[tail_high] = factor_high * exponential_distribution(x[tail_high] - mu, 1 / tau_high)
    y[tail_low] = factor_low * exponential_distribution(mu - x[tail_low], 1 / tau_low)
    return y

def landau(x, m, w):
    xs = (x - m) / w
    return 1 / np.sqrt(2 * np.pi) * np.exp(-(xs + np.exp(-xs)) / 2)

def novosibirsk(x, mu, sigma, k):
    xi = np.sqrt(np.log(4))
    normed = (x - mu) / sigma
    arg = 1 - normed * k
    arg_sel = arg > 0
    arg2 = arg_sel * np.log(np.maximum(arg, 1e-7))
    width = (np.arcsinh(k * xi) / xi)**2
    return np.exp(-arg2**2 / (2 * width) - width / 2) * arg_sel

def calculate_chisq(data, model, errors, n_parameters):
    nonzero = errors > 0
    residuals = (data[nonzero] - model[nonzero]) / errors[nonzero]
    chisq = np.sum(residuals**2)
    dof = nonzero.sum() - n_parameters
    return chisq, dof, chisq / dof

def calculate_residual(data, model, errors):
    nonzero = errors > 0
    residuals = (data[nonzero] - model[nonzero]) / errors[nonzero]
    return residuals


def fermi_function(x):
    return 1 / (np.exp(-x) + 1)

def inverse_fermi_function(x):
    return -np.log(1 / x - 1)

def scaled_fermi_function(x, n, m, w, y0):
    return n * fermi_function((x - m) / w) + y0

def shifted_fermi_function(x, m, w):
    return fermi_function((x - m) / w)


def bethe_bloch(beta, charge, k1, k2, k3):
    gamma = 1 / (1 - beta**2)
    return charge**2 / beta**2 * (k1 * np.log(beta * gamma) - k2 * beta**2 + k3)

def bethe_bloch_pm(momentum, mass, charge, k1, k2, k3):
    energy = np.sqrt(momentum**2 + mass**2)
    beta = momentum / energy
    gamma = energy / mass
    return charge**2 / beta**2 * (k1 * np.log(beta * gamma) - k2 * beta**2 + k3)


def calculate_efficiency(passed, all):
    return (passed + 1) / (all + 2)

def calculate_efficiency_error(passed, all):
    k = passed
    n = all
    return np.sqrt(((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - (k + 1)**2 / (n + 2)**2)

def calculate_efficiency_and_error(passed, all):
    return calculate_efficiency(passed, all), calculate_efficiency_error(passed, all)


def calculate_efficiency_weighted(passed_values, failed_values):
    return passed_values / (passed_values + failed_values)

def calculate_efficiency_error_weighted(passed_values, failed_values, passed_squared_values, failed_squared_values):
    return np.sqrt(passed_squared_values * failed_values**2 + failed_squared_values * passed_values**2) / (passed_values + failed_values)**2

def calculate_efficiency_and_error_weighted(passed_values, failed_values, passed_squared_values, failed_squared_values):
    return calculate_efficiency_weighted(passed_values, failed_values), calculate_efficiency_error_weighted(passed_values, failed_values, passed_squared_values, failed_squared_values)

def calculate_efficiency_and_rejection(signal_histogram, background_histogram):
    assert signal_histogram.dimensions == 1 and background_histogram.dimensions == 1
    assert signal_histogram.binnings[0] == background_histogram.binnings[0]
    cut_binning = signal_histogram.binnings[0]
    bin_edges = cut_binning.edges
    signal_mean, signal_std, _ = hist_mean_and_std(signal_histogram)
    background_mean, background_std, _ = hist_mean_and_std(background_histogram)
    if signal_mean > background_mean:
        cut_values = bin_edges[1:]
        signal_values = signal_histogram.values
        background_values = background_histogram.values
    else:
        cut_values = bin_edges[:0:-1]
        signal_values = signal_histogram.values[::-1]
        background_values = background_histogram.values[::-1]
    signal_cumulative = signal_values[::-1].cumsum()[::-1]
    background_cumulative = background_values[::-1].cumsum()[::-1]
    signal_total = signal_cumulative[0]
    background_total = background_cumulative[0]
    signal_efficiency, signal_efficiency_error = calculate_efficiency_and_error(signal_cumulative, signal_total)
    background_efficiency, background_efficiency_error = calculate_efficiency_and_error(background_cumulative, background_total)
    background_rejection = 1 / background_efficiency
    background_efficiency_relative_error = background_efficiency_error / background_efficiency
    background_rejection_error = background_efficiency_relative_error * background_rejection
    return signal_efficiency, signal_efficiency_error, background_rejection, background_rejection_error, cut_values

def calculate_signal_and_background_efficiency(signal_histogram, background_histogram):
    assert signal_histogram.dimensions == 1 and background_histogram.dimensions == 1
    assert signal_histogram.binnings[0] == background_histogram.binnings[0]
    cut_binning = signal_histogram.binnings[0]
    bin_edges = cut_binning.edges
    signal_mean, signal_std, _ = hist_mean_and_std(signal_histogram)
    background_mean, background_std, _ = hist_mean_and_std(background_histogram)
    if signal_mean > background_mean:
        cut_values = bin_edges[:-1]
        signal_values = signal_histogram.values
        signal_squared_values = signal_histogram.squared_values
        background_values = background_histogram.values
        background_squared_values = background_histogram.squared_values
    else:
        cut_values = bin_edges[:0:-1]
        signal_values = signal_histogram.values[::-1]
        signal_squared_values = signal_histogram.squared_values[::-1]
        background_values = background_histogram.values[::-1]
        background_squared_values = background_histogram.squared_values[::-1]
    signal_passed_cumulative = signal_values[::-1].cumsum()[::-1]
    signal_passed_cumulative_squared = signal_squared_values[::-1].cumsum()[::-1]
    signal_failed_cumulative = signal_values.cumsum()
    signal_failed_cumulative_squared = signal_squared_values.cumsum()
    background_passed_cumulative = background_values[::-1].cumsum()[::-1]
    background_passed_cumulative_squared = background_squared_values[::-1].cumsum()[::-1]
    background_failed_cumulative = background_values.cumsum()
    background_failed_cumulative_squared = background_squared_values.cumsum()
    signal_efficiency, signal_efficiency_error = calculate_efficiency_and_error_weighted(signal_passed_cumulative, signal_failed_cumulative, signal_passed_cumulative_squared, signal_failed_cumulative_squared)
    background_efficiency, background_efficiency_error = calculate_efficiency_and_error_weighted(background_passed_cumulative, background_failed_cumulative, background_passed_cumulative_squared, background_failed_cumulative_squared)
    return signal_efficiency, signal_efficiency_error, background_efficiency, background_efficiency_error, cut_values


def calculate_cut_value_for_efficiency(signal_histogram, target_efficiency):
    assert signal_histogram.dimensions == 1
    cut_binning = signal_histogram.binnings[0]
    bin_edges = cut_binning.edges
    cut_values = bin_edges[:-1]
    signal_values = signal_histogram.values
    signal_cumulative = signal_values[::-1].cumsum()[::-1]
    signal_total = signal_cumulative[0]
    signal_efficiency, signal_efficiency_error = calculate_efficiency_and_error(signal_cumulative, signal_total)
    return np.interp(target_efficiency, signal_efficiency[::-1], cut_values[::-1])


def weighted_mean(values, errors):
    weights = 1 / errors**2
    return (values * weights).sum() / weights.sum()


def lafferty_whyatt(edges, gamma):
    ex = 1 - gamma
    rmin = edges[:-1]
    rmax = edges[1:]
    return ((rmax - rmin) * ex / (rmax**ex - rmin**ex))**(1 / gamma)


def row_mean_and_std(bin_centers, weights, axis=1):
    mean = (np.expand_dims(bin_centers, axis=1-axis) * weights).sum(axis=axis) / weights.sum(axis=axis)
    std = np.sqrt(((np.expand_dims(bin_centers, axis=1-axis) - np.expand_dims(mean, axis=axis))**2 * weights).sum(axis=axis) / weights.sum(axis=axis))
    return mean, std, std / np.sqrt(weights.sum(axis=axis))

def weighted_row_mean_and_std(bin_centers, weights, squared_weights, axis=1):
    values = np.expand_dims(bin_centers, axis=1-axis)
    mean = (values * weights).sum(axis=axis) / weights.sum(axis=axis)
    n = weights.sum(axis=axis)
    residuals = ((values - np.expand_dims(mean, axis=axis))**2 * weights).sum(axis=axis)
    std = np.sqrt(1 / (n - 1) * residuals)
    sum_of_weights_squared = weights.sum(axis=axis)**2
    sum_of_squared_weights = squared_weights.sum(axis=axis)
    return mean, std, std / n * sum_of_squared_weights / sum_of_weights_squared


def hist_mean_and_std(hist, axis=1):
    if hist.dimensions > 1:
        bin_centers = hist.binnings[axis].bin_centers
        weights = hist.values
        squared_weights = hist.squared_values
        return weighted_row_mean_and_std(bin_centers[1:-1], weights[1:-1,1:-1], squared_weights[1:-1,1:-1], axis=axis)
    elif hist.dimensions == 1:
        bin_centers = hist.binnings[0].bin_centers[1:-1]
        weights = hist.values[1:-1]
        squared_weights = hist.squared_values[1:-1]
        mean = (bin_centers * weights).sum() / weights.sum()
        n = weights.sum()
        residuals = (((bin_centers - mean)**2 * weights)**2).sum()
        std = np.sqrt(1 / (n - 1) * residuals)
        sum_of_weights_squared = weights.sum()**2
        sum_of_squared_weights = squared_weights.sum()
        mean_error = std / n * sum_of_squared_weights / sum_of_weights_squared
        return mean, std, mean_error
    raise NotImplementedError


def hist_percentile(hist, axis=1, percentile=0.95, bin_point="center"):
    def _get_bin_values(binning):
        if bin_point == "center":
            return binning.bin_centers
        elif bin_point == "low":
            return binning.edges[:-1]
        elif bin_point == "high":
            return binning.edges[1:]
        raise NotImplementedError
    if hist.dimensions == 1:
        assert axis == 0
        bin_values = _get_bin_values(hist.binnings[axis])
        cdf = np.cumsum(hist.values, axis=axis) / np.sum(hist.values)
        return bin_values[np.argmin(np.abs(cdf - percentile), axis=axis)]
    if hist.dimensions == 2:
        bin_values = _get_bin_values(hist.binnings[axis])
        cdf = np.cumsum(hist.values, axis=axis) / np.expand_dims(np.sum(hist.values, axis=axis), axis=axis)
        return bin_values[np.argmin(np.abs(cdf - percentile), axis=axis)]
    raise NotImplementedError


def calculate_likelihood(distribution, values):
    norm = distribution.values.sum()
    return distribution.get(values) / norm


def random_powerlaw(E_min, E_max, gamma, n=100):
    ex = 1 - gamma
    return (np.random.random(n) * (E_max**ex - E_min**ex) + E_min**ex)**(1 / ex)

def integral_powerlaw(E_min, E_max, gamma, phi_0):
    ex = 1 - gamma
    return phi_0 / ex * (E_max**ex - E_min**ex)


def poisson_limit_lower(n, fraction):
    return scp_chi2.ppf(fraction, 2 * n) / 2

def poisson_limit_upper(n, fraction):
    return scp_chi2.ppf(fraction, 2 * (n + 1)) / 2

def poisson_interval(n, probability):
    remainder = 1 - probability
    lower_limit = poisson_limit_lower(n, remainder / 2)
    upper_limit = poisson_limit_lower(n, 1 - remainder / 2)
    return lower_limit, upper_limit

def poisson_limit(n, probability):
    return poisson_limit_upper(n, probability)

def n_sigmas_to_probability(n_sigmas):
    return 2 * scp_gaussian.cdf(n_sigmas) - 1


def approximate_upper_poisson_error(n, sigmas=1):
    return (n + 1) * (1 - 1 / (9 * (n + 1)) + sigmas / (3 * np.sqrt(n + 1)))**3 - n

poisson_parametrization_beta = {1: 0, 2: 0.062, 3: 0.222}
poisson_parametrization_gamma = {1: 0, 2: -2.19, 3: -1.88}

def approximate_lower_poisson_error(n, sigmas=1):
    beta = poisson_parametrization_beta[sigmas]
    gamma = poisson_parametrization_gamma[sigmas]
    return -n * (1 - 1 / (9 * np.maximum(n, 1e-7)) - sigmas / (3 * np.sqrt(np.maximum(n, 1e-7))) + beta * n**gamma)**3 + n
    

def smooth_additive(values, window=1):
    result = np.copy(values)
    count = np.ones_like(values)
    for shift in range(-window, window + 1):
        if shift < 0:
            result[:shift] += values[-shift:]
            count[:shift] += 1
        elif shift > 0:
            result[shift:] += values[:-shift]
            count[shift:] += 1
    return result / count


def calculate_correlation(x, y):
    x_central = x - np.mean(x)
    y_central = y - np.mean(y)
    return np.mean(x_central * y_central) / (np.std(x_central) * np.std(y_central))

def calculate_correlation_and_error(x, y):
    r = calculate_correlation(x, y)
    n = len(x)
    std = np.sqrt((1 - r**2) / (n - 2))
    return r, std


def draw_random_from_hist(histogram, size, seed=None):
    from .utilities import transform_overflow_edges
    probability = histogram.values / histogram.values.sum()
    cumulative = np.cumsum(probability)
    edges = transform_overflow_edges(histogram.binnings[0].edges)
    x_values = (edges[1:] + edges[:-1]) / 2
    first_index = np.max(np.arange(len(cumulative))[cumulative == 0]) if np.any(cumulative == 0) else 0
    last_index = np.min(np.arange(len(cumulative))[(1 - cumulative) < 1e-10])
    cumulative = cumulative[first_index:last_index + 1]
    x_values = x_values[first_index:last_index + 1]
    monotonic_sel = np.concatenate(([True], cumulative[1:] > cumulative[:-1]))
    cumulative = cumulative[monotonic_sel]
    x_values = x_values[monotonic_sel]
    spline = PchipInterpolator(cumulative, x_values)
    return spline(np.random.default_rng(seed=seed).random(size=size))
