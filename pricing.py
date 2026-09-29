"""European-call benchmarks and exact terminal Monte Carlo estimators.

Models assume constant parameters, no dividends, and a risk-neutral measure.
See METHODS.md for units, estimator uncertainty, and the mixture error bound.
"""
from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
from scipy.special import ndtr
from scipy.stats import poisson


@dataclass(frozen=True)
class Estimate:
    price: float
    standard_error: float
    independent_observations: int
    payoff_evaluations: int

    @property
    def confidence_interval(self):
        """Pointwise asymptotic 95% interval; not a model-risk interval."""
        width = 1.96 * self.standard_error
        return self.price - width, self.price + width


def _validate(spot, strike, rate, volatility, maturity):
    if not all(math.isfinite(x) for x in (spot, strike, rate, volatility, maturity)):
        raise ValueError('Model parameters must be finite.')
    if spot <= 0 or strike <= 0 or volatility < 0 or maturity < 0:
        raise ValueError('Spot/strike must be positive; volatility/maturity nonnegative.')


def _sample_count(n, minimum=2):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < minimum:
        raise ValueError(f'Sample count must be an integer >= {minimum}.')


def _jumps(intensity, jump_mean, jump_std):
    if not all(math.isfinite(x) for x in (intensity, jump_mean, jump_std)):
        raise ValueError('Jump parameters must be finite.')
    if intensity < 0 or jump_std < 0:
        raise ValueError('Jump intensity and log-jump standard deviation must be nonnegative.')


def _estimate(samples, evaluations):
    return Estimate(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(samples.size)),
                    int(samples.size), int(evaluations))


def black_scholes_call(spot, strike, rate, volatility, maturity):
    """Non-dividend-paying European call; maturity in years and annualized inputs."""
    _validate(spot, strike, rate, volatility, maturity)
    discounted_strike = strike * math.exp(-rate * maturity)
    if maturity == 0 or volatility == 0:
        return max(spot - discounted_strike, 0.0)
    scale = volatility * math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / scale
    return float(spot * ndtr(d1) - discounted_strike * ndtr(d1 - scale))


def gbm_terminal(spot, rate, volatility, maturity, n, rng):
    """Sample the terminal distribution exactly (no time-discretization error)."""
    _validate(spot, spot, rate, volatility, maturity)
    _sample_count(n)
    z = rng.standard_normal(n)
    return spot * np.exp((rate - 0.5 * volatility**2) * maturity
                         + volatility * np.sqrt(maturity) * z)


def plain_call(spot, strike, rate, volatility, maturity, n, rng):
    _validate(spot, strike, rate, volatility, maturity)
    terminal = gbm_terminal(spot, rate, volatility, maturity, n, rng)
    discounted = np.exp(-rate * maturity) * np.maximum(terminal - strike, 0.0)
    return _estimate(discounted, n)


def antithetic_call(spot, strike, rate, volatility, maturity, n, rng):
    """n payoff evaluations, n/2 independent pair means; n must be even >= 4."""
    _validate(spot, strike, rate, volatility, maturity)
    _sample_count(n, minimum=4)
    if n % 2:
        raise ValueError('Antithetic payoff budget must be even.')
    z = rng.standard_normal(n // 2)
    drift = (rate - 0.5 * volatility**2) * maturity
    shock = volatility * np.sqrt(maturity) * z
    plus = np.maximum(spot * np.exp(drift + shock) - strike, 0.0)
    minus = np.maximum(spot * np.exp(drift - shock) - strike, 0.0)
    pair_means = np.exp(-rate * maturity) * (plus + minus) / 2
    return _estimate(pair_means, n)


def merton_terminal(spot, rate, volatility, maturity, n, intensity, jump_mean, jump_std, rng):
    """Log jump sizes are iid Normal(jump_mean, jump_std**2)."""
    _validate(spot, spot, rate, volatility, maturity)
    _sample_count(n)
    _jumps(intensity, jump_mean, jump_std)
    counts = rng.poisson(intensity * maturity, n)
    diffusion = volatility * np.sqrt(maturity) * rng.standard_normal(n)
    log_jumps = jump_mean * counts + jump_std * np.sqrt(counts) * rng.standard_normal(n)
    kappa = math.expm1(jump_mean + 0.5 * jump_std**2)
    drift = (rate - intensity * kappa - 0.5 * volatility**2) * maturity
    return spot * np.exp(drift + diffusion + log_jumps)


def merton_call_mc(spot, strike, rate, volatility, maturity, n, intensity, jump_mean, jump_std, rng):
    _validate(spot, strike, rate, volatility, maturity)
    terminal = merton_terminal(spot, rate, volatility, maturity, n,
                               intensity, jump_mean, jump_std, rng)
    return _estimate(np.exp(-rate * maturity) * np.maximum(terminal - strike, 0.0), n)


def merton_call_benchmark(spot, strike, rate, volatility, maturity,
                          intensity, jump_mean, jump_std, tolerance=1e-10):
    """Poisson-mixture call price with an absolute omitted-tail bound.

    The omitted nonnegative call terms are bounded by the omitted discounted
    stock terms: spot * P(Poisson(intensity*T*E[J]) > last_count).
    Floating-point roundoff is separate from this summation error bound.
    """
    _validate(spot, strike, rate, volatility, maturity)
    _jumps(intensity, jump_mean, jump_std)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Tolerance must be positive and finite.')
    if maturity == 0 or intensity == 0:
        return black_scholes_call(spot, strike, rate, volatility, maturity)
    expected_jump = math.exp(jump_mean + 0.5 * jump_std**2)
    count_mean = intensity * maturity
    tilted_mean = count_mean * expected_jump
    if not math.isfinite(tilted_mean) or max(count_mean, tilted_mean) > 10_000:
        raise ValueError('This reference implementation supports Poisson means <= 10,000.')
    last = max(0, int(math.ceil(tilted_mean + 10 * math.sqrt(tilted_mean + 1))))
    while spot * poisson.sf(last, tilted_mean) > tolerance:
        last += 1
    counts = np.arange(last + 1)
    variance = volatility**2 * maturity + counts * jump_std**2
    log_mean = (math.log(spot) + (rate - intensity * (expected_jump - 1)
                - 0.5 * volatility**2) * maturity + counts * jump_mean)
    nonzero = variance > 0
    d2 = np.zeros_like(variance)
    d2[nonzero] = (log_mean[nonzero] - math.log(strike)) / np.sqrt(variance[nonzero])
    d1 = d2 + np.sqrt(variance)
    # Zero conditional variance: payoff is deterministic given the jump count.
    probability_itm = (log_mean > math.log(strike)).astype(float)
    stock_cdf = np.where(nonzero, ndtr(d1), probability_itm)
    strike_cdf = np.where(nonzero, ndtr(d2), probability_itm)
    terms = (spot * poisson.pmf(counts, tilted_mean) * stock_cdf
             - strike * math.exp(-rate * maturity) * poisson.pmf(counts, count_mean) * strike_cdf)
    return float(terms.sum())
