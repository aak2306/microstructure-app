"""Tests for the particle-size distributions module.

Empirical checks (large N) confirm each sampler reproduces its target
moments to within a tight Monte-Carlo tolerance — if the math drifts, the
test catches it.
"""

import math

import numpy as np
import pytest

from microstructure import distributions as dist


# ---------------------------------------------------------------------------
# Sampler determinism + ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", dist.DISTRIBUTIONS)
def test_sampler_deterministic_with_seed(name):
    np.random.seed(7)
    s1 = dist.make_size_sampler(name, nominal_rad_px=20.0)
    a = [s1() for _ in range(50)]
    np.random.seed(7)
    s2 = dist.make_size_sampler(name, nominal_rad_px=20.0)
    b = [s2() for _ in range(50)]
    assert a == b


def test_unknown_distribution_raises():
    with pytest.raises(ValueError, match="Unknown distribution"):
        dist.make_size_sampler("garbage", nominal_rad_px=10.0)


def test_unknown_distribution_raises_for_r2_factor():
    with pytest.raises(ValueError, match="Unknown distribution"):
        dist.expected_r2_factor("garbage")


# ---------------------------------------------------------------------------
# Distribution moments — large-N empirical sanity
# ---------------------------------------------------------------------------


def _sample(name: str, n: int, **kwargs) -> np.ndarray:
    np.random.seed(0)
    sampler = dist.make_size_sampler(name, nominal_rad_px=20.0, **kwargs)
    return np.array([sampler() for _ in range(n)])


def test_uniform_mean_is_nominal_and_range_is_correct():
    arr = _sample(dist.UNIFORM, n=20_000, uniform_pct=10.0)
    assert arr.mean() == pytest.approx(20.0, rel=0.01)
    # Uniform bounds: 20 ± 2.0
    assert arr.min() >= 20.0 * 0.9 - 1e-9
    assert arr.max() <= 20.0 * 1.1 + 1e-9


def test_lognormal_median_is_nominal_arithmetic_mean_is_larger():
    sigma_g = 1.5
    arr = _sample(dist.LOG_NORMAL, n=30_000, lognormal_sigma_g=sigma_g)
    # Median ≈ nominal
    assert np.median(arr) == pytest.approx(20.0, rel=0.03)
    # Arithmetic mean = nominal · exp(σ_ln² / 2) > median
    sigma_ln = math.log(sigma_g)
    expected_mean = 20.0 * math.exp(sigma_ln**2 / 2.0)
    assert arr.mean() == pytest.approx(expected_mean, rel=0.03)
    assert arr.mean() > np.median(arr)


def test_normal_mean_and_std_match_inputs():
    arr = _sample(dist.NORMAL, n=30_000, normal_sigma_pct=15.0)
    assert arr.mean() == pytest.approx(20.0, rel=0.01)
    assert arr.std() == pytest.approx(3.0, rel=0.05)  # 15% of 20


def test_rosin_rammler_63_2_percent_below_characteristic():
    """For Weibull(scale=d₆₃, shape=n), 63.2% of mass lies below d₆₃."""
    n = 2.5
    arr = _sample(dist.ROSIN_RAMMLER, n=30_000, rr_shape_n=n)
    frac_below = float((arr <= 20.0).mean())
    assert frac_below == pytest.approx(0.632, abs=0.015)


# ---------------------------------------------------------------------------
# expected_r2_factor agrees with the empirical second moment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,kwargs",
    [
        (dist.UNIFORM, {"uniform_pct": 30.0}),
        (dist.LOG_NORMAL, {"lognormal_sigma_g": 1.5}),
        (dist.NORMAL, {"normal_sigma_pct": 20.0}),
        (dist.ROSIN_RAMMLER, {"rr_shape_n": 2.5}),
    ],
)
def test_expected_r2_factor_matches_empirical(name, kwargs):
    nominal = 50.0
    sampler = dist.make_size_sampler(name, nominal_rad_px=nominal, **kwargs)
    np.random.seed(0)
    samples = np.array([sampler() for _ in range(40_000)])
    empirical = float((samples**2).mean() / nominal**2)
    analytical = dist.expected_r2_factor(name, **kwargs)
    # Looser tolerance for the long-tailed distributions.
    assert empirical == pytest.approx(analytical, rel=0.03)


def test_uniform_r2_factor_collapses_to_one_at_zero_pct():
    assert dist.expected_r2_factor(dist.UNIFORM, uniform_pct=0.0) == 1.0


def test_lognormal_r2_factor_collapses_to_one_at_unit_sigma_g():
    # σ_g = 1 → σ_ln = 0 → exp(0) = 1
    assert dist.expected_r2_factor(
        dist.LOG_NORMAL, lognormal_sigma_g=1.0
    ) == pytest.approx(1.0)
