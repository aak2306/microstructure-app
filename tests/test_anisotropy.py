"""Tests for the directional / anisotropy analysis."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from microstructure.anisotropy import (
    anisotropy_index,
    directional_intercept_density,
    elongation_direction_deg,
)


ANGLES = np.arange(0.0, 180.0, 10.0)


def _make_binary(size: int = 400) -> Image.Image:
    return Image.fromarray(np.zeros((size, size), dtype=np.uint8))


def test_empty_binary_returns_zeros():
    binary = np.zeros((100, 100), dtype=bool)
    p_l = directional_intercept_density(binary, ANGLES, pixel_per_um=1.0)
    assert np.all(p_l == 0.0)
    assert anisotropy_index(p_l) == 0.0
    assert elongation_direction_deg(ANGLES, p_l) == 0.0


def test_full_binary_returns_zeros():
    binary = np.ones((100, 100), dtype=bool)
    p_l = directional_intercept_density(binary, ANGLES, pixel_per_um=1.0)
    # All foreground, no boundaries anywhere → P_L = 0 in all directions.
    assert np.all(p_l == 0.0)
    assert anisotropy_index(p_l) == 0.0


def test_anisotropy_index_isotropic_disk_is_small():
    """A centered disk is rotationally symmetric → DA should be near zero,
    bounded by rasterization noise."""
    img = _make_binary(400)
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 100, 300, 300], fill=255)
    binary = np.array(img) > 0
    p_l = directional_intercept_density(binary, ANGLES, pixel_per_um=10.0)
    assert anisotropy_index(p_l) < 0.10


def test_horizontal_stripes_have_high_anisotropy_with_horizontal_elongation():
    """A pattern of horizontal stripes is elongated along 0°.
    Scan lines parallel to the stripes (θ=0°) cross few boundaries.
    Scan lines perpendicular (θ=90°) cross many. So P_L_min is at θ=0°."""
    H = W = 400
    arr = np.zeros((H, W), dtype=bool)
    period = 20
    for row in range(0, H, period):
        arr[row : row + period // 2, :] = True
    p_l = directional_intercept_density(arr, ANGLES, pixel_per_um=1.0)
    assert anisotropy_index(p_l) > 0.5
    elongation = elongation_direction_deg(ANGLES, p_l)
    # Allow ±20° tolerance (10° angular sampling + rotation discretization)
    assert min(elongation, 180.0 - elongation) <= 20.0


def test_vertical_stripes_have_elongation_near_90deg():
    H = W = 400
    arr = np.zeros((H, W), dtype=bool)
    period = 20
    for col in range(0, W, period):
        arr[:, col : col + period // 2] = True
    p_l = directional_intercept_density(arr, ANGLES, pixel_per_um=1.0)
    assert anisotropy_index(p_l) > 0.5
    elongation = elongation_direction_deg(ANGLES, p_l)
    assert abs(elongation - 90.0) <= 20.0


def test_p_l_symmetric_under_180_rotation():
    """P_L(θ) and P_L(θ + 180°) describe the same scan-line family, so
    must be equal. The inscribed-square crop also has this symmetry."""
    img = _make_binary(300)
    np.random.seed(0)
    draw = ImageDraw.Draw(img)
    # Scatter some particles
    for _ in range(40):
        cx, cy = np.random.randint(30, 270, size=2)
        r = np.random.randint(10, 20)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    binary = np.array(img) > 0
    angles = np.array([0.0, 30.0, 60.0, 90.0])
    p_l_forward = directional_intercept_density(binary, angles, 1.0)
    p_l_reverse = directional_intercept_density(binary, angles + 180.0, 1.0)
    for a, b in zip(p_l_forward, p_l_reverse):
        assert a == pytest.approx(b, rel=0.05, abs=1e-6)


def test_anisotropy_index_bounds():
    assert anisotropy_index(np.array([1.0, 1.0, 1.0])) == 0.0
    # max=1, min=0 → 1.0
    assert anisotropy_index(np.array([0.0, 1.0])) == 1.0
    assert anisotropy_index(np.array([])) == 0.0


def test_elongation_direction_picks_argmin():
    angles = np.array([0.0, 45.0, 90.0, 135.0])
    p_l = np.array([0.1, 0.5, 0.3, 0.6])  # min at 0°
    assert elongation_direction_deg(angles, p_l) == 0.0
    p_l2 = np.array([0.5, 0.5, 0.1, 0.5])  # min at 90°
    assert elongation_direction_deg(angles, p_l2) == 90.0
