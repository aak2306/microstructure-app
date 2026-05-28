"""Tests for interfacial-length and ratio metrics."""

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from microstructure.metrics import (
    interface_to_area_ratio_per_um,
    interfacial_length_um,
    mean_free_path_um,
    mean_intercept_length_um,
    measured_volume_fraction,
    specific_surface_area_per_um,
)


def test_empty_canvas_has_zero_interface():
    binary = np.zeros((100, 100), dtype=bool)
    assert interfacial_length_um(binary, pixel_per_um=10.0) == 0.0


def test_single_circle_perimeter_close_to_2pir():
    # large radius keeps quantization error well under 10%
    r_px = 30
    pixel_per_um = 10.0
    img = Image.new("L", (200, 200), 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse([100 - r_px, 100 - r_px, 100 + r_px, 100 + r_px], fill=255)
    binary = np.array(img) > 0

    measured_um = interfacial_length_um(binary, pixel_per_um=pixel_per_um)
    expected_um = (2 * math.pi * r_px) / pixel_per_um
    # skimage.measure.perimeter underestimates by ~5-10% on rasterized circles;
    # this tolerance just pins down current behavior without overfitting.
    assert 0.8 * expected_um <= measured_um <= 1.1 * expected_um


def test_two_separate_circles_sum():
    r_px = 20
    pixel_per_um = 10.0
    img = Image.new("L", (300, 300), 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse([50 - r_px, 150 - r_px, 50 + r_px, 150 + r_px], fill=255)
    draw.ellipse([250 - r_px, 150 - r_px, 250 + r_px, 150 + r_px], fill=255)
    binary = np.array(img) > 0
    two = interfacial_length_um(binary, pixel_per_um)

    img_one = Image.new("L", (300, 300), 0)
    ImageDraw.Draw(img_one).ellipse(
        [50 - r_px, 150 - r_px, 50 + r_px, 150 + r_px], fill=255
    )
    one = interfacial_length_um(np.array(img_one) > 0, pixel_per_um)

    assert abs(two - 2 * one) / (2 * one) < 0.05


def test_overlapping_circles_less_than_sum():
    r_px = 30
    pixel_per_um = 10.0
    img = Image.new("L", (200, 200), 0)
    draw = ImageDraw.Draw(img)
    # heavy overlap → single merged label
    draw.ellipse([80 - r_px, 100 - r_px, 80 + r_px, 100 + r_px], fill=255)
    draw.ellipse([120 - r_px, 100 - r_px, 120 + r_px, 100 + r_px], fill=255)
    merged = interfacial_length_um(np.array(img) > 0, pixel_per_um)

    single_img = Image.new("L", (200, 200), 0)
    ImageDraw.Draw(single_img).ellipse(
        [80 - r_px, 100 - r_px, 80 + r_px, 100 + r_px], fill=255
    )
    single = interfacial_length_um(np.array(single_img) > 0, pixel_per_um)

    assert merged < 2 * single


def test_ratio_units_and_division():
    interface_um = 100.0
    w, h = 50.0, 20.0
    assert interface_to_area_ratio_per_um(interface_um, w, h) == 100.0 / (50.0 * 20.0)


def test_measured_volume_fraction_empty():
    assert measured_volume_fraction(np.zeros((50, 50), dtype=bool)) == 0.0


def test_measured_volume_fraction_full():
    assert measured_volume_fraction(np.ones((50, 50), dtype=bool)) == 1.0


def test_measured_volume_fraction_quarter():
    binary = np.zeros((100, 100), dtype=bool)
    binary[:50, :50] = True
    assert measured_volume_fraction(binary) == 0.25


# ---------------------------------------------------------------------------
# Stereology
# ---------------------------------------------------------------------------


def test_specific_surface_area_is_4_over_pi_times_L_A():
    L_A = 0.1
    assert specific_surface_area_per_um(L_A) == (4.0 / math.pi) * L_A


def test_specific_surface_area_zero_when_no_interface():
    assert specific_surface_area_per_um(0.0) == 0.0


def test_mean_intercept_length_matches_4_V_over_S():
    V = 0.3
    S = 0.05
    assert mean_intercept_length_um(V, S) == pytest.approx(4.0 * V / S)


def test_mean_intercept_length_zero_when_no_surface():
    assert mean_intercept_length_um(volume_fraction=0.3, s_v_per_um=0.0) == 0.0


def test_mean_free_path_matches_4_1minusV_over_S():
    V = 0.3
    S = 0.05
    assert mean_free_path_um(V, S) == pytest.approx(4.0 * (1.0 - V) / S)


def test_mean_free_path_zero_when_no_surface():
    assert mean_free_path_um(volume_fraction=0.3, s_v_per_um=0.0) == 0.0


def test_intercept_plus_free_path_equals_4_over_S():
    """⟨L_α⟩ + λ = 4·V/S + 4·(1-V)/S = 4/S. Identity worth pinning."""
    V = 0.42
    S = 0.07
    total = mean_intercept_length_um(V, S) + mean_free_path_um(V, S)
    assert total == pytest.approx(4.0 / S)


def test_monodisperse_sphere_relation_sanity():
    """For monodisperse spheres of diameter D at volume fraction φ,
    S_V = 6φ/D. Working backwards from that: L_A = (π/4)·S_V = (3π/2)·φ/D.
    Round-tripping that L_A through our S_V function should recover S_V."""
    D = 10.0
    phi = 0.2
    s_v_theory = 6.0 * phi / D
    L_A_from_theory = (math.pi / 4.0) * s_v_theory
    s_v_back = specific_surface_area_per_um(L_A_from_theory)
    assert s_v_back == pytest.approx(s_v_theory)
