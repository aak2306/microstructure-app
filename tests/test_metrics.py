"""Tests for interfacial-length and ratio metrics."""

import math

import numpy as np
from PIL import Image, ImageDraw

from microstructure.metrics import (
    interface_to_area_ratio_per_um,
    interfacial_length_um,
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
