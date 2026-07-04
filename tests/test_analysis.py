"""Tests for micrograph analysis: segmentation, descriptors, classification."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from microstructure import generators as gen
from microstructure.analysis import (
    classify_shape,
    drop_fines,
    geometric_std,
    particle_descriptors,
    segment_particles,
    suggest_generator_settings,
)


def _circles_image(
    size: int = 400,
    radii: list[int] | None = None,
    n: int = 12,
    r: int = 20,
    bright_particles: bool = True,
) -> np.ndarray:
    """Grayscale image with a grid of non-touching, non-border circles."""
    img = Image.new("L", (size, size), 30 if bright_particles else 220)
    draw = ImageDraw.Draw(img)
    fill = 220 if bright_particles else 30
    radii = radii if radii is not None else [r] * n
    per_row = int(np.ceil(np.sqrt(len(radii))))
    pitch = size // (per_row + 1)
    for i, rr in enumerate(radii):
        cx = pitch * (1 + i % per_row)
        cy = pitch * (1 + i // per_row)
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=fill)
    return np.array(img)


def _ellipses_image(size: int = 400, rx: int = 30, ry: int = 12) -> np.ndarray:
    img = Image.new("L", (size, size), 30)
    draw = ImageDraw.Draw(img)
    for i in range(3):
        for j in range(3):
            cx, cy = 70 + i * 130, 70 + j * 130
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=220)
    return np.array(img)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def test_segment_auto_detects_bright_particles():
    gray = _circles_image(bright_particles=True)
    binary, used_bright = segment_particles(gray)
    assert used_bright is True
    assert 0.0 < binary.mean() < 0.5  # minority phase


def test_segment_auto_detects_dark_particles():
    gray = _circles_image(bright_particles=False)
    binary, used_bright = segment_particles(gray)
    assert used_bright is False
    assert 0.0 < binary.mean() < 0.5


def test_segment_explicit_polarity_overrides_auto():
    gray = _circles_image(bright_particles=True)
    binary, used_bright = segment_particles(gray, particles_are_bright=False)
    assert used_bright is False
    assert binary.mean() > 0.5  # picked the majority (matrix) phase


def test_segment_rejects_flat_image():
    with pytest.raises(ValueError):
        segment_particles(np.full((50, 50), 128, dtype=np.uint8))


def test_segment_rejects_non_2d():
    with pytest.raises(ValueError):
        segment_particles(np.zeros((10, 10, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

def test_descriptors_circles_have_high_circularity():
    binary, _ = segment_particles(_circles_image())
    desc = particle_descriptors([binary])
    assert desc.n_particles > 0
    assert np.median(desc.circularity) > 0.9
    assert np.median(desc.aspect_ratio) < 1.15


def test_descriptors_exclude_border_touching_particles():
    gray = _circles_image(n=9, r=20)
    # Add a circle cut by the left edge
    img = Image.fromarray(gray)
    ImageDraw.Draw(img).ellipse([-20, 180, 20, 220], fill=220)
    binary, _ = segment_particles(np.array(img))
    desc = particle_descriptors([binary])
    assert desc.n_particles == 9  # the cut circle is excluded


def test_descriptors_pool_across_images():
    b1, _ = segment_particles(_circles_image(n=9))
    b2, _ = segment_particles(_circles_image(n=9))
    pooled = particle_descriptors([b1, b2])
    single = particle_descriptors([b1])
    assert pooled.n_particles == 2 * single.n_particles


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_classify_circular():
    assert classify_shape(0.95, 1.05, 0.98) == gen.CIRCULAR


def test_classify_elliptical():
    assert classify_shape(0.75, 2.2, 0.97) == gen.ELLIPTICAL


def test_classify_rough_spheres():
    assert classify_shape(0.75, 1.1, 0.95) == gen.ROUGH_SPHERES


def test_classify_cracked_flakes_concave():
    assert classify_shape(0.55, 1.3, 0.75) == gen.CRACKED_FLAKES


def test_classify_irregular():
    assert classify_shape(0.5, 1.2, 0.9) == gen.IRREGULAR


def test_classify_angular_convex_flakes_not_elliptical():
    """Angular convex flakes: elongated (aspect 1.89) and solid (0.89),
    but circularity 0.67 is far below the ~0.85 a smooth ellipse of that
    aspect would have. Regression test for a real SiC micrograph that
    was misclassified as Elliptical."""
    assert classify_shape(0.67, 1.89, 0.89) == gen.CRACKED_FLAKES


def test_classify_moderately_elongated_angular_flakes():
    """Faceted but only mildly elongated (aspect < 1.6) — caught by the
    smoothness rule, not the aspect rule."""
    assert classify_shape(0.62, 1.45, 0.90) == gen.CRACKED_FLAKES


def test_classify_smooth_ellipse_stays_elliptical():
    """A genuinely smooth ellipse outline sits close to the theoretical
    circularity ceiling for its aspect ratio."""
    assert classify_shape(0.80, 1.9, 0.98) == gen.ELLIPTICAL


def test_end_to_end_circles_classified_circular():
    binary, _ = segment_particles(_circles_image())
    s = suggest_generator_settings([binary], pixel_per_um=2.0)
    assert s.shape == gen.CIRCULAR
    # radius 20 px → diameter 40 px → 20 µm at 2 px/µm
    assert s.diameter_um == pytest.approx(20.0, rel=0.10)
    assert s.sigma_g == pytest.approx(1.0, abs=0.05)


def test_end_to_end_ellipses_classified_elliptical():
    binary, _ = segment_particles(_ellipses_image())
    s = suggest_generator_settings([binary], pixel_per_um=2.0)
    assert s.shape == gen.ELLIPTICAL
    assert s.median_aspect > 1.6


# ---------------------------------------------------------------------------
# Fines exclusion
# ---------------------------------------------------------------------------

def _flakes_with_specks_image(size: int = 400) -> np.ndarray:
    """9 large circles (the real particles) plus a swarm of small specks."""
    img = Image.new("L", (size, size), 30)
    draw = ImageDraw.Draw(img)
    for i in range(3):
        for j in range(3):
            cx, cy = 70 + i * 130, 70 + j * 130
            draw.ellipse([cx - 25, cy - 25, cx + 25, cy + 25], fill=220)
    # ~60 specks of radius 3 px (28 px² each — above the noise floor,
    # tiny next to the 1963 px² circles)
    rng = np.random.default_rng(7)
    placed = 0
    while placed < 60:
        cx, cy = rng.integers(20, size - 20, size=2)
        near_circle = any(
            (cx - (70 + i * 130)) ** 2 + (cy - (70 + j * 130)) ** 2 < 45**2
            for i in range(3)
            for j in range(3)
        )
        if near_circle:
            continue
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=220)
        placed += 1
    return np.array(img)


def test_drop_fines_keeps_monodisperse_population_intact():
    binary, _ = segment_particles(_circles_image(n=12))
    desc = particle_descriptors([binary])
    filtered = drop_fines(desc)
    assert filtered.n_particles == desc.n_particles


def test_drop_fines_removes_specks():
    binary, _ = segment_particles(_flakes_with_specks_image())
    desc = particle_descriptors([binary])
    filtered = drop_fines(desc)
    assert desc.n_particles > 50  # specks were detected
    assert filtered.n_particles < 15  # ...but excluded from the stats


def test_suggestion_diameter_not_dragged_down_by_specks():
    """Regression: fine debris made the median diameter ~1 px-scale and
    the generated structure a dust cloud."""
    binary, _ = segment_particles(_flakes_with_specks_image())
    s = suggest_generator_settings([binary], pixel_per_um=2.0)
    # Real particles: radius 25 px → diameter 50 px → 25 µm at 2 px/µm
    assert s.diameter_um == pytest.approx(25.0, rel=0.10)
    assert s.n_particles < s.n_detected  # fines were excluded and reported


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_geometric_std_monodisperse_is_one():
    assert geometric_std(np.array([40.0] * 10)) == pytest.approx(1.0)


def test_geometric_std_polydisperse_is_above_one():
    rng = np.random.default_rng(0)
    d = np.exp(rng.normal(np.log(40), np.log(1.5), size=500))
    assert geometric_std(d) == pytest.approx(1.5, rel=0.05)


def test_geometric_std_single_particle_falls_back_to_one():
    assert geometric_std(np.array([40.0])) == 1.0


def test_suggestion_vf_matches_drawn_area():
    radii = [20] * 9
    gray = _circles_image(radii=radii)
    binary, _ = segment_particles(gray)
    s = suggest_generator_settings([binary], pixel_per_um=None)
    drawn = 9 * np.pi * 20**2 / (400 * 400) * 100
    assert s.volume_fraction_pct == pytest.approx(drawn, rel=0.05)
    assert s.diameter_um is None  # unknown scale
    assert s.diameter_px == pytest.approx(40.0, rel=0.05)


def test_suggestion_raises_when_no_particles():
    empty = np.zeros((100, 100), dtype=bool)
    with pytest.raises(ValueError):
        suggest_generator_settings([empty], pixel_per_um=1.0)
