"""Tests for micrograph analysis: segmentation, descriptors, classification."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from microstructure import generators as gen
from microstructure.analysis import (
    area_weighted_mean_diameter,
    classify_shape,
    drop_fines,
    geometric_std,
    particle_descriptors,
    remove_small_particles,
    segment_particles,
    split_touching_particles,
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
# Area-weighted diameter
# ---------------------------------------------------------------------------

def test_area_weighted_equals_value_for_monodisperse():
    d = np.full(20, 40.0)
    assert area_weighted_mean_diameter(d) == pytest.approx(40.0)


def test_area_weighted_far_exceeds_number_average_with_many_specks():
    """10 real particles plus 500 specks. The number statistics collapse
    to speck scale; the area-weighted mean stays within the same order
    of magnitude as the real particles."""
    both = np.concatenate([np.full(10, 100.0), np.full(500, 2.0)])
    assert np.median(both) == 2.0  # number median is pure speck
    assert np.mean(both) < 4.0  # number mean nearly as bad
    assert area_weighted_mean_diameter(both) == pytest.approx(51.0, rel=0.01)


def test_area_weighted_exceeds_number_median_for_skewed_population():
    d = np.concatenate([np.full(100, 5.0), np.full(5, 80.0)])
    assert area_weighted_mean_diameter(d) == pytest.approx(38.3, rel=0.01)
    assert area_weighted_mean_diameter(d) > 7 * np.median(d)


def test_area_weighted_handles_empty_and_zero():
    assert area_weighted_mean_diameter(np.array([])) == 0.0
    assert area_weighted_mean_diameter(np.zeros(5)) == 0.0


def test_area_weighted_diameter_reproduces_l_a_and_vf():
    """The defining property: n circles of diameter D with n·D = Σd
    match a polydisperse population's VF *and* L/A simultaneously."""
    rng = np.random.default_rng(0)
    d = np.exp(rng.normal(np.log(30), np.log(1.8), size=2000))
    D = area_weighted_mean_diameter(d)
    n = np.sum(d) / D  # count fixed by matching L/A
    assert n * D**2 == pytest.approx(np.sum(d**2), rel=1e-9)  # VF matches too


# ---------------------------------------------------------------------------
# Manual size floor
# ---------------------------------------------------------------------------

def test_remove_small_particles_drops_specks_keeps_particles():
    binary, _ = segment_particles(_flakes_with_specks_image())
    before = int(binary.sum())
    cleaned = remove_small_particles(binary, min_diameter_px=20.0)
    # The 9 circles of radius 25 survive; the radius-3 specks do not.
    assert particle_descriptors([cleaned]).n_particles == 9
    assert int(cleaned.sum()) < before


def test_remove_small_particles_zero_floor_is_a_no_op():
    binary, _ = segment_particles(_circles_image())
    assert np.array_equal(remove_small_particles(binary, 0.0), binary)


def test_remove_small_particles_cuts_perimeter_more_than_area():
    """The point of the filter: specks are a big share of interfacial
    length but a small share of phase area."""
    from microstructure.metrics import interfacial_length_um

    binary, _ = segment_particles(_flakes_with_specks_image())
    cleaned = remove_small_particles(binary, min_diameter_px=20.0)
    area_kept = cleaned.sum() / binary.sum()
    perim_kept = interfacial_length_um(cleaned, 1.0) / interfacial_length_um(
        binary, 1.0
    )
    # Specks are ~11% of the phase area but ~46% of the interfacial
    # length — which is exactly why they wreck a measured S/V.
    assert area_kept > 0.85
    assert perim_kept < 0.65
    assert perim_kept < area_kept


def test_min_diameter_px_filters_small_particles():
    binary, _ = segment_particles(_flakes_with_specks_image())
    unfiltered = particle_descriptors([binary])
    filtered = particle_descriptors([binary], min_diameter_px=20.0)
    assert filtered.n_particles < unfiltered.n_particles
    assert filtered.equivalent_diameter_px.min() >= 20.0


def test_min_diameter_px_zero_is_a_no_op():
    binary, _ = segment_particles(_circles_image())
    assert (
        particle_descriptors([binary], min_diameter_px=0.0).n_particles
        == particle_descriptors([binary]).n_particles
    )


def test_min_diameter_px_too_high_raises_helpful_error():
    binary, _ = segment_particles(_circles_image())
    with pytest.raises(ValueError, match="minimum particle size"):
        suggest_generator_settings([binary], 1.0, min_diameter_px=10_000.0)


def test_suggestion_reports_both_diameters():
    binary, _ = segment_particles(_flakes_with_specks_image())
    s = suggest_generator_settings([binary], pixel_per_um=2.0)
    # Real particles are radius 25 px → diameter 50 px
    assert s.diameter_px == pytest.approx(50.0, rel=0.10)
    assert s.median_diameter_px > 0
    assert s.diameters_px.size == s.n_particles


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


# ---------------------------------------------------------------------------
# Watershed splitting of touching particles
# ---------------------------------------------------------------------------

def _touching_pair_image(size: int = 300, r: int = 40) -> np.ndarray:
    """Two circles overlapping slightly — one blob under plain labelling."""
    img = Image.new("L", (size, size), 30)
    draw = ImageDraw.Draw(img)
    cy = size // 2
    for cx in (size // 2 - r + 8, size // 2 + r - 8):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=220)
    return np.array(img)


def test_plain_labelling_fuses_touching_particles():
    binary, _ = segment_particles(_touching_pair_image())
    assert particle_descriptors([binary], split_touching=False).n_particles == 1


def test_splitting_separates_touching_particles():
    binary, _ = segment_particles(_touching_pair_image())
    assert particle_descriptors([binary], split_touching=True).n_particles == 2


def test_splitting_recovers_true_particle_size():
    """The fused blob reads far too large; splitting restores ~2r."""
    r = 40
    binary, _ = segment_particles(_touching_pair_image(r=r))
    fused = particle_descriptors([binary], split_touching=False)
    split = particle_descriptors([binary], split_touching=True)
    assert fused.equivalent_diameter_px[0] > 1.3 * 2 * r
    assert np.median(split.equivalent_diameter_px) == pytest.approx(
        2 * r, rel=0.15
    )


def test_splitting_leaves_isolated_particles_alone():
    binary, _ = segment_particles(_circles_image(n=9, r=20))
    without = particle_descriptors([binary], split_touching=False)
    with_split = particle_descriptors([binary], split_touching=True)
    assert with_split.n_particles == without.n_particles
    assert np.median(with_split.equivalent_diameter_px) == pytest.approx(
        np.median(without.equivalent_diameter_px), rel=0.05
    )


def test_splitting_fixes_shape_classification_of_fused_cluster():
    """A fused pair looks elongated and non-circular; split, it is circular."""
    binary, _ = segment_particles(_touching_pair_image())
    fused = suggest_generator_settings([binary], 1.0, split_touching=False)
    split = suggest_generator_settings([binary], 1.0, split_touching=True)
    assert fused.median_aspect > 1.6
    assert split.median_aspect < 1.3
    assert split.shape in (gen.CIRCULAR, gen.ROUGH_SPHERES)


def test_split_touching_particles_handles_empty_binary():
    empty = np.zeros((50, 50), dtype=bool)
    assert split_touching_particles(empty).max() == 0


# ---------------------------------------------------------------------------
# Border cropping (removing burnt-in scale bars / labels)
# ---------------------------------------------------------------------------

def test_crop_borders_trims_expected_shape():
    from micro_app import _crop_borders

    a = np.zeros((200, 400), dtype=np.uint8)
    assert _crop_borders(a, 10, 10, 0, 0).shape == (160, 400)
    assert _crop_borders(a, 0, 0, 25, 25).shape == (200, 200)
    assert _crop_borders(a, 0, 0, 0, 0).shape == (200, 400)


def test_crop_borders_refuses_to_erase_image():
    from micro_app import _crop_borders

    # 40% off each side of 50 px leaves exactly 10 px, which is allowed;
    # anything that would leave under 10 px returns the image untouched.
    a = np.zeros((50, 50), dtype=np.uint8)
    assert _crop_borders(a, 40, 40, 40, 40).shape == (10, 10)
    b = np.zeros((20, 20), dtype=np.uint8)
    assert _crop_borders(b, 40, 40, 0, 0).shape == (20, 20)


def test_crop_removes_a_burnt_in_scale_bar():
    """A dark bar in the bottom margin reads as a particle until cropped."""
    img = Image.new("L", (400, 400), 30)
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 100, 200, 200], fill=220)   # one real particle
    draw.rectangle([20, 370, 260, 382], fill=220)  # burnt-in scale bar
    gray = np.array(img)

    from micro_app import _crop_borders

    binary_full, _ = segment_particles(gray)
    binary_crop, _ = segment_particles(_crop_borders(gray, 0, 12, 0, 0))
    assert particle_descriptors([binary_full]).n_particles == 2
    assert particle_descriptors([binary_crop]).n_particles == 1
