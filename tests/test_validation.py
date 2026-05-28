"""Tests for the validation sweep."""

import numpy as np
import pytest

from microstructure.validation import (
    ValidationPoint,
    _theory_L_A_per_um,
    validate_monodisperse_circles,
)


def test_theory_L_A_matches_closed_form():
    # 4 · 0.25 / 10 = 0.1 µm⁻¹
    assert _theory_L_A_per_um(phi=0.25, diameter_um=10.0) == pytest.approx(0.1)


def test_theory_L_A_zero_when_diameter_is_zero():
    assert _theory_L_A_per_um(phi=0.5, diameter_um=0.0) == 0.0


def test_validate_returns_one_point_per_phi():
    phis = [0.05, 0.10, 0.15]
    pts = validate_monodisperse_circles(
        canvas_um=200.0,
        diameter_um=10.0,
        pixel_per_um=8,
        phi_targets=phis,
        seed=0,
    )
    assert len(pts) == 3
    assert [p.phi_target for p in pts] == phis


def test_validate_point_fields_make_sense():
    pts = validate_monodisperse_circles(
        canvas_um=200.0,
        diameter_um=10.0,
        pixel_per_um=8,
        phi_targets=[0.10],
        seed=0,
    )
    p = pts[0]
    assert isinstance(p, ValidationPoint)
    assert p.particles_placed > 0
    assert p.particles_placed <= p.particles_requested
    # Achieved should land within 30% of target at this canvas size — looser
    # tolerance than the L_A test below because some particles fail to
    # place at the edge.
    assert 0.7 * p.phi_target <= p.phi_achieved <= 1.3 * p.phi_target


def test_measured_L_A_tracks_theory_at_low_phi():
    """The 4·φ/d relation should hold within ~10% at low φ where overlap
    rejection is rare and rasterization error is well-behaved."""
    pts = validate_monodisperse_circles(
        canvas_um=300.0,
        diameter_um=10.0,
        pixel_per_um=10,
        phi_targets=[0.05, 0.10, 0.15],
        seed=0,
    )
    for p in pts:
        assert p.L_A_theory > 0
        rel_err = abs(p.L_A_measured - p.L_A_theory) / p.L_A_theory
        assert rel_err < 0.10, (
            f"phi={p.phi_target}: measured {p.L_A_measured:.5f} vs theory "
            f"{p.L_A_theory:.5f} (err {rel_err:.2%})"
        )


def test_validate_is_deterministic_with_seed():
    pts1 = validate_monodisperse_circles(
        canvas_um=200.0,
        diameter_um=10.0,
        pixel_per_um=8,
        phi_targets=[0.05, 0.10],
        seed=42,
    )
    pts2 = validate_monodisperse_circles(
        canvas_um=200.0,
        diameter_um=10.0,
        pixel_per_um=8,
        phi_targets=[0.05, 0.10],
        seed=42,
    )
    for a, b in zip(pts1, pts2):
        assert a == b


def test_progress_callback_invoked_each_point():
    seen: list[float] = []
    validate_monodisperse_circles(
        canvas_um=150.0,
        diameter_um=10.0,
        pixel_per_um=6,
        phi_targets=[0.05, 0.10, 0.15],
        seed=0,
        progress_callback=seen.append,
    )
    assert len(seen) == 3
    assert seen[-1] == 1.0


def test_relative_error_zero_when_theory_zero():
    p = ValidationPoint(
        phi_target=0.0,
        phi_achieved=0.0,
        L_A_measured=0.0,
        L_A_theory=0.0,
        particles_requested=0,
        particles_placed=0,
    )
    assert p.relative_error_pct == 0.0
