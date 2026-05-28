"""Analytical validation: compare measured L_A against the closed-form
relation for monodisperse non-overlapping circles in 2D.

For N circles of diameter d randomly placed without overlap in area A:

    φ      = N · π · (d/2)² / A
    L_A    = N · π · d / A    = 4 · φ / d

The 4·φ/d identity is exact, independent of placement, so any deviation
between this and the app's measured L_A reflects:
- rasterization (pixel discretization of circle edges),
- the Crofton-perimeter estimator's bias on small-radius shapes,
- placement budget exhaustion at high φ (fewer particles actually placed
  than analytically required).

Use it as a built-in sanity check before trusting absolute numbers on more
complex shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
from PIL import Image

from . import generators as gen
from .metrics import (
    interface_to_area_ratio_per_um,
    interfacial_length_um,
    measured_volume_fraction,
)
from .placement import place_particles


@dataclass(frozen=True)
class ValidationPoint:
    phi_target: float
    phi_achieved: float
    L_A_measured: float  # µm⁻¹
    L_A_theory: float  # 4·φ_achieved / d, µm⁻¹
    particles_requested: int
    particles_placed: int

    @property
    def relative_error_pct(self) -> float:
        """Signed % error of measured L_A vs theory."""
        if self.L_A_theory == 0.0:
            return 0.0
        return 100.0 * (self.L_A_measured - self.L_A_theory) / self.L_A_theory


def _theory_L_A_per_um(phi: float, diameter_um: float) -> float:
    """L_A = 4·φ / d for monodisperse non-overlapping circles."""
    if diameter_um <= 0:
        return 0.0
    return 4.0 * phi / diameter_um


def validate_monodisperse_circles(
    canvas_um: float,
    diameter_um: float,
    pixel_per_um: float,
    phi_targets: Iterable[float],
    seed: int = 0,
    progress_callback: Optional[Callable[[float], None]] = None,
    periodic_boundaries: bool = False,
) -> list[ValidationPoint]:
    """Run a φ sweep and compute L_A_measured vs L_A_theory at each point.

    Uses ``place_particles`` with shape=Circular, size_variation=0,
    allow_overlap=False — so each particle has exactly the nominal radius
    and the analytical formula applies exactly to whichever particles do
    fit on the canvas.

    ``periodic_boundaries`` toggles the toroidal placement+drawing path.
    With PBC enabled the achieved φ tracks the target much more closely
    (no edge-margin depletion), so this is the sharpest test of whether
    the perimeter measurement itself is unbiased.
    """
    phi_list = [float(p) for p in phi_targets]
    width_px = height_px = int(canvas_um * pixel_per_um)
    avg_rad_px = max(1, int((diameter_um / 2.0) * pixel_per_um))
    points: list[ValidationPoint] = []

    np.random.seed(int(seed))
    for i, phi in enumerate(phi_list):
        # Estimate particle count from the closed-form relation (circles, so
        # the existing area_factor is exactly 1.0; no overcount).
        canvas = np.zeros((height_px, width_px), dtype=np.uint8)
        pil_img = Image.fromarray(canvas)
        circle_area_um2 = np.pi * (diameter_um / 2.0) ** 2
        num_requested = max(1, int(canvas_um * canvas_um * phi / circle_area_um2))

        particles = place_particles(
            pil_img=pil_img,
            shape=gen.CIRCULAR,
            num_particles=num_requested,
            avg_rad_px=avg_rad_px,
            size_variation=0.0,
            allow_overlap=False,
            bumpiness_pct=0.0,
            jitter_pct=0.0,
            mix_ratio=None,
            volume_fraction=phi * 100.0,
            periodic_boundaries=periodic_boundaries,
        )

        binary = np.array(pil_img) > 0
        interface_um = interfacial_length_um(binary, pixel_per_um)
        L_A_measured = interface_to_area_ratio_per_um(interface_um, canvas_um, canvas_um)
        phi_achieved = measured_volume_fraction(binary)
        L_A_theory = _theory_L_A_per_um(phi_achieved, diameter_um)

        points.append(
            ValidationPoint(
                phi_target=phi,
                phi_achieved=phi_achieved,
                L_A_measured=L_A_measured,
                L_A_theory=L_A_theory,
                particles_requested=num_requested,
                particles_placed=len(particles),
            )
        )

        if progress_callback is not None:
            progress_callback((i + 1) / len(phi_list))

    return points
