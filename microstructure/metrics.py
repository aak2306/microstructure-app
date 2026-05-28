"""Microstructure measurements computed from a binary image."""

from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops


def interfacial_length_um(binary: np.ndarray, pixel_per_um: float) -> float:
    """Total perimeter of all connected components in pixels, converted to µm.

    Uses ``perimeter_crofton``, which is more accurate than the naive
    boundary-pixel count on rasterized circular shapes (the naive method
    over- or under-counts depending on edge orientation; Crofton averages
    over four discretized directions).

    Iterating with ``regionprops`` is O(sum of component bounding-box areas)
    instead of O(N_components × full_image_area), which matters when the
    user generates many particles on a large canvas.
    """
    labels = label(binary)
    if labels.max() == 0:
        return 0.0
    interface_px = float(sum(r.perimeter_crofton for r in regionprops(labels)))
    return interface_px / pixel_per_um


def interface_to_area_ratio_per_um(
    interface_um: float, image_width_um: float, image_height_um: float
) -> float:
    """Ratio of interfacial length to image area, in µm⁻¹."""
    return interface_um / (image_width_um * image_height_um)


def measured_volume_fraction(binary: np.ndarray) -> float:
    """Fraction of pixels that are foreground, in 0..1.

    For 2D microstructure analysis, area fraction equals volume fraction
    under standard stereological assumptions (e.g. Delesse's principle).
    """
    if binary.size == 0:
        return 0.0
    return float(binary.sum()) / float(binary.size)


# ---------------------------------------------------------------------------
# Stereology (2D measurements → 3D estimates)
# ---------------------------------------------------------------------------
# These conversions assume the 2D image is an isotropic uniform random (IUR)
# section through a 3D structure. See Underwood, "Quantitative Stereology"
# (1970) and DeHoff & Rhines, "Quantitative Microscopy" (1968). If the image
# *is* the structure (genuinely 2D), the 3D quantities below are not
# physically meaningful for your sample — only the 2D L_A and V_V are.


def specific_surface_area_per_um(interface_to_area_ratio_per_um: float) -> float:
    """Specific surface area S_V (3D) from L_A (2D), in µm⁻¹.

    Standard stereological relation S_V = (4/π) · L_A. Holds exactly for
    an isotropic structure measured on a randomly oriented section.
    """
    return (4.0 / np.pi) * interface_to_area_ratio_per_um


def mean_intercept_length_um(volume_fraction: float, s_v_per_um: float) -> float:
    """Mean chord length through the dispersed phase, in µm.

    ⟨L_α⟩ = 4 · V_V / S_V. Average distance a random ray spends inside a
    particle before crossing back to the matrix. ``volume_fraction`` is
    a fraction in 0..1, not a percent.
    """
    if s_v_per_um <= 0.0:
        return 0.0
    return 4.0 * volume_fraction / s_v_per_um


def mean_free_path_um(volume_fraction: float, s_v_per_um: float) -> float:
    """Mean chord length through the matrix phase, in µm.

    λ = 4 · (1 - V_V) / S_V. Average distance between particle encounters
    along a random ray. Common proxy for diffusion path length and a
    classical input to dispersion-strengthening models.
    """
    if s_v_per_um <= 0.0:
        return 0.0
    return 4.0 * (1.0 - volume_fraction) / s_v_per_um
