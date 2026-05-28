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
