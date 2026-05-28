"""Microstructure measurements computed from a binary image."""

from __future__ import annotations

import numpy as np
from skimage.measure import label, perimeter


def interfacial_length_um(binary: np.ndarray, pixel_per_um: float) -> float:
    """Total perimeter of all connected components in pixels, converted to µm.

    Matches the original micro_app.py behavior, including the per-label loop
    (kept for behavior parity; will be vectorized in a later pass).
    """
    labels = label(binary)
    interface_px = float(
        np.sum([perimeter(labels == i) for i in range(1, labels.max() + 1)])
    )
    return interface_px / pixel_per_um


def interface_to_area_ratio_per_um(
    interface_um: float, image_width_um: float, image_height_um: float
) -> float:
    """Ratio of interfacial length to image area, in µm⁻¹."""
    return interface_um / (image_width_um * image_height_um)
