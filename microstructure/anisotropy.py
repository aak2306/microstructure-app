"""Directional analysis of the binary microstructure.

Detects whether the interfacial length is uniformly distributed across
orientations (isotropic) or biased toward some axis (textured). The
machinery is the same as the Mean Intercept Length (MIL) tensor of
quantitative stereology (Underwood 1970; Harrigan & Mann 1984): for each
direction θ, count phase transitions along parallel scan lines and
divide by line length.

By the Cauchy-Crofton identity the average of P_L(θ) over [0, π) is
proportional to L_A; the *variation* across θ is what tells you about
anisotropy. Long horizontal features mean fewer crossings for horizontal
scan lines and more crossings for vertical ones — so P_L peaks
perpendicular to the elongation direction.

Implementation notes:

- Rotation: ``skimage.transform.rotate`` with ``order=0`` keeps the
  binary truly binary (nearest-neighbor; no anti-alias gray).
- Padding: rotating leaves zero-valued triangular corners that would
  bias the line-length denominator. We sidestep that by counting only
  inside a centered square small enough to fit inside the un-rotated
  image at any angle — for a side-S image that's S/√2.
- PBC: the rotation isn't periodic-aware. If you generated with PBC,
  P_L(θ) still reads correctly *inside* the inscribed crop because the
  wrap doesn't reach there for any θ; the index is a fair estimate.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from skimage.transform import rotate


def directional_intercept_density(
    binary: np.ndarray,
    angles_deg: Iterable[float],
    pixel_per_um: float,
) -> np.ndarray:
    """Compute P_L(θ) in µm⁻¹ for each angle in ``angles_deg``.

    For each angle, rotate the binary by -θ so horizontal scan lines in
    the rotated frame correspond to lines at angle θ in the original.
    Count phase transitions along those scan lines, divide by the total
    scan-line length, and convert to physical units.
    """
    if binary.ndim != 2:
        raise ValueError("binary must be a 2D array")

    H, W = binary.shape
    side = min(H, W)
    inscribed = int(side / math.sqrt(2))
    cy, cx = H // 2, W // 2
    half = inscribed // 2
    if half <= 0:
        return np.zeros(len(list(angles_deg)), dtype=float)

    binary_f = binary.astype(np.float32)

    p_l_values: list[float] = []
    for theta in angles_deg:
        rotated = rotate(
            binary_f,
            -float(theta),
            resize=False,
            preserve_range=True,
            order=0,
        )
        rotated_bin = rotated > 0.5
        crop = rotated_bin[cy - half : cy + half, cx - half : cx + half]
        if crop.size == 0 or crop.shape[1] < 2:
            p_l_values.append(0.0)
            continue
        transitions = int((crop[:, :-1] != crop[:, 1:]).sum())
        line_length_px = crop.shape[0] * (crop.shape[1] - 1)
        p_l_values.append(transitions / line_length_px * pixel_per_um)

    return np.array(p_l_values, dtype=float)


def anisotropy_index(p_l_values: np.ndarray) -> float:
    """Degree of anisotropy DA = (max − min) / (max + min).

    0 → fully isotropic (P_L constant across angles).
    1 → fully oriented (some direction has no boundary at all).
    Independent of overall L_A magnitude — purely a shape metric.
    """
    if p_l_values.size == 0:
        return 0.0
    mx = float(p_l_values.max())
    mn = float(p_l_values.min())
    if mx + mn == 0:
        return 0.0
    return (mx - mn) / (mx + mn)


def elongation_direction_deg(
    angles_deg: np.ndarray, p_l_values: np.ndarray
) -> float:
    """Estimated elongation axis of foreground features, in degrees.

    P_L(θ) is highest *perpendicular* to a feature's long axis (lines
    crossing across it cut more interfaces than lines running along
    it). So the elongation direction equals the angle of P_L minimum.
    Returns 0 if undefined (e.g. empty binary).
    """
    if p_l_values.size == 0:
        return 0.0
    return float(np.asarray(angles_deg)[int(np.argmin(p_l_values))])
