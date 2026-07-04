"""Analysis of uploaded micrographs.

Segments a grayscale micrograph into a particle/matrix binary, measures
per-particle shape descriptors, and maps the pooled statistics onto the
closest generator preset so a synthetic microstructure can be produced
with matching geometry.

Descriptors used (all from ``skimage.measure.regionprops``):

- circularity  = 4πA / P²   (1.0 for a perfect circle; falls with
  boundary roughness and elongation)
- aspect ratio = major axis / minor axis of the fitted ellipse
- solidity     = A / A_convex (dips below ~0.9 for concave, fractured,
  or star-like outlines)
- equivalent diameter = diameter of the circle with the same area

Classification is deliberately rule-based rather than learned: the five
generator presets occupy well-separated regions of this descriptor
space, and rules keep the mapping transparent and testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from . import generators as gen

# Regions smaller than this many pixels are treated as noise (dust,
# JPEG speckle) and excluded from both the mask and the statistics.
MIN_PARTICLE_PX = 16


def segment_particles(
    gray: np.ndarray, particles_are_bright: bool | None = None
) -> tuple[np.ndarray, bool]:
    """Otsu-threshold a grayscale image into a particle mask.

    ``particles_are_bright=None`` auto-detects polarity by assuming the
    particles are the minority phase — the usual case for a dispersed
    second phase in a matrix. Returns ``(binary, particles_are_bright)``
    with the polarity actually used.
    """
    if gray.ndim != 2:
        raise ValueError("expected a 2D grayscale array")
    if gray.min() == gray.max():
        raise ValueError("image has a single gray level — nothing to segment")

    threshold = threshold_otsu(gray)
    bright = gray > threshold
    if particles_are_bright is None:
        particles_are_bright = bool(bright.mean() <= 0.5)
    binary = bright if particles_are_bright else ~bright
    binary = remove_small_objects(binary, min_size=MIN_PARTICLE_PX)
    return binary, particles_are_bright


@dataclass(frozen=True)
class ParticleDescriptors:
    """Pooled per-particle descriptor arrays for one or more binaries."""

    circularity: np.ndarray
    aspect_ratio: np.ndarray
    solidity: np.ndarray
    equivalent_diameter_px: np.ndarray

    @property
    def n_particles(self) -> int:
        return int(self.circularity.size)


def particle_descriptors(binaries: list[np.ndarray]) -> ParticleDescriptors:
    """Measure per-particle shape descriptors, pooled across images.

    Particles touching the image border are excluded from the *shape*
    statistics (their outlines are cut, which corrupts circularity and
    aspect ratio) but still count toward area-based metrics computed
    elsewhere.
    """
    circ: list[float] = []
    aspect: list[float] = []
    solidity: list[float] = []
    equiv_d: list[float] = []

    for binary in binaries:
        h, w = binary.shape
        for region in regionprops(label(binary)):
            if region.area < MIN_PARTICLE_PX:
                continue
            r0, c0, r1, c1 = region.bbox
            if r0 == 0 or c0 == 0 or r1 == h or c1 == w:
                continue  # cut by the image edge
            perimeter = region.perimeter_crofton
            if perimeter <= 0:
                continue
            circ.append(min(1.0, 4 * math.pi * region.area / perimeter**2))
            minor = region.axis_minor_length
            aspect.append(
                region.axis_major_length / minor if minor > 0 else 1.0
            )
            solidity.append(region.solidity)
            equiv_d.append(region.equivalent_diameter_area)

    return ParticleDescriptors(
        circularity=np.array(circ),
        aspect_ratio=np.array(aspect),
        solidity=np.array(solidity),
        equivalent_diameter_px=np.array(equiv_d),
    )


def classify_shape(
    median_circularity: float,
    median_aspect: float,
    median_solidity: float,
) -> str:
    """Map pooled descriptor medians onto the closest generator preset.

    Order matters: elongation is checked first because an elongated
    particle also has low circularity; concavity (solidity) next because
    fractured outlines depress every other descriptor too.
    """
    if median_aspect >= 1.6:
        return gen.ELLIPTICAL
    if median_solidity < 0.85:
        return gen.CRACKED_FLAKES
    if median_circularity >= 0.85:
        return gen.CIRCULAR
    if median_circularity >= 0.65:
        return gen.ROUGH_SPHERES
    return gen.IRREGULAR


def geometric_std(diameters: np.ndarray) -> float:
    """Geometric standard deviation σ_g = exp(std(ln d)); 1.0 = monodisperse."""
    if diameters.size < 2:
        return 1.0
    return float(np.exp(np.std(np.log(diameters))))


@dataclass(frozen=True)
class GeneratorSuggestion:
    """Generator settings inferred from uploaded micrographs."""

    shape: str
    volume_fraction_pct: float
    diameter_um: float | None  # None when the image scale is unknown
    diameter_px: float
    sigma_g: float
    bumpiness_pct: float
    n_particles: int
    median_circularity: float
    median_aspect: float
    median_solidity: float


def suggest_generator_settings(
    binaries: list[np.ndarray], pixel_per_um: float | None
) -> GeneratorSuggestion:
    """Pool descriptors across ``binaries`` and propose generator settings."""
    desc = particle_descriptors(binaries)
    if desc.n_particles == 0:
        raise ValueError(
            "no complete particles found — check the segmentation polarity "
            "or use an image where particles do not all touch the border"
        )

    circ_med = float(np.median(desc.circularity))
    aspect_med = float(np.median(desc.aspect_ratio))
    solidity_med = float(np.median(desc.solidity))
    shape = classify_shape(circ_med, aspect_med, solidity_med)

    total_px = sum(b.size for b in binaries)
    particle_px = sum(int(b.sum()) for b in binaries)
    vf_pct = 100.0 * particle_px / total_px

    d_px = float(np.median(desc.equivalent_diameter_px))
    d_um = d_px / pixel_per_um if pixel_per_um else None

    # Rough-sphere bumpiness from the circularity deficit: a smooth circle
    # sits near 1.0, and each % of radial noise costs roughly 1% of
    # circularity in this regime.
    bumpiness = float(np.clip((0.92 - circ_med) * 120.0, 3.0, 30.0))

    return GeneratorSuggestion(
        shape=shape,
        volume_fraction_pct=vf_pct,
        diameter_um=d_um,
        diameter_px=d_px,
        sigma_g=geometric_std(desc.equivalent_diameter_px),
        bumpiness_pct=bumpiness,
        n_particles=desc.n_particles,
        median_circularity=circ_med,
        median_aspect=aspect_med,
        median_solidity=solidity_med,
    )
