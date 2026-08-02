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


def remove_small_particles(
    binary: np.ndarray, min_diameter_px: float
) -> np.ndarray:
    """Drop connected components below ``min_diameter_px`` equivalent diameter.

    Applied to the binary *before* any measurement, so that a size floor
    the user sets is honoured consistently: the excluded specks vanish
    from the segmentation preview, from the volume fraction, and from
    L/A and S/V alike.

    This matters most for S/V. Fine debris has a very high
    perimeter-to-area ratio, so a swarm of specks can contribute a large
    share of the total interfacial length while being a negligible share
    of the phase area — inflating a "measured" S/V that is really an
    artefact of segmentation noise.
    """
    if min_diameter_px <= 0:
        return binary
    min_area = math.pi * (min_diameter_px / 2.0) ** 2
    return remove_small_objects(binary, min_size=int(math.ceil(min_area)))


@dataclass(frozen=True)
class ParticleDescriptors:
    """Pooled per-particle descriptor arrays for one or more binaries."""

    circularity: np.ndarray
    aspect_ratio: np.ndarray
    solidity: np.ndarray
    equivalent_diameter_px: np.ndarray
    area_px: np.ndarray

    @property
    def n_particles(self) -> int:
        return int(self.circularity.size)


def particle_descriptors(
    binaries: list[np.ndarray], min_diameter_px: float = 0.0
) -> ParticleDescriptors:
    """Measure per-particle shape descriptors, pooled across images.

    Particles touching the image border are excluded from the *shape*
    statistics (their outlines are cut, which corrupts circularity and
    aspect ratio) but still count toward area-based metrics computed
    elsewhere.

    ``min_diameter_px`` drops anything whose equivalent diameter is below
    that floor — a manual override for images where scratches, staining,
    or pitting survive thresholding as spurious "particles".
    """
    circ: list[float] = []
    aspect: list[float] = []
    solidity: list[float] = []
    equiv_d: list[float] = []
    area: list[float] = []

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
            if region.equivalent_diameter_area < min_diameter_px:
                continue
            circ.append(min(1.0, 4 * math.pi * region.area / perimeter**2))
            minor = region.axis_minor_length
            aspect.append(
                region.axis_major_length / minor if minor > 0 else 1.0
            )
            solidity.append(region.solidity)
            equiv_d.append(region.equivalent_diameter_area)
            area.append(float(region.area))

    return ParticleDescriptors(
        circularity=np.array(circ),
        aspect_ratio=np.array(aspect),
        solidity=np.array(solidity),
        equivalent_diameter_px=np.array(equiv_d),
        area_px=np.array(area),
    )


def drop_fines(
    desc: ParticleDescriptors, fines_area_ratio: float = 0.05
) -> ParticleDescriptors:
    """Exclude fine debris so it cannot dominate number-weighted statistics.

    Real micrographs carry polishing debris, dust, and sub-resolution
    speckle; each speck counts as one particle, so a few hundred of them
    drag the *median* size and shape toward noise even though together
    they are a sliver of the phase area.

    The reference is the area-weighted median particle — the one
    straddling 50% of cumulative area, which is always a representative
    "real" particle no matter how many specks there are. Particles
    smaller than ``fines_area_ratio`` of its area are excluded. A
    monodisperse population keeps everything; the 5% default corresponds
    to dropping particles below ~22% of the reference diameter.
    """
    if desc.n_particles == 0:
        return desc
    order = np.argsort(desc.area_px)
    cumulative = np.cumsum(desc.area_px[order])
    median_idx = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    reference_area = desc.area_px[order][min(median_idx, desc.n_particles - 1)]
    keep = desc.area_px >= fines_area_ratio * reference_area
    return ParticleDescriptors(
        circularity=desc.circularity[keep],
        aspect_ratio=desc.aspect_ratio[keep],
        solidity=desc.solidity[keep],
        equivalent_diameter_px=desc.equivalent_diameter_px[keep],
        area_px=desc.area_px[keep],
    )


def _smooth_ellipse_circularity(aspect: float) -> float:
    """Circularity 4πA/P² of a *smooth* ellipse with the given aspect ratio.

    Uses Ramanujan's perimeter approximation. This is the ceiling any
    outline with that elongation can reach; measured circularity far
    below it means the boundary is angular or rough, not just elongated.
    """
    a, b = max(aspect, 1.0), 1.0
    h = ((a - b) / (a + b)) ** 2
    perimeter = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
    return 4 * math.pi * (math.pi * a * b) / perimeter**2


def classify_shape(
    median_circularity: float,
    median_aspect: float,
    median_solidity: float,
) -> str:
    """Map pooled descriptor medians onto the closest generator preset.

    Elongation alone cannot separate ellipses from angular flakes — a
    convex flake can have solidity near 1 and any aspect ratio. The
    discriminator is *smoothness*: measured circularity relative to the
    smooth-ellipse ceiling for the same aspect ratio. Faceted outlines
    (straight edges, corners) fall well below that ceiling.
    """
    smoothness = median_circularity / _smooth_ellipse_circularity(median_aspect)
    if median_solidity < 0.85:
        return gen.CRACKED_FLAKES  # concave / fractured outlines
    if median_aspect >= 1.6:
        return gen.ELLIPTICAL if smoothness >= 0.85 else gen.CRACKED_FLAKES
    if median_circularity >= 0.85:
        return gen.CIRCULAR
    if smoothness < 0.72 and median_aspect >= 1.25:
        return gen.CRACKED_FLAKES  # angular but convex, moderately elongated
    if median_circularity >= 0.65:
        return gen.ROUGH_SPHERES
    return gen.IRREGULAR


def geometric_std(diameters: np.ndarray) -> float:
    """Geometric standard deviation σ_g = exp(std(ln d)); 1.0 = monodisperse."""
    if diameters.size < 2:
        return 1.0
    return float(np.exp(np.std(np.log(diameters))))


def area_weighted_mean_diameter(diameters: np.ndarray) -> float:
    """Area-weighted mean diameter D = Σd² / Σd — the 2D Sauter diameter.

    This is the diameter a *monodisperse* simulation must use to
    reproduce both the volume fraction and the interfacial length of a
    polydisperse population, which is what makes the simulated S/V match
    the micrograph's. For N circles of diameter dᵢ in an image of area A:

        L/A = π·Σdᵢ / A          VF = π·Σdᵢ² / (4A)

    Generating n circles of a single diameter D and demanding both match
    gives n·D = Σdᵢ and n·D² = Σdᵢ², hence D = Σdᵢ² / Σdᵢ.

    It is far more robust to fine debris than the number median, which a
    swarm of specks captures outright. It is not *immune*: fines carry
    real perimeter, so enough of them do pull it down — which is correct
    behaviour when the fines are real material, and why spurious ones
    are removed first by ``drop_fines`` and the manual size floor.
    """
    if diameters.size == 0:
        return 0.0
    total = float(np.sum(diameters))
    if total <= 0:
        return 0.0
    return float(np.sum(diameters**2) / total)


@dataclass(frozen=True)
class GeneratorSuggestion:
    """Generator settings inferred from uploaded micrographs."""

    shape: str
    volume_fraction_pct: float
    diameter_um: float | None  # area-weighted; None when scale is unknown
    diameter_px: float  # area-weighted mean diameter Σd²/Σd
    median_diameter_px: float  # number median, reported for comparison
    sigma_g: float
    bumpiness_pct: float
    n_particles: int  # particles used for shape/size stats (fines excluded)
    n_detected: int  # all complete particles found
    median_circularity: float
    median_aspect: float
    median_solidity: float
    diameters_px: np.ndarray  # kept particles, for the size histogram


def suggest_generator_settings(
    binaries: list[np.ndarray],
    pixel_per_um: float | None,
    min_diameter_px: float = 0.0,
) -> GeneratorSuggestion:
    """Pool descriptors across ``binaries`` and propose generator settings.

    Shape and size statistics are computed on the coarse population only
    (see ``drop_fines``); volume fraction uses every particle pixel, since
    fines are real phase area even when they shouldn't steer the size.

    The reported diameter is the *area-weighted* mean (see
    ``area_weighted_mean_diameter``) rather than the number median, so
    that a simulation built from it reproduces the micrograph's L/A and
    S/V — and so that surviving fine debris cannot deflate it.
    """
    all_desc = particle_descriptors(binaries, min_diameter_px)
    if all_desc.n_particles == 0:
        raise ValueError(
            "no complete particles found — check the segmentation polarity, "
            "lower the minimum particle size, or use an image where "
            "particles do not all touch the border"
        )
    desc = drop_fines(all_desc)

    circ_med = float(np.median(desc.circularity))
    aspect_med = float(np.median(desc.aspect_ratio))
    solidity_med = float(np.median(desc.solidity))
    shape = classify_shape(circ_med, aspect_med, solidity_med)

    total_px = sum(b.size for b in binaries)
    particle_px = sum(int(b.sum()) for b in binaries)
    vf_pct = 100.0 * particle_px / total_px

    d_px = area_weighted_mean_diameter(desc.equivalent_diameter_px)
    d_median_px = float(np.median(desc.equivalent_diameter_px))
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
        median_diameter_px=d_median_px,
        sigma_g=geometric_std(desc.equivalent_diameter_px),
        bumpiness_pct=bumpiness,
        n_particles=desc.n_particles,
        n_detected=all_desc.n_particles,
        median_circularity=circ_med,
        median_aspect=aspect_med,
        median_solidity=solidity_med,
        diameters_px=desc.equivalent_diameter_px,
    )
