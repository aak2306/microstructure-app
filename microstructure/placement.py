"""Particle-placement loop.

Wraps the per-shape generators with rejection sampling for overlap and a
retry budget for high volume fractions.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import numpy as np
from PIL import Image, ImageDraw

from . import generators as gen


class PlacedParticle(NamedTuple):
    """One placed particle.

    ``r_px`` is the nominal radius after size-variation jitter — the actual
    drawn shape's extent depends on the shape type (ellipse aspect, blob
    bezier curve, etc.). It's still the best per-particle size summary for
    histograms and CSV export.
    """

    cx: int
    cy: int
    r_px: int


def place_particles(
    pil_img: Image.Image,
    shape: str,
    num_particles: int,
    avg_rad_px: int,
    size_variation: float,
    allow_overlap: bool,
    bumpiness_pct: float,
    jitter_pct: float,
    mix_ratio: Optional[int],
    volume_fraction: float,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> list[PlacedParticle]:
    """Place ``num_particles`` particles into ``pil_img``.

    Returns the list of particles actually placed (as PlacedParticle).
    Mutates ``pil_img``. ``progress_callback`` (if provided) is called with
    a 0..1 fraction every 100 particles and at completion.
    """
    draw = ImageDraw.Draw(pil_img)
    width_px, height_px = pil_img.width, pil_img.height

    placed_particles: list[PlacedParticle] = []
    max_attempts = int(num_particles * (100 if volume_fraction > 80 else 20))
    attempts = 0

    while len(placed_particles) < num_particles and attempts < max_attempts:
        attempts += 1
        cx = np.random.randint(avg_rad_px, width_px - avg_rad_px)
        cy = np.random.randint(avg_rad_px, height_px - avg_rad_px)
        if (not allow_overlap) and any(
            (cx - p.cx) ** 2 + (cy - p.cy) ** 2 < (1.8 * avg_rad_px) ** 2
            for p in placed_particles
        ):
            continue

        variation_factor = 1 + np.random.uniform(
            -size_variation / 100, size_variation / 100
        )
        r_px = int(avg_rad_px * variation_factor)

        placed = False
        if shape == gen.CIRCULAR:
            gen.draw_circle(draw, cx, cy, r_px)
            placed = True
        elif shape == gen.ELLIPTICAL:
            gen.draw_ellipse(draw, cx, cy, r_px)
            placed = True
        elif shape == gen.IRREGULAR:
            placed = gen.paste_blob(pil_img, cx, cy, r_px)
        elif shape == gen.ROUGH_SPHERES:
            gen.draw_rough_sphere(draw, cx, cy, r_px, bumpiness_pct)
            placed = True
        elif shape == gen.CRACKED_FLAKES:
            gen.draw_cracked_flake(draw, cx, cy, r_px, jitter_pct)
            placed = True
        elif shape == gen.MIXED:
            rv = np.random.rand()
            if rv < (mix_ratio or 0) / 100:
                gen.draw_circle(draw, cx, cy, r_px)
                placed = True
            elif rv < ((mix_ratio or 0) + (100 - (mix_ratio or 0)) / 2) / 100:
                gen.draw_ellipse(draw, cx, cy, r_px)
                placed = True
            else:
                placed = gen.paste_blob(pil_img, cx, cy, r_px)

        if placed:
            placed_particles.append(PlacedParticle(cx, cy, r_px))
            if progress_callback is not None and (
                len(placed_particles) % 100 == 0
                or len(placed_particles) == num_particles
            ):
                progress_callback(
                    min(len(placed_particles) / num_particles, 1.0)
                )

    return placed_particles
