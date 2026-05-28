"""Particle-placement loop.

Wraps the per-shape generators with rejection sampling for overlap and a
retry budget for high volume fractions. Supports periodic boundary
conditions: centers can sit anywhere on the canvas and particles wrap
across the toroidal edges, removing the no-go margin that biases the
non-PBC case.
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


def _wrap_positions(
    cx: int, cy: int, r_px: int, width_px: int, height_px: int
) -> list[tuple[int, int]]:
    """Return the toroidal-copy centers whose bounding box intersects the canvas.

    A particle near a corner needs up to three wrapped copies (one across
    each adjacent edge plus one across the diagonal). One near a non-corner
    edge needs one. Particles fully inside the canvas need none.
    """
    out: list[tuple[int, int]] = []
    for dx in (-width_px, 0, width_px):
        for dy in (-height_px, 0, height_px):
            if dx == 0 and dy == 0:
                continue
            nx, ny = cx + dx, cy + dy
            if (
                nx + r_px >= 0
                and nx - r_px < width_px
                and ny + r_px >= 0
                and ny - r_px < height_px
            ):
                out.append((nx, ny))
    return out


def _periodic_distance_sq(
    x1: int, y1: int, x2: int, y2: int, width_px: int, height_px: int
) -> int:
    """Squared distance under the minimum image convention."""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dx > width_px - dx:
        dx = width_px - dx
    if dy > height_px - dy:
        dy = height_px - dy
    return dx * dx + dy * dy


def _draw_one(
    draw: ImageDraw.ImageDraw,
    pil_img: Image.Image,
    shape: str,
    cx: int,
    cy: int,
    r_px: int,
    bumpiness_pct: float,
    jitter_pct: float,
    mix_ratio: Optional[int],
    periodic: bool,
) -> bool:
    """Dispatch one draw call to the right generator. Returns whether the
    particle was actually placed."""
    if shape == gen.CIRCULAR:
        gen.draw_circle(draw, cx, cy, r_px)
        return True
    if shape == gen.ELLIPTICAL:
        gen.draw_ellipse(draw, cx, cy, r_px)
        return True
    if shape == gen.IRREGULAR:
        return gen.paste_blob(pil_img, cx, cy, r_px, periodic=periodic)
    if shape == gen.ROUGH_SPHERES:
        gen.draw_rough_sphere(draw, cx, cy, r_px, bumpiness_pct)
        return True
    if shape == gen.CRACKED_FLAKES:
        gen.draw_cracked_flake(draw, cx, cy, r_px, jitter_pct)
        return True
    if shape == gen.MIXED:
        rv = np.random.rand()
        if rv < (mix_ratio or 0) / 100:
            gen.draw_circle(draw, cx, cy, r_px)
            return True
        if rv < ((mix_ratio or 0) + (100 - (mix_ratio or 0)) / 2) / 100:
            gen.draw_ellipse(draw, cx, cy, r_px)
            return True
        return gen.paste_blob(pil_img, cx, cy, r_px, periodic=periodic)
    raise ValueError(f"Unknown shape: {shape!r}")


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
    size_sampler: Optional[Callable[[], float]] = None,
    periodic_boundaries: bool = False,
) -> list[PlacedParticle]:
    """Place ``num_particles`` particles into ``pil_img``.

    Returns the list of particles actually placed (as PlacedParticle).
    Mutates ``pil_img``. ``progress_callback`` (if provided) is called with
    a 0..1 fraction every 100 particles and at completion.

    When ``size_sampler`` is provided, it overrides the inline uniform
    ``size_variation`` jitter — the sampler is called once per placement
    attempt to draw a fresh radius in pixels.

    When ``periodic_boundaries`` is True, centers are drawn from the full
    canvas (instead of a margin-restricted range), the overlap check uses
    the minimum image convention, and each particle is composited at its
    primary position plus 1–3 toroidal wrap copies whose bbox touches the
    canvas. Non-deterministic shape generators (ellipse aspect, rough
    sphere noise, cracked flake jitter, blob bezier curve) are replicated
    at each wrap position by save/restoring ``np.random`` state — the
    wrapped halves are guaranteed to be the same shape.
    """
    draw = ImageDraw.Draw(pil_img)
    width_px, height_px = pil_img.width, pil_img.height

    placed_particles: list[PlacedParticle] = []
    max_attempts = int(num_particles * (100 if volume_fraction > 80 else 20))
    attempts = 0

    while len(placed_particles) < num_particles and attempts < max_attempts:
        attempts += 1
        if periodic_boundaries:
            cx = np.random.randint(0, width_px)
            cy = np.random.randint(0, height_px)
        else:
            cx = np.random.randint(avg_rad_px, width_px - avg_rad_px)
            cy = np.random.randint(avg_rad_px, height_px - avg_rad_px)

        if not allow_overlap:
            threshold_sq = (1.8 * avg_rad_px) ** 2
            if periodic_boundaries:
                hit = any(
                    _periodic_distance_sq(
                        cx, cy, p.cx, p.cy, width_px, height_px
                    )
                    < threshold_sq
                    for p in placed_particles
                )
            else:
                hit = any(
                    (cx - p.cx) ** 2 + (cy - p.cy) ** 2 < threshold_sq
                    for p in placed_particles
                )
            if hit:
                continue

        if size_sampler is not None:
            r_px = max(1, int(size_sampler()))
        else:
            variation_factor = 1 + np.random.uniform(
                -size_variation / 100, size_variation / 100
            )
            r_px = int(avg_rad_px * variation_factor)

        if periodic_boundaries:
            state_before = np.random.get_state()
            placed = _draw_one(
                draw,
                pil_img,
                shape,
                cx,
                cy,
                r_px,
                bumpiness_pct,
                jitter_pct,
                mix_ratio,
                periodic=True,
            )
            state_after = np.random.get_state()
            if placed:
                for wx, wy in _wrap_positions(
                    cx, cy, r_px, width_px, height_px
                ):
                    np.random.set_state(state_before)
                    _draw_one(
                        draw,
                        pil_img,
                        shape,
                        wx,
                        wy,
                        r_px,
                        bumpiness_pct,
                        jitter_pct,
                        mix_ratio,
                        periodic=True,
                    )
                # Restore the state that came out of the primary draw so
                # the overall RNG sequence is independent of how many wrap
                # copies happened to be needed.
                np.random.set_state(state_after)
        else:
            placed = _draw_one(
                draw,
                pil_img,
                shape,
                cx,
                cy,
                r_px,
                bumpiness_pct,
                jitter_pct,
                mix_ratio,
                periodic=False,
            )

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
