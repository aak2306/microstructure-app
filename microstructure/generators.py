"""Per-shape particle generators.

Each function draws a single particle into a target image (or returns False if
the particle cannot fit). All randomness uses the legacy ``np.random`` global
state to match the original micro_app.py behavior; seeding should be done by
the caller.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.special import binom

CIRCULAR = "Circular"
ELLIPTICAL = "Elliptical"
IRREGULAR = "Irregular (Blob)"
ROUGH_SPHERES = "Rough Spheres"
CRACKED_FLAKES = "Cracked Flakes"
MIXED = "Mixed (Circular + Elliptical + Irregular)"

SHAPES = [CIRCULAR, ELLIPTICAL, IRREGULAR, ROUGH_SPHERES, CRACKED_FLAKES, MIXED]


def draw_circle(draw: ImageDraw.ImageDraw, cx: int, cy: int, r_px: int) -> None:
    draw.ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px], fill=255)


def draw_ellipse(draw: ImageDraw.ImageDraw, cx: int, cy: int, r_px: int) -> None:
    ry = int(r_px * np.random.uniform(0.5, 1.2))
    draw.ellipse([cx - r_px, cy - ry, cx + r_px, cy + ry], fill=255)


def draw_rough_sphere(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    r_px: int,
    bumpiness_pct: float,
) -> None:
    num_pts = 90
    angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
    noise = np.random.normal(0, bumpiness_pct / 100.0, num_pts)
    noise = gaussian_filter(noise, sigma=3, mode="wrap")
    radii = r_px * (1 + noise)
    x = cx + radii * np.cos(angles)
    y = cy + radii * np.sin(angles)
    draw.polygon(list(zip(x, y)), fill=255)


def draw_cracked_flake(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    r_px: int,
    jitter_pct: float,
) -> None:
    n_vert = np.random.randint(6, 12)
    angles = np.sort(np.random.rand(n_vert) * 2 * np.pi)
    radii = r_px * (0.6 + 0.4 * np.random.rand(n_vert))
    x = cx + radii * np.cos(angles)
    y = cy + radii * np.sin(angles)
    jitter = (np.random.rand(n_vert, 2) - 0.5) * (jitter_pct / 100.0 * 2 * r_px)
    poly = list(zip(x + jitter[:, 0], y + jitter[:, 1]))
    draw.polygon(poly, fill=255)


def paste_blob(pil_img: Image.Image, cx: int, cy: int, r_px: int) -> bool:
    """Generate an irregular blob and composite it into ``pil_img``.

    Returns False if the blob falls outside the image bounds, in which case
    ``pil_img`` is left unchanged.
    """

    def ccw_sort(p: np.ndarray) -> np.ndarray:
        d = p - np.mean(p, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        return p[np.argsort(ang)]

    def bezier(ctrl: np.ndarray, num: int = 200) -> np.ndarray:
        n = len(ctrl) - 1
        t = np.linspace(0, 1, num)
        out = np.zeros((num, 2))
        for k in range(n + 1):
            out += (
                binom(n, k)
                * (t ** k)[:, None]
                * ((1 - t) ** (n - k))[:, None]
                * ctrl[k]
            )
        return out

    ctrl_pts = np.random.randn(6, 2)
    ctrl_pts = ccw_sort(ctrl_pts) * 0.8
    ctrl_pts = np.vstack([ctrl_pts, ctrl_pts[0]])
    curve = bezier(ctrl_pts)
    x, y = curve.T
    x -= x.min()
    y -= y.min()
    scale = 2 * r_px / max(x.max(), y.max())
    x *= scale
    y *= scale

    w, h = int(x.max()) + 2, int(y.max()) + 2
    blob_img = Image.new("L", (w, h), 0)
    draw_blob = ImageDraw.Draw(blob_img)
    draw_blob.polygon(list(zip(x, y)), fill=255)
    blob_img = blob_img.rotate(np.random.rand() * 360, expand=True, fillcolor=0)
    blob_arr = np.array(blob_img)
    bh, bw = blob_arr.shape

    height_px, width_px = pil_img.height, pil_img.width
    top = cy - bh // 2
    left = cx - bw // 2
    if top < 0 or left < 0 or top + bh > height_px or left + bw > width_px:
        return False

    temp = np.array(pil_img)
    region = temp[top : top + bh, left : left + bw]
    temp[top : top + bh, left : left + bw] = np.maximum(region, blob_arr)
    pil_img.paste(Image.fromarray(temp))
    return True
