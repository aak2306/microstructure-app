"""Tests for individual particle generators."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from microstructure import generators as gen


def _blank(w: int = 200, h: int = 200) -> Image.Image:
    return Image.fromarray(np.zeros((h, w), dtype=np.uint8))


def test_draw_circle_fills_pixels():
    img = _blank()
    draw = ImageDraw.Draw(img)
    gen.draw_circle(draw, cx=100, cy=100, r_px=20)
    arr = np.array(img)
    # roughly pi*r^2 white pixels, allow generous tolerance for raster edges
    assert 1000 < int((arr > 0).sum()) < 1600


def test_draw_circle_centered():
    img = _blank()
    draw = ImageDraw.Draw(img)
    gen.draw_circle(draw, cx=100, cy=100, r_px=15)
    arr = np.array(img)
    ys, xs = np.where(arr > 0)
    assert abs(int(xs.mean()) - 100) <= 1
    assert abs(int(ys.mean()) - 100) <= 1


def test_paste_blob_returns_false_when_too_close_to_edge():
    img = _blank()
    np.random.seed(0)
    placed = gen.paste_blob(img, cx=2, cy=2, r_px=30)
    assert placed is False
    # image must remain blank when placement fails
    assert int(np.array(img).sum()) == 0


def test_paste_blob_places_when_inside_bounds():
    img = _blank(400, 400)
    np.random.seed(1)
    placed = gen.paste_blob(img, cx=200, cy=200, r_px=30)
    assert placed is True
    assert int((np.array(img) > 0).sum()) > 0


def test_draw_rough_sphere_deterministic_with_seed():
    img1 = _blank()
    img2 = _blank()
    np.random.seed(42)
    gen.draw_rough_sphere(ImageDraw.Draw(img1), 100, 100, 20, bumpiness_pct=10.0)
    np.random.seed(42)
    gen.draw_rough_sphere(ImageDraw.Draw(img2), 100, 100, 20, bumpiness_pct=10.0)
    assert np.array_equal(np.array(img1), np.array(img2))


def test_draw_cracked_flake_deterministic_with_seed():
    img1 = _blank()
    img2 = _blank()
    np.random.seed(7)
    gen.draw_cracked_flake(ImageDraw.Draw(img1), 100, 100, 20, jitter_pct=15.0)
    np.random.seed(7)
    gen.draw_cracked_flake(ImageDraw.Draw(img2), 100, 100, 20, jitter_pct=15.0)
    assert np.array_equal(np.array(img1), np.array(img2))


@pytest.mark.parametrize("shape", gen.SHAPES)
def test_all_shape_names_are_unique_strings(shape):
    assert isinstance(shape, str)
    assert gen.SHAPES.count(shape) == 1
