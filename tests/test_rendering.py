"""Tests for image rendering helpers."""

import numpy as np
from PIL import Image

from microstructure.rendering import add_scale_bar


def test_scale_bar_extends_canvas_height():
    img = Image.fromarray(np.zeros((100, 200), dtype=np.uint8))
    out = add_scale_bar(img, image_width_um=200.0, image_height_um=100.0, pixel_per_um=1.0)
    assert out.width == img.width
    assert out.height > img.height


def test_scale_bar_does_not_modify_input():
    img = Image.fromarray(np.full((50, 50), 128, dtype=np.uint8))
    original_arr = np.array(img).copy()
    _ = add_scale_bar(img, 50.0, 50.0, 1.0)
    assert np.array_equal(np.array(img), original_arr)
