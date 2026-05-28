"""Tests for the particle-placement loop."""

import numpy as np
from PIL import Image

from microstructure import generators as gen
from microstructure.placement import PlacedParticle, place_particles


def _blank(w: int = 400, h: int = 400) -> Image.Image:
    return Image.fromarray(np.zeros((h, w), dtype=np.uint8))


def test_placement_deterministic_with_seed():
    img1, img2 = _blank(), _blank()
    np.random.seed(123)
    c1 = place_particles(
        pil_img=img1,
        shape=gen.CIRCULAR,
        num_particles=20,
        avg_rad_px=15,
        size_variation=5.0,
        allow_overlap=False,
        bumpiness_pct=10.0,
        jitter_pct=15.0,
        mix_ratio=None,
        volume_fraction=20.0,
    )
    np.random.seed(123)
    c2 = place_particles(
        pil_img=img2,
        shape=gen.CIRCULAR,
        num_particles=20,
        avg_rad_px=15,
        size_variation=5.0,
        allow_overlap=False,
        bumpiness_pct=10.0,
        jitter_pct=15.0,
        mix_ratio=None,
        volume_fraction=20.0,
    )
    assert c1 == c2
    assert np.array_equal(np.array(img1), np.array(img2))


def test_placement_respects_requested_count():
    img = _blank()
    np.random.seed(0)
    centers = place_particles(
        pil_img=img,
        shape=gen.CIRCULAR,
        num_particles=10,
        avg_rad_px=10,
        size_variation=0.0,
        allow_overlap=False,
        bumpiness_pct=10.0,
        jitter_pct=15.0,
        mix_ratio=None,
        volume_fraction=10.0,
    )
    assert len(centers) <= 10


def test_placement_centers_within_canvas_margin():
    img = _blank()
    np.random.seed(0)
    avg_rad_px = 10
    centers = place_particles(
        pil_img=img,
        shape=gen.CIRCULAR,
        num_particles=20,
        avg_rad_px=avg_rad_px,
        size_variation=0.0,
        allow_overlap=False,
        bumpiness_pct=10.0,
        jitter_pct=15.0,
        mix_ratio=None,
        volume_fraction=10.0,
    )
    for p in centers:
        assert avg_rad_px <= p.cx < img.width - avg_rad_px
        assert avg_rad_px <= p.cy < img.height - avg_rad_px
        assert p.r_px > 0


def test_placement_returns_placed_particle_namedtuples():
    img = _blank()
    np.random.seed(0)
    particles = place_particles(
        pil_img=img,
        shape=gen.CIRCULAR,
        num_particles=5,
        avg_rad_px=12,
        size_variation=10.0,
        allow_overlap=False,
        bumpiness_pct=10.0,
        jitter_pct=15.0,
        mix_ratio=None,
        volume_fraction=10.0,
    )
    assert len(particles) > 0
    for p in particles:
        assert isinstance(p, PlacedParticle)
        assert hasattr(p, "cx") and hasattr(p, "cy") and hasattr(p, "r_px")


def test_progress_callback_invoked_at_completion():
    img = _blank()
    seen: list[float] = []
    np.random.seed(0)
    place_particles(
        pil_img=img,
        shape=gen.CIRCULAR,
        num_particles=5,
        avg_rad_px=10,
        size_variation=0.0,
        allow_overlap=False,
        bumpiness_pct=10.0,
        jitter_pct=15.0,
        mix_ratio=None,
        volume_fraction=10.0,
        progress_callback=seen.append,
    )
    assert seen, "expected progress_callback to fire at least once"
    assert seen[-1] == 1.0
