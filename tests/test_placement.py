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


def test_size_sampler_overrides_uniform_jitter():
    """When size_sampler is provided, every placed radius should come from
    it — independent of the size_variation argument."""
    img = _blank()
    fixed_r = 7
    np.random.seed(0)
    particles = place_particles(
        pil_img=img,
        shape=gen.CIRCULAR,
        num_particles=8,
        avg_rad_px=fixed_r,
        size_variation=99.0,  # ignored when size_sampler is set
        allow_overlap=False,
        bumpiness_pct=10.0,
        jitter_pct=15.0,
        mix_ratio=None,
        volume_fraction=10.0,
        size_sampler=lambda: fixed_r,
    )
    assert all(p.r_px == fixed_r for p in particles)


def test_pbc_centers_can_hit_canvas_edges():
    """Without PBC, every cx is in [r, W-r]. With PBC, every cx is in [0, W)."""
    img = _blank(400, 400)
    np.random.seed(0)
    particles = place_particles(
        pil_img=img,
        shape=gen.CIRCULAR,
        num_particles=200,
        avg_rad_px=20,
        size_variation=0.0,
        allow_overlap=False,
        bumpiness_pct=10.0,
        jitter_pct=15.0,
        mix_ratio=None,
        volume_fraction=15.0,
        periodic_boundaries=True,
    )
    xs = np.array([p.cx for p in particles])
    ys = np.array([p.cy for p in particles])
    # Should see particles near (within r_px of) both edges
    assert xs.min() < 20  # left edge
    assert xs.max() >= 400 - 20  # right edge
    assert ys.min() < 20
    assert ys.max() >= 400 - 20


def test_pbc_deterministic_with_seed():
    img1, img2 = _blank(300, 300), _blank(300, 300)
    np.random.seed(99)
    p1 = place_particles(
        pil_img=img1,
        shape=gen.CIRCULAR,
        num_particles=30,
        avg_rad_px=12,
        size_variation=5.0,
        allow_overlap=False,
        bumpiness_pct=10.0,
        jitter_pct=15.0,
        mix_ratio=None,
        volume_fraction=15.0,
        periodic_boundaries=True,
    )
    np.random.seed(99)
    p2 = place_particles(
        pil_img=img2,
        shape=gen.CIRCULAR,
        num_particles=30,
        avg_rad_px=12,
        size_variation=5.0,
        allow_overlap=False,
        bumpiness_pct=10.0,
        jitter_pct=15.0,
        mix_ratio=None,
        volume_fraction=15.0,
        periodic_boundaries=True,
    )
    assert p1 == p2
    assert np.array_equal(np.array(img1), np.array(img2))


def test_pbc_wrap_draws_on_opposite_edge():
    """A particle placed near the right edge should produce foreground
    pixels near the left edge too (the wrapped half)."""
    img = _blank(200, 200)
    # Place a single circle right against the right edge by tightly
    # restricting the placement: we cheat by drawing directly via
    # place_particles with only 1 particle and overriding cx via the seed.
    # Instead, run a few PBC particles and confirm both edges accumulate
    # foreground pixels (without PBC, edges are guaranteed empty).
    np.random.seed(0)
    place_particles(
        pil_img=img,
        shape=gen.CIRCULAR,
        num_particles=30,
        avg_rad_px=15,
        size_variation=0.0,
        allow_overlap=False,
        bumpiness_pct=10.0,
        jitter_pct=15.0,
        mix_ratio=None,
        volume_fraction=20.0,
        periodic_boundaries=True,
    )
    arr = np.array(img)
    assert arr[:, 0].sum() > 0  # left edge column has foreground
    assert arr[:, -1].sum() > 0  # right edge column has foreground


def test_pbc_pushes_achieved_vf_closer_to_target():
    """The whole point of PBC: removing the edge margin pushes achieved
    VF up toward target. Particle:image-edge ratio is moderate so the
    bias is measurable but not extreme."""
    target_pct = 20.0
    args = dict(
        shape=gen.CIRCULAR,
        num_particles=100,
        avg_rad_px=15,
        size_variation=0.0,
        allow_overlap=False,
        bumpiness_pct=10.0,
        jitter_pct=15.0,
        mix_ratio=None,
        volume_fraction=target_pct,
    )

    img_nopbc = _blank(300, 300)
    np.random.seed(0)
    place_particles(pil_img=img_nopbc, **args, periodic_boundaries=False)
    vf_nopbc = float((np.array(img_nopbc) > 0).mean())

    img_pbc = _blank(300, 300)
    np.random.seed(0)
    place_particles(pil_img=img_pbc, **args, periodic_boundaries=True)
    vf_pbc = float((np.array(img_pbc) > 0).mean())

    # PBC should achieve a higher (closer to target) VF than non-PBC.
    assert vf_pbc > vf_nopbc


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
