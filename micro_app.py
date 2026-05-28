"""Streamlit entry point for the Microstructure Interface Area Calculator.

The actual generation and analysis lives in the ``microstructure`` package;
this module only wires inputs from the UI to those functions and displays
the results.
"""

from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from microstructure import generators as gen
from microstructure.metrics import (
    interface_to_area_ratio_per_um,
    interfacial_length_um,
    measured_volume_fraction,
)
from microstructure.placement import place_particles
from microstructure.rendering import add_scale_bar

st.set_page_config(page_title="Microstructure Generator", layout="centered")
st.title("Microstructure Interface Area Calculator")

# --- UI ---
col1, col2 = st.columns(2)
with col1:
    image_width_um = st.number_input("Image Width (µm)", 50.0, 1000.0, 200.0)
    particle_diameter_um = st.number_input(
        "Avg. Particle Diameter (µm)",
        min_value=0.1,
        max_value=100.0,
        value=10.0,
        step=0.1,
        format="%.1f",
    )
    pixel_per_um = st.slider(
        "Pixels per Micron", min_value=1, max_value=100, value=10, step=1
    )
    size_variation = st.number_input(
        "Size variation ± (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1
    )
    rng_seed = st.number_input(
        "Random Seed (-1 = random each run)",
        min_value=-1,
        max_value=9999,
        value=-1,
        step=1,
    )
with col2:
    image_height_um = st.number_input("Image Height (µm)", 50.0, 1000.0, 200.0)
    volume_fraction = st.number_input(
        "Volume Fraction (%)",
        min_value=0.1,
        max_value=99.9,
        value=20.0,
        step=0.1,
        format="%.1f",
    )
    shape = st.selectbox("Particle Shape", gen.SHAPES)

if shape == gen.ROUGH_SPHERES:
    bumpiness_pct = st.slider(
        "Bumpiness ± (%)", min_value=0.0, max_value=30.0, value=10.0, step=1.0
    )
else:
    bumpiness_pct = 10.0

if shape == gen.CRACKED_FLAKES:
    jitter_pct = st.slider(
        "Edge jitter ± (%)", min_value=0.0, max_value=30.0, value=15.0, step=1.0
    )
else:
    jitter_pct = 15.0

allow_overlap = st.checkbox("Allow particle overlap", value=False)

mix_ratio = (
    st.slider("% Circular in Mix", 0, 100, 33) if shape == gen.MIXED else None
)

calculate = st.button("Calculate")
if not calculate:
    st.stop()

if rng_seed >= 0:
    np.random.seed(int(rng_seed))

width_px = int(image_width_um * pixel_per_um)
height_px = int(image_height_um * pixel_per_um)
canvas = np.zeros((height_px, width_px), dtype=np.uint8)

rad_um = particle_diameter_um / 2
area_factor = gen.expected_area_factor(shape, mix_ratio)
shape_area_um2 = np.pi * rad_um**2 * area_factor
area_target_um2 = image_width_um * image_height_um * volume_fraction / 100
num_particles = max(1, int(area_target_um2 / shape_area_um2))

avg_rad_px = max(1, int(rad_um * pixel_per_um))
pil_img = Image.fromarray(canvas)

progress = st.progress(0.0)
centers = place_particles(
    pil_img=pil_img,
    shape=shape,
    num_particles=num_particles,
    avg_rad_px=avg_rad_px,
    size_variation=size_variation,
    allow_overlap=allow_overlap,
    bumpiness_pct=bumpiness_pct,
    jitter_pct=jitter_pct,
    mix_ratio=mix_ratio,
    volume_fraction=volume_fraction,
    progress_callback=progress.progress,
)

canvas = np.array(pil_img)
final = add_scale_bar(pil_img, image_width_um, image_height_um, pixel_per_um)

binary = canvas > 0
interface_um = interfacial_length_um(binary, pixel_per_um)
ratio = interface_to_area_ratio_per_um(interface_um, image_width_um, image_height_um)
achieved_vf_pct = measured_volume_fraction(binary) * 100.0

st.markdown("---")
st.subheader("Results")

if len(centers) < num_particles:
    st.warning(
        f"Placed only {len(centers)} of {num_particles} requested particles — "
        "the non-overlap budget was exhausted. Try enabling **Allow particle "
        "overlap**, lowering **Volume Fraction**, or increasing **Image** size."
    )

col_a, col_b = st.columns(2)
col_a.metric("Particles Placed", f"{len(centers)}")
col_a.metric("Interfacial Length", f"{interface_um:.2f} µm")
col_b.metric(
    "Volume Fraction (achieved)",
    f"{achieved_vf_pct:.2f}%",
    delta=f"{achieved_vf_pct - volume_fraction:+.2f}% vs target",
    delta_color="off",
)
col_b.metric("Interface / Area", f"{ratio:.5f} µm⁻¹")

st.image(
    np.array(final),
    caption="Simulated Microstructure",
    channels="GRAY",
    use_container_width=True,
)

buf = BytesIO()
final.save(buf, format="PNG")
st.download_button(
    "Download PNG",
    data=buf.getvalue(),
    file_name="microstructure.png",
    mime="image/png",
)
