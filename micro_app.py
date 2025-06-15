import streamlit as st
import numpy as np
import cv2
from skimage.measure import label, perimeter
from skimage.draw import ellipse
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Microstructure Generator", layout="centered")
st.title("Microstructure Interface Area Calculator")

# --- User Inputs ---
col1, col2 = st.columns(2)
with col1:
    image_width_um = st.number_input("Image Width (µm)", 50.0, 1000.0, 200.0)
    particle_diameter_um = st.number_input("Avg. Particle Diameter (µm)", 1.0, 100.0, 10.0)
    pixel_per_um = st.slider("Pixels per Micron", 1, 10, 2)
with col2:
    image_height_um = st.number_input("Image Height (µm)", 50.0, 1000.0, 200.0)
    volume_fraction = st.slider("Volume Fraction (%)", 1, 99, 20)
    shape = st.selectbox("Particle Shape", ["Circular", "Elliptical", "Irregular (Blob)"])

random_seed = 42
np.random.seed(random_seed)

# --- Derived Quantities ---
image_width_px = int(image_width_um * pixel_per_um)
image_height_px = int(image_height_um * pixel_per_um)
image = np.zeros((image_height_px, image_width_px), dtype=np.uint8)

particle_radius_um = particle_diameter_um / 2
particle_area_um2 = np.pi * particle_radius_um**2
image_area_um2 = image_width_um * image_height_um
num_particles = int((volume_fraction / 100.0) * image_area_um2 / particle_area_um2)
avg_radius_px = int(particle_radius_um * pixel_per_um)

# --- Particle Generation ---
for _ in range(num_particles):
    x = np.random.randint(avg_radius_px, image_width_px - avg_radius_px)
    y = np.random.randint(avg_radius_px, image_height_px - avg_radius_px)
    r = int(avg_radius_px * (1 + np.random.uniform(-0.3, 0.3)))

    if shape == "Circular":
        cv2.circle(image, (x, y), r, 255, -1)
    elif shape == "Elliptical":
        rx = int(r)
        ry = int(r * np.random.uniform(0.5, 1.5))
        angle = np.random.randint(0, 180)
        cv2.ellipse(image, (x, y), (rx, ry), angle, 0, 360, 255, -1)
    elif shape == "Irregular (Blob)":
        rr, cc = ellipse(x, y, r, r // 2, shape=image.shape)
        blob = np.zeros_like(image)
        blob[rr, cc] = 1
        blob = binary_fill_holes(blob).astype(np.uint8)
        angle = np.random.rand() * 360
        M = cv2.getRotationMatrix2D((y, x), angle, 1.0)
        rotated_blob = cv2.warpAffine(blob * 255, M, (image_width_px, image_height_px))
        image = np.maximum(image, rotated_blob.astype(np.uint8))

# --- Analysis ---
binary = image > 0
labeled = label(binary)
interface_length_px = np.sum([perimeter(labeled == i) for i in range(1, labeled.max()+1)])
interface_length_um = interface_length_px / pixel_per_um
ratio = interface_length_um / image_area_um2

# --- Output ---
st.markdown("""---""")
st.subheader("Results")
st.write(f"**Image Area:** {image_area_um2:.2f} µm²") 
st.write(f"**Particles Generated:** {num_particles}")
st.write(f"**Interfacial Length:** {interface_length_um:.2f} µm") 
st.write(f"**Interface/Area Ratio:** {ratio:.5f} µm⁻¹")

# --- Image Display ---
st.image(image, caption="Simulated Microstructure", channels="GRAY")

# --- Download ---
buf = BytesIO()
Image.fromarray(image).save(buf, format="PNG")
st.download_button("Download Image (PNG)", data=buf.getvalue(), file_name="microstructure.png", mime="image/png")