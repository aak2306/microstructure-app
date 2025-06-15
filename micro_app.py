import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, perimeter
from skimage.draw import ellipse
from scipy.ndimage import binary_fill_holes
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

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
pil_image = Image.fromarray(image)
draw = ImageDraw.Draw(pil_image)

for _ in range(num_particles):
    x = np.random.randint(avg_radius_px, image_width_px - avg_radius_px)
    y = np.random.randint(avg_radius_px, image_height_px - avg_radius_px)
    r = int(avg_radius_px * (1 + np.random.uniform(-0.3, 0.3)))

    if shape == "Circular":
        draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    elif shape == "Elliptical":
        rx = int(r)
        ry = int(r * np.random.uniform(0.5, 1.5))
        draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=255)

    elif shape == "Irregular (Blob)":
        rr, cc = ellipse(x, y, r, r // 2, shape=image.shape)
        blob = np.zeros_like(image)
        blob[rr, cc] = 1
        blob = binary_fill_holes(blob).astype(np.uint8) * 255
        image = np.maximum(np.array(pil_image), blob.astype(np.uint8))
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

# --- Add Info Box at the Bottom ---
info_box_height = 40
extended_height = image_height_px + info_box_height
extended_image = Image.new("L", (image_width_px, extended_height), color=255)
extended_image.paste(pil_image, (0, 0))
draw = ImageDraw.Draw(extended_image)

# Draw white rectangle info box with black border
draw.rectangle([(0, image_height_px), (image_width_px - 1, extended_height - 1)], outline=0, fill=255)

# Add scale info text centered
info_text = f"Image Size: {image_width_um:.1f} µm × {image_height_um:.1f} µm   |   Particle Size: {particle_diameter_um:.1f} µm   |   Volume Fraction: {volume_fraction}%"
font_size = 12
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except:
    font = ImageFont.load_default()
text_width, text_height = draw.textsize(info_text, font=font)
text_x = (image_width_px - text_width) // 2
text_y = image_height_px + (info_box_height - text_height) // 2

# Add black outline
for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        draw.text((text_x + dx, text_y + dy), info_text, fill=0, font=font)
# Main text in white
draw.text((text_x, text_y), info_text, fill=0, font=font)

image = np.array(extended_image)

# --- Analysis ---
binary = image[:image_height_px, :] > 0
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
