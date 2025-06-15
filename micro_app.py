import streamlit as st
import numpy as np
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
    pixel_per_um = st.slider("Pixels per Micron (resolution)", 1, 10, 3)
    random_seed = st.number_input("Random Seed (0 = random)", 0, 9999, 0)
with col2:
    image_height_um = st.number_input("Image Height (µm)", 50.0, 1000.0, 200.0)
    volume_fraction = st.slider("Volume Fraction (%)", 1, 99, 20)
    shape = st.selectbox("Particle Shape", ["Circular", "Elliptical", "Irregular (Blob)"])

# --- RNG Setup ---
if random_seed != 0:
    np.random.seed(random_seed)

# --- Derived Sizes ---
width_px, height_px = int(image_width_um * pixel_per_um), int(image_height_um * pixel_per_um)
canvas = np.zeros((height_px, width_px), dtype=np.uint8)

# Estimate number of particles for desired volume fraction (use shape‑specific area)
rad_um = particle_diameter_um / 2
if shape == "Elliptical":
    shape_area_um2 = np.pi * rad_um * (0.8 * rad_um)  # average ry ≈ 0.8 r
elif shape == "Irregular (Blob)":
    shape_area_um2 = np.pi * rad_um**2 * 1.2          # blobs a bit larger on avg
else:
    shape_area_um2 = np.pi * rad_um**2

area_target_um2 = image_width_um * image_height_um * (volume_fraction / 100)
num_particles = max(1, int(area_target_um2 / shape_area_um2))

# Convert typical radius to px for placement boundaries
avg_rad_px = int(rad_um * pixel_per_um)

pil_img = Image.fromarray(canvas)
d = ImageDraw.Draw(pil_img)

# --- Particle Generation (simple non‑overlap rejection to improve homogeneity) ---
placed_centers = []
max_attempts = num_particles * 10
attempts = 0
while len(placed_centers) < num_particles and attempts < max_attempts:
    attempts += 1
    cx = np.random.randint(avg_rad_px, width_px - avg_rad_px)
    cy = np.random.randint(avg_rad_px, height_px - avg_rad_px)
    # Ensure new center not too close to existing (Poisson‑like)
    if all((cx - x)**2 + (cy - y)**2 > (2*avg_rad_px)**2 for x, y in placed_centers):
        r_px = int(avg_rad_px * (1 + np.random.uniform(-0.3, 0.3)))
        if shape == "Circular":
            d.ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px], fill=255)
        elif shape == "Elliptical":
            rx = r_px
            ry = int(r_px * np.random.uniform(0.5, 1.2))
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
        else:  # Irregular blob
            pad = 2 * r_px
            blob_canvas = np.zeros((height_px + 2 * pad, width_px + 2 * pad), dtype=np.uint8)
            rr, cc = ellipse(cy + pad, cx + pad, r_px, int(0.6 * r_px), shape=blob_canvas.shape)
            blob_canvas[rr, cc] = 255

            angle = np.random.rand() * 360
            blob_img = Image.fromarray(blob_canvas)
            blob_img = blob_img.rotate(angle, expand=False, fillcolor=0)
            blob_cropped = np.array(blob_img)[pad:-pad, pad:-pad]  # center-crop back to image size

            pil_img = Image.fromarray(np.maximum(np.array(pil_img), blob_cropped))
            d = ImageDraw.Draw(pil_img)
        placed_centers.append((cx, cy))

canvas = np.array(pil_img)

# --- Create composite with separate white scale‑box ---
scale_um = image_width_um / 5
scale_px = int(scale_um * pixel_per_um)
bar_h = max(4, int(0.01 * height_px))
box_h = bar_h + 30
final = Image.new("L", (width_px, height_px + box_h), color=255)
final.paste(pil_img, (0, 0))

# Draw scale bar
draw_final = ImageDraw.Draw(final)
bar_x1 = (width_px - scale_px) // 2
bar_y1 = height_px + 8
draw_final.rectangle([bar_x1, bar_y1, bar_x1 + scale_px, bar_y1 + bar_h], fill=0)

scale_label = f"{int(scale_um)} μm"
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
except:
    font = ImageFont.load_default()

bbox = draw_final.textbbox((0, 0), scale_label, font=font)
text_w = bbox[2] - bbox[0]
text_x = (width_px - text_w) // 2
text_y = bar_y1 + bar_h + 4
draw_final.text((text_x, text_y), scale_label, fill=0, font=font)

# --- Metrics ---
binary = (canvas > 0).astype(bool)
labels = label(binary)
interface_px = np.sum([perimeter(labels == i) for i in range(1, labels.max()+1)])
interface_um = interface_px / pixel_per_um
ratio = interface_um / (image_width_um * image_height_um)

# --- Streamlit Display ---
st.markdown("""---""")
st.subheader("Results")
st.write(f"**Particles Placed:** {len(placed_centers)} | **Interfacial Length:** {interface_um:.2f} µm | **Interface/Area:** {ratio:.5f} µm⁻¹")

st.image(np.array(final), caption="Simulated Microstructure", channels="GRAY", use_container_width=True)

buf = BytesIO()
final.save(buf, format="PNG")
st.download_button("Download PNG", data=buf.getvalue(), file_name="microstructure.png", mime="image/png")
