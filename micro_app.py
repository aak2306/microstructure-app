import streamlit as st
import numpy as np
import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, Image as PILImage
from skimage.measure import label, perimeter
from skimage.draw import ellipse
from scipy.ndimage import gaussian_filter, binary_dilation, binary_fill_holes

st.set_page_config(page_title="Microstructure Generator", layout="centered")
st.title("Microstructure Interface Area Calculator")

# --- UI ---
col1, col2 = st.columns(2)
with col1:
    image_width_um = st.number_input("Image Width (µm)", 50.0, 1000.0, 200.0)
    particle_diameter_um = st.number_input("Avg. Particle Diameter (µm)", min_value=0.1, max_value=100.0, value=10.0, step=0.1, format="%.1f")
    pixel_per_um = st.slider("Pixels per Micron", min_value=1, max_value=100, value=10, step=1)
    rng_seed = st.number_input("Random Seed (0=random)", 0, 9999, 0)
with col2:
    image_height_um = st.number_input("Image Height (µm)", 50.0, 1000.0, 200.0)
    volume_fraction = st.number_input("Volume Fraction (%)", min_value=0.1, max_value=99.9, value=20.0, step=0.1, format="%.1f")
    shape = st.selectbox("Particle Shape", [
        "Circular", "Elliptical", "Irregular (Blob)", "Mixed (Circular + Elliptical + Irregular)"])

allow_overlap = st.checkbox("Allow particle overlap", value=False)

mix_ratio = (st.slider("% Circular in Mix", 0, 100, 33)
             if shape == "Mixed (Circular + Elliptical + Irregular)" else None)

# --- Run trigger ---
calculate = st.button("Calculate")
if not calculate:
    st.stop()

# --- RNG ---
if rng_seed:
    np.random.seed(int(rng_seed))

# --- Irregular mask generation disabled ---
blob_masks = []

# --- Derived sizes ---
width_px, height_px = int(image_width_um * pixel_per_um), int(image_height_um * pixel_per_um)
canvas = np.zeros((height_px, width_px), dtype=np.uint8)

rad_um = particle_diameter_um / 2
shape_area_um2 = np.pi * rad_um**2  # rough estimate for all shapes
area_target_um2 = image_width_um * image_height_um * volume_fraction / 100
num_particles = max(1, int(area_target_um2 / shape_area_um2))

avg_rad_px = max(1, int(rad_um * pixel_per_um))
pil_img = Image.fromarray(canvas)
draw = ImageDraw.Draw(pil_img)

# --- Helper: generate irregular blob ---
def paste_blob(center_x, center_y, r_px):
    """Generate a smooth Bezier blob mask ≈ 2*r_px in diameter and paste it.
    Returns True if pasted fully in-bounds, else False (so caller can retry).
    """
    # --- helper functions for Bezier blob generation ---
    def ccw_sort(p):
        d = p - np.mean(p, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        return p[np.argsort(ang)]

    from scipy.special import binom

    def bezier(ctrl, num=200):
        n = len(ctrl) - 1
        t = np.linspace(0, 1, num)
        out = np.zeros((num, 2))
        for k in range(n + 1):
            out += binom(n, k) * (t ** k)[:, None] * ((1 - t) ** (n - k))[:, None] * ctrl[k]
        return out

    # random control points ~polygon
    ctrl_pts = np.random.randn(6, 2)
    ctrl_pts = ccw_sort(ctrl_pts) * 0.8
    ctrl_pts = np.vstack([ctrl_pts, ctrl_pts[0]])  # close loop
    curve = bezier(ctrl_pts)
    x, y = curve.T
    # normalise to bounding box
    x -= x.min(); y -= y.min()
    scale = 2 * r_px / max(x.max(), y.max())
    x *= scale; y *= scale

    w, h = int(x.max()) + 2, int(y.max()) + 2
    blob_img = Image.new("L", (w, h), 0)
    draw_blob = ImageDraw.Draw(blob_img)
    draw_blob.polygon(list(zip(x, y)), fill=255)
    blob_img = blob_img.rotate(np.random.rand() * 360, expand=True, fillcolor=0)
    blob_arr = np.array(blob_img)
    bh, bw = blob_arr.shape

    top = center_y - bh // 2
    left = center_x - bw // 2
    if top < 0 or left < 0 or top + bh > height_px or left + bw > width_px:
        return False  # out of bounds, caller will retry

    temp = np.array(pil_img)
    region = temp[top:top + bh, left:left + bw]
    temp[top:top + bh, left:left + bw] = np.maximum(region, blob_arr)
    pil_img.paste(Image.fromarray(temp))
    return True

# --- Particle placement ---
progress_bar = st.progress(0.0)
progress_text = st.empty()
centers = []
max_attempts = int(num_particles * (100 if volume_fraction > 80 else 20))
attempts = 0
while len(centers) < num_particles and attempts < max_attempts:
    attempts += 1
    cx = np.random.randint(avg_rad_px, width_px - avg_rad_px)
    cy = np.random.randint(avg_rad_px, height_px - avg_rad_px)
    if (not allow_overlap) and any((cx-x)**2 + (cy-y)**2 < (1.8*avg_rad_px)**2 for x,y in centers):
        continue

    r_px = int(avg_rad_px * (1 + np.random.uniform(-0.05, 0.05)))

    placed = False
    if shape == "Circular":
        draw.ellipse([cx-r_px, cy-r_px, cx+r_px, cy+r_px], fill=255)
        placed = True
    elif shape == "Elliptical":
        ry = int(r_px * np.random.uniform(0.5, 1.2))
        draw.ellipse([cx-r_px, cy-ry, cx+r_px, cy+ry], fill=255)
        placed = True
    elif shape == "Irregular (Blob)":
        placed = paste_blob(cx, cy, r_px)
    else:  # Mixed
        rv = np.random.rand()
        if rv < mix_ratio/100:
            draw.ellipse([cx-r_px, cy-r_px, cx+r_px, cy+r_px], fill=255)
            placed = True
        elif rv < (mix_ratio + (100-mix_ratio)/2)/100:
            ry = int(r_px * np.random.uniform(0.5, 1.2))
            draw.ellipse([cx-r_px, cy-ry, cx+r_px, cy+ry], fill=255)
            placed = True
        else:
            placed = paste_blob(cx, cy, r_px)

    if placed:
        centers.append((cx, cy))
        if len(centers) % 100 == 0 or len(centers) == num_particles:
            pct = int(100 * len(centers) / num_particles)
            progress_bar.progress(pct / 100)
            progress_text.text(f"Placing particles: {pct}%")

progress_bar.progress(1.0)
progress_text.text("Rendering results ...")
canvas = np.array(pil_img)

# --- Scale bar ---
scale_um = image_width_um / 5
scale_px = int(scale_um * pixel_per_um)
bar_h = max(4, int(0.01*height_px))
box_h = bar_h + 30
final = Image.new("L", (width_px, height_px+box_h), color=255)
final.paste(pil_img, (0,0))

bar_y = height_px + 8
draw_final = ImageDraw.Draw(final)
bar_x1 = (width_px-scale_px)//2
draw_final.rectangle([bar_x1, bar_y, bar_x1+scale_px, bar_y+bar_h], fill=0)

label_txt = f"{int(scale_um)} μm"
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
except:
    font = ImageFont.load_default()

bbox = draw_final.textbbox((0, 0), label_txt, font=font)
w = bbox[2] - bbox[0]
h = bbox[3] - bbox[1]
text_x = (width_px - w) // 2
draw_final.text((text_x, bar_y+bar_h+4), label_txt, font=font, fill=0)

# --- Metrics ---
binary = (canvas>0)
labels = label(binary)
interface_px = np.sum([perimeter(labels==i) for i in range(1, labels.max()+1)])
interface_um = interface_px/pixel_per_um
ratio = interface_um/(image_width_um*image_height_um)

# --- Display ---
st.markdown("---")
st.subheader("Results")
st.write(f"**Particles Placed:** {len(centers)} | **Interfacial Length:** {interface_um:.2f} µm | **Interface/Area:** {ratio:.5f} µm⁻¹")

st.image(np.array(final), caption="Simulated Microstructure", channels="GRAY", use_container_width=True)

buf = BytesIO()
final.save(buf, format="PNG")
st.download_button("Download PNG", data=buf.getvalue(), file_name="microstructure.png", mime="image/png")

# --- Finish progress bar ---
progress_bar.empty()
progress_text.empty()
