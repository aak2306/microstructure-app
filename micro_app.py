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
    particle_diameter_um = st.number_input("Avg. Particle Diameter (µm)", 1.0, 100.0, 10.0)
    pixel_per_um = st.slider("Pixels per Micron", 1, 10, 3)
    rng_seed = st.number_input("Random Seed (0=random)", 0, 9999, 0)
with col2:
    image_height_um = st.number_input("Image Height (µm)", 50.0, 1000.0, 200.0)
    volume_fraction = st.slider("Volume Fraction (%)", 1, 99, 20)
    shape = st.selectbox("Particle Shape", [
        "Circular", "Elliptical", "Irregular (Blob)", "Mixed (Circular + Elliptical + Irregular)"])

mix_ratio = (st.slider("% Circular in Mix", 0, 100, 33)
             if shape == "Mixed (Circular + Elliptical + Irregular)" else None)

# --- RNG ---
if rng_seed:
    np.random.seed(int(rng_seed))

# --- Load irregular masks (optional) ---
project_dir = os.path.dirname(__file__)
mask_dir_local = os.path.join(project_dir, "blob_masks")
mask_dir_cloud = "/mount/data/blob_masks"
mask_dir = mask_dir_local if os.path.isdir(mask_dir_local) else mask_dir_cloud
blob_masks = []
if os.path.isdir(mask_dir):
    blob_masks = [np.array(PILImage.open(os.path.join(mask_dir, f)))
                  for f in os.listdir(mask_dir) if f.endswith(".png")]

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

# --- Helper: paste irregular blob mask ---
def paste_blob(center_x, center_y, r_px):
    if not blob_masks:
        return False  # no masks available
    blob = blob_masks[np.random.randint(len(blob_masks))]
    scale = np.random.uniform(0.5, 1.5)
    new_size = (max(1, int(blob.shape[1]*scale)), max(1, int(blob.shape[0]*scale)))
    blob_resized = np.array(PILImage.fromarray(blob).resize(new_size, PILImage.BILINEAR))
    blob_rot = np.array(PILImage.fromarray(blob_resized).rotate(np.random.rand()*360, expand=True, fillcolor=0))
    bh, bw = blob_rot.shape
    top = center_y - bh//2
    left = center_x - bw//2
    if top < 0 or left < 0 or top+bh > height_px or left+bw > width_px:
        return False  # out of bounds
    tmp = np.array(pil_img)
    region = tmp[top:top+bh, left:left+bw]
    tmp[top:top+bh, left:left+bw] = np.maximum(region, blob_rot)
    pil_img.paste(Image.fromarray(tmp))
    return True

# --- Particle placement ---
centers = []
max_attempts = num_particles * 20
attempts = 0
while len(centers) < num_particles and attempts < max_attempts:
    attempts += 1
    cx = np.random.randint(avg_rad_px, width_px - avg_rad_px)
    cy = np.random.randint(avg_rad_px, height_px - avg_rad_px)
    if any((cx-x)**2 + (cy-y)**2 < (2*avg_rad_px)**2 for x,y in centers):
        continue

    r_px = int(avg_rad_px * (1 + np.random.uniform(-0.3, 0.3)))

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
text_x = (width_px-w)//2
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
