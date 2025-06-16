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

allow_overlap = st.checkbox("Allow particle overlap", value=False)

mix_ratio = (st.slider("% Circular in Mix", 0, 100, 33)
             if shape == "Mixed (Circular + Elliptical + Irregular)" else None)

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
    def ccw_sort(p):
        d = p - np.mean(p, axis=0)
        s = np.arctan2(d[:, 0], d[:, 1])
        return p[np.argsort(s), :]

    def get_curve(points, **kw):
        segments = []
        for i in range(len(points) - 1):
            seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
            segments.append(seg)
        curve = np.concatenate([s.curve for s in segments])
        return segments, curve

    def get_bezier_curve(a, rad=0.2, edgy=0.15):
        p = np.arctan(edgy) / np.pi + .5
        a = ccw_sort(a)
        a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
        d = np.diff(a, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
        ang = f(ang)
        ang1 = ang
        ang2 = np.roll(ang, 1)
        ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
        ang = np.append(ang, [ang[0]])
        a = np.append(a, np.atleast_2d(ang).T, axis=1)
        s, c = get_curve(a, r=rad)
        return c.T

    class Segment:
        def __init__(self, p1, p2, angle1, angle2, **kw):
            self.p1 = p1
            self.p2 = p2
            self.angle1 = angle1
            self.angle2 = angle2
            self.numpoints = kw.get("numpoints", 100)
            r = kw.get("r", 0.3)
            d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
            self.r = r * d
            self.p = np.zeros((4, 2))
            self.p[0, :] = self.p1[:]
            self.p[3, :] = self.p2[:]
            self.calc_intermediate_points(self.r)

        def calc_intermediate_points(self, r):
            self.p[1, :] = self.p1 + np.array([
                self.r * np.cos(self.angle1),
                self.r * np.sin(self.angle1)])
            self.p[2, :] = self.p2 + np.array([
                self.r * np.cos(self.angle2 + np.pi),
                self.r * np.sin(self.angle2 + np.pi)])
            self.curve = bezier(self.p, self.numpoints)

    def bezier(points, num=200):
        from scipy.special import binom
        N = len(points)
        t = np.linspace(0, 1, num=num)
        curve = np.zeros((num, 2))
        for i in range(N):
            curve += np.outer(binom(N - 1, i) * t ** i * (1 - t) ** (N - 1 - i), points[i])
        return curve

    def get_random_points(n=6, scale=0.8):
        a = np.random.rand(n, 2) - 0.5
        a /= np.max(np.abs(a))
        return a * scale

    xy = get_bezier_curve(get_random_points(), rad=0.35)
    x, y = xy
    x = x - np.min(x)
    y = y - np.min(y)
    x = x / np.max(x) * 2 * r_px
    y = y / np.max(y) * 2 * r_px
    mask_w, mask_h = int(np.max(x)) + 2, int(np.max(y)) + 2
    blob = Image.new("L", (mask_w, mask_h), 0)
    draw = ImageDraw.Draw(blob)
    coords = list(zip(x, y))
    draw.polygon(coords, fill=255)
    blob_img = blob.rotate(np.random.rand() * 360, expand=True, fillcolor=0)
    blob_arr = np.array(blob_img)
    bh, bw = blob_arr.shape
    top = center_y - bh // 2
    left = center_x - bw // 2
    if top < 0 or left < 0 or top + bh > height_px or left + bw > width_px:
        return False
    temp_canvas = np.array(pil_img)
    crop = temp_canvas[top:top + bh, left:left + bw]
    temp_canvas[top:top + bh, left:left + bw] = np.maximum(crop, blob_arr)
    pil_img.paste(Image.fromarray(temp_canvas))
    return True

# --- Particle placement ---
centers = []
max_attempts = int(num_particles * (100 if volume_fraction > 80 else 20))
attempts = 0
while len(centers) < num_particles and attempts < max_attempts:
    attempts += 1
    cx = np.random.randint(avg_rad_px, width_px - avg_rad_px)
    cy = np.random.randint(avg_rad_px, height_px - avg_rad_px)
    if (not allow_overlap) and any((cx-x)**2 + (cy-y)**2 < (1.8*avg_rad_px)**2 for x,y in centers):
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
