"""Image composition helpers (currently just the scale bar)."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

# Candidate TrueType paths to try in order. Streamlit Cloud (Debian) ships
# DejaVu by name; Linux distros usually have the full path; macOS keeps it
# under /System; the bare name works wherever Pillow has it on its font
# search path.
_FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "Arial.ttf",
)


def _load_label_font(size: int = 18) -> ImageFont.ImageFont:
    """Try real TrueType fonts first, then a size-aware default.

    The original code fell straight through to ``ImageFont.load_default()``,
    which on older Pillow returns a tiny bitmap font that's nearly unreadable
    next to a microstructure image. Pillow ≥ 10 supports a size argument on
    the default, which is good enough as a last resort.
    """
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow ≥ 10
    except TypeError:
        return ImageFont.load_default()


def add_scale_bar(
    pil_img: Image.Image,
    image_width_um: float,
    image_height_um: float,
    pixel_per_um: float,
) -> Image.Image:
    """Return a new image with a horizontal scale bar appended below ``pil_img``."""
    width_px, height_px = pil_img.width, pil_img.height
    scale_um = image_width_um / 5
    scale_px = int(scale_um * pixel_per_um)
    bar_h = max(4, int(0.01 * height_px))
    box_h = bar_h + 30

    final = Image.new("L", (width_px, height_px + box_h), color=255)
    final.paste(pil_img, (0, 0))

    bar_y = height_px + 8
    draw_final = ImageDraw.Draw(final)
    bar_x1 = (width_px - scale_px) // 2
    draw_final.rectangle(
        [bar_x1, bar_y, bar_x1 + scale_px, bar_y + bar_h], fill=0
    )

    label_txt = f"{int(scale_um)} μm"
    font = _load_label_font(size=18)

    bbox = draw_final.textbbox((0, 0), label_txt, font=font)
    w = bbox[2] - bbox[0]
    text_x = (width_px - w) // 2
    draw_final.text((text_x, bar_y + bar_h + 4), label_txt, font=font, fill=0)

    return final
