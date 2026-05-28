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


def _load_label_font(size: int = 18) -> tuple[ImageFont.ImageFont, bool]:
    """Try real TrueType fonts first, then a size-aware default.

    Returns ``(font, is_truetype)``. The flag tells the caller whether
    non-ASCII glyphs (specifically µ) are safe to use — older Pillow's
    legacy bitmap default font is ASCII-only and would render µ as a
    missing-glyph box.
    """
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size), True
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size), True  # Pillow ≥ 10
    except TypeError:
        return ImageFont.load_default(), False


def add_scale_bar(
    pil_img: Image.Image,
    image_width_um: float,
    image_height_um: float,
    pixel_per_um: float,
) -> Image.Image:
    """Return a new image with a horizontal scale bar appended below ``pil_img``.

    The bar, the label font, and the surrounding strip all scale with the
    image's shorter side so that everything stays readable after Streamlit
    rescales the rendered image to fit a column. With a fixed 18 px font
    on a 2000 px-tall canvas the label was ~6 px once displayed; here the
    font is at least 1/40 of the shorter image side.
    """
    width_px, height_px = pil_img.width, pil_img.height
    short_side = min(width_px, height_px)

    # Sizes proportional to the image so the printed result stays legible
    # at any pixel_per_um. Floors keep small canvases from disappearing.
    font_size_px = max(20, short_side // 40)
    bar_h = max(6, short_side // 90)
    padding = max(6, font_size_px // 3)
    box_h = bar_h + font_size_px + padding * 3

    scale_um = image_width_um / 5
    scale_px = int(scale_um * pixel_per_um)

    final = Image.new("L", (width_px, height_px + box_h), color=255)
    final.paste(pil_img, (0, 0))

    draw_final = ImageDraw.Draw(final)
    bar_y = height_px + padding
    bar_x1 = (width_px - scale_px) // 2
    draw_final.rectangle(
        [bar_x1, bar_y, bar_x1 + scale_px, bar_y + bar_h], fill=0
    )

    font, is_truetype = _load_label_font(size=font_size_px)
    # Use U+00B5 MICRO SIGN (Latin-1) rather than U+03BC GREEK SMALL LETTER
    # MU. Both look identical when both are present, but only U+00B5 is
    # guaranteed in every Latin-coverage font; some stripped-down system
    # fonts omit the Greek block, which is what the live deploy was hitting.
    unit = "µm" if is_truetype else "um"
    label_txt = f"{int(scale_um)} {unit}"

    bbox = draw_final.textbbox((0, 0), label_txt, font=font)
    text_w = bbox[2] - bbox[0]
    text_x = (width_px - text_w) // 2
    draw_final.text(
        (text_x, bar_y + bar_h + padding), label_txt, font=font, fill=0
    )

    return final
