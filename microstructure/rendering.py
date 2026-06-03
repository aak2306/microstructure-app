"""Image composition helpers (currently just the scale bar)."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

# Candidate TrueType paths to try in order.
# packages.txt installs fonts-dejavu-core on Streamlit Cloud so the
# absolute Debian path is reliable there; the bare name works wherever
# Pillow has the font on its search path; macOS paths are tried last.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian / Streamlit Cloud
    "DejaVuSans.ttf",                                    # Pillow font search path
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",      # macOS
    "/Library/Fonts/Arial.ttf",
    "Arial.ttf",
)


def _load_label_font(size: int = 18) -> tuple[ImageFont.ImageFont, bool]:
    """Try real TrueType fonts first, then a size-aware default.

    Returns ``(font, can_use_micro)`` where the flag is True only when
    the loaded font can actually render the µ glyph (U+00B5). Older
    Pillow bitmap fonts and stripped-down TrueType fonts silently
    substitute a box; we detect this and fall back to ASCII "um".
    """
    for path in _FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, size)
            if _glyph_renders(font, "µ"):
                return font, True
        except OSError:
            continue
    try:
        font = ImageFont.load_default(size=size)  # Pillow ≥ 10
        return font, _glyph_renders(font, "µ")
    except TypeError:
        return ImageFont.load_default(), False


def _glyph_renders(font: ImageFont.ImageFont, char: str) -> bool:
    """Return True if *char* draws as a visible, non-.notdef glyph.

    Renders *char* into a small scratch image and checks that at least
    one pixel was darkened. A missing glyph either produces zero pixels
    (no outline) or a hollow .notdef box whose interior stays white;
    either way the minimum pixel value stays near 255.
    """
    try:
        canvas_side = 64
        tmp = Image.new("L", (canvas_side, canvas_side), 255)
        ImageDraw.Draw(tmp).text((4, 4), char, font=font, fill=0)
        return min(tmp.getdata()) < 200  # at least one dark pixel drawn
    except Exception:
        return False


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

    # Scale-bar strip: always white background, black bar + text so the
    # annotation stays legible regardless of the image's colour scheme.
    mode = pil_img.mode  # "L" (grayscale) or "RGB" (colour)
    strip_bg: int | tuple[int, int, int] = 255 if mode == "L" else (255, 255, 255)
    ink:      int | tuple[int, int, int] = 0   if mode == "L" else (0, 0, 0)

    final = Image.new(mode, (width_px, height_px + box_h), color=strip_bg)
    final.paste(pil_img, (0, 0))

    draw_final = ImageDraw.Draw(final)
    bar_y = height_px + padding
    bar_x1 = (width_px - scale_px) // 2
    draw_final.rectangle(
        [bar_x1, bar_y, bar_x1 + scale_px, bar_y + bar_h], fill=ink
    )

    font, can_use_micro = _load_label_font(size=font_size_px)
    unit = "µm" if can_use_micro else "um"
    label_txt = f"{int(scale_um)} {unit}"

    bbox = draw_final.textbbox((0, 0), label_txt, font=font)
    text_w = bbox[2] - bbox[0]
    text_x = (width_px - text_w) // 2
    draw_final.text(
        (text_x, bar_y + bar_h + padding), label_txt, font=font, fill=ink
    )

    return final
