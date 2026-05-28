"""Image composition helpers (currently just the scale bar)."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


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
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw_final.textbbox((0, 0), label_txt, font=font)
    w = bbox[2] - bbox[0]
    text_x = (width_px - w) // 2
    draw_final.text((text_x, bar_y + bar_h + 4), label_txt, font=font, fill=0)

    return final
