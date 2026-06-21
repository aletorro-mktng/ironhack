"""Render branded quote cards as social-ready image graphics.

Produces a cinematic, Mortal Vengeance-styled text graphic for a single pull
quote, exported in the aspect ratios used across stories, reels, feed posts, and
link previews so the same quote can be adapted for every platform.
"""

import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from character_assets import resolve_character_name_asset, resolve_character_portrait_asset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = PROJECT_ROOT / "assets" / "fonts"

# Each format maps a user-facing label to (width, height) and the platforms it fits.
QUOTE_GRAPHIC_FORMATS = {
    "Instagram Reel (9:16)": {
        "size": (1080, 1920),
        "platforms": "Instagram Reels",
        "slug": "instagram_reel_9x16",
    },
    "Instagram Story (9:16)": {
        "size": (1080, 1920),
        "platforms": "Instagram Stories",
        "slug": "instagram_story_9x16",
    },
    "Instagram Post (4:5)": {
        "size": (1080, 1350),
        "platforms": "Instagram feed post",
        "slug": "instagram_post_4x5",
    },
    "YouTube Image Cover (16:9)": {
        "size": (1280, 720),
        "platforms": "YouTube thumbnail / image cover",
        "slug": "youtube_cover_16x9",
    },
    "YouTube Post (1:1)": {
        "size": (1080, 1080),
        "platforms": "YouTube community post",
        "slug": "youtube_post_1x1",
    },
    "Blog Cover (16:9)": {
        "size": (1600, 900),
        "platforms": "Blog hero / cover image",
        "slug": "blog_cover_16x9",
    },
    "Blog Square (1:1)": {
        "size": (1080, 1080),
        "platforms": "Blog square image",
        "slug": "blog_square_1x1",
    },
    "Blog Horizontal (1.91:1)": {
        "size": (1200, 628),
        "platforms": "Blog link/social horizontal image",
        "slug": "blog_horizontal_1.91x1",
    },
    "LinkedIn Post Square (1:1)": {
        "size": (1080, 1080),
        "platforms": "LinkedIn post square",
        "slug": "linkedin_post_square_1x1",
    },
    "LinkedIn Post Horizontal (1.91:1)": {
        "size": (1200, 628),
        "platforms": "LinkedIn post horizontal",
        "slug": "linkedin_post_horizontal_1.91x1",
    },
    "LinkedIn Article Square (1:1)": {
        "size": (1080, 1080),
        "platforms": "LinkedIn article square image",
        "slug": "linkedin_article_square_1x1",
    },
    "LinkedIn Article Horizontal (1.91:1)": {
        "size": (1200, 628),
        "platforms": "LinkedIn article horizontal image",
        "slug": "linkedin_article_horizontal_1.91x1",
    },
    # Backward-compatible labels used by earlier generated campaigns.
    "Story / Reel (9:16)": {
        "size": (1080, 1920),
        "platforms": "Instagram Stories and Reels",
        "slug": "story_reel_9x16",
    },
    "Instagram / Facebook Post (4:5)": {
        "size": (1080, 1350),
        "platforms": "Instagram feed, Facebook feed",
        "slug": "post_4x5",
    },
    "Square Post (1:1)": {
        "size": (1080, 1080),
        "platforms": "Instagram, Facebook, LinkedIn feed",
        "slug": "square_1x1",
    },
    "Link Preview (1.91:1)": {
        "size": (1200, 628),
        "platforms": "LinkedIn, Facebook, Twitter/X link cards",
        "slug": "link_preview_1.91x1",
    },
    # Instagram carousel slides (4:5). Registered so base-image compositing and the
    # local quote-card fallback both work per slide (matches _carousel_slide_labels).
    **{
        f"Carousel Slide {i} (4:5)": {
            "size": (1080, 1350),
            "platforms": "Instagram carousel",
            "slug": f"carousel_slide_{i}_4x5",
        }
        for i in range(1, 11)
    },
}


# Selectable visual themes. Each maps a display label to a full palette plus the
# font roles used for the quote and the supporting labels.
QUOTE_GRAPHIC_THEMES = {
    "Modern": {
        "background_top": (18, 18, 20),
        "background_bottom": (18, 18, 20),
        "accent": (236, 236, 238),
        "quote_color": (240, 240, 242),
        "attribution_color": (150, 150, 156),
        "brand_color": (110, 110, 116),
        "quote_mark_color": (42, 42, 46),
        "quote_font": "milonet",
        "label_font": "unthinkers",
    },
    "Press": {
        "background_top": (248, 246, 243),
        "background_bottom": (236, 230, 223),
        "accent": (158, 31, 46),
        "quote_color": (28, 24, 26),
        "attribution_color": (132, 96, 40),
        "brand_color": (124, 114, 110),
        "quote_mark_color": (222, 208, 198),
        "quote_font": "milonet",
        "label_font": "unthinkers",
    },
    "Noir": {
        "background_top": (5, 6, 8),
        "background_bottom": (34, 34, 34),
        "accent": (235, 235, 228),
        "quote_color": (246, 246, 238),
        "attribution_color": (188, 188, 178),
        "brand_color": (112, 112, 108),
        "quote_mark_color": (44, 44, 44),
        "quote_font": "milonet",
        "label_font": "unthinkers",
    },
    "Retro": {
        "background_top": (34, 28, 42),
        "background_bottom": (168, 74, 62),
        "accent": (250, 202, 92),
        "quote_color": (255, 238, 188),
        "attribution_color": (255, 176, 117),
        "brand_color": (230, 184, 132),
        "quote_mark_color": (93, 45, 72),
        "quote_font": "milonet",
        "label_font": "unthinkers",
    },
    "Vintage": {
        "background_top": (236, 224, 200),
        "background_bottom": (212, 194, 162),
        "accent": (124, 58, 40),
        "quote_color": (46, 34, 26),
        "attribution_color": (110, 70, 36),
        "brand_color": (118, 94, 68),
        "quote_mark_color": (202, 184, 150),
        "quote_font": "milonet",
        "label_font": "unthinkers",
    },
    "80s": {
        "background_top": (18, 8, 42),
        "background_bottom": (16, 94, 112),
        "accent": (255, 67, 150),
        "quote_color": (250, 247, 255),
        "attribution_color": (84, 230, 255),
        "brand_color": (255, 190, 80),
        "quote_mark_color": (68, 26, 96),
        "quote_font": "milonet",
        "label_font": "unthinkers",
    },
    "90s Slasher": {
        "background_top": (12, 12, 14),
        "background_bottom": (72, 3, 9),
        "accent": (255, 30, 38),
        "quote_color": (255, 250, 245),
        "attribution_color": (255, 40, 47),
        "brand_color": (190, 190, 186),
        "quote_mark_color": (86, 0, 7),
        "quote_font": "milonet",
        "label_font": "unthinkers",
    },
    "Gothic": {
        "background_top": (16, 14, 20),
        "background_bottom": (38, 12, 18),
        "accent": (158, 31, 46),
        "quote_color": (244, 240, 236),
        "attribution_color": (196, 168, 122),
        "brand_color": (150, 150, 160),
        "quote_mark_color": (74, 30, 38),
        "quote_font": "milonet",
        "label_font": "unthinkers",
    },
    "Asylum": {
        "background_top": (196, 201, 194),
        "background_bottom": (58, 68, 64),
        "accent": (114, 19, 28),
        "quote_color": (18, 24, 22),
        "attribution_color": (93, 22, 30),
        "brand_color": (46, 54, 50),
        "quote_mark_color": (156, 164, 154),
        "quote_font": "milonet",
        "label_font": "unthinkers",
    },
    "Cinematic Dark": {
        "background_top": (16, 14, 20),
        "background_bottom": (38, 12, 18),
        "accent": (158, 31, 46),
        "quote_color": (244, 240, 236),
        "attribution_color": (196, 168, 122),
        "brand_color": (150, 150, 160),
        "quote_mark_color": (74, 30, 38),
        "quote_font": "milonet",
        "label_font": "unthinkers",
    },
    "Blood Crimson": {
        "background_top": (66, 10, 16),
        "background_bottom": (22, 4, 8),
        "accent": (214, 170, 96),
        "quote_color": (246, 236, 232),
        "attribution_color": (226, 198, 152),
        "brand_color": (196, 156, 156),
        "quote_mark_color": (120, 26, 32),
        "quote_font": "milonet",
        "label_font": "unthinkers",
    },
    "Ivory Editorial": {
        "background_top": (248, 246, 243),
        "background_bottom": (236, 230, 223),
        "accent": (158, 31, 46),
        "quote_color": (28, 24, 26),
        "attribution_color": (132, 96, 40),
        "brand_color": (124, 114, 110),
        "quote_mark_color": (222, 208, 198),
        "quote_font": "milonet",
        "label_font": "unthinkers",
    },
    "Parchment Folklore": {
        "background_top": (236, 224, 200),
        "background_bottom": (212, 194, 162),
        "accent": (124, 58, 40),
        "quote_color": (46, 34, 26),
        "attribution_color": (110, 70, 36),
        "brand_color": (118, 94, 68),
        "quote_mark_color": (202, 184, 150),
        "quote_font": "milonet",
        "label_font": "unthinkers",
    },
    "Modern Mono": {
        "background_top": (18, 18, 20),
        "background_bottom": (18, 18, 20),
        "accent": (236, 236, 238),
        "quote_color": (240, 240, 242),
        "attribution_color": (150, 150, 156),
        "brand_color": (110, 110, 116),
        "quote_mark_color": (42, 42, 46),
        "quote_font": "milonet",
        "label_font": "unthinkers",
    },
}

QUOTE_GRAPHIC_DESTINATIONS = {
    "Instagram": ["Instagram Reel (9:16)", "Instagram Post (4:5)", "Instagram Story (9:16)"],
    "YouTube": ["YouTube Image Cover (16:9)", "YouTube Post (1:1)"],
    "Blog": ["Blog Cover (16:9)", "Blog Square (1:1)", "Blog Horizontal (1.91:1)"],
    "LinkedIn": [
        "LinkedIn Post Square (1:1)",
        "LinkedIn Post Horizontal (1.91:1)",
        "LinkedIn Article Square (1:1)",
        "LinkedIn Article Horizontal (1.91:1)",
    ],
}

DEFAULT_THEME = "Gothic"


def quote_graphic_format_options():
    """Return the preferred destination-specific format labels."""
    labels = []
    for destination_labels in QUOTE_GRAPHIC_DESTINATIONS.values():
        labels.extend(destination_labels)
    return labels


_FONT_CANDIDATES = {
    "milonet": [
        FONT_DIR / "Milonet.otf",
    ],
    "unthinkers": [
        FONT_DIR / "Unthinkers.otf",
    ],
    "serif": [
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/Library/Fonts/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ],
    "serif_italic": [
        "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
    ],
    "serif_bold": [
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    ],
    "sans": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ],
    "sans_bold": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ],
}

def _load_font(kind, size):
    """Load the first available TrueType font for a role, with a safe fallback."""
    for path in _FONT_CANDIDATES.get(kind, []):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _transparent_black_background(image):
    """Make black name-graphic backgrounds transparent while preserving red/white lettering."""
    image = image.convert("RGBA")
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            red, green, blue, alpha = pixels[x, y]
            if red < 22 and green < 22 and blue < 22:
                pixels[x, y] = (red, green, blue, 0)
            elif alpha:
                pixels[x, y] = (red, green, blue, alpha)
    bbox = image.getbbox()
    return image.crop(bbox) if bbox else image


def _paste_character_name(image, character_asset_path, margin, content_width, y, max_height):
    """Composite the selected character's uploaded name mark onto the quote card."""
    if not character_asset_path:
        return 0
    try:
        mark = Image.open(character_asset_path)
    except OSError:
        return 0

    mark = _transparent_black_background(mark)
    if not mark.width or not mark.height:
        return 0

    max_width = int(content_width * 0.70)
    scale = min(max_width / mark.width, max_height / mark.height, 1.0)
    mark = mark.resize((max(1, int(mark.width * scale)), max(1, int(mark.height * scale))), Image.Resampling.LANCZOS)
    x = int(margin + (content_width - mark.width) / 2)
    image.alpha_composite(mark, (x, y))
    return mark.height


def _cover_resize(image, size):
    """Resize an image to cover a target size, cropping the overflow."""
    target_width, target_height = size
    scale = max(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - target_width) // 2)
    top = max(0, (resized.height - target_height) // 2)
    return resized.crop((left, top, left + target_width, top + target_height))


def _paste_character_portrait(image, portrait_asset_path, theme):
    """Blend a selected character portrait into the card background."""
    if not portrait_asset_path:
        return
    try:
        portrait = Image.open(portrait_asset_path).convert("RGBA")
    except OSError:
        return

    width, height = image.size
    portrait_width = int(width * (0.46 if width <= height else 0.38))
    portrait = _cover_resize(portrait, (portrait_width, height))

    alpha = Image.new("L", (portrait_width, height), 0)
    alpha_pixels = alpha.load()
    for x in range(portrait_width):
        x_ratio = x / max(portrait_width - 1, 1)
        for y in range(height):
            y_ratio = y / max(height - 1, 1)
            edge_fade = min(1.0, x_ratio * 2.8)
            vertical_fade = min(1.0, 1.25 - abs(y_ratio - 0.50) * 0.85)
            alpha_pixels[x, y] = int(150 * edge_fade * vertical_fade)

    portrait.putalpha(alpha)
    x = width - portrait_width
    image.alpha_composite(portrait, (x, 0))

    veil = Image.new("RGBA", image.size, (*theme["background_top"], 0))
    veil_draw = ImageDraw.Draw(veil)
    veil_draw.rectangle((0, 0, width, height), fill=(*theme["background_top"], 34))
    image.alpha_composite(veil)


def _vertical_gradient(size, top_color, bottom_color):
    """Build a soft top-to-bottom gradient background."""
    width, height = size
    base = Image.new("RGB", (1, height))
    for y in range(height):
        ratio = y / max(height - 1, 1)
        base.putpixel(
            (0, y),
            tuple(
                int(top_color[c] + (bottom_color[c] - top_color[c]) * ratio)
                for c in range(3)
            ),
        )
    return base.resize(size)


def _wrap_text(draw, text, font, max_width):
    """Greedy word-wrap a string to a pixel width."""
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if _text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _text_width(draw, text, font):
    """Measure text with glyph overhangs included."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _centered_text_x(draw, text, font, left, width):
    """Return an x coordinate that visually centers a font's full bounding box."""
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    return left + (width - text_width) / 2 - bbox[0]


def _fit_quote_font(draw, text, max_width, max_height, start_size, font_kind, min_size=28):
    """Shrink the quote font until the wrapped block fits the available box."""
    size = start_size
    while size >= min_size:
        font = _load_font(font_kind, size)
        lines = _wrap_text(draw, text, font, max_width)
        line_height = int(size * 1.32)
        if len(lines) * line_height <= max_height:
            return font, lines, line_height
        size -= 4
    font = _load_font(font_kind, min_size)
    return font, _wrap_text(draw, text, font, max_width), int(min_size * 1.32)


def _render_card(quote, attribution, brand_title, size, theme, character_name_asset=None, character_portrait_asset=None, background_image_path=None, show_brand=False):
    """Render a single quote card image for one format size and theme.

    When ``background_image_path`` is given, that image (cover-cropped + darkened for
    legibility) is used as the background instead of the theme gradient.
    """
    width, height = size
    base_used = False
    if background_image_path and Path(str(background_image_path)).exists():
        try:
            base = Image.open(str(background_image_path)).convert("RGBA")
            image = _cover_resize(base, size).convert("RGBA")
            scrim = Image.new("RGBA", size, (0, 0, 0, 0))
            scrim_draw = ImageDraw.Draw(scrim)
            for line_y in range(height):
                alpha = int(70 + 150 * (line_y / max(1, height - 1)))
                scrim_draw.line([(0, line_y), (width, line_y)], fill=(8, 5, 8, alpha))
            image = Image.alpha_composite(image, scrim)
            base_used = True
        except Exception:
            image = _vertical_gradient(size, theme["background_top"], theme["background_bottom"]).convert("RGBA")
    else:
        image = _vertical_gradient(size, theme["background_top"], theme["background_bottom"]).convert("RGBA")
    if not base_used:
        _paste_character_portrait(image, character_portrait_asset, theme)
    draw = ImageDraw.Draw(image)

    if base_used:
        # Light text reads on the darkened photo regardless of the theme's palette.
        theme = {
            **theme,
            "quote_color": (255, 255, 255),
            "attribution_color": (231, 201, 207),
            "brand_color": (236, 224, 224),
            "quote_mark_color": (255, 255, 255),
        }

    margin = int(width * 0.085)
    content_width = width - 2 * margin
    is_wide = width > height
    label_font_kind = theme["label_font"]

    # Accent rule near the top.
    accent_y = int(height * 0.12)
    draw.line(
        [(margin, accent_y), (margin + int(content_width * 0.18), accent_y)],
        fill=theme["accent"],
        width=max(4, int(height * 0.006)),
    )

    # Oversized opening quotation mark.
    quote_mark_font = _load_font("serif_bold", int(height * (0.16 if is_wide else 0.18)))
    draw.text((margin - int(width * 0.01), accent_y + int(height * 0.01)),
              "“", font=quote_mark_font, fill=theme["quote_mark_color"])

    name_mark_height = _paste_character_name(
        image=image,
        character_asset_path=character_name_asset,
        margin=margin,
        content_width=content_width,
        y=int(height * (0.155 if not is_wide else 0.135)),
        max_height=int(height * (0.12 if not is_wide else 0.16)),
    )

    # Fit and place the quote text in the central band.
    quote_top = int(height * (0.34 if name_mark_height and not is_wide else 0.30 if not is_wide else 0.30 if name_mark_height else 0.26))
    quote_bottom = int(height * 0.80)
    start_size = int(height * (0.072 if not is_wide else 0.10))
    quote_text_width = int(content_width * 0.88)
    quote_font, lines, line_height = _fit_quote_font(
        draw, quote.strip().strip('"“”'),
        quote_text_width, quote_bottom - quote_top, start_size, theme["quote_font"],
    )

    block_height = len(lines) * line_height
    y = quote_top + max(0, ((quote_bottom - quote_top) - block_height) // 2)
    for line in lines:
        x = _centered_text_x(draw, line, quote_font, margin, content_width)
        draw.text((x, y), line, font=quote_font, fill=theme["quote_color"])
        y += line_height

    # Attribution — skipped when it merely repeats the book/brand title, otherwise the
    # book name would appear twice (here AND in the bottom brand mark below).
    brand_book = (brand_title or "MORTAL VENGEANCE").strip().upper()
    attribution_clean = (attribution or "").strip().lstrip("—").strip()
    if attribution_clean and attribution_clean.upper() != brand_book:
        attribution_font = _load_font(label_font_kind, int(height * (0.026 if not is_wide else 0.04)))
        attribution_text = f"— {attribution_clean}"
        draw.text(
            (_centered_text_x(draw, attribution_text, attribution_font, margin, content_width), y + int(height * 0.02)),
            attribution_text, font=attribution_font, fill=theme["attribution_color"],
        )

    # Brand mark, bottom-centered: "TELL TALES INK · <BOOK>". Off by default — generated images
    # carry no app/book watermark at the bottom unless a caller opts in via show_brand.
    if show_brand:
        brand_font = _load_font(label_font_kind, int(height * (0.020 if not is_wide else 0.032)))
        book_text = (brand_title or "MORTAL VENGEANCE").upper()
        brand_y = height - int(height * 0.075)
        if "TELL TALES INK" in book_text:
            draw.text(
                (_centered_text_x(draw, book_text, brand_font, margin, content_width), brand_y),
                book_text, font=brand_font, fill=theme["brand_color"],
            )
        else:
            left_text = "TELL TALES INK"
            gap = int(width * 0.022)
            dot_r = max(2, int(height * 0.0045))
            left_w = _text_width(draw, left_text, brand_font)
            start_x = margin + (content_width - (left_w + gap + 2 * dot_r + gap + _text_width(draw, book_text, brand_font))) // 2
            draw.text((start_x, brand_y), left_text, font=brand_font, fill=theme["brand_color"])
            line_bbox = draw.textbbox((start_x, brand_y), left_text, font=brand_font)
            dot_cx = start_x + left_w + gap + dot_r
            dot_cy = (line_bbox[1] + line_bbox[3]) // 2
            draw.ellipse(
                [(dot_cx - dot_r, dot_cy - dot_r), (dot_cx + dot_r, dot_cy + dot_r)],
                fill=theme["brand_color"],
            )
            draw.text((dot_cx + dot_r + gap, brand_y), book_text, font=brand_font, fill=theme["brand_color"])

    return image.convert("RGB")


def render_quote_cards(
    quote,
    attribution,
    format_labels,
    brand_title,
    output_dir,
    file_stem,
    theme_name=DEFAULT_THEME,
    character_name=None,
    background_image_path=None,
    slide_texts=None,
    show_brand=False,
    book=None,
):
    """
    Render the quote in every requested format and bundle the PNGs into a zip.

    ``slide_texts`` (optional) maps a format label -> distinct quote, so a multi-slide
    carousel renders different text per slide while every other format in the same set
    falls back to ``quote`` (label-keyed, so non-carousel sizes are never affected).

    Returns {"paths": [...], "zip_path": Path, "manifest": [(label, path), ...]}.
    """
    if not quote or not quote.strip():
        raise ValueError("A quote is required to render quote graphics.")

    selected = [label for label in (format_labels or []) if label in QUOTE_GRAPHIC_FORMATS]
    if not selected:
        raise ValueError("Select at least one graphic format.")

    theme = QUOTE_GRAPHIC_THEMES.get(theme_name, QUOTE_GRAPHIC_THEMES[DEFAULT_THEME])
    character_name_asset = resolve_character_name_asset(character_name)
    character_portrait_asset = resolve_character_portrait_asset(character_name, book=book)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    manifest = []
    for label in selected:
        spec = QUOTE_GRAPHIC_FORMATS[label]
        card_quote = quote
        if isinstance(slide_texts, dict) and slide_texts.get(label):
            card_quote = slide_texts[label]
        image = _render_card(
            card_quote,
            attribution,
            brand_title,
            spec["size"],
            theme,
            character_name_asset,
            character_portrait_asset,
            background_image_path=background_image_path,
            show_brand=show_brand,
        )
        image_path = output_dir / f"{file_stem}_{spec['slug']}.png"
        image.save(image_path, format="PNG")
        paths.append(image_path)
        manifest.append((label, image_path))

    zip_path = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image_path in paths:
            archive.write(image_path, arcname=image_path.name)

    return {"paths": paths, "zip_path": zip_path, "manifest": manifest}


if __name__ == "__main__":
    demo = render_quote_cards(
        quote="Stories are the most powerful weapon of all.",
        attribution="Mortal Vengeance",
        format_labels=list(QUOTE_GRAPHIC_FORMATS.keys()),
        brand_title="Mortal Vengeance",
        output_dir=Path("outputs") / "quote_graphics_demo",
        file_stem="demo_quote",
    )
    print("Rendered:")
    for label, path in demo["manifest"]:
        print(f"- {label}: {path}")
    print(f"Zip: {demo['zip_path']}")
