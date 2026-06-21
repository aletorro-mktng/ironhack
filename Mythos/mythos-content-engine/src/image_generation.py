"""External image-generation routing for Tell Tales Ink visual assets."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
IDEOGRAM_MODEL = os.getenv("IDEOGRAM_MODEL", "V_3")
IDEOGRAM_ENDPOINT = os.getenv("IDEOGRAM_ENDPOINT", "https://api.ideogram.ai/v1/ideogram-v3/generate")


def external_images_enabled() -> bool:
    return os.getenv("USE_EXTERNAL_IMAGE_APIS", "true").strip().lower() not in {"0", "false", "no", "off"}


def preferred_provider(use_case: str, content_kind: str = "") -> str:
    """Choose the external image provider by real-world asset type.

    A global ``IMAGE_PROVIDER`` env var (``openai`` or ``ideogram``) overrides the
    heuristic when set, so a deployment without an Ideogram key can force OpenAI.
    """
    override = os.getenv("IMAGE_PROVIDER", "").strip().lower()
    if override in {"openai", "ideogram"}:
        return override
    key = f"{use_case} {content_kind}".lower()
    # Instagram / social posts now use OpenAI (gpt-image-1) so quote graphics are
    # generated as fresh AI images rather than routed to Ideogram.
    if "instagram post" in key or "social post" in key:
        return "openai"
    if any(term in key for term in ["quote", "typography", "font", "type", "linkedin banner"]):
        return "ideogram"
    if "blog hero" in key:
        return "openai"
    if "book cover" in key:
        return "ideogram"
    if any(term in key for term in ["character", "portrait", "illustration"]):
        return "openai"
    if "advertisement" in key or "youtube cover" in key or "youtube image" in key:
        return "ideogram"
    if "linkedin image" in key:
        return "ideogram"
    return "openai"


def aspect_ratio_for_format(label: str) -> str:
    text = str(label or "").lower()
    if "9:16" in text or "story" in text or "reel" in text:
        return "9:16"
    if "4:5" in text:
        return "4:5"
    if "16:9" in text or "cover" in text or "thumbnail" in text:
        return "16:9"
    if "1.91" in text or "horizontal" in text:
        return "16:9"
    return "1:1"


def openai_size_for_aspect(aspect_ratio: str) -> str:
    if aspect_ratio in {"9:16", "4:5"}:
        return "1024x1536"
    if aspect_ratio == "16:9":
        return "1536x1024"
    return "1024x1024"


def ideogram_aspect_for_aspect(aspect_ratio: str) -> str:
    # Ideogram v3 accepts human-readable ratios such as 1:1, 16:9, 9:16, and 4:5.
    return aspect_ratio or "1:1"


def slugify_filename(value: str, fallback: str = "generated_image") -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").lower()).strip("_")
    return slug[:90] or fallback


def save_base64_png(encoded: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(encoded))
    return output_path


def download_image(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "TellTalesInk/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content_type = response.headers.get("content-type", "")
        extension = mimetypes.guess_extension(content_type.split(";")[0].strip()) or output_path.suffix or ".png"
        final_path = output_path.with_suffix(extension)
        final_path.write_bytes(response.read())
        return final_path


def generate_openai_image(prompt: str, output_path: Path, aspect_ratio: str = "1:1") -> Path:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY was not found. Add it to your .env file.")
    client = OpenAI()
    response = client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        size=openai_size_for_aspect(aspect_ratio),
        quality=os.getenv("OPENAI_IMAGE_QUALITY", "high"),
        n=1,
    )
    image_data = response.data[0]
    encoded = getattr(image_data, "b64_json", None)
    if encoded:
        return save_base64_png(encoded, output_path.with_suffix(".png"))
    url = getattr(image_data, "url", None)
    if url:
        return download_image(url, output_path.with_suffix(".png"))
    raise RuntimeError("OpenAI image response did not include b64_json or url.")


def _find_ideogram_image(payload) -> Optional[str]:
    if isinstance(payload, dict):
        for key in ("url", "image_url", "image", "b64_json", "base64"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        for value in payload.values():
            found = _find_ideogram_image(value)
            if found:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _find_ideogram_image(item)
            if found:
                return found
    return None


def generate_ideogram_image(prompt: str, output_path: Path, aspect_ratio: str = "1:1") -> Path:
    api_key = os.getenv("IDEOGRAM_API_KEY")
    if not api_key:
        raise EnvironmentError("IDEOGRAM_API_KEY was not found. Add it to your .env file.")

    form = {
        "prompt": prompt,
        "aspect_ratio": ideogram_aspect_for_aspect(aspect_ratio),
        "rendering_speed": os.getenv("IDEOGRAM_RENDERING_SPEED", "DEFAULT"),
        "model": IDEOGRAM_MODEL,
    }
    encoded_form = urllib.parse.urlencode(form).encode("utf-8")
    request = urllib.request.Request(
        IDEOGRAM_ENDPOINT,
        data=encoded_form,
        headers={
            "Api-Key": api_key,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ideogram image generation failed: {exc.code} {detail}") from exc

    image_value = _find_ideogram_image(payload)
    if not image_value:
        raise RuntimeError("Ideogram response did not include an image URL or base64 payload.")
    if image_value.startswith("http"):
        return download_image(image_value, output_path.with_suffix(".png"))
    if image_value.startswith("data:image"):
        image_value = image_value.split(",", 1)[-1]
    return save_base64_png(image_value, output_path.with_suffix(".png"))


def build_visual_prompt(
    *,
    use_case: str,
    content: str,
    topic: str = "",
    style: str = "",
    attribution: str = "",
    format_label: str = "",
    provider: str = "",
    render_text: bool = False,
) -> str:
    if render_text:
        attribution_clause = (
            f"Clearly include the attribution/speaker name '{attribution}' beneath or beside the quote."
            if attribution
            else "Include a tasteful attribution line if a speaker is named in the text."
        )
        typography_note = (
            "Render the exact short quote text as legible, well-kerned custom typography integrated into the design. "
            + attribution_clause
        )
    else:
        typography_note = (
            "Use strong, legible custom typography and preserve exact short quote text."
            if provider == "ideogram"
            else "Create a polished cinematic illustration; avoid adding misspelled text unless explicitly requested."
        )
    return "\n".join(
        part
        for part in [
            f"Create a brand-new Tell Tales Ink visual asset for: {use_case}.",
            f"Destination format: {format_label}.",
            f"Visual style: {style or 'Gothic, literary, cinematic, premium'}." ,
            f"Topic/context: {topic}".strip(),
            f"Primary text or creative direction: {content}".strip(),
            f"Attribution/character/source: {attribution}".strip(),
            typography_note,
            "Brand direction: dark literary suspense, burgundy, ivory, cinematic contrast, sophisticated horror, no generic stock look.",
            "Do not include watermarks, UI chrome, platform logos, or unrelated text.",
        ]
        if part
    )


def generate_external_visual(
    *,
    use_case: str,
    content: str,
    topic: str = "",
    style: str = "",
    attribution: str = "",
    format_label: str = "",
    output_dir: Path,
    file_stem: str,
    content_kind: str = "",
    render_text: Optional[bool] = None,
) -> dict:
    if not external_images_enabled():
        return {"error": "External image APIs are disabled."}
    provider = preferred_provider(use_case, content_kind)
    aspect_ratio = aspect_ratio_for_format(format_label)
    key = f"{use_case} {content_kind}".lower()
    if render_text is None:
        render_text = any(term in key for term in ["quote", "typography", "social post", "instagram"]) and not any(
            term in key for term in ["portrait", "abstract", "mood"]
        )
    prompt = build_visual_prompt(
        use_case=use_case,
        content=content,
        topic=topic,
        style=style,
        attribution=attribution,
        format_label=format_label,
        provider=provider,
        render_text=render_text,
    )
    output_path = output_dir / f"{slugify_filename(file_stem)}_{provider}_{slugify_filename(format_label, 'image')}.png"
    try:
        if provider == "ideogram":
            path = generate_ideogram_image(prompt, output_path, aspect_ratio)
        else:
            path = generate_openai_image(prompt, output_path, aspect_ratio)
        return {"path": str(path), "provider": provider, "prompt": prompt}
    except Exception as exc:
        return {"error": str(exc), "provider": provider, "prompt": prompt}


def new_visual_output_dir(prefix: str = "external_visuals") -> Path:
    return PROJECT_ROOT / "outputs" / prefix / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
