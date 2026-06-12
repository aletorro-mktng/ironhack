"""NiceGUI user interface for Mythos Content Engine."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
import sys
import zipfile
from datetime import datetime
from html import escape
from pathlib import Path

from nicegui import ui
from pydub import AudioSegment

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.chdir(PROJECT_ROOT)

from content_pipeline import create_generation_prompt, run_pipeline, save_output
from context_filter import select_relevant_context
from draft_store import (
    get_saved_draft,
    list_saved_drafts,
    read_saved_draft_content,
    render_saved_drafts_markdown,
    save_draft,
    update_saved_draft,
)
from elevenlabs_integration import list_voices, strip_production_tags, synthesize_speech
from llm_integration import DEFAULT_MODEL, generate_text, generate_text_with_image
from image_generation import generate_external_visual
from prompt_templates import list_supported_content_types
from character_assets import character_asset_note, resolve_character_portrait_asset
from quote_graphics import (
    QUOTE_GRAPHIC_FORMATS,
    QUOTE_GRAPHIC_THEMES,
    quote_graphic_format_options,
    render_quote_cards,
)
from selection_options import (
    AUDIENCE_OPTIONS,
    BLOG_LENGTH_OPTIONS,
    CHARACTER_TAGS,
    CONSTRAINT_OPTIONS,
    ELEVENLABS_MODEL_OPTIONS,
    INSTAGRAM_FORMAT_OPTIONS,
    INSTAGRAM_HASHTAG_OPTIONS,
    LINKEDIN_CONTENT_FORMAT_OPTIONS,
    NEWSLETTER_STRUCTURE_OPTIONS,
    PLATFORM_OPTIONS,
    PODCAST_DESTINATION_OPTIONS,
    PODCAST_FORMAT_OPTIONS,
    PODCAST_LENGTH_OPTIONS,
    PODCAST_TONE_OPTIONS,
    PRESS_RELEASE_TIMING_OPTIONS,
    QUOTE_BOOK_OPTIONS,
    QUOTE_MOOD_TAGS,
    SOCIAL_OBJECTIVES,
    YOUTUBE_CONTENT_FORMAT_OPTIONS,
)


APP_TITLE = "Mythos Content Engine"
PRESS_PROFILE_PATH = PROJECT_ROOT / "outputs" / "press_profiles.json"
PRESS_RELEASE_DESTINATION = "PR Distribution Services"
PR_ASSET_OPTIONS = [
    "Media kit",
    "Pitch email",
    "Author bio",
    "Book description",
    "Boilerplate",
    "Fact sheet",
    "Cover image brief",
    "Press kit checklist",
]


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or seconds == float("inf"):
        return "calculating"
    seconds = int(round(seconds))
    minutes, remaining = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {remaining:02d}s"
    return f"{remaining}s"


async def set_generation_progress(
    *,
    progress,
    label,
    status=None,
    started_at: datetime,
    completed_steps: int,
    total_steps: int,
    completed_assets: int,
    total_assets: int,
    message: str,
) -> None:
    total_steps = max(1, int(total_steps or 1))
    total_assets = max(1, int(total_assets or 1))
    completed_steps = max(0, min(int(completed_steps), total_steps))
    completed_assets = max(0, min(int(completed_assets), total_assets))
    ratio = completed_steps / total_steps
    elapsed = max(0.1, (datetime.now() - started_at).total_seconds())
    eta = _format_eta((elapsed / ratio) - elapsed if ratio > 0 else 0)
    progress.visible = True
    progress.value = ratio
    detail = f"{int(round(ratio * 100))}% complete · ETA {eta} · Assets {completed_assets}/{total_assets} · {message}"
    if hasattr(label, "set_text"):
        label.set_text(detail)
    else:
        label.value = detail
    if status is not None:
        status.value = message
    await asyncio.sleep(0)


async def finish_generation_progress(
    *,
    progress,
    label,
    status=None,
    total_assets: int = 1,
    message: str = "Complete",
) -> None:
    progress.visible = True
    progress.value = 1
    detail = f"100% complete · ETA 0s · Assets {max(1, total_assets)}/{max(1, total_assets)} · {message}"
    if hasattr(label, "set_text"):
        label.set_text(detail)
    else:
        label.value = detail
    if status is not None:
        status.value = message
    await asyncio.sleep(0)
CHATGPT_COMPARISON_MODEL_OPTIONS = list(dict.fromkeys([
    os.getenv("CHATGPT_COMPARISON_MODEL", DEFAULT_MODEL),
    DEFAULT_MODEL,
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.4-instant",
]))

CAMPAIGN_FORMATS = [
    ("podcast", "Podcast"),
    ("instagram_caption", "Instagram"),
    ("youtube_content", "YouTube"),
    ("linkedin_content", "LinkedIn"),
    ("blog_post", "Blog Post"),
    ("newsletter_blurb", "Newsletter"),
    ("quote_post", "Quote Post"),
    ("review_pull_quote", "Review Pull Quote"),
    ("character_spotlight", "Character Spotlight"),
    ("press_release", "Press Release"),
]

CAMPAIGN_FORMAT_LABELS = {key: label for key, label in CAMPAIGN_FORMATS}
CAMPAIGN_QUANTITY_OPTIONS = ["1", "2", "3", "4", "5"]
CAMPAIGN_STYLE_OPTIONS = {
    "podcast": ["cinematic monologue", "host + guest interview", "lore deep dive", "news-style segment"],
    "instagram_caption": ["punchy caption", "reel script", "carousel copy", "story sequence"],
    "youtube_content": ["SEO title set", "description", "community post", "talking points"],
    "linkedin_content": ["authority post", "article", "reflection post", "announcement"],
    "blog_post": ["editorial essay", "listicle", "reader guide", "how-to guide", "SEO explainer"],
    "newsletter_blurb": ["announcement", "personal note", "digest", "behind-the-scenes"],
    "quote_post": ["character quote", "theme quote", "spoiler-free quote", "dark academia quote"],
    "review_pull_quote": ["critic pull quote", "reader praise", "short testimonial", "media kit blurb"],
    "character_spotlight": ["profile", "arc analysis", "spoiler-safe intro", "fan-facing deep dive"],
    "press_release": ["FOR IMMEDIATE RELEASE", "media advisory", "award announcement", "launch announcement"],
}

BLOG_FORMAT_OPTIONS = [
    "character deep-dive",
    "listicle",
    "interview",
    "best of",
    "news",
    "photo/gallery slideshow",
    "comparison",
    "op-ed",
    "curated content",
    "critics review roundup",
    "polls / survey / quiz",
]
BLOG_STRUCTURE_CHECKLIST_OPTIONS = [
    "strong hook",
    "canon context",
    "spoiler warning",
    "character stakes",
    "reader takeaway",
    "image/visual break",
    "quote pullout",
    "CTA",
    "SEO FAQ",
]
CHARACTER_SPOTLIGHT_PLATFORM_FORMATS = [
    "Instagram post",
    "Instagram reel",
    "Instagram story",
    "Instagram carousel",
    "YouTube cover",
    "YouTube post",
    "LinkedIn post square",
    "LinkedIn post horizontal",
    "Blog cover",
    "Blog square",
    "Blog horizontal",
]
CHARACTER_IMAGE_MODE_OPTIONS = [
    "use uploaded image",
    "use character portrait",
    "text-only highlight image",
]
QUOTE_IMAGE_STYLE_OPTIONS = list(QUOTE_GRAPHIC_THEMES.keys())
INSTAGRAM_VISUAL_FORMAT_OPTIONS = [
    "Instagram Post (4:5)",
    "Instagram Reel (9:16)",
    "Instagram Story (9:16)",
    "Square Post (1:1)",
]
BLOG_VISUAL_FORMAT_OPTIONS = [
    "Blog Cover (16:9)",
    "Blog Square (1:1)",
    "Blog Horizontal (1.91:1)",
    "LinkedIn Article Horizontal (1.91:1)",
]
BLOG_GALLERY_VISUAL_FORMAT_OPTIONS = [
    "Blog Cover (16:9)",
    "Blog Square (1:1)",
    "Blog Horizontal (1.91:1)",
    "Instagram Post (4:5)",
    "Square Post (1:1)",
]

PODCAST_PREVIEW_MAX_SEGMENTS = 3
PODCAST_TTS_MAX_CHARS = 2400
PODCAST_OUTPUT_FORMAT = "mp3_44100_128"
IMAGE_UPLOAD_DIR = PROJECT_ROOT / "outputs" / "uploads"
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_AWARE_CONTENT_TYPES = {"instagram_caption", "youtube_content", "linkedin_content", "character_spotlight", "blog_post"}


def normalize_selected(value) -> str:
    if value is None:
        return "Not specified"
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or "Not specified"
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        if not items:
            return "Not specified"
        return ", ".join(dict.fromkeys(items))
    cleaned = str(value).strip()
    return cleaned or "Not specified"


def load_press_profiles() -> dict:
    if not PRESS_PROFILE_PATH.exists():
        return {}
    try:
        return json.loads(PRESS_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_press_profiles(profiles: dict) -> None:
    PRESS_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRESS_PROFILE_PATH.write_text(json.dumps(profiles, indent=2, sort_keys=True), encoding="utf-8")


def press_profile_options() -> list[str]:
    return sorted(load_press_profiles().keys())


def first_nonempty_line(text: str) -> str:
    for line in str(text or "").splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if cleaned:
            return cleaned
    return ""


def brief_line(label: str, value) -> str:
    return f"{label}: {normalize_selected(value)}"


def safe_upload_filename(name: str) -> str:
    stem = Path(name or "uploaded_image").stem
    suffix = Path(name or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        suffix = ".png"
    stem = re.sub(r"[^\w\s-]", "", stem, flags=re.UNICODE)
    stem = re.sub(r"[-\s]+", "_", stem).strip("_").lower() or "uploaded_image"
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{stem[:70]}{suffix}"


def generate_press_release_assets(
    *,
    selected_assets,
    topic: str,
    draft: str,
    contact: dict[str, str],
    news_angle: str,
) -> dict[str, object]:
    assets = [asset for asset in (selected_assets or []) if str(asset).strip()]
    if not assets:
        return {}
    output_dir = PROJECT_ROOT / "outputs" / "press_release_assets" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    errors: list[str] = []
    for asset in assets:
        prompt = f"""Create this press-release support asset for Mythos Content Engine.

Asset: {asset}
Topic: {topic}
News angle: {news_angle}
Media contact: {json.dumps(contact, ensure_ascii=False)}

Press release draft:
{draft}

Return a polished, usable asset. If the selected asset is a checklist, use checkboxes. If it is a pitch email, include subject line and body. If it is a media kit, include sections and reusable copy."""
        try:
            content = generate_text(prompt)
            path = output_dir / f"{slugify_filename(asset, 'press_asset')}.md"
            path.write_text(content, encoding="utf-8")
            paths.append(str(path))
        except Exception as exc:
            errors.append(f"{asset}: {exc}")
    zip_path = output_dir / "press_release_assets.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            file_path = Path(path)
            if file_path.exists():
                archive.write(file_path, arcname=file_path.name)
    result: dict[str, object] = {"paths": paths, "zip_path": str(zip_path) if paths else ""}
    if errors:
        result["error"] = "; ".join(errors)
    return result


def image_data_url(image_path: str | Path | None) -> str:
    if not image_path:
        return ""
    path = Path(str(image_path))
    if not path.exists():
        return ""
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def social_image_instructions(content_type: str) -> str:
    if content_type == "instagram_caption":
        return """
IMAGE-AWARE OUTPUT REQUIREMENTS:
- Look carefully at the uploaded image and write content that reflects what is visibly present.
- Generate: 1) Instagram-ready caption, 2) Reel/Post/Story/Carousel text depending on selected formats, 3) SEO-friendly image title, 4) image description / alt text, 5) targeted hashtags.
- Use bookish, horror, YA thriller, dark academia, and Bookstagram audience language only when it fits the image and brief.
- Do not claim character identity, awards, reviews, or plot events unless supported by the brief or knowledge context.
"""
    if content_type == "youtube_content":
        return """
IMAGE-AWARE OUTPUT REQUIREMENTS:
- Treat the uploaded image as the visual basis for a YouTube thumbnail, community post, or video asset.
- Generate: 1) SEO title options, 2) YouTube description, 3) thumbnail/image title, 4) image description / alt text, 5) relevant hashtags/tags.
- Make the title clickable without becoming clickbait. Keep claims grounded in the brief and knowledge context.
"""
    if content_type == "linkedin_content":
        return """
IMAGE-AWARE OUTPUT REQUIREMENTS:
- Treat the uploaded image as a professional post/article visual.
- Generate: 1) LinkedIn post or article copy based on selected format, 2) professional title, 3) image description / alt text, 4) meta description, 5) restrained hashtags.
- Keep the tone credible, polished, and author-brand appropriate.
"""
    if content_type == "character_spotlight":
        return """
IMAGE-AWARE OUTPUT REQUIREMENTS:
- Use the uploaded image or selected character portrait as the visual basis when provided.
- Generate platform-native character spotlight text for the selected destination format.
- For Instagram, YouTube, and LinkedIn post formats, keep on-image and caption text short, punchy, and scannable.
- For blog destination formats, the description may be longer and more analytical.
- Include a concise visual headline/highlight that can fit on a generated image.
"""
    if content_type == "blog_post":
        return """
IMAGE-AWARE OUTPUT REQUIREMENTS:
- If an uploaded or character image is provided, use it as a blog visual and generate image title, alt title, caption, and meta description.
- Suggest 1-3 basic companion visuals such as quote-text graphics, character cards, comparison cards, or simple graph/chart ideas when useful.
- Keep blog visuals grounded in the selected book, character, and knowledge-base context.
"""
    return ""


def build_chatgpt_baseline_prompt(content_type: str, structured_brief: str) -> str:
    """Build a fair baseline prompt without Mythos knowledge-base context."""
    return f"""You are ChatGPT responding in a fresh, general-purpose chat.

Create the requested {content_type} using only the user's topic and selections below.

Important comparison rules:
- Do not use the Mythos Content Engine knowledge bases, brand playbooks, templates, quote libraries, or hidden project context.
- Do not mention this is a comparison.
- Match the requested content type, selected formats, quantity, style, platform, audience, constraints, and CTA as closely as possible.
- If the brief asks for verifiable quotes, reviews, awards, or manuscript details but does not provide exact evidence, avoid inventing them.
- Produce polished, publication-ready copy.

User topic and selections:
{structured_brief}
"""


def comparison_side_by_side_html(mythos_content: str = "", chatgpt_content: str = "", model: str = "") -> str:
    """Render Mythos and ChatGPT outputs side by side for human judging."""
    if not mythos_content and not chatgpt_content:
        return """
        <div style="border:1px dashed rgba(17,17,20,.18);border-radius:20px;padding:18px;color:#6b6b6b;background:#fff;">
            Generate a Mythos draft first, then create the ChatGPT baseline here.
        </div>
        """

    safe_mythos = escape(mythos_content or "No Mythos draft generated yet.")
    safe_chatgpt = escape(chatgpt_content or "No ChatGPT baseline generated yet.")
    safe_model = escape(model or "Selected ChatGPT model")
    return f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;">
        <article style="border:1px solid rgba(142,31,47,.18);border-radius:18px;background:#fff;padding:14px;">
            <div style="font-weight:800;color:#8E1F2F;margin-bottom:4px;">Mythos</div>
            <div style="font-size:12px;color:#6b6b6b;margin-bottom:10px;">Knowledge base + templates + brand workflow</div>
            <pre style="white-space:pre-wrap;font-family:inherit;font-size:13px;line-height:1.5;margin:0;">{safe_mythos}</pre>
        </article>
        <article style="border:1px solid rgba(17,17,20,.12);border-radius:18px;background:#fff;padding:14px;">
            <div style="font-weight:800;color:#111114;margin-bottom:4px;">ChatGPT</div>
            <div style="font-size:12px;color:#6b6b6b;margin-bottom:10px;">{safe_model}</div>
            <pre style="white-space:pre-wrap;font-family:inherit;font-size:13px;line-height:1.5;margin:0;">{safe_chatgpt}</pre>
        </article>
    </div>
    """


async def generate_chatgpt_comparison(
    *,
    content_type: str,
    structured_brief: str,
    mythos_content: str,
    model: str,
    comparison_status,
    comparison_view,
    chatgpt_output,
    chatgpt_prompt_path,
    chatgpt_draft_path,
) -> None:
    """Generate and display a fresh ChatGPT baseline next to Mythos."""
    if not mythos_content or not mythos_content.strip():
        comparison_status.value = "Generate a Mythos draft first."
        ui.notify(comparison_status.value, type="warning")
        return
    if not structured_brief or not structured_brief.strip():
        comparison_status.value = "No structured brief captured yet. Generate a Mythos draft first."
        ui.notify(comparison_status.value, type="warning")
        return

    comparison_status.value = "Generating ChatGPT baseline..."
    prompt = build_chatgpt_baseline_prompt(content_type, structured_brief)
    selected_model = model or DEFAULT_MODEL
    try:
        baseline = await asyncio.to_thread(generate_text, prompt, selected_model)
        model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", selected_model).strip("_") or "chatgpt"
        saved_prompt = save_output(prompt, "chatgpt_comparison", f"{content_type}_{model_slug}_prompt")
        saved_draft = save_output(baseline, "chatgpt_comparison", f"{content_type}_{model_slug}_draft")
    except Exception as exc:
        comparison_status.value = f"ChatGPT comparison failed: {exc}"
        ui.notify(comparison_status.value, type="negative")
        return

    chatgpt_output.value = baseline
    chatgpt_prompt_path.value = str(saved_prompt)
    chatgpt_draft_path.value = str(saved_draft)
    comparison_view.content = comparison_side_by_side_html(mythos_content, baseline, selected_model)
    comparison_status.value = "Comparison ready. Pick the stronger result below."
    ui.notify("Mythos vs ChatGPT comparison ready.", type="positive")


def save_comparison_preference(
    *,
    preference: str,
    notes: str,
    content_type: str,
    structured_brief: str,
    mythos_content: str,
    chatgpt_model: str,
    chatgpt_content: str,
    chatgpt_draft_path: str,
    chatgpt_prompt_path: str,
    comparison_save_status,
    comparison_vote_path,
) -> None:
    """Save the human judge's comparison decision."""
    if not preference:
        comparison_save_status.value = "Choose Mythos, ChatGPT, or Tie before saving."
        ui.notify(comparison_save_status.value, type="warning")
        return

    report = f"""# Mythos vs ChatGPT Human Preference

- Content type: {content_type or "Not specified"}
- Preference: {preference}
- ChatGPT model: {chatgpt_model or DEFAULT_MODEL}
- ChatGPT draft path: {chatgpt_draft_path or "Not saved"}
- ChatGPT prompt path: {chatgpt_prompt_path or "Not saved"}

## Judge Notes

{notes or "_No notes provided._"}

## Structured Brief

```text
{structured_brief or "No structured brief captured."}
```

## Mythos Output

{mythos_content or "_No Mythos output captured._"}

## ChatGPT Output

{chatgpt_content or "_No ChatGPT output captured._"}
"""
    saved_path = save_output(report, "mythos_vs_chatgpt", "human_preference")
    comparison_vote_path.value = str(saved_path)
    comparison_save_status.value = "Comparison judgment saved."
    ui.notify("Comparison judgment saved.", type="positive")


def build_brief(
    content_type: str,
    topic: str,
    related_book: str,
    platform: str,
    social_objectives,
    audience,
    cta: str,
    constraints,
    quote_book: str,
    quote_moods,
    character_tags,
    podcast_format: str,
    podcast_speakers: str,
    podcast_roles: str,
    podcast_tone,
    podcast_length: str,
    elevenlabs_model: str,
    elevenlabs_voice_ids: str,
    podcast_show_title: str = "",
    podcast_episode_title: str = "",
    podcast_episode_number: str = "",
    blog_length: str = "",
    blog_format: str = "",
    blog_sections: str = "",
    blog_structure_options=None,
    blog_seo_keywords: str = "",
    blog_image_mode: str = "",
    blog_character: str = "",
    instagram_formats=None,
    instagram_hashtags: str = "",
    instagram_hook: str = "",
    cs_character: str = "",
    cs_focus: str = "",
    cs_platform_format: str = "",
    cs_image_mode: str = "",
    nl_subject: str = "",
    nl_preview: str = "",
    nl_structure: str = "",
    pr_timing: str = "",
    pr_embargo_date: str = "",
    pr_city: str = "",
    pr_state: str = "",
    pr_release_date: str = "",
    pr_contact_name: str = "",
    pr_contact_title: str = "",
    pr_organization: str = "",
    pr_contact_email: str = "",
    pr_contact_phone: str = "",
    pr_website: str = "",
    pr_news_angle: str = "",
    pr_release_goal: str = "",
    pr_primary_announcement: str = "",
    pr_supporting_proof: str = "",
    pr_quote_source: str = "",
    pr_target_media: str = "",
    pr_required_assets: str = "",
    uploaded_image_path: str = "",
) -> str:
    lines = [
        brief_line("Topic", topic),
        brief_line("Related book/source", related_book),
        brief_line("Content type", content_type),
        brief_line("CTA", cta),
        brief_line("Constraints", constraints),
    ]
    if content_type != "press_release":
        lines[3:3] = [
            brief_line("Platform", platform),
            brief_line("Social objectives", social_objectives),
            brief_line("Audience", audience),
        ]

    if content_type in IMAGE_AWARE_CONTENT_TYPES:
        lines.extend(
            [
                brief_line("Uploaded image path", uploaded_image_path),
                "Instruction: If an uploaded image is provided, make the content respond directly to the visible image. If topic text is blank, use the image as the primary creative brief.",
            ]
        )

    if content_type == "instagram_caption":
        lines.extend(
            [
                brief_line("Instagram post formats", instagram_formats),
                brief_line("Instagram hashtag strategy", instagram_hashtags),
                brief_line("Instagram opening hook", instagram_hook),
                "Instruction: If Instagram opening hook is blank, generate a short hook/first line and include it clearly as Hook / First line.",
                "Instruction: Include caption/post copy, hashtag set, image text suggestion, and a short visual direction for each selected Instagram format.",
            ]
        )

    if content_type == "quote_post":
        lines.extend(
            [
                brief_line("Requested quote book/source", quote_book),
                brief_line("Requested quote mood/category tags", quote_moods),
                brief_line("Requested character tags", character_tags),
                brief_line("Available character visual assets", character_asset_note(character_tags)),
            ]
        )

    if content_type == "podcast":
        lines.extend(
            [
                brief_line("Podcast format", podcast_format),
                brief_line("Podcast speaker count", podcast_speakers),
                brief_line("Podcast speaker roles/names", podcast_roles),
                brief_line("Podcast tone", podcast_tone),
                brief_line("Podcast target length", podcast_length),
                brief_line("ElevenLabs model", elevenlabs_model),
                brief_line("ElevenLabs voice IDs", elevenlabs_voice_ids),
                brief_line("Podcast show title", podcast_show_title),
                brief_line("Podcast episode title", podcast_episode_title),
                brief_line("Podcast episode number", podcast_episode_number),
            ]
        )

    if content_type == "blog_post":
        lines.extend(
            [
                brief_line("Blog target length", blog_length),
                brief_line("Blog post format", blog_format),
                brief_line("Blog desired sections/outline", blog_sections),
                brief_line("Blog structure checklist", blog_structure_options),
                brief_line("Blog SEO keywords", blog_seo_keywords),
                brief_line("Blog image mode", blog_image_mode),
                brief_line("Blog selected character portrait", blog_character),
            ]
        )
        if str(blog_format or "").strip().lower() == "photo/gallery slideshow":
            lines.append(
                "Instruction: Generate a photo/gallery slideshow plan with multiple image concepts. Include image title, image caption, alt title, SEO description, and visual direction for each slide/image."
            )

    if content_type == "character_spotlight":
        lines.extend(
            [
                brief_line("Character spotlight subject", cs_character),
                brief_line("Character spotlight angle/focus", cs_focus),
                brief_line("Character spotlight platform format", cs_platform_format),
                brief_line("Character spotlight image mode", cs_image_mode),
                brief_line("Available character visual assets", character_asset_note(cs_character)),
            ]
        )

    if content_type == "newsletter_blurb":
        lines.extend(
            [
                brief_line("Newsletter subject line", nl_subject),
                brief_line("Newsletter preview text", nl_preview),
                brief_line("Newsletter structure", nl_structure),
            ]
        )

    if content_type == "press_release":
        lines.extend(
            [
                brief_line("Press release timing", pr_timing),
                brief_line("Press release distribution", PRESS_RELEASE_DESTINATION),
                brief_line("Press release embargo date", pr_embargo_date),
                brief_line("Press release dateline city", pr_city),
                brief_line("Press release dateline state/region", pr_state),
                brief_line("Press release date", pr_release_date),
                brief_line("Press release media contact name", pr_contact_name),
                brief_line("Press release media contact title", pr_contact_title),
                brief_line("Press release organization/imprint", pr_organization),
                brief_line("Press release media contact email", pr_contact_email),
                brief_line("Press release media contact phone", pr_contact_phone),
                brief_line("Press release website", pr_website),
                brief_line("Press release news angle", pr_news_angle),
                brief_line("Press release supporting proof", pr_supporting_proof),
                brief_line("Press release quote source", pr_quote_source),
                brief_line("Press release target media", pr_target_media),
                brief_line("Required PR assets", pr_required_assets),
            ]
        )

    return "\n".join(lines)


def run_pipeline_with_image(content_type: str, topic: str, image_path: str | Path) -> dict:
    """Run the Mythos pipeline with markdown context plus an uploaded image."""

    filtered_context = select_relevant_context(
        content_type=content_type,
        topic=topic,
    )
    filtered_context_path = save_output(
        content=filtered_context,
        content_type=content_type,
        label="filtered_context",
    )

    generation_prompt = create_generation_prompt(
        content_type=content_type,
        topic=topic,
        filtered_context=filtered_context,
    )
    generation_prompt = "\n\n".join(
        [
            generation_prompt,
            social_image_instructions(content_type),
            "Return organized, publication-ready output with clear labels for title, description, caption/post copy, hashtags, and image alt text where relevant.",
        ]
    )
    prompt_path = save_output(
        content=generation_prompt,
        content_type=content_type,
        label="generation_prompt",
    )

    generated_content = generate_text_with_image(generation_prompt, image_path)
    draft_path = save_output(
        content=generated_content,
        content_type=content_type,
        label="draft",
    )

    return {
        "filtered_context_path": filtered_context_path,
        "prompt_path": prompt_path,
        "draft_path": draft_path,
        "generated_content": generated_content,
    }


def platform_choices(content_type: str) -> list[str]:
    return list(PODCAST_DESTINATION_OPTIONS if content_type == "podcast" else PLATFORM_OPTIONS)


def style_choices(content_type: str) -> list[str]:
    return CAMPAIGN_STYLE_OPTIONS.get(content_type, ["default"])


def default_style(content_type: str) -> str:
    choices = style_choices(content_type)
    return choices[0] if choices else "default"


def build_campaign_asset_brief(
    *,
    content_type: str,
    asset_number: int,
    asset_count: int,
    campaign_topic: str,
    campaign_book: str,
    campaign_tone: str,
    campaign_objectives,
    campaign_audience,
    campaign_cta: str,
    campaign_constraints,
    style: str,
    quantity: str,
    instagram_formats=None,
    youtube_formats=None,
    linkedin_formats=None,
    blog_length: str = "",
    blog_format: str = "",
    blog_sections: str = "",
    blog_structure_options=None,
    blog_seo_keywords: str = "",
    blog_image_mode: str = "",
    podcast_format: str = "",
    podcast_length: str = "",
    podcast_tone: str = "",
    quote_book: str = "",
    quote_moods=None,
    quote_characters=None,
    quote_graphic_formats=None,
    quote_graphic_theme: str = "",
    pull_quote_graphic_formats=None,
    pull_quote_graphic_theme: str = "",
    press_timing: str = "",
    press_news_angle: str = "",
    press_release_goal: str = "",
    newsletter_structure: str = "",
    character_subject: str = "",
    character_format: str = "",
    character_image_mode: str = "",
) -> str:
    """Build a deterministic per-asset campaign brief from user selections."""
    lines = [
        "Campaign Mode Asset Brief",
        brief_line("Campaign topic", campaign_topic),
        brief_line("Campaign related book/source", campaign_book),
        brief_line("Selected campaign output", content_type),
        brief_line("Asset number", f"{asset_number} of {asset_count}"),
        brief_line("Requested quantity for this content type", quantity),
        brief_line("Style / variant", style),
        brief_line("Overall campaign tone", campaign_tone),
        brief_line("Campaign objectives", campaign_objectives),
        brief_line("Campaign audience", campaign_audience),
        brief_line("Campaign CTA", campaign_cta),
        brief_line("Campaign constraints", campaign_constraints),
        "Instruction: This asset is part of a coordinated campaign. Make it native to its platform, but keep the same core message, audience promise, and brand voice as the campaign.",
    ]

    if content_type == "instagram_caption":
        lines.extend(
            [
                brief_line("Instagram selected post formats", instagram_formats),
                brief_line("Hashtag strategy", "Use audience-relevant book community hashtags only when requested."),
            ]
        )
    elif content_type == "youtube_content":
        lines.append(brief_line("YouTube selected deliverables", youtube_formats))
    elif content_type == "linkedin_content":
        lines.append(brief_line("LinkedIn selected deliverables", linkedin_formats))
    elif content_type == "blog_post":
        lines.extend(
            [
                brief_line("Blog target length", blog_length),
                brief_line("Blog post format", blog_format),
                brief_line("Blog structure checklist", blog_structure_options),
                brief_line("Blog sections / outline", blog_sections),
                brief_line("Blog SEO keywords", blog_seo_keywords),
                brief_line("Blog image mode", blog_image_mode),
            ]
        )
    elif content_type == "podcast":
        lines.extend(
            [
                brief_line("Podcast format", podcast_format),
                brief_line("Podcast length", podcast_length),
                brief_line("Podcast delivery tone", podcast_tone),
            ]
        )
    elif content_type == "quote_post":
        lines.extend(
            [
                brief_line("Quote book/source", quote_book),
                brief_line("Quote mood/category tags", quote_moods),
                brief_line("Quote character tags", quote_characters),
                brief_line("Quote image destination formats", quote_graphic_formats),
                brief_line("Quote image color style", quote_graphic_theme),
                brief_line("Available character visual assets", character_asset_note(quote_characters)),
            ]
        )
    elif content_type == "review_pull_quote":
        lines.extend(
            [
                brief_line("Review source rule", "Use only real review text from the knowledge base."),
                brief_line("Pull quote style", style),
                brief_line("Pull quote image destination formats", pull_quote_graphic_formats),
                brief_line("Pull quote image color style", pull_quote_graphic_theme),
            ]
        )
    elif content_type == "press_release":
        lines.extend(
            [
                brief_line("Press release timing", press_timing),
                brief_line("News angle", press_news_angle),
            ]
        )
    elif content_type == "newsletter_blurb":
        lines.append(brief_line("Newsletter structure", newsletter_structure))
    elif content_type == "character_spotlight":
        lines.extend(
            [
                brief_line("Character spotlight subject", character_subject),
                brief_line("Character spotlight destination format", character_format),
                brief_line("Character spotlight image mode", character_image_mode),
                brief_line("Available character visual assets", character_asset_note(character_subject)),
            ]
        )

    return "\n".join(lines)


async def generate_campaign_from_fields(
    *,
    campaign_topic,
    campaign_book,
    campaign_tone,
    campaign_objectives,
    campaign_audience,
    campaign_cta,
    campaign_constraints,
    campaign_formats,
    campaign_widgets,
    status,
    campaign_output,
    campaign_preview,
    campaign_brief_output,
    campaign_path,
    saved_draft_path=None,
    campaign_progress=None,
    campaign_progress_label=None,
) -> None:
    selected = [content_type for content_type, _label in CAMPAIGN_FORMATS if content_type in (campaign_formats or [])]
    if not campaign_topic or not str(campaign_topic).strip():
        status.value = "Enter a campaign topic first."
        ui.notify(status.value, type="warning")
        return
    if not selected:
        status.value = "Select at least one campaign content type."
        ui.notify(status.value, type="warning")
        return

    status.value = "Generating campaign..."
    campaign_output.value = ""
    campaign_preview.content = build_campaign_preview_html([])
    campaign_path.value = ""
    if saved_draft_path is not None:
        saved_draft_path.value = ""
    all_briefs: list[str] = []
    results: list[str] = []
    asset_records: list[dict[str, object]] = []
    asset_total = sum(int(campaign_widgets[content_type]["quantity"].value or "1") for content_type in selected)
    completed_assets = 0
    started_at = datetime.now()

    try:
        for content_type in selected:
            widgets = campaign_widgets[content_type]
            quantity = widgets["quantity"].value or "1"
            count = int(quantity)
            for index in range(1, count + 1):
                label = CAMPAIGN_FORMAT_LABELS.get(content_type, content_type)
                if campaign_progress is not None and campaign_progress_label is not None:
                    await set_generation_progress(
                        progress=campaign_progress,
                        label=campaign_progress_label,
                        status=status,
                        started_at=started_at,
                        completed_steps=completed_assets,
                        total_steps=asset_total + 1,
                        completed_assets=completed_assets,
                        total_assets=asset_total,
                        message=f"Generating {label} asset {index} of {count}...",
                    )
                brief = build_campaign_asset_brief(
                    content_type=content_type,
                    asset_number=index,
                    asset_count=count,
                    campaign_topic=campaign_topic,
                    campaign_book=campaign_book,
                    campaign_tone=campaign_tone,
                    campaign_objectives=campaign_objectives,
                    campaign_audience=campaign_audience,
                    campaign_cta=campaign_cta,
                    campaign_constraints=campaign_constraints,
                    style=widgets["style"].value,
                    quantity=quantity,
                    instagram_formats=campaign_widgets["instagram_caption"].get("formats").value,
                    youtube_formats=campaign_widgets["youtube_content"].get("formats").value,
                    linkedin_formats=campaign_widgets["linkedin_content"].get("formats").value,
                    blog_length=campaign_widgets["blog_post"].get("length").value,
                    blog_format=campaign_widgets["blog_post"].get("format").value,
                    blog_sections=campaign_widgets["blog_post"].get("sections").value,
                    blog_structure_options=campaign_widgets["blog_post"].get("structure").value,
                    blog_seo_keywords=campaign_widgets["blog_post"].get("seo").value,
                    blog_image_mode=campaign_widgets["blog_post"].get("image_mode").value,
                    podcast_format=campaign_widgets["podcast"].get("format").value,
                    podcast_length=campaign_widgets["podcast"].get("length").value,
                    podcast_tone=campaign_widgets["podcast"].get("tone").value,
                    quote_book=campaign_widgets["quote_post"].get("book").value,
                    quote_moods=campaign_widgets["quote_post"].get("moods").value,
                    quote_characters=campaign_widgets["quote_post"].get("characters").value,
                    quote_graphic_formats=campaign_widgets["quote_post"].get("graphic_formats").value,
                    quote_graphic_theme=campaign_widgets["quote_post"].get("graphic_theme").value,
                    pull_quote_graphic_formats=campaign_widgets["review_pull_quote"].get("graphic_formats").value,
                    pull_quote_graphic_theme=campaign_widgets["review_pull_quote"].get("graphic_theme").value,
                    press_timing=campaign_widgets["press_release"].get("timing").value,
                    press_news_angle=campaign_widgets["press_release"].get("news_angle").value,
                    press_release_goal="",
                    newsletter_structure=campaign_widgets["newsletter_blurb"].get("structure").value,
                    character_subject=campaign_widgets["character_spotlight"].get("character").value,
                    character_format=campaign_widgets["character_spotlight"].get("format").value,
                    character_image_mode=campaign_widgets["character_spotlight"].get("image_mode").value,
                )
                all_briefs.append(brief)
                result = await asyncio.to_thread(run_pipeline, content_type=content_type, topic=brief)
                heading = f"## {label} {index}" if count > 1 else f"## {label}"
                generated_content = result["generated_content"]
                results.append(f"{heading}\n\n{generated_content}")
                preview_fields = {
                    "instagram_formats": campaign_widgets["instagram_caption"].get("formats").value,
                    "instagram_hashtags": "Use selected audience hashtags",
                    "instagram_hook": widgets["style"].value,
                    "youtube_formats": campaign_widgets["youtube_content"].get("formats").value,
                    "youtube_keywords": campaign_widgets["youtube_content"].get("style").value,
                    "youtube_link": campaign_cta,
                    "linkedin_formats": campaign_widgets["linkedin_content"].get("formats").value,
                    "linkedin_angle": campaign_widgets["linkedin_content"].get("style").value,
                    "linkedin_cta": campaign_cta,
                    "quote_moods": campaign_widgets["quote_post"].get("moods").value,
                    "quote_characters": campaign_widgets["quote_post"].get("characters").value,
                    "character_subject": campaign_widgets["character_spotlight"].get("character").value,
                    "character_format": campaign_widgets["character_spotlight"].get("format").value,
                    "character_image_mode": campaign_widgets["character_spotlight"].get("image_mode").value,
                    "blog_format": campaign_widgets["blog_post"].get("format").value,
                    "blog_length": campaign_widgets["blog_post"].get("length").value,
                    "blog_structure": campaign_widgets["blog_post"].get("sections").value,
                    "blog_structure_checklist": campaign_widgets["blog_post"].get("structure").value,
                    "blog_image_mode": campaign_widgets["blog_post"].get("image_mode").value,
                }
                quote_graphic = {}
                if content_type in {"quote_post", "review_pull_quote", "character_spotlight", "blog_post"}:
                    quote_graphic = render_campaign_quote_graphic(
                        content_type=content_type,
                        content=generated_content,
                        widgets=widgets,
                        campaign_topic=str(campaign_topic or "campaign"),
                        asset_number=index,
                    )
                asset_records.append(
                    {
                        "content_type": content_type,
                        "label": label,
                        "asset_label": f"Asset {index} of {count}",
                        "style": widgets["style"].value,
                        "content": generated_content,
                        "preview_fields": preview_fields,
                        "quote_graphic": quote_graphic,
                        "widgets": widgets,
                    }
                )
                completed_assets += 1
                if campaign_progress is not None and campaign_progress_label is not None:
                    await set_generation_progress(
                        progress=campaign_progress,
                        label=campaign_progress_label,
                        status=status,
                        started_at=started_at,
                        completed_steps=completed_assets,
                        total_steps=asset_total + 1,
                        completed_assets=completed_assets,
                        total_assets=asset_total,
                        message=f"Completed {label} asset {index} of {count}.",
                    )
    except Exception as exc:
        status.value = f"Campaign generation failed: {exc}"
        ui.notify(status.value, type="negative")
        return

    if campaign_progress is not None and campaign_progress_label is not None:
        await set_generation_progress(
            progress=campaign_progress,
            label=campaign_progress_label,
            status=status,
            started_at=started_at,
            completed_steps=asset_total,
            total_steps=asset_total + 1,
            completed_assets=completed_assets,
            total_assets=asset_total,
            message="Saving campaign bundle...",
        )
    combined = "\n\n---\n\n".join(results)
    saved_path = save_output(combined, "campaign_mode", "bundle")
    draft_record = save_draft(
        title=str(campaign_topic).strip()[:90] or "Campaign Mode",
        content_type="campaign_mode",
        content=combined,
        source_path=saved_path,
        metadata={
            "formats": selected,
            "asset_count": len(results),
            "topic": str(campaign_topic or "").strip(),
            "related_book": normalize_selected(campaign_book),
            "brief": "\n\n---\n\n".join(all_briefs),
            "objectives": normalize_selected(campaign_objectives),
            "audience": normalize_selected(campaign_audience),
        },
    )
    campaign_output.value = combined
    campaign_preview.content = build_campaign_preview_html(asset_records)
    campaign_brief_output.value = "\n\n---\n\n".join(all_briefs)
    campaign_path.value = str(saved_path)
    if saved_draft_path is not None:
        saved_draft_path.value = draft_record["path"]
    status.value = f"Campaign ready: {len(results)} assets generated."
    if campaign_progress is not None and campaign_progress_label is not None:
        await finish_generation_progress(
            progress=campaign_progress,
            label=campaign_progress_label,
            status=status,
            total_assets=asset_total,
            message=f"Campaign ready: {len(results)} assets generated.",
        )
    ui.notify(status.value, type="positive")


def slugify_filename(value: str, fallback: str = "podcast") -> str:
    value = (value or "").strip() or fallback
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[-\s]+", "_", value).strip("_").lower()
    return value or fallback


def normalize_speaker_label(value: str) -> str:
    value = re.sub(r"[*_`#>]", "", value or "")
    value = value.strip().strip("[]").strip()
    value = re.sub(r"[^A-Za-z0-9_ -]", "", value)
    value = re.sub(r"[-\s]+", "_", value).strip("_")
    return value.upper() or "HOST"


def parse_podcast_segments(script: str) -> list[tuple[str, str]]:
    """Extract speaker-labeled turns, falling back to host narration."""
    segments: list[tuple[str, str]] = []
    current_label = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_label, current_lines
        text = strip_production_tags("\n".join(current_lines).strip())
        if current_label and text:
            segments.append((current_label, text))
        current_lines = []

    for raw_line in (script or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        bracket_match = re.match(
            r"^\[(?!SFX:|INTRO MUSIC:|OUTRO MUSIC:|MUSIC BED:|PAUSE:|BEAT|CUT IF|OPTIONAL|USER EDIT:)([A-Za-z0-9_ -]+)\]\s*(.*)$",
            line,
            flags=re.IGNORECASE,
        )
        if bracket_match:
            flush()
            current_label = normalize_speaker_label(bracket_match.group(1))
            remainder = bracket_match.group(2).strip(" :-")
            current_lines = [remainder] if remainder else []
            continue

        colon_match = re.match(r"^([A-Za-z][A-Za-z0-9_ -]{1,40}):\s+(.+)$", line)
        if colon_match and not line.startswith("["):
            flush()
            current_label = normalize_speaker_label(colon_match.group(1))
            current_lines = [colon_match.group(2).strip()]
            continue

        if current_label:
            current_lines.append(line)

    flush()

    if segments:
        return segments

    fallback_lines = []
    for raw_line in (script or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        if re.match(r"^\[(SFX:|INTRO MUSIC:|OUTRO MUSIC:|MUSIC BED:|PAUSE:|BEAT)", line, flags=re.IGNORECASE):
            continue
        fallback_lines.append(line)
    fallback_text = strip_production_tags("\n".join(fallback_lines).strip())
    return [("HOST", fallback_text)] if fallback_text else []


def split_tts_text(text: str, max_chars: int = PODCAST_TTS_MAX_CHARS) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        proposed = f"{current} {sentence}".strip()
        if len(proposed) <= max_chars:
            current = proposed
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_chars:
            current = sentence
            continue
        words = sentence.split()
        current = ""
        for word in words:
            proposed_word = f"{current} {word}".strip()
            if len(proposed_word) > max_chars and current:
                chunks.append(current)
                current = word
            else:
                current = proposed_word
    if current:
        chunks.append(current)
    return chunks


def compact_podcast_segments(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    compacted: list[tuple[str, str]] = []
    pending_label = ""
    pending_text = ""

    def flush() -> None:
        nonlocal pending_label, pending_text
        if pending_label and pending_text.strip():
            for chunk in split_tts_text(pending_text):
                compacted.append((pending_label, chunk))
        pending_label = ""
        pending_text = ""

    for label, text in segments:
        text = text.strip()
        proposed = f"{pending_text}\n\n{text}".strip() if pending_text else text
        if label == pending_label and len(proposed) <= PODCAST_TTS_MAX_CHARS:
            pending_text = proposed
            continue
        flush()
        pending_label = label
        pending_text = text
    flush()
    return compacted


def voice_options(voices: list[dict]) -> dict[str, str]:
    return {
        voice["voice_id"]: f"{voice.get('name', 'Unnamed')} · {voice.get('category', 'voice')}"
        for voice in voices
        if voice.get("voice_id")
    }


def voice_preview_url(voice_id: str, voices: list[dict]) -> str:
    for voice in voices:
        if voice.get("voice_id") == voice_id:
            return voice.get("preview_url", "") or ""
    return ""


def build_voice_casting_from_selects(host_voice: str, guest_voice: str, guest_2_voice: str) -> str:
    lines = []
    if host_voice:
        lines.append(f"HOST={host_voice}")
        lines.append(f"NARRATOR={host_voice}")
    if guest_voice:
        lines.extend([f"GUEST={guest_voice}", f"GUEST_1={guest_voice}", f"AUTHOR={guest_voice}"])
    if guest_2_voice:
        lines.extend([f"GUEST_2={guest_2_voice}", f"COHOST={guest_2_voice}"])
    return "\n".join(lines)


def parse_voice_casting(value: str) -> dict[str, str]:
    casting: dict[str, str] = {}
    for raw_line in (value or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" in line:
            label, voice_id = line.split("=", 1)
        elif ":" in line:
            label, voice_id = line.split(":", 1)
        else:
            continue
        casting[normalize_speaker_label(label)] = voice_id.strip()
    return casting


def resolve_voice_for_label(label: str, casting: dict[str, str], fallback_ids: list[str]) -> str:
    normalized = normalize_speaker_label(label)
    if normalized in casting:
        return casting[normalized]
    if normalized.startswith("GUEST_2") or normalized in {"COHOST", "SECOND_GUEST"}:
        return casting.get("GUEST_2") or (fallback_ids[2] if len(fallback_ids) > 2 else fallback_ids[-1])
    if normalized.startswith("GUEST") or normalized in {"AUTHOR", "CRITIC", "INTERVIEWEE"}:
        return casting.get("GUEST") or casting.get("GUEST_1") or (fallback_ids[1] if len(fallback_ids) > 1 else fallback_ids[0])
    return casting.get("HOST") or casting.get("NARRATOR") or fallback_ids[0]


async def render_podcast_audio_native(
    *,
    script: str,
    document_name: str,
    voices: list[dict],
    host_voice: str,
    guest_voice: str,
    guest_2_voice: str,
    model_id: str,
    stability: float,
    similarity_boost: float,
    style: float,
    speed: float,
    speaker_boost: bool,
    preview_only: bool,
    status,
    audio_player,
    full_audio_path,
    package_path,
    saved_draft_path=None,
    progress=None,
    progress_label=None,
) -> None:
    if not script or not script.strip():
        status.value = "Generate or paste a podcast script first."
        ui.notify(status.value, type="warning")
        return
    if not os.getenv("ELEVENLABS_API_KEY"):
        status.value = "ELEVENLABS_API_KEY is missing. Add it to .env and restart the server."
        ui.notify(status.value, type="negative")
        return
    if not voices:
        status.value = "No ElevenLabs voices are loaded yet."
        ui.notify(status.value, type="warning")
        return

    selected_voice_ids = [voice_id for voice_id in [host_voice, guest_voice, guest_2_voice] if voice_id]
    if not selected_voice_ids:
        selected_voice_ids = [voice["voice_id"] for voice in voices if voice.get("voice_id")][:1]
    if not selected_voice_ids:
        status.value = "No usable ElevenLabs voices found."
        ui.notify(status.value, type="negative")
        return

    segments = compact_podcast_segments(parse_podcast_segments(script))
    if preview_only:
        segments = segments[:PODCAST_PREVIEW_MAX_SEGMENTS]
    if not segments:
        status.value = "No renderable spoken text found in the podcast script."
        ui.notify(status.value, type="warning")
        return

    status.value = "Rendering ElevenLabs audio..."
    casting = parse_voice_casting(build_voice_casting_from_selects(host_voice, guest_voice, guest_2_voice))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stem = slugify_filename(document_name or "podcast_audio")
    render_mode = "preview" if preview_only else "full"
    output_dir = PROJECT_ROOT / "outputs" / f"{file_stem}_{render_mode}_audio_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_paths: list[Path] = []
    started_at = datetime.now()
    try:
        if progress is not None and progress_label is not None:
            await set_generation_progress(
                progress=progress,
                label=progress_label,
                status=status,
                started_at=started_at,
                completed_steps=0,
                total_steps=len(segments) + 2,
                completed_assets=0,
                total_assets=len(segments),
                message=f"Preparing {render_mode} audio render...",
            )
        for index, (label, text) in enumerate(segments, start=1):
            voice_id = resolve_voice_for_label(label, casting, selected_voice_ids)
            segment_path = output_dir / f"{index:03d}_{label.lower()}.mp3"
            if progress is not None and progress_label is not None:
                await set_generation_progress(
                    progress=progress,
                    label=progress_label,
                    status=status,
                    started_at=started_at,
                    completed_steps=index - 1,
                    total_steps=len(segments) + 2,
                    completed_assets=index - 1,
                    total_assets=len(segments),
                    message=f"Rendering voice segment {index} of {len(segments)}...",
                )
            await asyncio.to_thread(
                synthesize_speech,
                text=text,
                voice_id=voice_id,
                output_path=segment_path,
                model_id=model_id or ELEVENLABS_MODEL_OPTIONS[0],
                output_format=PODCAST_OUTPUT_FORMAT,
                strip_tags=True,
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                speed=speed,
                use_speaker_boost=speaker_boost,
            )
            rendered_paths.append(segment_path)
            if progress is not None and progress_label is not None:
                await set_generation_progress(
                    progress=progress,
                    label=progress_label,
                    status=status,
                    started_at=started_at,
                    completed_steps=index,
                    total_steps=len(segments) + 2,
                    completed_assets=index,
                    total_assets=len(segments),
                    message=f"Completed voice segment {index} of {len(segments)}.",
                )

        if progress is not None and progress_label is not None:
            await set_generation_progress(
                progress=progress,
                label=progress_label,
                status=status,
                started_at=started_at,
                completed_steps=len(segments) + 1,
                total_steps=len(segments) + 2,
                completed_assets=len(rendered_paths),
                total_assets=len(segments),
                message="Combining segments into MP3...",
            )
        combined = AudioSegment.empty()
        pause = AudioSegment.silent(duration=650)
        for index, segment_path in enumerate(rendered_paths):
            if index:
                combined += pause
            combined += AudioSegment.from_file(segment_path)

        output_path = output_dir / f"{file_stem}_{render_mode}.mp3"
        combined.export(output_path, format="mp3", bitrate="128k")

        manifest_path = output_dir / "podcast_audio_manifest.md"
        manifest_path.write_text(
            "\n".join(
                [
                    "# Podcast Audio Manifest",
                    "",
                    f"Mode: {render_mode}",
                    f"Document: {document_name or file_stem}",
                    f"Model: {model_id or ELEVENLABS_MODEL_OPTIONS[0]}",
                    f"Rendered segments: {len(rendered_paths)}",
                    "",
                    "## Voice Casting",
                    build_voice_casting_from_selects(host_voice, guest_voice, guest_2_voice) or "Auto-assigned first available voice.",
                    "",
                    "## Segment Files",
                    *[f"- {path.name}" for path in rendered_paths],
                    "",
                    f"Combined MP3: {output_path.name}",
                ]
            ),
            encoding="utf-8",
        )

        zip_path = output_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output_path, arcname=output_path.name)
            archive.write(manifest_path, arcname=manifest_path.name)
            for segment_path in rendered_paths:
                archive.write(segment_path, arcname=f"segments/{segment_path.name}")
    except Exception as exc:
        status.value = f"Audio render failed: {exc}"
        ui.notify(status.value, type="negative")
        return

    full_audio_path.value = str(output_path)
    package_path.value = str(zip_path)
    audio_draft_record = save_draft(
        title=f"{document_name or file_stem} {render_mode} audio",
        content_type="podcast_audio",
        content=manifest_path.read_text(encoding="utf-8"),
        source_path=output_path,
        metadata={
            "mode": render_mode,
            "package_path": str(zip_path),
            "segment_count": len(rendered_paths),
            "model": model_id or ELEVENLABS_MODEL_OPTIONS[0],
        },
    )
    if saved_draft_path is not None:
        saved_draft_path.value = audio_draft_record["path"]
    audio_player.set_source(output_path)
    status.value = f"Rendered {render_mode} MP3 from {len(rendered_paths)} voice segment(s)."
    if progress is not None and progress_label is not None:
        await finish_generation_progress(
            progress=progress,
            label=progress_label,
            status=status,
            total_assets=len(rendered_paths),
            message=f"Rendered {render_mode} MP3 from {len(rendered_paths)} voice segment(s).",
        )
    ui.notify(status.value, type="positive")


def _split_markdown_list(value: str) -> list[str]:
    if not value:
        return []
    cleaned = re.sub(r"^[\s*-]+", "", value.strip(), flags=re.MULTILINE)
    parts = re.split(r",|\n", cleaned)
    return [part.strip(" -*`") for part in parts if part.strip(" -*`")]


def _markdown_field(block: str, field_name: str) -> str:
    pattern = rf"###\s+{re.escape(field_name)}\s*\n+(.*?)(?=\n###|\n---|\Z)"
    match = re.search(pattern, block, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def read_quote_bank_entries() -> list[dict[str, object]]:
    path = PROJECT_ROOT / "knowledge_base" / "primary" / "quote_bank.md"
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=##\s+Quote\b)", text)
    entries: list[dict[str, object]] = []
    for block in blocks:
        if "### Quote" not in block:
            continue
        heading = re.search(r"##\s+(Quote\s+[^\n]+)", block)
        entry_id = heading.group(1).strip() if heading else "Quote"
        quote_text = _markdown_field(block, "Quote")
        if not quote_text:
            continue
        entries.append(
            {
                "id": entry_id,
                "book": _markdown_field(block, "Book") or "Unknown book",
                "speaker": _markdown_field(block, "Speaker / Source") or "Unknown source",
                "characters": _split_markdown_list(_markdown_field(block, "Character Tags")),
                "moods": _split_markdown_list(_markdown_field(block, "Mood Tags")),
                "spoiler": _markdown_field(block, "Spoiler Level") or "Unspecified",
                "quote": quote_text.strip().strip('"'),
            }
        )
    return entries


def quote_book_options() -> list[str]:
    books = [entry["book"] for entry in read_quote_bank_entries() if entry.get("book")]
    ordered = list(dict.fromkeys([str(book) for book in books]))
    return ["All books", *ordered] if ordered else ["All books"]


def quote_entries_for_book(book_source: str) -> list[dict[str, object]]:
    entries = read_quote_bank_entries()
    if not book_source or book_source == "All books":
        return entries
    return [entry for entry in entries if str(entry.get("book", "")).lower() == str(book_source).lower()]


def quote_options_for_book(book_source: str) -> list[str]:
    return [entry["quote"] for entry in quote_entries_for_book(book_source)]


def quote_characters_for_book(book_source: str) -> list[str]:
    available: list[str] = []
    for entry in quote_entries_for_book(book_source):
        available.extend(entry.get("characters") or [])
    available_set = {name for name in available if name}
    ordered = [name for name in CHARACTER_TAGS if name in available_set]
    extras = sorted(available_set.difference(ordered))
    return ordered + extras or list(CHARACTER_TAGS)


def read_real_review_entries() -> list[dict[str, object]]:
    path = PROJECT_ROOT / "knowledge_base" / "secondary" / "real_reviews.md"
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=##\s+(?:Review Source|Reader Review Source)\b)", text)
    entries: list[dict[str, object]] = []
    for block in blocks:
        heading = re.search(r"##\s+(.+)", block)
        if not heading:
            continue
        review_id = heading.group(1).strip()
        source = _markdown_field(block, "Source") or "Unknown source"
        reviewer = _markdown_field(block, "Reviewer") or "Unknown reviewer"
        quotes_block = _markdown_field(block, "Real Quotes")
        quotes = [line.strip(" -*\"") for line in quotes_block.splitlines() if line.strip().startswith("-")]
        if not quotes:
            continue
        entries.append(
            {
                "id": review_id,
                "book": _markdown_field(block, "Book") or "Unknown book",
                "source": source,
                "reviewer": reviewer,
                "review_type": _markdown_field(block, "Review Type") or "Unspecified",
                "rating": _markdown_field(block, "Rating / Award") or _markdown_field(block, "Rating") or "",
                "date": _markdown_field(block, "Date") or "",
                "summary": _markdown_field(block, "Summary of Review") or "",
                "link": _markdown_field(block, "Source Link") or "",
                "quotes": quotes,
            }
        )
    return entries


def review_book_options() -> list[str]:
    books = [entry["book"] for entry in read_real_review_entries() if entry.get("book")]
    ordered = list(dict.fromkeys([str(book) for book in books]))
    return ["All books", *ordered] if ordered else ["All books"]


def review_entries_for_book(book_source: str) -> list[dict[str, object]]:
    entries = read_real_review_entries()
    if not book_source or book_source == "All books":
        return entries
    return [entry for entry in entries if str(entry.get("book", "")).lower() == str(book_source).lower()]


def review_source_options(book_source: str) -> list[str]:
    return [f"{entry['source']} — {entry['reviewer']}" for entry in review_entries_for_book(book_source)]


def review_quotes_for_source(book_source: str, review_source: str) -> list[str]:
    entries = review_entries_for_book(book_source)
    if review_source:
        entries = [entry for entry in entries if f"{entry['source']} — {entry['reviewer']}" == review_source]
    quotes: list[str] = []
    for entry in entries:
        quotes.extend(entry.get("quotes") or [])
    return list(dict.fromkeys(quotes))


def build_quote_preview_html(content_type: str, fields: dict[str, object]) -> str:
    if content_type == "quote_post":
        quote_text = normalize_selected(fields.get("quote_text"))
        source = normalize_selected(fields.get("quote_source"))
        moods = normalize_selected(fields.get("quote_moods"))
        characters = normalize_selected(fields.get("quote_characters"))
        spoiler = normalize_selected(fields.get("quote_spoiler"))
        request = normalize_selected(fields.get("quote_request"))
        return f"""
        <div style="border-radius:20px;padding:18px;background:linear-gradient(180deg,#fff,#fbf6f7);border:1px solid rgba(142,31,47,.14);box-shadow:0 18px 36px rgba(61,31,41,.08);">
            <div style="text-transform:uppercase;letter-spacing:.18em;font-size:11px;color:#8E1F2F;margin-bottom:12px;">Quote Post Preview</div>
            <div style="font-size:18px;line-height:1.55;font-weight:700;margin-bottom:12px;white-space:pre-wrap;">“{quote_text}”</div>
            <div style="font-size:13px;color:#6b6b6b;margin-bottom:6px;"><strong>Source:</strong> {source}</div>
            <div style="font-size:13px;color:#6b6b6b;margin-bottom:6px;"><strong>Moods:</strong> {moods}</div>
            <div style="font-size:13px;color:#6b6b6b;margin-bottom:6px;"><strong>Characters:</strong> {characters}</div>
            <div style="font-size:13px;color:#6b6b6b;margin-bottom:6px;"><strong>Spoiler:</strong> {spoiler}</div>
            <div style="font-size:13px;color:#6b6b6b;"><strong>Request:</strong> {request}</div>
        </div>
        """

    if content_type == "review_pull_quote":
        mode = normalize_selected(fields.get("review_mode"))
        quote_text = normalize_selected(fields.get("review_quote_text"))
        source = normalize_selected(fields.get("review_source"))
        attribution = normalize_selected(fields.get("review_attribution"))
        promo_line = normalize_selected(fields.get("review_promo_line"))
        return f"""
        <div style="border-radius:20px;padding:18px;background:linear-gradient(180deg,#fff,#f5f7fb);border:1px solid rgba(17,17,20,.12);box-shadow:0 18px 36px rgba(61,31,41,.08);">
            <div style="text-transform:uppercase;letter-spacing:.18em;font-size:11px;color:#8E1F2F;margin-bottom:12px;">Review Pull Quote Preview</div>
            <div style="font-size:18px;line-height:1.55;font-weight:700;margin-bottom:12px;white-space:pre-wrap;">{quote_text}</div>
            <div style="font-size:13px;color:#6b6b6b;margin-bottom:6px;"><strong>Mode:</strong> {mode}</div>
            <div style="font-size:13px;color:#6b6b6b;margin-bottom:6px;"><strong>Source:</strong> {source}</div>
            <div style="font-size:13px;color:#6b6b6b;margin-bottom:6px;"><strong>Attribution:</strong> {attribution}</div>
            <div style="font-size:13px;color:#6b6b6b;"><strong>Brand promo line:</strong> {promo_line}</div>
        </div>
        """

    return """
    <div style="border-radius:20px;padding:20px;background:#ffffff;border:1px dashed rgba(17,17,20,.16);color:#6b6b6b;">
        Select a quote workflow to preview grounded quote selections.
    </div>
    """


def build_preview_html(content_type: str, fields: dict[str, object]) -> str:
    def text(value) -> str:
        return normalize_selected(value)

    def safe(value) -> str:
        return escape(text(value))

    def image_markup(image_path, fallback_label: str) -> str:
        url = image_data_url(str(image_path or ""))
        if url:
            return f'<img src="{escape(url, quote=True)}" alt="Uploaded creative" style="width:100%;height:100%;object-fit:cover;display:block;">'
        return f"""
        <div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#1c1418,#6f182c);color:#f8efe8;font-weight:800;letter-spacing:.08em;text-align:center;padding:20px;">
            {escape(fallback_label)}
        </div>
        """

    def first_line(value, fallback: str) -> str:
        raw = text(value)
        if raw == "Not specified":
            return fallback
        for line in raw.splitlines():
            cleaned = line.strip().strip("#*- ")
            if cleaned:
                return cleaned[:110]
        return fallback

    if content_type == "quote_post":
        image_path = str(fields.get("image_path") or "")
        if image_path and Path(image_path).exists():
            return quote_platform_mockup_html([image_path], str(fields.get("generated_visual_package_path") or ""))
        candidates = extract_quote_candidates(str(fields.get("draft") or ""))
        quote_text = candidates[0] if candidates else first_line(fields.get("draft"), "Select quote cards to generate images.")
        return build_quote_preview_html(
            "quote_post",
            {
                "quote_text": quote_text,
                "quote_source": fields.get("quote_source") or "Generated quote draft",
                "quote_moods": fields.get("quote_moods"),
                "quote_characters": fields.get("quote_characters"),
                "quote_spoiler": "Grounded selection",
                "quote_request": "Use the Quote Image Builder below the draft to select cards and render platform mockups.",
            },
        )

    if content_type == "instagram_caption":
        formats = safe(fields.get("instagram_formats"))
        hashtags = safe(fields.get("instagram_hashtags"))
        hook = safe(fields.get("instagram_hook"))
        visual_formats = safe(fields.get("instagram_visual_formats"))
        visual_style = safe(fields.get("instagram_visual_style"))
        caption = safe(fields.get("draft") or fields.get("caption"))
        post_type = first_line(fields.get("instagram_formats"), "Post")
        return f"""
        <div style="max-width:430px;margin:0 auto;border-radius:28px;background:#fff;border:1px solid rgba(0,0,0,.10);box-shadow:0 18px 40px rgba(61,31,41,.10);overflow:hidden;">
            <div style="padding:14px 16px;display:flex;align-items:center;gap:12px;">
                <div style="width:42px;height:42px;border-radius:50%;background:linear-gradient(135deg,#2a0712,#8E1F2F);color:#fff;display:flex;align-items:center;justify-content:center;font-family:Georgia,serif;font-size:24px;">M</div>
                <div style="flex:1;">
                    <div style="font-weight:800;font-size:14px;">mortalvengeance <span style="color:#2f80ed;">✓</span></div>
                <div style="font-size:11px;color:#777;">{escape(post_type)} preview</div>
                </div>
                <div style="font-size:24px;line-height:1;">•••</div>
            </div>
            <div style="aspect-ratio:1/1;background:#160f13;">{image_markup(fields.get("image_path"), "UPLOAD IMAGE")}</div>
            <div style="padding:12px 16px;">
                <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:10px;color:#333;font-weight:800;"><span>Like · Comment · Share</span><span>Save</span></div>
                <div style="font-size:13px;margin-bottom:8px;">Liked by <strong>bookish.souls</strong> and <strong>1,243 others</strong></div>
                <div style="font-size:13px;line-height:1.45;white-space:pre-wrap;"><strong>mortalvengeance</strong> {caption}</div>
                <div style="font-size:12px;color:#2454a6;line-height:1.45;margin-top:8px;">{hashtags}</div>
                <div style="font-size:11px;color:#777;margin-top:10px;"><strong>Hook:</strong> {hook} · <strong>Formats:</strong> {formats}</div>
                <div style="font-size:11px;color:#777;margin-top:4px;"><strong>Image package:</strong> {visual_style} · {visual_formats}</div>
            </div>
        </div>
        """

    if content_type == "linkedin_content":
        formats = safe(fields.get("linkedin_formats"))
        angle = safe(fields.get("linkedin_angle"))
        cta = safe(fields.get("linkedin_cta"))
        draft = safe(fields.get("draft"))
        return f"""
        <div style="max-width:520px;margin:0 auto;border-radius:16px;background:#fff;border:1px solid rgba(0,0,0,.12);box-shadow:0 18px 40px rgba(61,31,41,.08);overflow:hidden;">
            <div style="padding:16px;display:flex;gap:12px;align-items:center;">
                <div style="width:48px;height:48px;border-radius:4px;background:#0a66c2;color:white;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:22px;">in</div>
                <div style="flex:1;">
                    <div style="font-weight:800;font-size:14px;">Alejandro Torres De La Rocha</div>
                    <div style="font-size:12px;color:#666;">Author · Mortal Vengeance</div>
                    <div style="font-size:11px;color:#777;">Now · Public</div>
                </div>
            </div>
            <div style="padding:0 16px 14px;font-size:14px;line-height:1.5;white-space:pre-wrap;">{draft}</div>
            <div style="aspect-ratio:1.91/1;background:#111;">{image_markup(fields.get("image_path"), "LINKEDIN VISUAL")}</div>
            <div style="padding:12px 16px;border-top:1px solid #eee;color:#666;font-size:12px;">
                <strong>Angle:</strong> {angle} · <strong>CTA:</strong> {cta} · <strong>Formats:</strong> {formats}
            </div>
            <div style="display:flex;justify-content:space-around;border-top:1px solid #eee;padding:10px 0;color:#555;font-size:13px;font-weight:700;">
                <span>Like</span><span>Comment</span><span>Repost</span><span>Send</span>
            </div>
        </div>
        """

    if content_type == "youtube_content":
        formats = safe(fields.get("youtube_formats"))
        keywords = safe(fields.get("youtube_keywords"))
        link = safe(fields.get("youtube_link"))
        draft = safe(fields.get("draft"))
        title = escape(first_line(fields.get("draft"), "Mortal Vengeance: A Grim Tale"))
        return f"""
        <div style="max-width:560px;margin:0 auto;border-radius:18px;background:#fff;border:1px solid rgba(0,0,0,.12);box-shadow:0 18px 40px rgba(61,31,41,.08);overflow:hidden;">
            <div style="aspect-ratio:16/9;background:#111;position:relative;">{image_markup(fields.get("image_path"), "YOUTUBE THUMBNAIL")}
                <div style="position:absolute;right:10px;bottom:10px;background:rgba(0,0,0,.82);color:#fff;border-radius:4px;padding:3px 6px;font-size:12px;">8:42</div>
            </div>
            <div style="padding:14px 16px;display:flex;gap:12px;">
                <div style="width:42px;height:42px;border-radius:50%;background:#8E1F2F;color:#fff;display:flex;align-items:center;justify-content:center;font-family:Georgia,serif;font-size:23px;">M</div>
                <div style="flex:1;">
                    <div style="font-size:16px;font-weight:800;line-height:1.3;margin-bottom:5px;">{title}</div>
                    <div style="font-size:12px;color:#666;margin-bottom:10px;">Mortal Vengeance · 1.2K views · just now</div>
                    <div style="font-size:13px;line-height:1.45;white-space:pre-wrap;color:#333;">{draft}</div>
                    <div style="font-size:12px;color:#666;margin-top:10px;"><strong>Formats:</strong> {formats} · <strong>Keywords:</strong> {keywords} · <strong>Link:</strong> {link}</div>
                </div>
            </div>
        </div>
        """

    if content_type == "press_release":
        draft_raw = text(fields.get("draft") or fields.get("topic"))
        lines = [line.strip() for line in draft_raw.splitlines() if line.strip()]
        headline = escape(first_line(draft_raw, "Press Release Headline"))
        body = safe("\n\n".join(lines[1:8]) if len(lines) > 1 else draft_raw)
        dateline_parts = [
            text(fields.get("pr_city")) if text(fields.get("pr_city")) != "Not specified" else "",
            text(fields.get("pr_state")) if text(fields.get("pr_state")) != "Not specified" else "",
        ]
        dateline = safe(", ".join(part for part in dateline_parts if part) or "Dateline")
        release_date = safe(fields.get("pr_release_date"))
        contact = safe(fields.get("pr_contact_name"))
        website = safe(fields.get("pr_website"))
        news_angle = safe(fields.get("pr_news_angle"))
        assets = safe(fields.get("pr_required_assets"))
        return f"""
        <div style="max-width:720px;margin:0 auto;background:#f7f3ea;border:1px solid rgba(44,24,31,.18);box-shadow:0 20px 42px rgba(61,31,41,.14);padding:18px;">
            <div style="border-top:5px solid #1e1b19;border-bottom:2px solid #1e1b19;text-align:center;padding:10px 0 8px;margin-bottom:14px;">
                <div style="font-family:Georgia,serif;font-size:42px;line-height:1;font-weight:900;letter-spacing:.02em;color:#1d1a18;">THE MYTHOS PRESS</div>
                <div style="display:flex;justify-content:space-between;font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:#5c524b;margin-top:8px;">
                    <span>FOR IMMEDIATE RELEASE</span><span>{release_date}</span><span>{PRESS_RELEASE_DESTINATION}</span>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:2fr 1fr;gap:18px;">
                <div>
                    <div style="font-size:11px;text-transform:uppercase;letter-spacing:.16em;color:#8E1F2F;font-weight:900;margin-bottom:8px;">{dateline}</div>
                    <h1 style="font-family:Georgia,serif;font-size:38px;line-height:1.02;margin:0 0 10px;color:#17120f;">{headline}</h1>
                    <div style="font-size:15px;line-height:1.58;white-space:pre-wrap;color:#26211e;column-count:2;column-gap:20px;">{body[:1800]}</div>
                </div>
                <aside style="border-left:1px solid rgba(44,24,31,.22);padding-left:14px;">
                    <div style="font-size:12px;text-transform:uppercase;letter-spacing:.14em;font-weight:900;border-bottom:1px solid rgba(44,24,31,.2);padding-bottom:6px;margin-bottom:10px;">Newswire Notes</div>
                    <div style="font-size:13px;line-height:1.45;color:#322b27;"><strong>News angle</strong><br>{news_angle}</div>
                    <div style="font-size:13px;line-height:1.45;color:#322b27;margin-top:12px;"><strong>Media contact</strong><br>{contact}<br>{website}</div>
                    <div style="font-size:13px;line-height:1.45;color:#322b27;margin-top:12px;"><strong>Assets</strong><br>{assets}</div>
                </aside>
            </div>
        </div>
        """

    if content_type == "blog_post":
        blog_format = safe(fields.get("blog_format"))
        blog_length = safe(fields.get("blog_length"))
        structure = safe(fields.get("blog_structure_options") or fields.get("blog_structure_checklist") or fields.get("blog_sections") or fields.get("blog_structure"))
        seo = safe(fields.get("blog_seo_keywords"))
        image_mode = safe(fields.get("blog_image_mode"))
        blog_character = safe(fields.get("blog_character"))
        visual_style = safe(fields.get("blog_visual_style"))
        visual_formats = safe(fields.get("blog_visual_formats"))
        draft = safe(fields.get("draft") or fields.get("topic"))
        title = escape(first_line(fields.get("draft") or fields.get("topic"), "Mortal Vengeance Feature"))
        return f"""
        <article style="max-width:680px;margin:0 auto;border-radius:22px;background:#fffdfb;border:1px solid rgba(44,24,31,.13);box-shadow:0 18px 40px rgba(61,31,41,.09);overflow:hidden;">
            <div style="aspect-ratio:16/9;background:#120d10;">{image_markup(fields.get("image_path"), "BLOG VISUAL")}</div>
            <div style="padding:24px;">
                <div style="text-transform:uppercase;letter-spacing:.16em;font-size:11px;color:#8E1F2F;font-weight:900;margin-bottom:10px;">{blog_format}</div>
                <h2 style="font-family:Georgia,serif;font-size:34px;line-height:1.08;margin:0 0 12px;color:#171217;">{title}</h2>
                <div style="font-size:13px;color:#71686a;margin-bottom:14px;"><strong>Length:</strong> {blog_length} · <strong>Image:</strong> {image_mode} · <strong>Character:</strong> {blog_character}</div>
                <div style="font-size:14px;line-height:1.58;white-space:pre-wrap;color:#2f292b;margin-bottom:14px;">{draft[:700]}</div>
                <div style="font-size:12px;color:#71686a;"><strong>Structure:</strong> {structure}</div>
                <div style="font-size:12px;color:#71686a;margin-top:6px;"><strong>SEO:</strong> {seo}</div>
                <div style="font-size:12px;color:#71686a;margin-top:6px;"><strong>Visuals:</strong> {visual_style} · {visual_formats}</div>
            </div>
        </article>
        """

    if content_type == "character_spotlight":
        subject = text(fields.get("character_subject") or fields.get("cs_character"))
        focus = safe(fields.get("character_focus") or fields.get("cs_focus"))
        format_label = text(fields.get("character_format") or fields.get("cs_platform_format"))
        image_mode = text(fields.get("character_image_mode") or fields.get("cs_image_mode"))
        draft = safe(fields.get("draft"))
        portrait = resolve_character_portrait_asset(subject) if image_mode != "use uploaded image" else None
        portrait_url = image_data_url(portrait) if portrait else ""
        if image_mode == "text-only highlight image":
            portrait_markup = image_markup("", escape(subject).upper())
        elif portrait_url:
            portrait_markup = f'<img src="{escape(portrait_url, quote=True)}" alt="{escape(subject, quote=True)} portrait" style="width:100%;height:100%;object-fit:cover;display:block;">'
        else:
            portrait_markup = image_markup(fields.get("image_path"), "CHARACTER PORTRAIT")

        lookup = format_label.lower()
        if "story" in lookup or "reel" in lookup:
            container_style = "max-width:360px;aspect-ratio:9/16"
            layout = "display:flex;flex-direction:column"
            image_style = "height:58%;"
            text_style = "padding:18px;"
        elif "youtube" in lookup or "horizontal" in lookup or "blog cover" in lookup:
            container_style = "max-width:620px;aspect-ratio:16/9"
            layout = "display:grid;grid-template-columns:46% 1fr"
            image_style = "height:100%;"
            text_style = "padding:20px;"
        else:
            container_style = "max-width:500px;aspect-ratio:1/1"
            layout = "display:grid;grid-template-columns:46% 1fr"
            image_style = "height:100%;"
            text_style = "padding:20px;"

        max_copy = 520 if "blog" in lookup else 240
        display_draft = draft[:max_copy] + ("..." if len(draft) > max_copy else "")
        return f"""
        <div style="{container_style};margin:0 auto;border-radius:22px;background:#fffdfb;border:1px solid rgba(44,24,31,.13);box-shadow:0 18px 40px rgba(61,31,41,.09);overflow:hidden;">
            <div style="{layout};width:100%;height:100%;">
                <div style="background:#090908;{image_style}">{portrait_markup}</div>
                <div style="{text_style}display:flex;flex-direction:column;gap:10px;overflow:hidden;">
                    <div style="text-transform:uppercase;letter-spacing:.18em;font-size:11px;color:#8E1F2F;font-weight:800;">Character Spotlight</div>
                    <div style="font-family:Georgia,serif;font-size:clamp(24px,4vw,34px);line-height:1.05;font-weight:900;color:#171217;">{escape(subject)}</div>
                    <div style="font-size:12px;color:#71686a;"><strong>{escape(format_label)}</strong> · {escape(image_mode)}</div>
                    <div style="font-size:12px;color:#71686a;"><strong>Focus:</strong> {focus}</div>
                    <div style="font-size:14px;line-height:1.45;white-space:pre-wrap;color:#2f292b;">{display_draft}</div>
                </div>
            </div>
        </div>
        """

    return f"""
    <div style="border-radius:20px;padding:20px;background:#ffffff;border:1px dashed rgba(17,17,20,.16);color:#6b6b6b;">
        Select a content type to see a platform-style preview.
    </div>
    """


def extract_quote_for_graphic(content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""
    label_match = re.search(
        r"(?im)^\s*(?:quote|pull quote|review quote|caption text|graphic text)\s*:\s*(.+)$",
        text,
    )
    if label_match:
        return label_match.group(1).strip().strip('"“”')[:260]
    curly_match = re.search(r"[“\"]([^”\"]{18,260})[”\"]", text)
    if curly_match:
        return curly_match.group(1).strip()
    for line in text.splitlines():
        cleaned = line.strip().strip("#*- ").strip('"“”')
        if 18 <= len(cleaned) <= 260:
            return cleaned
    return text[:260]


def clean_quote_candidate(value: str) -> str:
    cleaned = str(value or "").strip().strip('"“”')
    cleaned = re.sub(r"^\s*(?:quote|pull quote|graphic text|caption text)\s*\d*\s*[:\-]\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^\s*(?:\d+[\).\]]|[-*•])\s*", "", cleaned)
    return cleaned.strip().strip('"“”')


def extract_quote_candidates(content: str, limit: int = 12) -> list[str]:
    """Extract multiple plausible quote-card lines from generated content."""
    text = str(content or "").strip()
    if not text:
        return []
    candidates: list[str] = []

    for match in re.finditer(
        r"(?im)^\s*(?:quote|quote option|pull quote|graphic text|caption text)\s*\d*\s*[:\-]\s*(.+)$",
        text,
    ):
        candidates.append(clean_quote_candidate(match.group(1)))

    for match in re.finditer(r"[“\"]([^”\"]{18,260})[”\"]", text):
        candidates.append(clean_quote_candidate(match.group(1)))

    for line in text.splitlines():
        cleaned = clean_quote_candidate(line)
        lower = cleaned.lower()
        if (
            18 <= len(cleaned) <= 260
            and not lower.startswith(("caption", "hashtags", "cta", "source", "mood", "character", "image", "alt text"))
            and ":" not in cleaned[:22]
        ):
            candidates.append(cleaned)

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = re.sub(r"\s+", " ", candidate.lower()).strip()
        if candidate and key not in seen:
            seen.add(key)
            unique.append(candidate)
        if len(unique) >= limit:
            break
    return unique


def formats_for_character_spotlight(selection) -> list[str]:
    selected = normalize_selected(selection)
    lookup = selected.lower()
    if "instagram reel" in lookup:
        return ["Instagram Reel (9:16)"]
    if "instagram story" in lookup:
        return ["Instagram Story (9:16)"]
    if "instagram carousel" in lookup:
        return ["Instagram Post (4:5)", "Square Post (1:1)"]
    if "instagram post" in lookup:
        return ["Instagram Post (4:5)"]
    if "youtube cover" in lookup:
        return ["YouTube Image Cover (16:9)"]
    if "youtube post" in lookup:
        return ["YouTube Post (1:1)"]
    if "linkedin post square" in lookup:
        return ["LinkedIn Post Square (1:1)"]
    if "linkedin post horizontal" in lookup:
        return ["LinkedIn Post Horizontal (1.91:1)"]
    if "blog cover" in lookup:
        return ["Blog Cover (16:9)"]
    if "blog square" in lookup:
        return ["Blog Square (1:1)"]
    if "blog horizontal" in lookup:
        return ["Blog Horizontal (1.91:1)"]
    return ["Instagram Post (4:5)"]


def render_generated_visual_package(
    *,
    content_type: str,
    content: str,
    topic: str,
    character_name: str = "",
    format_labels=None,
    theme_name: str = "",
) -> dict[str, object]:
    """Render a basic downloadable image package for visual content workflows."""
    if content_type not in {"quote_post", "review_pull_quote", "character_spotlight", "blog_post", "instagram_caption"}:
        return {}

    quote = extract_quote_for_graphic(content)
    if not quote:
        quote = extract_quote_for_graphic(topic)
    if not quote and content_type in {"instagram_caption", "blog_post", "character_spotlight"}:
        quote = first_nonempty_line(content) or first_nonempty_line(topic)
    if not quote:
        return {}

    output_dir = PROJECT_ROOT / "outputs" / "generated_visuals" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_stem = slugify_filename(f"{content_type}_{quote}", "generated_visual")
    if not format_labels:
        if content_type == "blog_post":
            format_labels = ["Blog Cover (16:9)", "Blog Square (1:1)", "Blog Horizontal (1.91:1)"]
        elif content_type == "instagram_caption":
            format_labels = ["Instagram Post (4:5)"]
        else:
            format_labels = ["Instagram Post (4:5)"]

    use_case_lookup = {
        "blog_post": "Blog hero images",
        "instagram_caption": "Instagram posts",
        "character_spotlight": "Character illustrations",
        "review_pull_quote": "Social quote cards",
    }
    content_kind_lookup = {
        "blog_post": "character" if character_name else "hero",
        "instagram_caption": "social post",
        "character_spotlight": "character",
        "review_pull_quote": "quote typography",
    }
    external_paths: list[str] = []
    external_errors: list[str] = []
    external_prompts: list[str] = []
    for label in format_labels:
        external = generate_external_visual(
            use_case=use_case_lookup.get(content_type, "Social posts"),
            content=quote,
            topic=topic,
            style=theme_name,
            attribution=character_name or ("Real reader review" if content_type == "review_pull_quote" else "Mortal Vengeance"),
            format_label=label,
            output_dir=output_dir,
            file_stem=file_stem,
            content_kind=content_kind_lookup.get(content_type, ""),
        )
        if external.get("path"):
            external_paths.append(str(external["path"]))
        else:
            external_errors.append(f"{label}: {external.get('provider', 'external')} - {external.get('error', 'No image returned')}")
        if external.get("prompt"):
            external_prompts.append(str(external["prompt"]))

    if external_paths:
        zip_path = output_dir / f"{file_stem}_external_images.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in external_paths:
                file_path = Path(path)
                if file_path.exists():
                    archive.write(file_path, arcname=file_path.name)
        result = {
            "paths": external_paths,
            "zip_path": str(zip_path),
            "quote": quote,
            "provider": "external",
            "prompts": external_prompts,
        }
        if external_errors:
            result["error"] = "; ".join(external_errors)
        return result

    try:
        return render_quote_cards(
            quote=quote,
            attribution=character_name or ("Real reader review" if content_type == "review_pull_quote" else "Mortal Vengeance"),
            format_labels=format_labels,
            brand_title="Mortal Vengeance",
            output_dir=output_dir,
            file_stem=file_stem,
            theme_name=theme_name or "Gothic",
            character_name=character_name,
        )
    except Exception as exc:
        return {"error": str(exc), "quote": quote}


def render_selected_quote_visual_package(
    *,
    quotes: list[str],
    attribution: str,
    format_labels=None,
    theme_name: str = "",
) -> dict[str, object]:
    """Render a downloadable image package for selected quote candidates."""
    selected_quotes = [clean_quote_candidate(quote) for quote in quotes if clean_quote_candidate(quote)]
    if not selected_quotes:
        return {"error": "Select at least one quote card to generate images."}
    output_dir = PROJECT_ROOT / "outputs" / "generated_visuals" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_paths: list[str] = []
    errors: list[str] = []
    formats = format_labels or ["Instagram Post (4:5)"]
    speaker = "" if normalize_selected(attribution) == "Not specified" else normalize_selected(attribution)

    for index, quote in enumerate(selected_quotes, start=1):
        external_rendered = False
        quote_external_errors: list[str] = []
        for label in formats:
            external = generate_external_visual(
                use_case="Instagram quote cards",
                content=quote,
                topic=quote,
                style=theme_name,
                attribution=speaker or "Mortal Vengeance",
                format_label=label,
                output_dir=output_dir,
                file_stem=slugify_filename(f"quote_card_{index}_{quote}", f"quote_card_{index}"),
                content_kind="quote typography",
            )
            if external.get("path"):
                all_paths.append(str(external["path"]))
                external_rendered = True
            else:
                quote_external_errors.append(
                    f"{quote[:60]} / {label}: {external.get('provider', 'external')} - {external.get('error', 'No image returned')}"
                )
        if external_rendered:
            errors.extend(quote_external_errors)
            continue
        try:
            package = render_quote_cards(
                quote=quote,
                attribution=speaker or "Mortal Vengeance",
                format_labels=formats,
                brand_title="Mortal Vengeance",
                output_dir=output_dir,
                file_stem=slugify_filename(f"quote_card_{index}_{quote}", f"quote_card_{index}"),
                theme_name=theme_name or "Gothic",
                character_name=speaker,
            )
            all_paths.extend(str(path) for path in package.get("paths", []))
        except Exception as exc:
            errors.extend(quote_external_errors)
            errors.append(f"{quote[:60]}: {exc}")

    zip_path = output_dir / "selected_quote_cards.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in all_paths:
            file_path = Path(path)
            if file_path.exists():
                archive.write(file_path, arcname=file_path.name)

    result: dict[str, object] = {
        "paths": all_paths,
        "zip_path": str(zip_path),
        "quotes": selected_quotes,
    }
    if errors:
        result["error"] = "; ".join(errors)
    return result


def quote_platform_mockup_html(paths: list[str], package_path: str = "", error: str = "") -> str:
    if error and not paths:
        return f"""
        <div style="border-radius:18px;background:#fff7f7;border:1px solid rgba(142,31,47,.22);padding:16px;color:#8E1F2F;">
            {escape(error)}
        </div>
        """
    if not paths:
        return """
        <div style="border-radius:18px;background:#fff;border:1px dashed rgba(44,24,31,.18);padding:16px;color:#71686a;">
            Select quote cards and generate images to preview them here.
        </div>
        """

    cards = []
    for path in paths[:12]:
        file_path = Path(path)
        label = file_path.stem.replace("_", " ").title()
        src = image_data_url(file_path)
        slug = file_path.stem.lower()
        if "youtube" in slug or "cover" in slug:
            frame = "YouTube thumbnail"
            aspect = "16/9"
            chrome = '<div style="height:26px;background:#0f0f0f;color:white;font-size:12px;padding:6px 10px;">Mortal Vengeance · Thumbnail preview</div>'
        elif "linkedin" in slug or "horizontal" in slug or "link" in slug:
            frame = "LinkedIn / link preview"
            aspect = "1.91/1"
            chrome = '<div style="height:34px;background:#fff;border-bottom:1px solid #e6e2df;color:#0a66c2;font-weight:800;font-size:13px;padding:8px 10px;">LinkedIn post preview</div>'
        elif "story" in slug or "reel" in slug or "9x16" in slug:
            frame = "Instagram story / reel"
            aspect = "9/16"
            chrome = '<div style="height:34px;background:#111;color:white;font-size:12px;padding:8px 10px;">@mortalvengeance</div>'
        else:
            frame = "Instagram post"
            aspect = "4/5" if "4x5" in slug or "post" in slug else "1/1"
            chrome = '<div style="height:38px;background:#fff;border-bottom:1px solid #eee;font-weight:800;font-size:13px;padding:10px;">mortalvengeance</div>'
        cards.append(
            f"""
            <article style="border:1px solid rgba(44,24,31,.14);border-radius:18px;background:#fff;overflow:hidden;box-shadow:0 14px 28px rgba(61,31,41,.07);">
                {chrome}
                <div style="aspect-ratio:{aspect};background:#120d10;display:flex;align-items:center;justify-content:center;">
                    <img src="{escape(src, quote=True)}" alt="{escape(label, quote=True)}" style="width:100%;height:100%;object-fit:contain;display:block;">
                </div>
                <div style="padding:10px 12px;font-size:12px;color:#71686a;">
                    <strong>{escape(frame)}</strong><br>{escape(file_path.name)}
                </div>
            </article>
            """
        )
    package_note = (
        f'<div style="font-size:12px;color:#71686a;margin-top:10px;"><strong>Download package:</strong> {escape(package_path)}</div>'
        if package_path
        else ""
    )
    error_note = (
        f'<div style="font-size:12px;color:#8E1F2F;margin-top:8px;"><strong>Render note:</strong> {escape(error)}</div>'
        if error
        else ""
    )
    return f"""
    <div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;">
            {''.join(cards)}
        </div>
        {package_note}
        {error_note}
    </div>
    """


def attribution_for_campaign_asset(content_type: str, widgets: dict[str, object]) -> str:
    if content_type == "review_pull_quote":
        return "Real reader review"
    if content_type == "quote_post":
        characters = normalize_selected(widgets.get("characters").value if widgets.get("characters") else "")
        if characters != "Not specified":
            return characters
    if content_type == "character_spotlight":
        character = normalize_selected(widgets.get("character").value if widgets.get("character") else "")
        if character != "Not specified":
            return character
    return "Mortal Vengeance"


def campaign_graphic_formats(content_type: str, widgets: dict[str, object]) -> list[str]:
    if content_type == "quote_post" and widgets.get("graphic_formats"):
        formats = widgets["graphic_formats"].value or []
        return list(formats) if isinstance(formats, list) else [str(formats)]
    if content_type == "review_pull_quote" and widgets.get("graphic_formats"):
        formats = widgets["graphic_formats"].value or []
        return list(formats) if isinstance(formats, list) else [str(formats)]
    if content_type == "character_spotlight" and widgets.get("format"):
        return formats_for_character_spotlight(widgets["format"].value)
    if content_type == "blog_post":
        return ["Blog Cover (16:9)", "Blog Square (1:1)", "Blog Horizontal (1.91:1)"]
    return ["Instagram Post (4:5)"]


def campaign_graphic_theme(content_type: str, widgets: dict[str, object]) -> str:
    if content_type in {"quote_post", "review_pull_quote"} and widgets.get("graphic_theme"):
        return str(widgets["graphic_theme"].value or "Gothic")
    if content_type == "blog_post":
        return "Press"
    return "Gothic"


def render_campaign_quote_graphic(
    *,
    content_type: str,
    content: str,
    widgets: dict[str, object],
    campaign_topic: str,
    asset_number: int,
) -> dict[str, object]:
    quote = extract_quote_for_graphic(content)
    if not quote and content_type in {"character_spotlight", "blog_post"}:
        quote = first_nonempty_line(content) or campaign_topic
    if not quote:
        return {}
    output_dir = PROJECT_ROOT / "outputs" / "campaign_quote_graphics" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_stem = slugify_filename(f"{content_type}_{asset_number}_{campaign_topic}", "campaign_quote")
    try:
        return render_quote_cards(
            quote=quote,
            attribution=attribution_for_campaign_asset(content_type, widgets),
            format_labels=campaign_graphic_formats(content_type, widgets),
            brand_title="Mortal Vengeance",
            output_dir=output_dir,
            file_stem=file_stem,
            theme_name=campaign_graphic_theme(content_type, widgets),
            character_name=normalize_selected(
                widgets.get("characters").value
                if widgets.get("characters")
                else widgets.get("character").value
                if widgets.get("character")
                else ""
            ),
        )
    except Exception as exc:
        return {"error": str(exc), "quote": quote}


def campaign_asset_preview_html(asset: dict[str, object]) -> str:
    content_type = str(asset.get("content_type") or "")
    label = escape(str(asset.get("label") or content_type))
    content = str(asset.get("content") or "")
    style = escape(str(asset.get("style") or "default"))
    quantity_label = escape(str(asset.get("asset_label") or "Asset"))
    escaped_content = escape(content)
    fields = asset.get("preview_fields") if isinstance(asset.get("preview_fields"), dict) else {}
    quote_graphic = asset.get("quote_graphic") if isinstance(asset.get("quote_graphic"), dict) else {}

    if content_type in {"instagram_caption", "youtube_content", "linkedin_content", "character_spotlight"}:
        preview = build_preview_html(content_type, {"draft": content, **fields})
    elif content_type in {"quote_post", "review_pull_quote"}:
        quote_text = extract_quote_for_graphic(content)
        preview = build_quote_preview_html(
            "quote_post" if content_type == "quote_post" else "review_pull_quote",
            {
                "quote_text": quote_text,
                "quote_source": quote_graphic.get("zip_path") or "Campaign generated quote",
                "quote_moods": fields.get("quote_moods", ""),
                "quote_characters": fields.get("quote_characters", ""),
                "quote_spoiler": "Campaign asset",
                "quote_request": style,
                "review_quote_text": quote_text,
                "review_source": "Campaign generated from review grounding",
                "review_attribution": attribution_for_campaign_asset(content_type, asset.get("widgets", {}) if isinstance(asset.get("widgets"), dict) else {}),
                "review_mode": style,
                "review_promo_line": "Mortal Vengeance",
            },
        )
        paths = quote_graphic.get("paths") or []
        first_path = str(paths[0]) if paths else ""
        if first_path and Path(first_path).exists():
            preview = f"""
            {preview}
            <div style="margin-top:14px;border-radius:16px;overflow:hidden;border:1px solid rgba(142,31,47,.18);">
                <img src="{escape(image_data_url(first_path), quote=True)}" alt="Generated quote graphic" style="width:100%;display:block;">
            </div>
            <div style="font-size:12px;color:#6b6b6b;margin-top:8px;"><strong>Graphic:</strong> {escape(first_path)}</div>
            <div style="font-size:12px;color:#6b6b6b;"><strong>Package:</strong> {escape(str(quote_graphic.get("zip_path") or ""))}</div>
            """
        elif quote_graphic.get("error"):
            preview = f'{preview}<div style="font-size:12px;color:#8E1F2F;margin-top:8px;">Graphic render note: {escape(str(quote_graphic.get("error")))}</div>'
    else:
        preview = f"""
        <div style="border-radius:18px;background:#fff;border:1px solid rgba(44,24,31,.12);box-shadow:0 14px 30px rgba(61,31,41,.07);padding:18px;">
            <div style="font-size:13px;line-height:1.55;white-space:pre-wrap;color:#2f292b;">{escaped_content}</div>
        </div>
        """

    return f"""
    <article style="border-radius:22px;background:#fffdfb;border:1px solid rgba(44,24,31,.13);box-shadow:0 18px 36px rgba(61,31,41,.08);padding:18px;margin-bottom:18px;">
        <div style="display:flex;gap:12px;align-items:flex-start;justify-content:space-between;margin-bottom:14px;">
            <div>
                <div style="font-family:Georgia,serif;font-size:22px;font-weight:800;color:#171217;">{label}</div>
                <div style="font-size:12px;color:#71686a;">{quantity_label} · {style}</div>
            </div>
            <div style="border-radius:999px;background:#f8eceb;color:#8E1F2F;padding:6px 10px;font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;">{escape(content_type)}</div>
        </div>
        {preview}
        <details style="margin-top:14px;">
            <summary style="cursor:pointer;color:#8E1F2F;font-weight:800;">Generated text</summary>
            <pre style="white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:12px;line-height:1.5;background:#fbf7f4;border:1px solid rgba(44,24,31,.10);border-radius:12px;padding:12px;overflow:auto;">{escaped_content}</pre>
        </details>
    </article>
    """


def build_campaign_preview_html(assets: list[dict[str, object]]) -> str:
    if not assets:
        return """
        <div style="border-radius:20px;padding:20px;background:#ffffff;border:1px dashed rgba(17,17,20,.16);color:#6b6b6b;">
            Campaign previews will appear here separated by content type.
        </div>
        """
    grouped: dict[str, list[dict[str, object]]] = {}
    for asset in assets:
        grouped.setdefault(str(asset.get("content_type") or "content"), []).append(asset)
    sections = []
    for content_type, content_assets in grouped.items():
        label = CAMPAIGN_FORMAT_LABELS.get(content_type, content_type)
        cards = "\n".join(campaign_asset_preview_html(asset) for asset in content_assets)
        sections.append(
            f"""
            <section style="margin-bottom:22px;">
                <div style="font-family:Georgia,serif;font-size:24px;font-weight:800;margin:0 0 12px;color:#171217;">{escape(label)}</div>
                {cards}
            </section>
            """
        )
    return "\n".join(sections)


async def generate_draft_from_fields(
    *,
    content_type: str,
    topic,
    related_book,
    platform,
    social_objectives,
    audience,
    cta,
    constraints,
    quote_book,
    quote_moods,
    character_tags,
    podcast_format,
    podcast_speakers,
    podcast_roles,
    podcast_tone,
    podcast_length,
    elevenlabs_model,
    elevenlabs_voice_ids,
    podcast_show_title,
    podcast_episode_title,
    podcast_episode_number,
    blog_length,
    blog_format,
    blog_sections,
    blog_structure_options,
    blog_seo_keywords,
    blog_image_mode,
    blog_character,
    instagram_formats,
    instagram_hashtags,
    instagram_hook,
    cs_character,
    cs_focus,
    cs_platform_format,
    cs_image_mode,
    nl_subject,
    nl_preview,
    nl_structure,
    pr_timing,
    pr_embargo_date,
    pr_city,
    pr_state,
    pr_release_date,
    pr_contact_name,
    pr_contact_title,
    pr_organization,
    pr_contact_email,
    pr_contact_phone,
    pr_website,
    pr_news_angle,
    pr_release_goal,
    pr_primary_announcement,
    pr_supporting_proof,
    pr_quote_source,
    pr_target_media,
    pr_required_assets,
    status,
    filtered_context_path,
    prompt_path,
    draft_path,
    generated_output,
    visual_format_labels=None,
    visual_theme_name: str = "",
    saved_draft_path=None,
    structured_brief_output=None,
    uploaded_image_path: str = "",
    generated_visual_path=None,
    generated_visual_package_path=None,
) -> None:
    status.value = "Generating..."
    generated_output.value = ""
    filtered_context_path.value = ""
    prompt_path.value = ""
    draft_path.value = ""
    if saved_draft_path is not None:
        saved_draft_path.value = ""
    if generated_visual_path is not None:
        generated_visual_path.value = ""
    if generated_visual_package_path is not None:
        generated_visual_package_path.value = ""

    brief = build_brief(
        content_type=content_type,
        topic=topic,
        related_book=related_book,
        platform=platform,
        social_objectives=social_objectives,
        audience=audience,
        cta=cta,
        constraints=constraints,
        quote_book=quote_book,
        quote_moods=quote_moods,
        character_tags=character_tags,
        podcast_format=podcast_format,
        podcast_speakers=podcast_speakers,
        podcast_roles=podcast_roles,
        podcast_tone=podcast_tone,
        podcast_length=podcast_length,
        elevenlabs_model=elevenlabs_model,
        elevenlabs_voice_ids=elevenlabs_voice_ids,
        podcast_show_title=podcast_show_title,
        podcast_episode_title=podcast_episode_title,
        podcast_episode_number=podcast_episode_number,
        blog_length=blog_length,
        blog_format=blog_format,
        blog_sections=blog_sections,
        blog_structure_options=blog_structure_options,
        blog_seo_keywords=blog_seo_keywords,
        blog_image_mode=blog_image_mode,
        blog_character=blog_character,
        instagram_formats=instagram_formats,
        instagram_hashtags=instagram_hashtags,
        instagram_hook=instagram_hook,
        cs_character=cs_character,
        cs_focus=cs_focus,
        cs_platform_format=cs_platform_format,
        cs_image_mode=cs_image_mode,
        nl_subject=nl_subject,
        nl_preview=nl_preview,
        nl_structure=nl_structure,
        pr_timing=pr_timing,
        pr_embargo_date=pr_embargo_date,
        pr_city=pr_city,
        pr_state=pr_state,
        pr_release_date=pr_release_date,
        pr_contact_name=pr_contact_name,
        pr_contact_title=pr_contact_title,
        pr_organization=pr_organization,
        pr_contact_email=pr_contact_email,
        pr_contact_phone=pr_contact_phone,
        pr_website=pr_website,
        pr_news_angle=pr_news_angle,
        pr_release_goal=pr_release_goal,
        pr_primary_announcement=pr_primary_announcement,
        pr_supporting_proof=pr_supporting_proof,
        pr_quote_source=pr_quote_source,
        pr_target_media=pr_target_media,
        pr_required_assets=pr_required_assets,
        uploaded_image_path=uploaded_image_path,
    )
    if structured_brief_output is not None:
        structured_brief_output.value = brief

    try:
        if content_type in IMAGE_AWARE_CONTENT_TYPES and uploaded_image_path:
            result = await asyncio.to_thread(
                run_pipeline_with_image,
                content_type=content_type,
                topic=brief,
                image_path=uploaded_image_path,
            )
        else:
            result = await asyncio.to_thread(
                run_pipeline,
                content_type=content_type,
                topic=brief,
            )
    except Exception as exc:
        status.value = f"Generation failed: {exc}"
        ui.notify(status.value, type="negative")
        return

    status.value = "Draft ready"
    filtered_context_path.value = str(result["filtered_context_path"])
    prompt_path.value = str(result["prompt_path"])
    draft_path.value = str(result["draft_path"])
    generated_output.value = result["generated_content"]
    character_for_visual = normalize_selected(
        cs_character
        if content_type == "character_spotlight"
        else blog_character
        if content_type == "blog_post" and "character portrait" in str(blog_image_mode or "").lower()
        else character_tags
        if content_type == "quote_post"
        else ""
    )
    visual_package = (
        {}
        if content_type == "quote_post"
        else render_generated_visual_package(
            content_type=content_type,
            content=result["generated_content"],
            topic=brief,
            character_name="" if character_for_visual == "Not specified" else character_for_visual,
            format_labels=visual_format_labels,
            theme_name=visual_theme_name,
        )
    )
    if generated_visual_path is not None:
        paths = visual_package.get("paths") if isinstance(visual_package, dict) else []
        generated_visual_path.value = str(paths[0]) if paths else str(visual_package.get("error", "") if isinstance(visual_package, dict) else "")
    if generated_visual_package_path is not None and isinstance(visual_package, dict):
        generated_visual_package_path.value = str(visual_package.get("zip_path") or "")
    topic_lines = str(topic or "").strip().splitlines()
    draft_title = (
        str(podcast_episode_title or "").strip()
        or str(podcast_show_title or "").strip()
        or (topic_lines[0][:90] if topic_lines else "")
        or f"{content_type} draft"
    )
    draft_record = save_draft(
        title=draft_title,
        content_type=content_type,
        content=result["generated_content"],
        source_path=result["draft_path"],
        metadata={
            "topic": str(topic or "").strip(),
            "related_book": normalize_selected(related_book),
            "brief": brief,
            "platform": normalize_selected(platform),
            "audience": normalize_selected(audience),
            "constraints": normalize_selected(constraints),
            "uploaded_image_path": uploaded_image_path,
        },
    )
    if saved_draft_path is not None:
        saved_draft_path.value = draft_record["path"]
    ui.notify("Content generated successfully.", type="positive")




NICEGUI_APP_CSS = r"""
:root {
    --mce-bg: #f8f3ee;
    --mce-bg-2: #fbf8f4;
    --mce-card: rgba(255, 255, 255, 0.94);
    --mce-card-solid: #fffdfb;
    --mce-ink: #171217;
    --mce-muted: #71686a;
    --mce-border: rgba(44, 24, 31, 0.13);
    --mce-border-strong: rgba(123, 21, 48, 0.25);
    --mce-accent: #8e1f2f;
    --mce-accent-dark: #4d1020;
    --mce-accent-soft: #f7e4e7;
    --mce-gold: #ead5ad;
    --mce-success: #2e7d4f;
    --mce-shadow: 0 22px 70px rgba(41, 24, 31, 0.11);
    --mce-shadow-soft: 0 12px 36px rgba(41, 24, 31, 0.08);
}

html, body, #app, .q-layout, .q-page, .nicegui-content {
    min-height: 100%;
    background:
        radial-gradient(900px 420px at 7% 0%, rgba(142, 31, 47, 0.13), transparent 56%),
        radial-gradient(780px 360px at 96% 4%, rgba(234, 213, 173, 0.22), transparent 52%),
        linear-gradient(180deg, var(--mce-bg-2) 0%, var(--mce-bg) 100%) !important;
    color: var(--mce-ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.nicegui-content {
    padding: 0 !important;
    max-width: none !important;
}

.mce-shell {
    width: 100%;
    max-width: 1320px;
    margin: 0 auto;
    padding: 34px 28px 56px;
    gap: 22px;
}

.mce-hero {
    position: relative;
    isolation: isolate;
    width: 100%;
    min-height: 154px;
    overflow: hidden;
    border-radius: 30px;
    padding: 30px 34px;
    color: #fffaf5;
    background:
        radial-gradient(620px 260px at 84% 8%, rgba(255, 255, 255, 0.14), transparent 50%),
        radial-gradient(420px 220px at 7% 94%, rgba(234, 213, 173, 0.16), transparent 58%),
        linear-gradient(135deg, #181116 0%, #441321 52%, #9c1f38 100%);
    box-shadow: var(--mce-shadow);
}

.mce-hero::after {
    content: "";
    position: absolute;
    inset: auto -80px -150px auto;
    width: 360px;
    height: 360px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    opacity: 0.7;
    z-index: -1;
}

.mce-kicker {
    font-size: 11px;
    line-height: 1;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    opacity: 0.74;
    font-weight: 850;
}

.mce-hero-title {
    margin-top: 17px;
    max-width: 880px;
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(25px, 3.1vw, 41px);
    line-height: 1.04;
    font-weight: 850;
    letter-spacing: -0.045em;
}

.mce-hero-subtitle {
    margin-top: 12px;
    max-width: 820px;
    color: rgba(255, 255, 255, 0.78);
    font-size: 15px;
    line-height: 1.5;
}

.mce-tabs-wrap {
    width: 100%;
    display: flex;
    justify-content: center;
    margin-top: 2px;
}

.mce-tabs {
    width: auto;
    background: rgba(255, 255, 255, 0.42);
    border: 1px solid var(--mce-border);
    border-radius: 999px;
    padding: 4px;
    box-shadow: 0 8px 26px rgba(41, 24, 31, 0.05);
}

.mce-tabs .q-tab {
    min-height: 40px;
    border-radius: 999px;
    padding: 0 18px;
    color: var(--mce-muted);
    font-size: 12px;
    font-weight: 850;
    letter-spacing: 0.04em;
}

.mce-tabs .q-tab--active {
    color: var(--mce-accent) !important;
    background: #fff;
    box-shadow: 0 8px 18px rgba(44, 24, 31, 0.06);
}

.mce-tabs .q-tab__indicator {
    display: none;
}

.mce-tab-panels,
.mce-panel,
.q-tab-panel {
    background: transparent !important;
    box-shadow: none !important;
}

.mce-panel {
    padding: 0 !important;
}

.mce-grid {
    width: 100%;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(360px, 430px);
    gap: 24px;
    align-items: start;
}

.mce-card {
    width: 100%;
    min-width: 0;
    border: 1px solid var(--mce-border);
    border-radius: 26px;
    background: var(--mce-card);
    box-shadow: var(--mce-shadow-soft);
    padding: 24px;
    backdrop-filter: blur(10px);
}

.mce-sticky {
    position: sticky;
    top: 22px;
    max-height: calc(100vh - 44px);
    overflow: auto;
}

.mce-card-title {
    margin: 0;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 25px;
    line-height: 1.08;
    font-weight: 850;
    letter-spacing: -0.04em;
    color: var(--mce-ink);
}

.mce-card-subtitle {
    margin-top: 8px;
    color: var(--mce-muted);
    font-size: 13px;
    line-height: 1.45;
}

.mce-section-title {
    color: var(--mce-accent-dark);
    font-size: 12px;
    font-weight: 900;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}

.mce-stack {
    gap: 14px;
}

.mce-stack .q-field,
.mce-stack .q-select,
.mce-stack .q-textarea,
.mce-stack .q-input {
    width: 100%;
}

.mce-stack .q-field__control {
    border-radius: 15px !important;
    background: rgba(255, 255, 255, 0.72) !important;
}

.mce-stack .q-field__native,
.mce-stack .q-field__input,
.mce-stack textarea,
.mce-stack input {
    color: var(--mce-ink) !important;
}

.mce-stack textarea {
    resize: vertical;
}

.mce-subcard {
    width: 100%;
    border: 1px solid var(--mce-border);
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.68);
    box-shadow: none;
    padding: 16px;
}

.mce-subcard .q-card__section {
    padding: 0;
}

.mce-two-col {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    width: 100%;
}

.mce-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    padding-top: 6px;
}

.mce-actions .q-btn,
.q-btn.mce-primary,
.q-btn.mce-secondary {
    min-height: 42px;
    border-radius: 13px;
    font-weight: 850;
    letter-spacing: 0.01em;
}

.q-btn.mce-primary,
.mce-primary {
    background: linear-gradient(180deg, var(--mce-accent), var(--mce-accent-dark)) !important;
    color: white !important;
    box-shadow: 0 12px 24px rgba(142, 31, 47, 0.26);
}

.q-btn.mce-secondary,
.mce-secondary {
    background: rgba(142, 31, 47, 0.08) !important;
    color: var(--mce-accent-dark) !important;
    border: 1px solid rgba(142, 31, 47, 0.16) !important;
}

.mce-primary:hover,
.mce-secondary:hover {
    transform: translateY(-1px);
}

.mce-preview {
    width: 100%;
    min-height: 150px;
    max-height: 470px;
    overflow: auto;
    border: 1px dashed var(--mce-border-strong);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.64);
    padding: 14px;
}

.mce-output-textarea textarea {
    min-height: 230px !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 12.5px !important;
    line-height: 1.58 !important;
}

.mce-script-textarea textarea {
    min-height: 320px !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 12.8px !important;
    line-height: 1.62 !important;
}

.mce-path-field input,
.mce-path-field textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 12px !important;
}

.mce-expansion {
    width: 100%;
    border: 1px solid var(--mce-border);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.58);
    overflow: hidden;
}

.mce-expansion .q-item {
    min-height: 48px;
    color: var(--mce-accent-dark);
    font-weight: 850;
}

.mce-note {
    padding: 12px 14px;
    border: 1px solid rgba(142, 31, 47, 0.14);
    border-radius: 16px;
    background: #fff8f8;
    color: #5d3a40;
    font-size: 13px;
    line-height: 1.45;
}

.mce-status-row {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
}

.mce-status-chip {
    padding: 12px 14px;
    border: 1px solid var(--mce-border);
    border-radius: 16px;
    background: #fff;
}

.mce-status-chip small {
    display: block;
    color: var(--mce-muted);
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.mce-status-chip strong {
    color: var(--mce-success);
}

.mce-audio {
    width: 100%;
    border-radius: 999px;
}

.mce-divider {
    height: 1px;
    width: 100%;
    margin: 4px 0;
    background: var(--mce-border);
}

.mce-muted {
    color: var(--mce-muted);
}

@media (max-width: 1100px) {
    .mce-grid {
        grid-template-columns: 1fr;
    }

    .mce-sticky {
        position: static;
        max-height: none;
    }
}

@media (max-width: 720px) {
    .mce-shell {
        padding: 20px 13px 42px;
    }

    .mce-hero {
        border-radius: 22px;
        padding: 24px;
    }

    .mce-tabs-wrap {
        justify-content: stretch;
    }

    .mce-tabs {
        width: 100%;
        border-radius: 18px;
    }

    .mce-tabs .q-tabs__content {
        flex-wrap: wrap;
    }

    .mce-card {
        border-radius: 22px;
        padding: 18px;
    }

    .mce-two-col,
    .mce-status-row {
        grid-template-columns: 1fr;
    }
}
"""


def install_nicegui_theme() -> None:
    ui.colors(primary="#8E1F2F", secondary="#111114", accent="#EAD5AD", positive="#2E7D4F")
    ui.add_head_html(f"<style>{NICEGUI_APP_CSS}</style>")


def make_primary_button(label: str, on_click):
    return ui.button(label, on_click=on_click).props("unelevated no-caps").classes("mce-primary")


def make_secondary_button(label: str, on_click):
    return ui.button(label, on_click=on_click).props("unelevated no-caps").classes("mce-secondary")


def section_heading(title: str, subtitle: str = "") -> None:
    ui.label(title).classes("mce-card-title")
    if subtitle:
        ui.label(subtitle).classes("mce-card-subtitle")
    ui.element("div").classes("mce-divider")


def subcard_heading(title: str) -> None:
    ui.label(title).classes("mce-section-title")


def apply_field_props(component, props: str = "outlined dense"):
    return component.props(props).classes("w-full")


def readonly_input(label: str, value: str = ""):
    return ui.input(label=label, value=value).props("readonly outlined dense").classes("w-full mce-path-field")


def readonly_textarea(label: str, value: str = ""):
    return ui.textarea(label=label, value=value).props("readonly outlined autogrow").classes("w-full mce-output-textarea")


@ui.page("/")
def index() -> None:
    install_nicegui_theme()

    content_types = list_supported_content_types()
    generator_content_types = [content_type for content_type in content_types if content_type != "podcast"]
    default_content_type = "blog_post" if "blog_post" in generator_content_types else generator_content_types[0]

    with ui.column().classes("mce-shell"):
        with ui.column().classes("mce-hero"):
            ui.label("MYTHOS CONTENT ENGINE").classes("mce-kicker")
            ui.label("A cleaner local workspace for briefs, drafts, and podcast production.").classes("mce-hero-title")
            ui.label(
                "NiceGUI powers the responsive studio shell while the content pipeline stays local, traceable, and fast."
            ).classes("mce-hero-subtitle")

        with ui.row().classes("mce-tabs-wrap"):
            with ui.tabs().classes("mce-tabs") as tabs:
                generator_tab = ui.tab("Generator")
                campaign_tab = ui.tab("Campaign Mode")
                podcast_tab = ui.tab("Podcast Studio")
                saved_drafts_tab = ui.tab("Saved Drafts")

        with ui.tab_panels(tabs, value=generator_tab).classes("w-full mce-tab-panels"):
            with ui.tab_panel(generator_tab).classes("mce-panel"):
                with ui.element("section").classes("mce-grid"):
                    with ui.card().classes("mce-card"):
                        section_heading("Content Brief", "Shape the prompt, then generate a draft with the existing pipeline.")
                        with ui.column().classes("mce-stack w-full"):
                            content_type = apply_field_props(
                                ui.select(generator_content_types, value=default_content_type, label="Content type"),
                            )
                            topic = apply_field_props(
                                ui.textarea(label="Topic", placeholder="What should this content be about?"),
                                "outlined autogrow",
                            )
                            related_book = apply_field_props(
                                ui.select(
                                    QUOTE_BOOK_OPTIONS,
                                    value=QUOTE_BOOK_OPTIONS[0] if QUOTE_BOOK_OPTIONS else None,
                                    label="Related book/source (optional)",
                                ),
                            )
                            platform = apply_field_props(
                                ui.select(PLATFORM_OPTIONS, value=PLATFORM_OPTIONS[0], label="Platform / destination"),
                            )
                            social_objectives = apply_field_props(
                                ui.select(SOCIAL_OBJECTIVES, multiple=True, label="Social objectives"),
                                "outlined dense use-chips clearable",
                            )
                            audience = apply_field_props(
                                ui.select(AUDIENCE_OPTIONS, multiple=True, label="Audience"),
                                "outlined dense use-chips clearable",
                            )
                            cta = apply_field_props(
                                ui.input(label="CTA / next step", placeholder="What should the reader do next?"),
                            )
                            constraints = apply_field_props(
                                ui.select(CONSTRAINT_OPTIONS, multiple=True, label="Constraints"),
                                "outlined dense use-chips clearable",
                            )

                            preview_refresh: dict[str, object] = {"fn": None}

                            async def save_social_image(event, path_input, status_label, platform_name: str) -> None:
                                suffix = Path(event.file.name or "").suffix.lower()
                                if suffix not in ALLOWED_IMAGE_EXTENSIONS:
                                    status_label.set_text("Use PNG, JPG, JPEG, or WEBP.")
                                    ui.notify("Please upload a PNG, JPG, JPEG, or WEBP image.", type="warning")
                                    return
                                IMAGE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                                destination = IMAGE_UPLOAD_DIR / safe_upload_filename(event.file.name)
                                await event.file.save(destination)
                                path_input.value = str(destination)
                                status_label.set_text(f"{platform_name} image ready: {event.file.name}")
                                refresh = preview_refresh.get("fn")
                                if callable(refresh):
                                    refresh()
                                ui.notify(f"{platform_name} image uploaded.", type="positive")

                            with ui.card().classes("mce-subcard") as instagram_card:
                                subcard_heading("Instagram Settings")
                                instagram_formats = apply_field_props(
                                    ui.select(INSTAGRAM_FORMAT_OPTIONS, multiple=True, label="Instagram formats"),
                                    "outlined dense use-chips clearable",
                                )
                                instagram_hashtags = apply_field_props(
                                    ui.select(INSTAGRAM_HASHTAG_OPTIONS, label="Instagram hashtags"),
                                )
                                instagram_hook = apply_field_props(
                                    ui.textarea(label="Instagram hook", placeholder="Opening hook or callout"),
                                    "outlined autogrow",
                                )
                                instagram_generate_images = apply_field_props(
                                    ui.select(["yes", "no"], value="yes", label="Generate image package"),
                                )
                                instagram_visual_formats = apply_field_props(
                                    ui.select(INSTAGRAM_VISUAL_FORMAT_OPTIONS, multiple=True, label="Image type / format"),
                                    "outlined dense use-chips clearable",
                                )
                                instagram_visual_style = apply_field_props(
                                    ui.select(QUOTE_IMAGE_STYLE_OPTIONS, value="Gothic", label="Visual style"),
                                )
                                instagram_hook_status = ui.label("").classes("mce-muted")

                                async def suggest_instagram_hook() -> None:
                                    instagram_hook_status.set_text("Suggesting hook...")
                                    prompt = f"""Suggest one short Instagram hook / first line for Mythos Content Engine.

Topic: {topic.value}
Related book/source: {related_book.value}
Selected formats: {normalize_selected(instagram_formats.value)}
Audience: {normalize_selected(audience.value)}
Objective: {normalize_selected(social_objectives.value)}

Return only the hook. Keep it under 14 words. Make it bookish, specific, and non-generic."""
                                    try:
                                        instagram_hook.value = (await asyncio.to_thread(generate_text, prompt)).strip().splitlines()[0][:120]
                                        instagram_hook_status.set_text("Hook suggested.")
                                    except Exception as exc:
                                        instagram_hook_status.set_text(f"Could not suggest hook: {exc}")

                                make_secondary_button("Suggest Hook", suggest_instagram_hook)
                                instagram_image_path = readonly_input("Instagram image", "")
                                instagram_image_status = ui.label("No image selected").classes("mce-muted")

                                async def handle_instagram_image_upload(event) -> None:
                                    await save_social_image(event, instagram_image_path, instagram_image_status, "Instagram")

                                ui.upload(on_upload=handle_instagram_image_upload, auto_upload=True).props(
                                    'accept=".png,.jpg,.jpeg,.webp" max-files=1 label="Upload Instagram image"'
                                ).classes("w-full")

                            with ui.card().classes("mce-subcard") as linkedin_card:
                                subcard_heading("LinkedIn Settings")
                                linkedin_formats = apply_field_props(
                                    ui.select(LINKEDIN_CONTENT_FORMAT_OPTIONS, multiple=True, label="LinkedIn formats"),
                                    "outlined dense use-chips clearable",
                                )
                                linkedin_angle = apply_field_props(
                                    ui.textarea(label="LinkedIn angle", placeholder="Professional angle or framing"),
                                    "outlined autogrow",
                                )
                                linkedin_cta = apply_field_props(
                                    ui.input(label="LinkedIn CTA", placeholder="Connect, comment, click, or follow-up"),
                                )
                                linkedin_image_path = readonly_input("LinkedIn image", "")
                                linkedin_image_status = ui.label("No image selected").classes("mce-muted")

                                async def handle_linkedin_image_upload(event) -> None:
                                    await save_social_image(event, linkedin_image_path, linkedin_image_status, "LinkedIn")

                                ui.upload(on_upload=handle_linkedin_image_upload, auto_upload=True).props(
                                    'accept=".png,.jpg,.jpeg,.webp" max-files=1 label="Upload LinkedIn image"'
                                ).classes("w-full")

                            with ui.card().classes("mce-subcard") as youtube_card:
                                subcard_heading("YouTube Settings")
                                youtube_formats = apply_field_props(
                                    ui.select(YOUTUBE_CONTENT_FORMAT_OPTIONS, multiple=True, label="YouTube formats"),
                                    "outlined dense use-chips clearable",
                                )
                                youtube_keywords = apply_field_props(
                                    ui.input(label="YouTube keywords", placeholder="Primary search keywords"),
                                )
                                youtube_link = apply_field_props(
                                    ui.input(label="Video or channel link", placeholder="Optional URL to include"),
                                )
                                youtube_image_path = readonly_input("YouTube image", "")
                                youtube_image_status = ui.label("No image selected").classes("mce-muted")

                                async def handle_youtube_image_upload(event) -> None:
                                    await save_social_image(event, youtube_image_path, youtube_image_status, "YouTube")

                                ui.upload(on_upload=handle_youtube_image_upload, auto_upload=True).props(
                                    'accept=".png,.jpg,.jpeg,.webp" max-files=1 label="Upload YouTube image"'
                                ).classes("w-full")

                            with ui.card().classes("mce-subcard") as quote_card:
                                subcard_heading("Quote Post Settings")
                                quote_book = apply_field_props(
                                    ui.select(QUOTE_BOOK_OPTIONS, value=QUOTE_BOOK_OPTIONS[0] if QUOTE_BOOK_OPTIONS else None, label="Quote book/source"),
                                )
                                quote_moods = apply_field_props(
                                    ui.select(QUOTE_MOOD_TAGS, multiple=True, label="Quote mood/category tags"),
                                    "outlined dense use-chips clearable",
                                )
                                character_tags = apply_field_props(
                                    ui.select(CHARACTER_TAGS, multiple=True, label="Character tags"),
                                    "outlined dense use-chips clearable",
                                )
                                quote_visual_formats = apply_field_props(
                                    ui.select(quote_graphic_format_options(), multiple=True, label="Quote image destination formats"),
                                    "outlined dense use-chips clearable",
                                )
                                quote_visual_style = apply_field_props(
                                    ui.select(QUOTE_IMAGE_STYLE_OPTIONS, value="Gothic", label="Quote image color style"),
                                )

                            with ui.card().classes("mce-subcard") as blog_card:
                                subcard_heading("Blog Post Settings")
                                blog_length = apply_field_props(
                                    ui.select(BLOG_LENGTH_OPTIONS, value=BLOG_LENGTH_OPTIONS[1] if len(BLOG_LENGTH_OPTIONS) > 1 else BLOG_LENGTH_OPTIONS[0], label="Blog length"),
                                )
                                blog_format = apply_field_props(
                                    ui.select(BLOG_FORMAT_OPTIONS, value=BLOG_FORMAT_OPTIONS[0], label="Blog format"),
                                )
                                blog_structure_options = apply_field_props(
                                    ui.select(BLOG_STRUCTURE_CHECKLIST_OPTIONS, multiple=True, label="Structure checklist"),
                                    "outlined dense use-chips clearable",
                                )
                                blog_sections = apply_field_props(
                                    ui.textarea(label="Optional structure notes", placeholder="Add any specific order, angle, or section notes"),
                                    "outlined autogrow",
                                )
                                blog_structure_status = ui.label("").classes("mce-muted")

                                async def suggest_blog_structure() -> None:
                                    blog_structure_status.set_text("Suggesting structure...")
                                    prompt = f"""Suggest a concise blog structure for Mythos Content Engine.

Topic: {topic.value}
Related book/source: {related_book.value}
Blog format: {blog_format.value}
Length: {blog_length.value}
Selected checklist: {normalize_selected(blog_structure_options.value)}

Return only a practical section outline with 5-8 bullets."""
                                    try:
                                        blog_sections.value = await asyncio.to_thread(generate_text, prompt)
                                        blog_structure_status.set_text("Structure suggested.")
                                    except Exception as exc:
                                        blog_structure_status.set_text(f"Could not suggest structure: {exc}")

                                make_secondary_button("Suggest Structure", suggest_blog_structure)
                                blog_seo_keywords = apply_field_props(
                                    ui.input(label="Blog SEO keywords", placeholder="Comma-separated keywords"),
                                )
                                blog_image_mode = apply_field_props(
                                    ui.select(["use uploaded image", "use selected character portrait", "text-only / quote graphic"], value="text-only / quote graphic", label="Blog image mode"),
                                )
                                blog_character = apply_field_props(
                                    ui.select(CHARACTER_TAGS, label="Selected character portrait"),
                                )
                                blog_visual_formats = apply_field_props(
                                    ui.select(BLOG_VISUAL_FORMAT_OPTIONS, multiple=True, label="Blog image formats"),
                                    "outlined dense use-chips clearable",
                                )
                                blog_visual_style = apply_field_props(
                                    ui.select(QUOTE_IMAGE_STYLE_OPTIONS, value="Press", label="Blog visual style"),
                                )

                                def sync_blog_visual_defaults(event=None) -> None:
                                    if blog_format.value == "photo/gallery slideshow" and not blog_visual_formats.value:
                                        blog_visual_formats.value = BLOG_GALLERY_VISUAL_FORMAT_OPTIONS

                                blog_format.on_value_change(sync_blog_visual_defaults)
                                blog_image_path = readonly_input("Blog image", "")
                                blog_image_status = ui.label("No image selected").classes("mce-muted")

                                async def handle_blog_image_upload(event) -> None:
                                    await save_social_image(event, blog_image_path, blog_image_status, "Blog")

                                ui.upload(on_upload=handle_blog_image_upload, auto_upload=True).props(
                                    'accept=".png,.jpg,.jpeg,.webp" max-files=1 label="Upload blog image"'
                                ).classes("w-full")

                            with ui.card().classes("mce-subcard") as newsletter_card:
                                subcard_heading("Newsletter Settings")
                                nl_subject = apply_field_props(
                                    ui.input(label="Newsletter subject line", placeholder="What is the subject line?"),
                                )
                                nl_preview = apply_field_props(
                                    ui.textarea(label="Preview text", placeholder="Short preview text shown in inboxes"),
                                    "outlined autogrow",
                                )
                                nl_structure = apply_field_props(
                                    ui.select(
                                        ["announcement", "roundup / digest", "personal note", "behind-the-scenes"],
                                        label="Newsletter structure",
                                    ),
                                )

                            with ui.card().classes("mce-subcard") as character_card:
                                subcard_heading("Character Spotlight Settings")
                                cs_character = apply_field_props(
                                    ui.select(CHARACTER_TAGS, label="Character spotlight subject"),
                                )
                                cs_platform_format = apply_field_props(
                                    ui.select(CHARACTER_SPOTLIGHT_PLATFORM_FORMATS, value=CHARACTER_SPOTLIGHT_PLATFORM_FORMATS[0], label="Destination format"),
                                )
                                cs_image_mode = apply_field_props(
                                    ui.select(CHARACTER_IMAGE_MODE_OPTIONS, value=CHARACTER_IMAGE_MODE_OPTIONS[1], label="Image source"),
                                )
                                cs_focus = apply_field_props(
                                    ui.textarea(label="Spotlight angle / focus", placeholder="Why this character matters, what to highlight"),
                                    "outlined autogrow",
                                )
                                cs_image_path = readonly_input("Character spotlight image", "")
                                cs_image_status = ui.label("No image selected").classes("mce-muted")

                                async def handle_character_image_upload(event) -> None:
                                    await save_social_image(event, cs_image_path, cs_image_status, "Character Spotlight")

                                ui.upload(on_upload=handle_character_image_upload, auto_upload=True).props(
                                    'accept=".png,.jpg,.jpeg,.webp" max-files=1 label="Upload character spotlight image"'
                                ).classes("w-full")

                            with ui.card().classes("mce-subcard") as press_release_card:
                                subcard_heading("Press Release Settings")
                                pr_profile = apply_field_props(
                                    ui.select(press_profile_options(), label="Saved press profile"),
                                    "outlined dense clearable",
                                )
                                pr_profile_name = apply_field_props(
                                    ui.input(label="Profile name", placeholder="Author / publisher profile name"),
                                )
                                pr_destination = readonly_input("Distribution", PRESS_RELEASE_DESTINATION)
                                with ui.element("div").classes("mce-two-col"):
                                    pr_timing = apply_field_props(ui.select(PRESS_RELEASE_TIMING_OPTIONS, label="Press release timing"))
                                    pr_embargo_date = apply_field_props(ui.input(label="Embargo date", placeholder="If embargoed, provide the date"))
                                    pr_city = apply_field_props(ui.input(label="Dateline city", placeholder="City for the dateline"))
                                    pr_state = apply_field_props(ui.input(label="Dateline state / region", placeholder="State or region"))
                                    pr_release_date = apply_field_props(ui.input(label="Release date").props("type=date"))
                                    pr_contact_name = apply_field_props(ui.input(label="Media contact name", placeholder="Contact person name"))
                                    pr_contact_title = apply_field_props(ui.input(label="Media contact title", placeholder="Contact person's title"))
                                    pr_organization = apply_field_props(ui.input(label="Organization / imprint", placeholder="Organization or imprint"))
                                    pr_contact_email = apply_field_props(ui.input(label="Media contact email", placeholder="contact@example.com"))
                                    pr_contact_phone = apply_field_props(ui.input(label="Media contact phone", placeholder="Phone number"))
                                pr_website = apply_field_props(ui.input(label="Website", placeholder="Organization website"))
                                pr_news_angle = apply_field_props(ui.textarea(label="News angle", placeholder="Why this is newsworthy"), "outlined autogrow")
                                pr_profile_actions = ui.row().classes("mce-actions")
                                pr_supporting_proof = apply_field_props(ui.textarea(label="Supporting proof", placeholder="Verified facts, awards, or evidence"), "outlined autogrow")
                                pr_quote_source = apply_field_props(ui.textarea(label="Quote source", placeholder="Approved spokesperson quote source"), "outlined autogrow")
                                pr_target_media = apply_field_props(ui.textarea(label="Target media", placeholder="Journalists, reviewers, outlets, or PR channels"), "outlined autogrow")
                                pr_required_assets = apply_field_props(
                                    ui.select(PR_ASSET_OPTIONS, multiple=True, label="Required PR assets to generate"),
                                    "outlined dense use-chips clearable",
                                )
                                pr_asset_status = readonly_input("Generated PR assets", "")

                            generator_actions = ui.row().classes("mce-actions")

                    with ui.card().classes("mce-card mce-sticky"):
                        section_heading("Output Studio", "Preview the draft, trace artifact paths, and compare against a clean baseline.")
                        with ui.column().classes("mce-stack w-full"):
                            status = readonly_input("Status", "Ready")
                            generation_progress = ui.linear_progress(value=0).classes("w-full")
                            generation_progress.visible = False
                            generation_progress_label = ui.label("Ready").classes("mce-muted")
                            platform_preview = ui.html(
                                build_preview_html(content_type.value, {"draft": "Select a content type to see a platform-style preview."}),
                                sanitize=False,
                            ).classes("mce-preview")
                            generated_output = readonly_textarea("Generated content", "")
                            result_actions = ui.row().classes("mce-actions")
                            with ui.card().classes("mce-subcard") as quote_image_builder:
                                subcard_heading("Quote Image Builder")
                                ui.label("Select the quote cards you want to render, then generate platform-ready image mockups.").classes("mce-muted")
                                quote_candidate_select = apply_field_props(
                                    ui.select([], multiple=True, label="Quote cards to render"),
                                    "outlined dense use-chips clearable",
                                )
                                quote_candidate_select.visible = False
                                quote_candidate_cards = ui.column().classes("w-full mce-stack")
                                quote_candidate_actions = ui.row().classes("mce-actions")
                                quote_image_status = readonly_input("Quote image status", "Generate a quote draft first.")
                                quote_platform_mockup = ui.html(quote_platform_mockup_html([]), sanitize=False).classes("w-full")
                                quote_image_actions = ui.row().classes("mce-actions")

                            with ui.expansion("Artifact paths and structured brief", icon="inventory_2").classes("mce-expansion"):
                                with ui.column().classes("mce-stack w-full"):
                                    filtered_context_path = readonly_input("Filtered context", "")
                                    prompt_path = readonly_input("Generation prompt", "")
                                    draft_path = readonly_input("Draft output", "")
                                    saved_draft_path = readonly_input("Saved draft", "")
                                    generated_visual_path = readonly_input("Generated image preview", "")
                                    generated_visual_package_path = readonly_input("Generated image package", "")
                                    structured_brief_output = readonly_textarea("Structured brief used for comparison", "")

                            with ui.expansion("Compare Mythos vs ChatGPT", icon="compare_arrows").classes("mce-expansion"):
                                with ui.column().classes("mce-stack w-full"):
                                    ui.label("Charisma. Uniqueness. Nerve and Talent.").classes("mce-section-title")
                                    ui.label("Generate a Mythos draft first, then create a fresh baseline from the same selections.").classes("mce-muted")
                                    chatgpt_model = apply_field_props(
                                        ui.select(
                                            CHATGPT_COMPARISON_MODEL_OPTIONS,
                                            value=CHATGPT_COMPARISON_MODEL_OPTIONS[0],
                                            label="ChatGPT model",
                                        )
                                    )
                                    comparison_status = readonly_input("Comparison status", "Generate a Mythos draft first.")
                                    comparison_view = ui.html(comparison_side_by_side_html(), sanitize=False).classes("w-full")
                                    chatgpt_output = readonly_textarea("ChatGPT baseline output", "")
                                    chatgpt_prompt_path = readonly_input("ChatGPT prompt path", "")
                                    chatgpt_draft_path = readonly_input("ChatGPT draft path", "")
                                    comparison_preference = apply_field_props(
                                        ui.select(["Mythos", "ChatGPT", "Tie / needs revision"], label="Which result do you prefer?")
                                    )
                                    comparison_notes = apply_field_props(
                                        ui.textarea(
                                            label="Judge notes",
                                            placeholder="Example: Mythos used the brand world more specifically; ChatGPT was clean but generic.",
                                        ),
                                        "outlined autogrow",
                                    )
                                    comparison_save_status = readonly_input("Saved judgment status", "")
                                    comparison_vote_path = readonly_input("Saved judgment path", "")
                                    comparison_actions = ui.row().classes("mce-actions")

                    def current_uploaded_image_path() -> str:
                        if content_type.value in {"instagram_caption", "blog_post", "character_spotlight", "quote_post"} and generated_visual_path.value:
                            generated_path = Path(str(generated_visual_path.value))
                            if generated_path.exists():
                                return str(generated_path)
                        if (
                            content_type.value == "blog_post"
                            and "selected character portrait" in str(blog_image_mode.value or "").lower()
                            and blog_character.value
                        ):
                            portrait = resolve_character_portrait_asset(blog_character.value)
                            if portrait:
                                return str(portrait)
                        return str(
                            {
                                "instagram_caption": instagram_image_path.value,
                                "linkedin_content": linkedin_image_path.value,
                                "youtube_content": youtube_image_path.value,
                                "blog_post": blog_image_path.value,
                                "character_spotlight": cs_image_path.value,
                            }.get(content_type.value, "")
                            or ""
                        )

                    def preview_fields() -> dict[str, object]:
                        return {
                            "caption": topic.value,
                            "draft": generated_output.value or topic.value,
                            "instagram_formats": instagram_formats.value,
                            "instagram_hashtags": instagram_hashtags.value,
                            "instagram_hook": instagram_hook.value,
                            "instagram_visual_formats": instagram_visual_formats.value,
                            "instagram_visual_style": instagram_visual_style.value,
                            "linkedin_formats": linkedin_formats.value,
                            "linkedin_angle": linkedin_angle.value,
                            "linkedin_cta": linkedin_cta.value,
                            "youtube_formats": youtube_formats.value,
                            "youtube_keywords": youtube_keywords.value,
                            "youtube_link": youtube_link.value,
                            "quote_moods": quote_moods.value,
                            "quote_characters": character_tags.value,
                            "quote_source": quote_book.value,
                            "generated_visual_package_path": generated_visual_package_path.value,
                            "blog_format": blog_format.value,
                            "blog_length": blog_length.value,
                            "blog_structure": blog_sections.value,
                            "blog_structure_checklist": blog_structure_options.value,
                            "blog_image_mode": blog_image_mode.value,
                            "blog_character": blog_character.value,
                            "blog_visual_formats": blog_visual_formats.value,
                            "blog_visual_style": blog_visual_style.value,
                            "character_subject": cs_character.value,
                            "character_format": cs_platform_format.value,
                            "character_image_mode": cs_image_mode.value,
                            "pr_city": pr_city.value,
                            "pr_state": pr_state.value,
                            "pr_release_date": pr_release_date.value,
                            "pr_contact_name": pr_contact_name.value,
                            "pr_website": pr_website.value,
                            "pr_news_angle": pr_news_angle.value,
                            "pr_required_assets": pr_required_assets.value,
                            "image_path": current_uploaded_image_path(),
                        }

                    def refresh_platform_preview() -> None:
                        platform_preview.content = build_preview_html(content_type.value, preview_fields())

                    def refresh_press_profile_options() -> None:
                        pr_profile.set_options(press_profile_options(), value=pr_profile.value if pr_profile.value in press_profile_options() else None)

                    def apply_press_profile() -> None:
                        profile_name = pr_profile.value
                        profiles = load_press_profiles()
                        data = profiles.get(profile_name or "")
                        if not data:
                            ui.notify("Select a saved press profile first.", type="warning")
                            return
                        pr_profile_name.value = profile_name
                        pr_contact_name.value = data.get("contact_name", "")
                        pr_contact_title.value = data.get("contact_title", "")
                        pr_organization.value = data.get("organization", "")
                        pr_contact_email.value = data.get("email", "")
                        pr_contact_phone.value = data.get("phone", "")
                        pr_website.value = data.get("website", "")
                        ui.notify("Press profile loaded.", type="positive")
                        refresh_platform_preview()

                    def save_press_profile_from_fields() -> None:
                        name = str(pr_profile_name.value or pr_contact_name.value or pr_organization.value or "").strip()
                        if not name:
                            ui.notify("Add a profile name before saving.", type="warning")
                            return
                        profiles = load_press_profiles()
                        profiles[name] = {
                            "contact_name": pr_contact_name.value,
                            "contact_title": pr_contact_title.value,
                            "organization": pr_organization.value,
                            "email": pr_contact_email.value,
                            "phone": pr_contact_phone.value,
                            "website": pr_website.value,
                        }
                        save_press_profiles(profiles)
                        refresh_press_profile_options()
                        pr_profile.value = name
                        ui.notify("Press profile saved.", type="positive")

                    async def suggest_press_news_angle() -> None:
                        pr_news_angle.value = "Suggesting news angle..."
                        prompt = f"""Suggest 3 strong, journalist-friendly news angles for this press release.

Topic: {topic.value}
Related book/source: {related_book.value}
Supporting proof: {pr_supporting_proof.value}
Target media: {pr_target_media.value}

Return concise angle options with why each is newsworthy."""
                        try:
                            pr_news_angle.value = await asyncio.to_thread(generate_text, prompt)
                            ui.notify("News angle suggested.", type="positive")
                        except Exception as exc:
                            pr_news_angle.value = f"Could not suggest news angle: {exc}"
                            ui.notify(pr_news_angle.value, type="negative")
                        refresh_platform_preview()

                    def quote_attribution_value() -> str:
                        selected = normalize_selected(character_tags.value)
                        return "" if selected == "Not specified" else selected

                    quote_candidate_checks = []

                    def selected_quote_card_values() -> list[str]:
                        selected = [quote for quote, checkbox in quote_candidate_checks if checkbox.value]
                        if selected:
                            return selected
                        fallback = quote_candidate_select.value or []
                        return [fallback] if isinstance(fallback, str) else list(fallback)

                    def sync_quote_select_from_cards() -> None:
                        selected = [quote for quote, checkbox in quote_candidate_checks if checkbox.value]
                        quote_candidate_select.value = selected
                        quote_image_status.value = f"Selected {len(selected)} quote card(s)."

                    def render_quote_candidate_cards(candidates: list[str], selected: list[str]) -> None:
                        quote_candidate_cards.clear()
                        quote_candidate_checks.clear()
                        selected_set = set(selected)
                        with quote_candidate_cards:
                            for index, candidate in enumerate(candidates, start=1):
                                with ui.card().classes("mce-subcard w-full"):
                                    checkbox = ui.checkbox(f"Quote card {index}", value=candidate in selected_set)
                                    ui.label(candidate).classes("mce-output-text")
                                    quote_candidate_checks.append((candidate, checkbox))
                                    checkbox.on_value_change(lambda _event: sync_quote_select_from_cards())

                    def refresh_quote_candidates(select_all: bool = True) -> None:
                        candidates = extract_quote_candidates(generated_output.value)
                        selected = candidates if select_all else []
                        quote_candidate_select.set_options(candidates, value=selected)
                        render_quote_candidate_cards(candidates, selected)
                        if candidates:
                            quote_image_status.value = f"{len(candidates)} quote card option(s) ready."
                            quote_platform_mockup.content = quote_platform_mockup_html([])
                        else:
                            quote_image_status.value = "No quote card candidates found yet."
                            quote_platform_mockup.content = quote_platform_mockup_html(
                                [],
                                error="No selectable quote cards were found in the generated draft. Regenerate with clearer quote options.",
                            )

                    def select_all_quote_candidates() -> None:
                        options = list(quote_candidate_select.options or [])
                        quote_candidate_select.value = options
                        for _, checkbox in quote_candidate_checks:
                            checkbox.value = True
                        quote_image_status.value = f"Selected {len(options)} quote card(s)."

                    def select_no_quote_candidates() -> None:
                        quote_candidate_select.value = []
                        for _, checkbox in quote_candidate_checks:
                            checkbox.value = False
                        quote_image_status.value = "No quote cards selected."

                    async def generate_selected_quote_images() -> None:
                        selected_quotes = selected_quote_card_values()
                        if not selected_quotes:
                            quote_image_status.value = "Select at least one quote card first."
                            ui.notify(quote_image_status.value, type="warning")
                            return
                        quote_image_status.value = "Generating quote card images..."
                        try:
                            package = await asyncio.to_thread(
                                render_selected_quote_visual_package,
                                quotes=list(selected_quotes),
                                attribution=quote_attribution_value(),
                                format_labels=quote_visual_formats.value or ["Instagram Post (4:5)"],
                                theme_name=quote_visual_style.value or "Gothic",
                            )
                        except Exception as exc:
                            quote_image_status.value = f"Quote image generation failed: {exc}"
                            ui.notify(quote_image_status.value, type="negative")
                            return

                        paths = [str(path) for path in package.get("paths", [])] if isinstance(package, dict) else []
                        error = str(package.get("error", "")) if isinstance(package, dict) else ""
                        package_path = str(package.get("zip_path", "")) if isinstance(package, dict) else ""
                        generated_visual_path.value = paths[0] if paths else error
                        generated_visual_package_path.value = package_path
                        quote_platform_mockup.content = quote_platform_mockup_html(paths, package_path, error)
                        quote_image_status.value = f"Generated {len(paths)} image file(s)." if paths else error or "No images generated."
                        refresh_platform_preview()
                        ui.notify(quote_image_status.value, type="positive" if paths else "warning")

                    preview_refresh["fn"] = refresh_platform_preview

                    def apply_visibility(selected_type: str) -> None:
                        is_press_release = selected_type == "press_release"
                        instagram_card.visible = selected_type == "instagram_caption"
                        linkedin_card.visible = selected_type == "linkedin_content"
                        youtube_card.visible = selected_type == "youtube_content"
                        quote_card.visible = selected_type == "quote_post"
                        blog_card.visible = selected_type == "blog_post"
                        newsletter_card.visible = selected_type == "newsletter_blurb"
                        character_card.visible = selected_type == "character_spotlight"
                        press_release_card.visible = is_press_release
                        quote_image_builder.visible = selected_type == "quote_post"
                        platform.visible = not is_press_release
                        social_objectives.visible = not is_press_release
                        audience.visible = not is_press_release
                        if is_press_release:
                            platform.value = PRESS_RELEASE_DESTINATION
                            social_objectives.value = []
                            audience.value = []
                        if not is_press_release and platform.value not in PLATFORM_OPTIONS:
                            platform.value = PLATFORM_OPTIONS[0]
                        platform_preview.content = build_preview_html(selected_type, preview_fields())

                    def on_content_type_change(event) -> None:
                        apply_visibility(event.value)

                    content_type.on_value_change(on_content_type_change)
                    apply_visibility(content_type.value)

                    async def generate_content() -> None:
                        started_at = datetime.now()
                        content_asset_total = 1
                        if content_type.value == "press_release" and pr_required_assets.value:
                            content_asset_total += len(pr_required_assets.value)
                        elif content_type.value == "quote_post":
                            content_asset_total += len(quote_visual_formats.value or [])
                        elif content_type.value == "instagram_caption" and instagram_generate_images.value == "yes":
                            content_asset_total += len(instagram_visual_formats.value or INSTAGRAM_VISUAL_FORMAT_OPTIONS)
                        elif content_type.value == "blog_post":
                            content_asset_total += len(
                                blog_visual_formats.value
                                or (
                                    BLOG_GALLERY_VISUAL_FORMAT_OPTIONS
                                    if blog_format.value == "photo/gallery slideshow"
                                    else BLOG_VISUAL_FORMAT_OPTIONS
                                )
                            )
                        elif content_type.value == "character_spotlight" and cs_image_mode.value != "text-only highlight card":
                            content_asset_total += len(formats_for_character_spotlight(cs_platform_format.value))
                        generation_progress.visible = True
                        generation_progress.value = 0
                        status.value = "Generating..."
                        try:
                            await set_generation_progress(
                                progress=generation_progress,
                                label=generation_progress_label,
                                status=status,
                                started_at=started_at,
                                completed_steps=0,
                                total_steps=4,
                                completed_assets=0,
                                total_assets=content_asset_total,
                                message="Preparing content brief...",
                            )
                            apply_visibility(content_type.value)
                            if content_type.value == "instagram_caption" and not str(instagram_hook.value or "").strip():
                                await set_generation_progress(
                                    progress=generation_progress,
                                    label=generation_progress_label,
                                    status=status,
                                    started_at=started_at,
                                    completed_steps=1,
                                    total_steps=4,
                                    completed_assets=0,
                                    total_assets=content_asset_total,
                                    message="Suggesting Instagram hook...",
                                )
                                await suggest_instagram_hook()
                            selected_blog_visual_formats = blog_visual_formats.value or (
                                BLOG_GALLERY_VISUAL_FORMAT_OPTIONS
                                if blog_format.value == "photo/gallery slideshow"
                                else BLOG_VISUAL_FORMAT_OPTIONS
                            )
                            await set_generation_progress(
                                progress=generation_progress,
                                label=generation_progress_label,
                                status=status,
                                started_at=started_at,
                                completed_steps=1,
                                total_steps=4,
                                completed_assets=0,
                                total_assets=content_asset_total,
                                message="Generating draft and selected media assets...",
                            )
                            await generate_draft_from_fields(
                                content_type=content_type.value,
                                topic=topic.value,
                                related_book=related_book.value,
                                platform=platform.value,
                                social_objectives=social_objectives.value,
                                audience=audience.value,
                                cta=cta.value,
                                constraints=constraints.value,
                                quote_book=quote_book.value,
                                quote_moods=quote_moods.value,
                                character_tags=character_tags.value,
                                podcast_format="",
                                podcast_speakers="",
                                podcast_roles="",
                                podcast_tone="",
                                podcast_length="",
                                elevenlabs_model="",
                                elevenlabs_voice_ids="",
                                podcast_show_title="",
                                podcast_episode_title="",
                                podcast_episode_number="",
                                blog_length=blog_length.value,
                                blog_format=blog_format.value,
                                blog_sections=blog_sections.value,
                                blog_structure_options=blog_structure_options.value,
                                blog_seo_keywords=blog_seo_keywords.value,
                                blog_image_mode=blog_image_mode.value,
                                blog_character=blog_character.value,
                                instagram_formats=instagram_formats.value,
                                instagram_hashtags=instagram_hashtags.value,
                                instagram_hook=instagram_hook.value,
                                cs_character=cs_character.value,
                                cs_focus=cs_focus.value,
                                cs_platform_format=cs_platform_format.value,
                                cs_image_mode=cs_image_mode.value,
                                nl_subject=nl_subject.value,
                                nl_preview=nl_preview.value,
                                nl_structure=nl_structure.value,
                                pr_timing=pr_timing.value,
                                pr_embargo_date=pr_embargo_date.value,
                                pr_city=pr_city.value,
                                pr_state=pr_state.value,
                                pr_release_date=pr_release_date.value,
                                pr_contact_name=pr_contact_name.value,
                                pr_contact_title=pr_contact_title.value,
                                pr_organization=pr_organization.value,
                                pr_contact_email=pr_contact_email.value,
                                pr_contact_phone=pr_contact_phone.value,
                                pr_website=pr_website.value,
                                pr_news_angle=pr_news_angle.value,
                                pr_release_goal="",
                                pr_primary_announcement="",
                                pr_supporting_proof=pr_supporting_proof.value,
                                pr_quote_source=pr_quote_source.value,
                                pr_target_media=pr_target_media.value,
                                pr_required_assets=pr_required_assets.value,
                                status=status,
                                filtered_context_path=filtered_context_path,
                                prompt_path=prompt_path,
                                draft_path=draft_path,
                                generated_output=generated_output,
                                visual_format_labels=(
                                    quote_visual_formats.value
                                    if content_type.value == "quote_post"
                                    else instagram_visual_formats.value or INSTAGRAM_VISUAL_FORMAT_OPTIONS
                                    if content_type.value == "instagram_caption" and instagram_generate_images.value == "yes"
                                    else formats_for_character_spotlight(cs_platform_format.value)
                                    if content_type.value == "character_spotlight"
                                    else selected_blog_visual_formats
                                    if content_type.value == "blog_post"
                                    else None
                                ),
                                visual_theme_name=(
                                    quote_visual_style.value
                                    if content_type.value == "quote_post"
                                    else instagram_visual_style.value
                                    if content_type.value == "instagram_caption"
                                    else blog_visual_style.value
                                    if content_type.value == "blog_post"
                                    else "Gothic"
                                ),
                                saved_draft_path=saved_draft_path,
                                structured_brief_output=structured_brief_output,
                                uploaded_image_path=current_uploaded_image_path(),
                                generated_visual_path=generated_visual_path,
                                generated_visual_package_path=generated_visual_package_path,
                            )
                            completed_assets = 1
                            generated_package_value = str(generated_visual_package_path.value or "").strip()
                            generated_package = Path(generated_package_value) if generated_package_value else None
                            if generated_package and generated_package.exists() and content_asset_total > 1 and content_type.value != "press_release":
                                completed_assets = content_asset_total
                            await set_generation_progress(
                                progress=generation_progress,
                                label=generation_progress_label,
                                status=status,
                                started_at=started_at,
                                completed_steps=3,
                                total_steps=4,
                                completed_assets=completed_assets,
                                total_assets=content_asset_total,
                                message="Draft ready. Finalizing assets...",
                            )
                            if content_type.value == "press_release" and pr_required_assets.value:
                                await set_generation_progress(
                                    progress=generation_progress,
                                    label=generation_progress_label,
                                    status=status,
                                    started_at=started_at,
                                    completed_steps=3,
                                    total_steps=4,
                                    completed_assets=1,
                                    total_assets=content_asset_total,
                                    message="Generating selected PR assets...",
                                )
                                assets = await asyncio.to_thread(
                                    generate_press_release_assets,
                                    selected_assets=pr_required_assets.value,
                                    topic=topic.value,
                                    draft=generated_output.value,
                                    contact={
                                        "name": pr_contact_name.value,
                                        "title": pr_contact_title.value,
                                        "organization": pr_organization.value,
                                        "email": pr_contact_email.value,
                                        "phone": pr_contact_phone.value,
                                        "website": pr_website.value,
                                    },
                                    news_angle=pr_news_angle.value,
                                )
                                if assets:
                                    pr_asset_status.value = str(assets.get("zip_path") or assets.get("error") or "")
                                    generated_visual_package_path.value = str(assets.get("zip_path") or generated_visual_package_path.value)
                                    completed_assets = content_asset_total if assets.get("paths") else 1
                                    ui.notify("PR assets generated.", type="positive" if assets.get("paths") else "warning")
                            if content_type.value == "quote_post":
                                refresh_quote_candidates(select_all=True)
                            refresh_platform_preview()
                            await finish_generation_progress(
                                progress=generation_progress,
                                label=generation_progress_label,
                                status=status,
                                total_assets=content_asset_total,
                                message="Draft ready.",
                            )
                        finally:
                            pass

                    async def run_comparison() -> None:
                        await generate_chatgpt_comparison(
                            content_type=content_type.value,
                            structured_brief=structured_brief_output.value,
                            mythos_content=generated_output.value,
                            model=chatgpt_model.value,
                            comparison_status=comparison_status,
                            comparison_view=comparison_view,
                            chatgpt_output=chatgpt_output,
                            chatgpt_prompt_path=chatgpt_prompt_path,
                            chatgpt_draft_path=chatgpt_draft_path,
                        )

                    def save_preference() -> None:
                        save_comparison_preference(
                            preference=comparison_preference.value,
                            notes=comparison_notes.value,
                            content_type=content_type.value,
                            structured_brief=structured_brief_output.value,
                            mythos_content=generated_output.value,
                            chatgpt_model=chatgpt_model.value,
                            chatgpt_content=chatgpt_output.value,
                            chatgpt_draft_path=chatgpt_draft_path.value,
                            chatgpt_prompt_path=chatgpt_prompt_path.value,
                            comparison_save_status=comparison_save_status,
                            comparison_vote_path=comparison_vote_path,
                        )

                    def save_current_draft() -> None:
                        if not str(generated_output.value or "").strip():
                            status.value = "Generate content before saving a draft."
                            ui.notify(status.value, type="warning")
                            return
                        title_source = (
                            first_nonempty_line(generated_output.value)
                            or first_nonempty_line(topic.value)
                            or f"{content_type.value} draft"
                        )
                        draft_record = save_draft(
                            title=title_source[:90],
                            content_type=content_type.value,
                            content=generated_output.value,
                            source_path=draft_path.value,
                            metadata={
                                "topic": str(topic.value or "").strip(),
                                "related_book": normalize_selected(related_book.value),
                                "platform": normalize_selected(platform.value),
                                "structured_brief": structured_brief_output.value,
                                "generated_visual_path": generated_visual_path.value,
                                "generated_visual_package_path": generated_visual_package_path.value,
                                "manual_save": True,
                            },
                        )
                        saved_draft_path.value = draft_record["path"]
                        status.value = "Draft saved."
                        ui.notify("Draft saved.", type="positive")

                    def download_generated_draft() -> None:
                        if draft_path.value and Path(str(draft_path.value)).exists():
                            ui.download(draft_path.value)
                            return
                        if saved_draft_path.value and Path(str(saved_draft_path.value)).exists():
                            ui.download(saved_draft_path.value)
                            return
                        if str(generated_output.value or "").strip():
                            saved_path = save_output(generated_output.value, content_type.value, "download")
                            draft_path.value = str(saved_path)
                            ui.download(str(saved_path))
                            return
                        ui.notify("Generate content before downloading.", type="warning")

                    def download_generated_media_package() -> None:
                        package = str(generated_visual_package_path.value or "").strip()
                        if package and Path(package).exists():
                            ui.download(package)
                            return
                        preview = str(generated_visual_path.value or "").strip()
                        if preview and Path(preview).exists():
                            ui.download(preview)
                            return
                        ui.notify("No generated image or asset package is available yet.", type="warning")

                    with quote_candidate_actions:
                        make_secondary_button("Select All", select_all_quote_candidates)
                        make_secondary_button("Select None", select_no_quote_candidates)
                    with quote_image_actions:
                        make_primary_button("Generate Selected Quote Images", generate_selected_quote_images)
                    with pr_profile_actions:
                        make_secondary_button("Load Profile", apply_press_profile)
                        make_secondary_button("Save Profile", save_press_profile_from_fields)
                        make_secondary_button("Suggest News Angle", suggest_press_news_angle)
                    with generator_actions:
                        make_primary_button("Generate Draft", generate_content)
                        make_secondary_button("Save Draft", save_current_draft)
                    with result_actions:
                        make_secondary_button("Retry / Regenerate", generate_content)
                        make_secondary_button("Download Draft", download_generated_draft)
                        make_secondary_button("Download Images / Assets", download_generated_media_package)
                        make_secondary_button("Save Draft", save_current_draft)
                    with comparison_actions:
                        make_secondary_button("Generate ChatGPT Baseline", run_comparison)
                        make_secondary_button("Save Judgment", save_preference)

            with ui.tab_panel(campaign_tab).classes("mce-panel"):
                with ui.element("section").classes("mce-grid"):
                    with ui.card().classes("mce-card"):
                        section_heading(
                            "Campaign Mode",
                            "One brief, multiple configured assets. Select outputs first, then tune each content type.",
                        )
                        with ui.column().classes("mce-stack w-full"):
                            campaign_topic = apply_field_props(
                                ui.textarea(
                                    label="Campaign topic",
                                    placeholder="Example: Announce Mortal Vengeance winning an award and drive readers to the book page.",
                                ),
                                "outlined autogrow",
                            )
                            campaign_book = apply_field_props(
                                ui.select(
                                    QUOTE_BOOK_OPTIONS,
                                    value=QUOTE_BOOK_OPTIONS[0] if QUOTE_BOOK_OPTIONS else None,
                                    label="Campaign related book/source (optional)",
                                ),
                            )
                            campaign_tone = apply_field_props(
                                ui.input(label="Campaign tone", placeholder="Example: cinematic, literary, darkly funny, media-ready"),
                            )
                            campaign_objectives = apply_field_props(
                                ui.select(SOCIAL_OBJECTIVES, multiple=True, label="Campaign objectives"),
                                "outlined dense use-chips clearable",
                            )
                            campaign_audience = apply_field_props(
                                ui.select(AUDIENCE_OPTIONS, multiple=True, label="Campaign audience"),
                                "outlined dense use-chips clearable",
                            )
                            campaign_cta = apply_field_props(
                                ui.input(label="Campaign CTA", placeholder="What should every asset ultimately drive people to do?"),
                            )
                            campaign_constraints = apply_field_props(
                                ui.select(CONSTRAINT_OPTIONS, multiple=True, label="Campaign constraints"),
                                "outlined dense use-chips clearable",
                            )
                            campaign_formats = apply_field_props(
                                ui.select({key: label for key, label in CAMPAIGN_FORMATS}, multiple=True, label="Campaign content types"),
                                "outlined dense use-chips clearable",
                            )

                            ui.label("Selected Content Type Settings").classes("mce-section-title")
                            ui.label("Only cards for selected formats appear below. One tiny mercy from the UI gods.").classes("mce-muted")

                            campaign_widgets: dict[str, dict[str, object]] = {}
                            campaign_cards: dict[str, object] = {}

                            def add_base_campaign_fields(content_type: str) -> dict[str, object]:
                                return {
                                    "style": apply_field_props(
                                        ui.select(style_choices(content_type), value=default_style(content_type), label="Style / variant"),
                                    ),
                                    "quantity": apply_field_props(
                                        ui.select(CAMPAIGN_QUANTITY_OPTIONS, value="1", label="Quantity"),
                                    ),
                                }

                            with ui.card().classes("mce-subcard") as campaign_podcast_card:
                                subcard_heading("Podcast")
                                campaign_widgets["podcast"] = add_base_campaign_fields("podcast")
                                campaign_widgets["podcast"]["format"] = apply_field_props(ui.select(PODCAST_FORMAT_OPTIONS, value=PODCAST_FORMAT_OPTIONS[0], label="Podcast format"))
                                campaign_widgets["podcast"]["length"] = apply_field_props(ui.select(PODCAST_LENGTH_OPTIONS, value=PODCAST_LENGTH_OPTIONS[0], label="Podcast length"))
                                campaign_widgets["podcast"]["tone"] = apply_field_props(ui.select(PODCAST_TONE_OPTIONS, value=PODCAST_TONE_OPTIONS[0], label="Podcast delivery tone"))
                            campaign_cards["podcast"] = campaign_podcast_card

                            with ui.card().classes("mce-subcard") as campaign_instagram_card:
                                subcard_heading("Instagram")
                                campaign_widgets["instagram_caption"] = add_base_campaign_fields("instagram_caption")
                                campaign_widgets["instagram_caption"]["formats"] = apply_field_props(ui.select(INSTAGRAM_FORMAT_OPTIONS, multiple=True, label="Instagram formats"), "outlined dense use-chips clearable")
                            campaign_cards["instagram_caption"] = campaign_instagram_card

                            with ui.card().classes("mce-subcard") as campaign_youtube_card:
                                subcard_heading("YouTube")
                                campaign_widgets["youtube_content"] = add_base_campaign_fields("youtube_content")
                                campaign_widgets["youtube_content"]["formats"] = apply_field_props(ui.select(YOUTUBE_CONTENT_FORMAT_OPTIONS, multiple=True, label="YouTube deliverables"), "outlined dense use-chips clearable")
                            campaign_cards["youtube_content"] = campaign_youtube_card

                            with ui.card().classes("mce-subcard") as campaign_linkedin_card:
                                subcard_heading("LinkedIn")
                                campaign_widgets["linkedin_content"] = add_base_campaign_fields("linkedin_content")
                                campaign_widgets["linkedin_content"]["formats"] = apply_field_props(ui.select(LINKEDIN_CONTENT_FORMAT_OPTIONS, multiple=True, label="LinkedIn deliverables"), "outlined dense use-chips clearable")
                            campaign_cards["linkedin_content"] = campaign_linkedin_card

                            with ui.card().classes("mce-subcard") as campaign_blog_card:
                                subcard_heading("Blog Post")
                                campaign_widgets["blog_post"] = add_base_campaign_fields("blog_post")
                                campaign_widgets["blog_post"]["format"] = apply_field_props(ui.select(BLOG_FORMAT_OPTIONS, value=BLOG_FORMAT_OPTIONS[0], label="Blog format"))
                                campaign_widgets["blog_post"]["length"] = apply_field_props(ui.select(BLOG_LENGTH_OPTIONS, value=BLOG_LENGTH_OPTIONS[1] if len(BLOG_LENGTH_OPTIONS) > 1 else BLOG_LENGTH_OPTIONS[0], label="Blog length"))
                                campaign_widgets["blog_post"]["structure"] = apply_field_props(ui.select(BLOG_STRUCTURE_CHECKLIST_OPTIONS, multiple=True, label="Structure checklist"), "outlined dense use-chips clearable")
                                campaign_widgets["blog_post"]["sections"] = apply_field_props(ui.textarea(label="Desired sections / outline", placeholder="Optional section plan for campaign blog assets"), "outlined autogrow")
                                campaign_widgets["blog_post"]["seo"] = apply_field_props(ui.input(label="SEO keywords", placeholder="Comma-separated keywords"))
                                campaign_widgets["blog_post"]["image_mode"] = apply_field_props(ui.select(["use selected character portrait", "text-only / quote graphic", "repurpose uploaded assets"], value="text-only / quote graphic", label="Blog image mode"))
                            campaign_cards["blog_post"] = campaign_blog_card

                            with ui.card().classes("mce-subcard") as campaign_newsletter_card:
                                subcard_heading("Newsletter")
                                campaign_widgets["newsletter_blurb"] = add_base_campaign_fields("newsletter_blurb")
                                campaign_widgets["newsletter_blurb"]["structure"] = apply_field_props(ui.select(NEWSLETTER_STRUCTURE_OPTIONS, value=NEWSLETTER_STRUCTURE_OPTIONS[0], label="Newsletter structure"))
                            campaign_cards["newsletter_blurb"] = campaign_newsletter_card

                            with ui.card().classes("mce-subcard") as campaign_quote_card:
                                subcard_heading("Quote Post")
                                campaign_widgets["quote_post"] = add_base_campaign_fields("quote_post")
                                campaign_widgets["quote_post"]["book"] = apply_field_props(ui.select(QUOTE_BOOK_OPTIONS, value=QUOTE_BOOK_OPTIONS[0] if QUOTE_BOOK_OPTIONS else None, label="Quote book/source"))
                                campaign_widgets["quote_post"]["moods"] = apply_field_props(ui.select(QUOTE_MOOD_TAGS, multiple=True, label="Quote mood/category tags"), "outlined dense use-chips clearable")
                                campaign_widgets["quote_post"]["characters"] = apply_field_props(ui.select(CHARACTER_TAGS, multiple=True, label="Character tags"), "outlined dense use-chips clearable")
                                campaign_widgets["quote_post"]["graphic_formats"] = apply_field_props(ui.select(quote_graphic_format_options(), multiple=True, label="Quote image destination formats"), "outlined dense use-chips clearable")
                                campaign_widgets["quote_post"]["graphic_theme"] = apply_field_props(ui.select(QUOTE_IMAGE_STYLE_OPTIONS, value="Gothic", label="Quote image color style"))
                            campaign_cards["quote_post"] = campaign_quote_card

                            with ui.card().classes("mce-subcard") as campaign_pull_quote_card:
                                subcard_heading("Review Pull Quote")
                                campaign_widgets["review_pull_quote"] = add_base_campaign_fields("review_pull_quote")
                                ui.label("Grounding rule: use only real review excerpts from the knowledge base.").classes("mce-muted")
                                campaign_widgets["review_pull_quote"]["graphic_formats"] = apply_field_props(ui.select(quote_graphic_format_options(), multiple=True, label="Pull quote image destination formats"), "outlined dense use-chips clearable")
                                campaign_widgets["review_pull_quote"]["graphic_theme"] = apply_field_props(ui.select(QUOTE_IMAGE_STYLE_OPTIONS, value="Press", label="Pull quote image color style"))
                            campaign_cards["review_pull_quote"] = campaign_pull_quote_card

                            with ui.card().classes("mce-subcard") as campaign_character_card:
                                subcard_heading("Character Spotlight")
                                campaign_widgets["character_spotlight"] = add_base_campaign_fields("character_spotlight")
                                campaign_widgets["character_spotlight"]["character"] = apply_field_props(ui.select(CHARACTER_TAGS, label="Character subject"))
                                campaign_widgets["character_spotlight"]["format"] = apply_field_props(ui.select(CHARACTER_SPOTLIGHT_PLATFORM_FORMATS, value=CHARACTER_SPOTLIGHT_PLATFORM_FORMATS[0], label="Destination format"))
                                campaign_widgets["character_spotlight"]["image_mode"] = apply_field_props(ui.select(CHARACTER_IMAGE_MODE_OPTIONS, value=CHARACTER_IMAGE_MODE_OPTIONS[1], label="Image source"))
                            campaign_cards["character_spotlight"] = campaign_character_card

                            with ui.card().classes("mce-subcard") as campaign_press_card:
                                subcard_heading("Press Release")
                                campaign_widgets["press_release"] = add_base_campaign_fields("press_release")
                                campaign_widgets["press_release"]["timing"] = apply_field_props(ui.select(PRESS_RELEASE_TIMING_OPTIONS, value=PRESS_RELEASE_TIMING_OPTIONS[0], label="Release timing"))
                                campaign_widgets["press_release"]["news_angle"] = apply_field_props(ui.textarea(label="News angle", placeholder="Why this announcement is newsworthy"), "outlined autogrow")
                            campaign_cards["press_release"] = campaign_press_card

                            campaign_actions = ui.row().classes("mce-actions")

                    with ui.card().classes("mce-card mce-sticky"):
                        section_heading("Campaign Results", "Campaign outputs appear as a bundle with each selected asset separated for review.")
                        with ui.column().classes("mce-stack w-full"):
                            campaign_status = readonly_input("Campaign status", "Ready")
                            campaign_progress = ui.linear_progress(value=0).classes("w-full")
                            campaign_progress.visible = False
                            campaign_progress_label = ui.label("Ready").classes("mce-muted")
                            campaign_preview = ui.html(build_campaign_preview_html([]), sanitize=False).classes("mce-preview")
                            campaign_output = readonly_textarea("Generated campaign", "")
                            campaign_path = readonly_input("Combined campaign path", "")
                            campaign_saved_draft_path = readonly_input("Saved campaign draft", "")
                            campaign_result_actions = ui.row().classes("mce-actions")
                            with ui.expansion("Structured campaign brief", icon="subject").classes("mce-expansion"):
                                campaign_brief_output = readonly_textarea("Structured campaign brief used for comparison", "")

                            with ui.expansion("Compare campaign vs ChatGPT", icon="compare_arrows").classes("mce-expansion"):
                                with ui.column().classes("mce-stack w-full"):
                                    campaign_chatgpt_model = apply_field_props(
                                        ui.select(CHATGPT_COMPARISON_MODEL_OPTIONS, value=CHATGPT_COMPARISON_MODEL_OPTIONS[0], label="ChatGPT model")
                                    )
                                    campaign_comparison_status = readonly_input("Comparison status", "Generate a Mythos campaign first.")
                                    campaign_comparison_view = ui.html(comparison_side_by_side_html(), sanitize=False).classes("w-full")
                                    campaign_chatgpt_output = readonly_textarea("ChatGPT campaign baseline output", "")
                                    campaign_chatgpt_prompt_path = readonly_input("ChatGPT prompt path", "")
                                    campaign_chatgpt_draft_path = readonly_input("ChatGPT draft path", "")
                                    campaign_preference = apply_field_props(ui.select(["Mythos", "ChatGPT", "Tie / needs revision"], label="Which campaign do you prefer?"))
                                    campaign_notes = apply_field_props(ui.textarea(label="Judge notes", placeholder="What made one campaign stronger, more specific, or less generic?"), "outlined autogrow")
                                    campaign_save_status = readonly_input("Saved judgment status", "")
                                    campaign_vote_path = readonly_input("Saved judgment path", "")
                                    campaign_comparison_actions = ui.row().classes("mce-actions")

                    def apply_campaign_visibility(selected_formats=None) -> None:
                        selected_set = set(selected_formats or [])
                        for campaign_type, card in campaign_cards.items():
                            card.visible = campaign_type in selected_set

                    apply_campaign_visibility(campaign_formats.value)
                    campaign_formats.on_value_change(lambda event: apply_campaign_visibility(event.value))

                    async def generate_campaign() -> None:
                        campaign_progress.visible = True
                        campaign_status.value = "Generating campaign..."
                        try:
                            await generate_campaign_from_fields(
                                campaign_topic=campaign_topic.value,
                                campaign_book=campaign_book.value,
                                campaign_tone=campaign_tone.value,
                                campaign_objectives=campaign_objectives.value,
                                campaign_audience=campaign_audience.value,
                                campaign_cta=campaign_cta.value,
                                campaign_constraints=campaign_constraints.value,
                                campaign_formats=campaign_formats.value,
                                campaign_widgets=campaign_widgets,
                                status=campaign_status,
                                campaign_output=campaign_output,
                                campaign_preview=campaign_preview,
                                campaign_brief_output=campaign_brief_output,
                                campaign_path=campaign_path,
                                saved_draft_path=campaign_saved_draft_path,
                                campaign_progress=campaign_progress,
                                campaign_progress_label=campaign_progress_label,
                            )
                        finally:
                            pass

                    async def run_campaign_comparison() -> None:
                        await generate_chatgpt_comparison(
                            content_type="campaign_mode",
                            structured_brief=campaign_brief_output.value,
                            mythos_content=campaign_output.value,
                            model=campaign_chatgpt_model.value,
                            comparison_status=campaign_comparison_status,
                            comparison_view=campaign_comparison_view,
                            chatgpt_output=campaign_chatgpt_output,
                            chatgpt_prompt_path=campaign_chatgpt_prompt_path,
                            chatgpt_draft_path=campaign_chatgpt_draft_path,
                        )

                    def save_campaign_preference() -> None:
                        save_comparison_preference(
                            preference=campaign_preference.value,
                            notes=campaign_notes.value,
                            content_type="campaign_mode",
                            structured_brief=campaign_brief_output.value,
                            mythos_content=campaign_output.value,
                            chatgpt_model=campaign_chatgpt_model.value,
                            chatgpt_content=campaign_chatgpt_output.value,
                            chatgpt_draft_path=campaign_chatgpt_draft_path.value,
                            chatgpt_prompt_path=campaign_chatgpt_prompt_path.value,
                            comparison_save_status=campaign_save_status,
                            comparison_vote_path=campaign_vote_path,
                        )

                    def download_campaign_bundle() -> None:
                        if campaign_path.value and Path(str(campaign_path.value)).exists():
                            ui.download(campaign_path.value)
                            return
                        if campaign_saved_draft_path.value and Path(str(campaign_saved_draft_path.value)).exists():
                            ui.download(campaign_saved_draft_path.value)
                            return
                        if str(campaign_output.value or "").strip():
                            saved_path = save_output(campaign_output.value, "campaign_mode", "download")
                            campaign_path.value = str(saved_path)
                            ui.download(str(saved_path))
                            return
                        ui.notify("Generate a campaign before downloading.", type="warning")

                    with campaign_actions:
                        make_primary_button("Generate Campaign", generate_campaign)
                    with campaign_result_actions:
                        make_secondary_button("Retry / Regenerate Campaign", generate_campaign)
                        make_secondary_button("Download Campaign Bundle", download_campaign_bundle)
                    with campaign_comparison_actions:
                        make_secondary_button("Generate ChatGPT Campaign", run_campaign_comparison)
                        make_secondary_button("Save Judgment", save_campaign_preference)

            with ui.tab_panel(podcast_tab).classes("mce-panel"):
                with ui.element("section").classes("mce-grid"):
                    with ui.card().classes("mce-card"):
                        section_heading("Podcast Studio", "Create a podcast brief, generate a script, and render audio directly in NiceGUI.")
                        with ui.column().classes("mce-stack w-full"):
                            podcast_topic = apply_field_props(
                                ui.textarea(label="Podcast topic", placeholder="What should the episode cover?"),
                                "outlined autogrow",
                            )
                            podcast_platform = apply_field_props(
                                ui.select(PODCAST_DESTINATION_OPTIONS, value=PODCAST_DESTINATION_OPTIONS[0], label="Podcast destination"),
                            )
                            podcast_related_book = apply_field_props(
                                ui.select(
                                    QUOTE_BOOK_OPTIONS,
                                    value=QUOTE_BOOK_OPTIONS[0] if QUOTE_BOOK_OPTIONS else None,
                                    label="Related book/source (optional)",
                                ),
                            )
                            podcast_social_objectives = apply_field_props(
                                ui.select(SOCIAL_OBJECTIVES, multiple=True, label="Social objectives"),
                                "outlined dense use-chips clearable",
                            )
                            podcast_audience = apply_field_props(
                                ui.select(AUDIENCE_OPTIONS, multiple=True, label="Audience"),
                                "outlined dense use-chips clearable",
                            )
                            podcast_cta = apply_field_props(
                                ui.input(label="CTA / next step", placeholder="What should the listener do next?"),
                            )
                            podcast_constraints = apply_field_props(
                                ui.select(CONSTRAINT_OPTIONS, multiple=True, label="Constraints"),
                                "outlined dense use-chips clearable",
                            )

                            with ui.card().classes("mce-subcard"):
                                subcard_heading("Format and metadata")
                                podcast_format_native = apply_field_props(ui.select(PODCAST_FORMAT_OPTIONS, value=PODCAST_FORMAT_OPTIONS[0], label="Podcast format"))
                                podcast_speakers_native = apply_field_props(ui.input(label="Podcast speaker count", placeholder="For example: 2"))
                                podcast_roles_native = apply_field_props(ui.input(label="Speaker roles / names", placeholder="Host, Author, Critic"))
                                podcast_tone_native = apply_field_props(ui.select(PODCAST_TONE_OPTIONS, multiple=True, label="Podcast tone"), "outlined dense use-chips clearable")
                                podcast_length_native = apply_field_props(ui.select(PODCAST_LENGTH_OPTIONS, value=PODCAST_LENGTH_OPTIONS[0], label="Podcast target length"))
                                podcast_show_title_native = apply_field_props(ui.input(label="Podcast show title", placeholder="Series or show name"))
                                podcast_episode_title_native = apply_field_props(ui.input(label="Episode title", placeholder="Specific episode title"))
                                podcast_episode_number_native = apply_field_props(ui.input(label="Episode number", placeholder="For example: 12"))

                            with ui.expansion("Advanced voice settings", icon="record_voice_over").classes("mce-expansion"):
                                with ui.column().classes("mce-stack w-full"):
                                    podcast_model_native = apply_field_props(ui.select(ELEVENLABS_MODEL_OPTIONS, value=ELEVENLABS_MODEL_OPTIONS[0], label="ElevenLabs model"))
                                    podcast_voice_ids_native = apply_field_props(
                                        ui.input(label="Resolved voice casting", placeholder="Host=voice_id, Guest=voice_id"),
                                    ).props("readonly outlined dense")
                                    voice_load_status = readonly_input("Voice library status", "Loading ElevenLabs voices...")
                                    podcast_host_voice = apply_field_props(ui.select({}, label="Host voice"))
                                    podcast_guest_voice = apply_field_props(ui.select({}, label="Guest voice"))
                                    podcast_guest_2_voice = apply_field_props(ui.select({}, label="Guest 2 / co-host voice"))
                                    selected_voice_preview = ui.audio("", controls=True).classes("mce-audio")
                                    podcast_stability = ui.slider(min=0, max=1, value=0.5, step=0.05).props("label-always").classes("w-full")
                                    ui.label("Stability").classes("mce-muted")
                                    podcast_similarity = ui.slider(min=0, max=1, value=0.75, step=0.05).props("label-always").classes("w-full")
                                    ui.label("Similarity").classes("mce-muted")
                                    podcast_style = ui.slider(min=0, max=1, value=0.0, step=0.05).props("label-always").classes("w-full")
                                    ui.label("Style").classes("mce-muted")
                                    podcast_speed = ui.slider(min=0.7, max=1.2, value=1.0, step=0.05).props("label-always").classes("w-full")
                                    ui.label("Speed").classes("mce-muted")
                                    podcast_speaker_boost = ui.switch("Speaker boost", value=True)

                            podcast_actions = ui.row().classes("mce-actions")

                    with ui.card().classes("mce-card mce-sticky"):
                        section_heading("Podcast Output", "Edit the generated script, preview audio, and download the finished package.")
                        with ui.column().classes("mce-stack w-full"):
                            podcast_status = readonly_input("Status", "Ready")
                            podcast_draft_progress = ui.linear_progress(value=0).classes("w-full")
                            podcast_draft_progress.visible = False
                            podcast_draft_progress_label = ui.label("Ready").classes("mce-muted")
                            podcast_generated_output = ui.textarea(label="Script editor", value="").props("outlined autogrow").classes("w-full mce-script-textarea")
                            podcast_result_actions = ui.row().classes("mce-actions")
                            podcast_audio_status = readonly_input("Audio status / errors", "Generate a preview or full podcast when the script is ready.")
                            podcast_audio_progress = ui.linear_progress(value=0).classes("w-full")
                            podcast_audio_progress.visible = False
                            podcast_audio_progress_label = ui.label("Ready").classes("mce-muted")
                            podcast_audio_player = ui.audio("", controls=True).classes("mce-audio")
                            podcast_audio_path = readonly_input("Podcast MP3 path", "")
                            podcast_audio_package_path = readonly_input("Audio package path", "")
                            podcast_audio_saved_draft_path = readonly_input("Saved audio draft", "")
                            with ui.expansion("Draft artifacts", icon="inventory_2").classes("mce-expansion"):
                                with ui.column().classes("mce-stack w-full"):
                                    podcast_filtered_context_path = readonly_input("Filtered context", "")
                                    podcast_prompt_path = readonly_input("Generation prompt", "")
                                    podcast_draft_path = readonly_input("Draft output", "")
                                    podcast_saved_draft_path = readonly_input("Saved podcast draft", "")
                            podcast_audio_actions = ui.row().classes("mce-actions")

                    podcast_voices_state: list[dict] = []

                    def sync_voice_casting() -> None:
                        podcast_voice_ids_native.value = build_voice_casting_from_selects(
                            podcast_host_voice.value,
                            podcast_guest_voice.value,
                            podcast_guest_2_voice.value,
                        )

                    def update_voice_preview(voice_id: str | None) -> None:
                        url = voice_preview_url(voice_id or "", podcast_voices_state)
                        if url:
                            selected_voice_preview.set_source(url)
                            return
                        selected_voice_preview.set_source("")

                    async def load_podcast_voices() -> None:
                        nonlocal podcast_voices_state
                        try:
                            voices = await asyncio.to_thread(list_voices)
                        except Exception as exc:
                            voice_load_status.value = f"Could not load voices: {exc}"
                            ui.notify(voice_load_status.value, type="warning")
                            return
                        podcast_voices_state = voices
                        options = voice_options(voices)
                        default_ids = list(options.keys())
                        podcast_host_voice.set_options(options, value=default_ids[0] if default_ids else None)
                        podcast_guest_voice.set_options(options, value=default_ids[1] if len(default_ids) > 1 else (default_ids[0] if default_ids else None))
                        podcast_guest_2_voice.set_options(options, value=default_ids[2] if len(default_ids) > 2 else (default_ids[0] if default_ids else None))
                        sync_voice_casting()
                        update_voice_preview(podcast_host_voice.value)
                        voice_load_status.value = f"Loaded {len(voices)} ElevenLabs voice(s)."

                    podcast_host_voice.on_value_change(lambda event: (sync_voice_casting(), update_voice_preview(event.value)))
                    podcast_guest_voice.on_value_change(lambda event: (sync_voice_casting(), update_voice_preview(event.value)))
                    podcast_guest_2_voice.on_value_change(lambda event: (sync_voice_casting(), update_voice_preview(event.value)))
                    ui.timer(0.25, load_podcast_voices, once=True)

                    async def generate_podcast_content() -> None:
                        started_at = datetime.now()
                        podcast_draft_progress.visible = True
                        podcast_draft_progress.value = 0
                        podcast_status.value = "Generating podcast draft..."
                        try:
                            await set_generation_progress(
                                progress=podcast_draft_progress,
                                label=podcast_draft_progress_label,
                                status=podcast_status,
                                started_at=started_at,
                                completed_steps=0,
                                total_steps=4,
                                completed_assets=0,
                                total_assets=1,
                                message="Preparing podcast brief...",
                            )
                            await set_generation_progress(
                                progress=podcast_draft_progress,
                                label=podcast_draft_progress_label,
                                status=podcast_status,
                                started_at=started_at,
                                completed_steps=1,
                                total_steps=4,
                                completed_assets=0,
                                total_assets=1,
                                message="Generating podcast script...",
                            )
                            await generate_draft_from_fields(
                                content_type="podcast",
                                topic=podcast_topic.value,
                                related_book=podcast_related_book.value,
                                platform=podcast_platform.value,
                                social_objectives=podcast_social_objectives.value,
                                audience=podcast_audience.value,
                                cta=podcast_cta.value,
                                constraints=podcast_constraints.value,
                                quote_book="Not applicable",
                                quote_moods="Not applicable",
                                character_tags="Not applicable",
                                podcast_format=podcast_format_native.value,
                                podcast_speakers=podcast_speakers_native.value,
                                podcast_roles=podcast_roles_native.value,
                                podcast_tone=podcast_tone_native.value,
                                podcast_length=podcast_length_native.value,
                                elevenlabs_model=podcast_model_native.value,
                                elevenlabs_voice_ids=podcast_voice_ids_native.value,
                                podcast_show_title=podcast_show_title_native.value,
                                podcast_episode_title=podcast_episode_title_native.value,
                                podcast_episode_number=podcast_episode_number_native.value,
                                blog_length="",
                                blog_format="",
                                blog_sections="",
                                blog_structure_options="",
                                blog_seo_keywords="",
                                blog_image_mode="",
                                blog_character="",
                                instagram_formats="",
                                instagram_hashtags="",
                                instagram_hook="",
                                cs_character="",
                                cs_focus="",
                                cs_platform_format="",
                                cs_image_mode="",
                                nl_subject="",
                                nl_preview="",
                                nl_structure="",
                                pr_timing="",
                                pr_embargo_date="",
                                pr_city="",
                                pr_state="",
                                pr_release_date="",
                                pr_contact_name="",
                                pr_contact_title="",
                                pr_organization="",
                                pr_contact_email="",
                                pr_contact_phone="",
                                pr_website="",
                                pr_news_angle="",
                                pr_release_goal="",
                                pr_primary_announcement="",
                                pr_supporting_proof="",
                                pr_quote_source="",
                                pr_target_media="",
                                pr_required_assets="",
                                status=podcast_status,
                                filtered_context_path=podcast_filtered_context_path,
                                prompt_path=podcast_prompt_path,
                                draft_path=podcast_draft_path,
                                generated_output=podcast_generated_output,
                                visual_format_labels=None,
                                visual_theme_name="",
                                saved_draft_path=podcast_saved_draft_path,
                            )
                            await set_generation_progress(
                                progress=podcast_draft_progress,
                                label=podcast_draft_progress_label,
                                status=podcast_status,
                                started_at=started_at,
                                completed_steps=3,
                                total_steps=4,
                                completed_assets=1,
                                total_assets=1,
                                message="Saving podcast draft...",
                            )
                            sync_voice_casting()
                            await finish_generation_progress(
                                progress=podcast_draft_progress,
                                label=podcast_draft_progress_label,
                                status=podcast_status,
                                total_assets=1,
                                message="Podcast draft ready.",
                            )
                        finally:
                            pass

                    async def generate_audio_preview() -> None:
                        podcast_audio_progress.value = 0
                        podcast_audio_progress.visible = True
                        podcast_audio_status.value = "Generating audio preview..."
                        try:
                            await render_podcast_audio_native(
                                script=podcast_generated_output.value,
                                document_name=podcast_episode_title_native.value or podcast_show_title_native.value or "podcast_preview",
                                voices=podcast_voices_state,
                                host_voice=podcast_host_voice.value,
                                guest_voice=podcast_guest_voice.value,
                                guest_2_voice=podcast_guest_2_voice.value,
                                model_id=podcast_model_native.value,
                                stability=podcast_stability.value or 0.5,
                                similarity_boost=podcast_similarity.value or 0.75,
                                style=podcast_style.value or 0.0,
                                speed=podcast_speed.value or 1.0,
                                speaker_boost=bool(podcast_speaker_boost.value),
                                preview_only=True,
                                status=podcast_audio_status,
                                audio_player=podcast_audio_player,
                                full_audio_path=podcast_audio_path,
                                package_path=podcast_audio_package_path,
                                saved_draft_path=podcast_audio_saved_draft_path,
                                progress=podcast_audio_progress,
                                progress_label=podcast_audio_progress_label,
                            )
                        finally:
                            pass

                    async def generate_full_podcast_audio() -> None:
                        podcast_audio_progress.value = 0
                        podcast_audio_progress.visible = True
                        podcast_audio_status.value = "Generating full podcast audio..."
                        try:
                            await render_podcast_audio_native(
                                script=podcast_generated_output.value,
                                document_name=podcast_episode_title_native.value or podcast_show_title_native.value or "podcast",
                                voices=podcast_voices_state,
                                host_voice=podcast_host_voice.value,
                                guest_voice=podcast_guest_voice.value,
                                guest_2_voice=podcast_guest_2_voice.value,
                                model_id=podcast_model_native.value,
                                stability=podcast_stability.value or 0.5,
                                similarity_boost=podcast_similarity.value or 0.75,
                                style=podcast_style.value or 0.0,
                                speed=podcast_speed.value or 1.0,
                                speaker_boost=bool(podcast_speaker_boost.value),
                                preview_only=False,
                                status=podcast_audio_status,
                                audio_player=podcast_audio_player,
                                full_audio_path=podcast_audio_path,
                                package_path=podcast_audio_package_path,
                                saved_draft_path=podcast_audio_saved_draft_path,
                                progress=podcast_audio_progress,
                                progress_label=podcast_audio_progress_label,
                            )
                        finally:
                            pass

                    def download_podcast_script() -> None:
                        if podcast_draft_path.value and Path(str(podcast_draft_path.value)).exists():
                            ui.download(podcast_draft_path.value)
                            return
                        if podcast_saved_draft_path.value and Path(str(podcast_saved_draft_path.value)).exists():
                            ui.download(podcast_saved_draft_path.value)
                            return
                        if str(podcast_generated_output.value or "").strip():
                            saved_path = save_output(podcast_generated_output.value, "podcast", "script_download")
                            podcast_draft_path.value = str(saved_path)
                            ui.download(str(saved_path))
                            return
                        ui.notify("Generate a podcast script before downloading.", type="warning")

                    with podcast_actions:
                        make_primary_button("Generate Podcast Draft", generate_podcast_content)
                    with podcast_result_actions:
                        make_secondary_button("Retry / Regenerate Script", generate_podcast_content)
                        make_secondary_button("Download Script", download_podcast_script)
                    with podcast_audio_actions:
                        make_secondary_button("Generate Audio Preview", generate_audio_preview)
                        make_secondary_button("Generate Full Podcast", generate_full_podcast_audio)
                        make_secondary_button("Retry Audio Preview", generate_audio_preview)
                        make_secondary_button("Retry Full Audio", generate_full_podcast_audio)
                        make_secondary_button(
                            "Download MP3",
                            lambda: ui.download(podcast_audio_path.value) if podcast_audio_path.value else ui.notify("Generate audio first.", type="warning"),
                        )
                        make_secondary_button(
                            "Download Package",
                            lambda: ui.download(podcast_audio_package_path.value) if podcast_audio_package_path.value else ui.notify("Generate audio first.", type="warning"),
                        )

            with ui.tab_panel(saved_drafts_tab).classes("mce-panel"):
                with ui.card().classes("mce-card"):
                    section_heading(
                        "Saved Drafts",
                        "Open, edit, and keep generated topics, campaign bundles, and podcast drafts inside the workspace.",
                    )

                    saved_draft_lookup: dict[str, str] = {}

                    def saved_draft_options() -> list[str]:
                        saved_draft_lookup.clear()
                        options: list[str] = []
                        for record in list_saved_drafts(100):
                            title = record.get("title") or "Untitled draft"
                            content_type_label = record.get("content_type") or "content"
                            updated = record.get("updated_at") or record.get("created_at") or ""
                            label = f"{title} · {content_type_label} · {updated}"
                            if label in saved_draft_lookup:
                                label = f"{label} · {str(record.get('id'))[-8:]}"
                            saved_draft_lookup[label] = str(record.get("id") or "")
                            options.append(label)
                        return options

                    with ui.column().classes("mce-stack w-full"):
                        saved_draft_select = apply_field_props(
                            ui.select(saved_draft_options(), label="Open saved draft"),
                        )
                        with ui.element("div").classes("mce-two-col"):
                            saved_editor_title = apply_field_props(ui.input(label="Draft title"))
                            saved_editor_status = apply_field_props(
                                ui.select(["Draft", "Edited", "Approved", "Needs revision"], value="Draft", label="Revision status")
                            )
                        with ui.element("div").classes("mce-two-col"):
                            saved_editor_type = readonly_input("Content type", "")
                            saved_editor_updated = readonly_input("Last saved", "")
                        saved_editor_draft_id = readonly_input("Draft ID", "")
                        saved_editor_topic = apply_field_props(
                            ui.textarea(label="Original topic / query", placeholder="The prompt or campaign topic used to generate this draft."),
                            "outlined autogrow",
                        )
                        saved_editor_content = apply_field_props(
                            ui.textarea(label="Editable generated result", placeholder="Open a saved draft to edit it here."),
                            "outlined autogrow",
                        )
                        saved_editor_path = readonly_input("Draft file", "")
                        saved_editor_message = readonly_input("Editor status", "Select a draft to open it.")
                        with ui.expansion("Stored brief / generation context", icon="notes").classes("mce-expansion"):
                            saved_editor_brief = readonly_textarea("Brief", "")

                    def load_saved_draft(draft_id: str | None = None) -> None:
                        selected_value = draft_id or saved_draft_select.value
                        selected_id = saved_draft_lookup.get(str(selected_value), str(selected_value or ""))
                        if not selected_id:
                            saved_editor_message.value = "Select a draft to open it."
                            return
                        record = get_saved_draft(str(selected_id))
                        if not record:
                            saved_editor_message.value = "That saved draft could not be found."
                            ui.notify(saved_editor_message.value, type="warning")
                            return
                        metadata = record.get("metadata") or {}
                        saved_editor_title.value = record.get("title") or ""
                        saved_editor_status.value = record.get("status") or "Draft"
                        saved_editor_type.value = record.get("content_type") or ""
                        saved_editor_updated.value = record.get("updated_at") or record.get("created_at") or ""
                        saved_editor_draft_id.value = str(selected_id)
                        saved_editor_topic.value = str(metadata.get("topic") or metadata.get("query") or "")
                        saved_editor_brief.value = str(metadata.get("brief") or "")
                        saved_editor_content.value = read_saved_draft_content(str(selected_id))
                        saved_editor_path.value = record.get("path") or ""
                        saved_editor_message.value = "Draft opened. Edits stay here when you save revisions."

                    def refresh_saved_drafts() -> None:
                        current_label = saved_draft_select.value
                        current_id = saved_editor_draft_id.value
                        options = saved_draft_options()
                        next_value = current_label if current_label in options else None
                        if not next_value and current_id:
                            for label, draft_id in saved_draft_lookup.items():
                                if draft_id == current_id:
                                    next_value = label
                                    break
                        saved_draft_select.set_options(options, value=next_value)
                        if saved_draft_select.value:
                            load_saved_draft(saved_draft_select.value)
                        else:
                            saved_editor_message.value = "No saved drafts found yet."
                        ui.notify("Saved drafts refreshed.", type="positive")

                    def save_saved_draft_revision() -> None:
                        selected_id = saved_editor_draft_id.value or saved_draft_lookup.get(str(saved_draft_select.value), "")
                        if not selected_id:
                            saved_editor_message.value = "Select a draft before saving edits."
                            ui.notify(saved_editor_message.value, type="warning")
                            return
                        updated = update_saved_draft(
                            draft_id=str(selected_id),
                            title=saved_editor_title.value,
                            content=saved_editor_content.value,
                            status=saved_editor_status.value,
                            metadata={
                                "topic": saved_editor_topic.value,
                                "edited_in_platform": True,
                            },
                        )
                        saved_editor_path.value = updated.get("path") or ""
                        saved_editor_updated.value = updated.get("updated_at") or ""
                        options = saved_draft_options()
                        selected_label = None
                        for label, draft_id in saved_draft_lookup.items():
                            if draft_id == updated.get("id"):
                                selected_label = label
                                break
                        saved_draft_select.set_options(options, value=selected_label)
                        saved_editor_draft_id.value = updated.get("id") or ""
                        saved_editor_message.value = "Revision saved."
                        ui.notify("Draft revision saved.", type="positive")

                    def download_saved_draft() -> None:
                        if saved_editor_path.value:
                            ui.download(saved_editor_path.value)
                            return
                        ui.notify("Open a saved draft first.", type="warning")

                    saved_draft_select.on_value_change(lambda event: load_saved_draft(event.value))

                    with ui.row().classes("mce-actions"):
                        make_primary_button("Save Revision", save_saved_draft_revision)
                        make_secondary_button("Refresh Saved Drafts", refresh_saved_drafts)
                        make_secondary_button("Download Draft", download_saved_draft)


if __name__ in {"__main__", "__mp_main__"}:
    Path("outputs").mkdir(exist_ok=True)
    ui.run(
        host="127.0.0.1",
        port=int(os.getenv("NICEGUI_SERVER_PORT", "7860")),
        title=APP_TITLE,
        reload=False,
    )
