from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ContentGenerationRequest(BaseModel):
    content_type: str
    topic: str
    related_book: str = ""
    related_books: list[str] = Field(default_factory=list)
    platform: str = ""
    audience: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    cta: str = ""
    formats: list[str] = Field(default_factory=list)
    visual_formats: list[str] = Field(default_factory=list)
    visual_style: str = ""
    image_source: str = ""
    hashtag_mode: str = ""
    blog_length: str = ""
    blog_format: str = ""
    structure: list[str] = Field(default_factory=list)
    spoiler_level: str = ""
    source_restrictions: str = ""
    length_preference: str = ""
    supporting_image: str = ""
    image_generation: str = ""
    word_count: str = ""
    seo_keyword: str = ""
    heading_depth: str = ""
    metadata_controls: str = ""
    knowledge_sources: list[str] = Field(default_factory=list)
    source_focus: str = ""
    press_timing: str = ""
    publication_date: str = ""
    byline_city: str = ""
    news_angle: str = ""
    auto_news_angle: bool = False
    boilerplate: str = ""
    auto_boilerplate: bool = False
    media_contact: str = ""
    media_contact_name: str = ""
    media_contact_email: str = ""
    media_contact_phone: str = ""
    media_contact_website: str = ""
    companion_assets: list[str] = Field(default_factory=list)


class CampaignGenerationRequest(BaseModel):
    topic: str
    content_types: list[str]
    related_book: str = ""
    platform: str = ""
    audience: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    cta: str = ""


class ChapterPromoRequest(BaseModel):
    books: list[str]
    chapters: list[str]
    platforms: list[str] = Field(default_factory=lambda: ["Instagram", "TikTok", "YouTube", "Newsletter"])
    mode: str = "Tease It"
    genre: str = "Horror"
    promotional_goal: str = "Read the Next Chapter"
    moods: list[str] = Field(default_factory=lambda: ["Ominous"])
    style_variant: str = "spoiler_safe_teaser"
    spoiler_level: str = "Almost Nothing"
    spoiler_notes: str = ""
    cta: str = ""
    optional_quote: str = ""
    hooks: list[str] = Field(default_factory=lambda: ["Choose the Strongest Hook for Me"])
    focus_character: str = ""
    teaser_pillar: str = "Choose the Best Structure for Me"
    promo_duration: str = "30-Second Chapter Promo"
    genre_promo_mode: str = "Match the Chapter Automatically"
    promo_outputs: list[str] = Field(default_factory=list)


class PodcastGenerationRequest(BaseModel):
    topic: str
    source_title: str = ""
    show_title: str = ""
    episode_title: str = ""
    episode_number: str = ""
    destination: str = "Spotify"
    podcast_format: str = "roundtable discussion"
    speakers: str = "3"
    speaker_roles: str = "Host, Co-host, Guest"
    tone: list[str] = Field(default_factory=lambda: ["cinematic"])
    audience: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    performance_cues: list[str] = Field(default_factory=list)
    custom_constraints: str = ""
    model_id: str = "eleven_multilingual_v2"
    target_length: str = "5-7 minutes"
    cta: str = ""
    knowledge_sources: list[str] = Field(default_factory=list)
    source_focus: str = ""


class PodcastRenderRequest(BaseModel):
    voice_map: dict[str, str] = Field(default_factory=dict)
    host_voice_id: str = ""
    guest_voice_id: str = ""
    guest_2_voice_id: str = ""
    model_id: str = "eleven_multilingual_v2"
    stability: float | None = None
    similarity_boost: float | None = None
    style: float | None = None
    speed: float | None = None
    use_speaker_boost: bool | None = None


class DraftUpdateRequest(BaseModel):
    title: str
    content: str
    status: str = "Draft"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DraftCreateRequest(DraftUpdateRequest):
    content_type: str = "content"
    source_path: str = ""


class ExportRequest(BaseModel):
    title: str
    content: str
    format: Literal["docx", "markdown", "txt"] = "docx"


class VoicePreviewRequest(BaseModel):
    text: str = "This is a TellTales Ink voice preview."


class PerformanceCue(BaseModel):
    id: str
    type: str
    value: str = ""


class PodcastSpeaker(BaseModel):
    id: str
    name: str
    role: str = ""
    voice_id: str | None = None


class PodcastTurn(BaseModel):
    id: str
    speaker_id: str
    text: str
    cues: list[PerformanceCue] = Field(default_factory=list)
    estimated_duration_seconds: float | None = None


class PodcastSegment(BaseModel):
    id: str
    title: str
    turns: list[PodcastTurn]


class Draft(BaseModel):
    id: str
    title: str
    content_type: str
    status: str = "Draft"
    created_at: str = ""
    updated_at: str = ""
    path: str = ""
    source_path: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: str = ""


class PodcastDraft(Draft):
    content_type: str = "podcast"
    show_title: str = ""
    episode_title: str = ""
    speakers: list[PodcastSpeaker] = Field(default_factory=list)
    segments: list[PodcastSegment] = Field(default_factory=list)
    raw_script: str | None = None


class Artifact(BaseModel):
    id: str
    path: str
    filename: str
    media_type: str = "application/octet-stream"


class Job(BaseModel):
    id: str
    status: Literal["queued", "running", "complete", "failed"] = "queued"
    message: str = ""
    progress: float = 0.0
    result: dict[str, Any] = Field(default_factory=dict)
