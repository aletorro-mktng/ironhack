import type { components } from "./api.generated";

export type Artifact = components["schemas"]["Artifact"];
export type DraftRecord = components["schemas"]["Draft"];
export type Job = components["schemas"]["Job"];
export type PerformanceCue = components["schemas"]["PerformanceCue"];
export type PodcastDraft = components["schemas"]["PodcastDraft"];
export type PodcastGenerationRequest = components["schemas"]["PodcastGenerationRequest"];
export type PodcastRenderRequest = components["schemas"]["PodcastRenderRequest"];
export type PodcastSegment = components["schemas"]["PodcastSegment"];
export type PodcastSpeaker = components["schemas"]["PodcastSpeaker"];
export type PodcastTurn = components["schemas"]["PodcastTurn"];

export type OptionsResponse = {
  contentTypes: string[];
  books: string[];
  chapters: { id: string; book: string; label: string }[];
  platforms: string[];
  characters: string[];
  audiences: string[];
  objectives: string[];
  constraints: string[];
  podcast: { destinations: string[]; formats: string[]; tones: string[]; lengths: string[] };
  chapterPromos: {
    modes: string[];
    goals: string[];
    moods: string[];
    hooks: string[];
    genres: string[];
    teaserPillars: string[];
    durations: string[];
    genrePromoModes: string[];
    styleVariants: { value: string; label: string }[];
    promoOutputs: { value: string; label: string }[];
  };
};

export type GalleryImage = {
  path: string;
  filename: string;
  type_label: string;
  topic: string;
  format: string;
  aspect: string;
  date: string;
  size_kb: number;
  url?: string;
};

export type AudioEpisode = {
  title: string;
  mode: string;
  model: string;
  segments: number;
  mp3_path: string;
  date: string;
  size_mb: number;
  url?: string;
};

export type Voice = {
  voice_id: string;
  name: string;
  category: string;
  preview_url?: string;
};

export type Deliverable = {
  id: string;
  label: string;
  type: string;
  platform?: string;
  title: string;
  content: string;
  characterCount: number;
  status: "generated" | "edited" | "pending" | "error";
};

export type GeneratedDraft = {
  title: string;
  contentType: string;
  deliverables: Deliverable[];
  rawSource: string;
};

export type ChapterPromoForm = {
  books: string[];
  chapters: string[];
  platforms: string[];
  mode: string;
  genre: string;
  promotional_goal: string;
  moods: string[];
  style_variant: string;
  spoiler_level: string;
  spoiler_notes: string;
  cta: string;
  optional_quote: string;
  hooks: string[];
  focus_character: string;
  teaser_pillar: string;
  promo_duration: string;
  genre_promo_mode: string;
  promo_outputs: string[];
};

export type GenericForm = {
  content_type: string;
  topic: string;
  related_book: string;
  related_books?: string[];
  platform: string;
  audience: string[];
  objectives: string[];
  constraints: string[];
  cta: string;
  formats?: string[];
  visual_formats?: string[];
  visual_style?: string;
  image_source?: string;
  hashtag_mode?: string;
  blog_length?: string;
  blog_format?: string;
  structure?: string[];
  press_timing?: string;
  companion_assets?: string[];
  quote_moods?: string[];
  campaign_goal?: string;
  campaign_name?: string;
  duration?: string;
  cadence?: string;
  tone?: string[];
  quantities?: Record<string, string>;
  destination_ids?: string[];
  spoiler_level?: string;
  source_restrictions?: string;
  length_preference?: string;
  required_cta?: string;
  caption_length?: string;
  emoji_preference?: string;
  supporting_image?: string;
  platform_adaptations?: string;
  slide_count?: string;
  carousel_structure?: string;
  cover_hook?: string;
  final_slide_cta?: string;
  image_generation?: string;
  quote_source?: string;
  attribution?: string;
  graphic_orientation?: string;
  supporting_caption?: string;
  hook_style?: string;
  shot_density?: string;
  voice_over?: string;
  word_count?: string;
  seo_keyword?: string;
  heading_depth?: string;
  metadata_controls?: string;
  news_angle?: string;
  dateline?: string;
  boilerplate?: string;
  media_contact?: string;
  media_contact_name?: string;
  media_contact_email?: string;
  media_contact_phone?: string;
  media_contact_website?: string;
  byline_city?: string;
  publication_date?: string;
  auto_news_angle?: boolean;
  auto_boilerplate?: boolean;
  subject_style?: string;
  preview_text?: string;
  content_sections?: string;
  sender_voice?: string;
  podcast_format?: string;
  speaker_count?: string;
  roles?: string;
  runtime?: string;
  voice_settings?: string;
  knowledge_sources?: string[];
  source_focus?: string;
};

export type PodcastForm = {
  topic: string;
  show_title: string;
  episode_title: string;
  episode_number: string;
  destination: string;
  podcast_format: string;
  speakers: string;
  speaker_roles: string;
  tone: string[];
  target_length: string;
  cta: string;
  audience?: string[];
  constraints?: string[];
  model_id?: string;
  performance_cues?: string[];
  custom_constraints?: string;
  knowledge_sources?: string[];
  source_focus?: string;
};
