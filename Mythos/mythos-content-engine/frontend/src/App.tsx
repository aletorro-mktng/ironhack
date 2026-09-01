import {
  ArrowLeft,
  Bookmark,
  Calendar,
  CheckCircle2,
  ChevronRight,
  Clapperboard,
  Code2,
  Copy,
  Download,
  Edit3,
  FileAudio,
  FileText,
  Flame,
  GalleryHorizontalEnd,
  Hash,
  Eye,
  Image as ImageIcon,
  Layers,
  LayoutDashboard,
  Mail,
  Megaphone,
  Mic2,
  MoreVertical,
  Play,
  Plus,
  RefreshCw,
  Rocket,
  Search,
  Save,
  Scissors,
  Sparkles,
  Bell,
  BookOpen,
  MessageSquare,
  Settings,
  Target,
  Users,
  Video,
  Wand2
} from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import { MultiSelect, SingleSelect } from "./components/Fields";
import { labelize, markdownToHtml, normalizeDraft, optionize, stripMarkdown } from "./features/generator/draftModel";
import { useJobPolling } from "./hooks/useJobPolling";
import type {
  AudioEpisode,
  ChapterPromoForm,
  Deliverable,
  DraftRecord,
  GalleryImage,
  GeneratedDraft,
  GenericForm,
  Job,
  OptionsResponse,
  PodcastDraft,
  PodcastForm,
  Voice
} from "./types";

const INSTAGRAM_FORMATS = ["Feed post", "Reel", "Story", "Carousel"];
const INSTAGRAM_HASHTAGS = ["no hashtags", "3-5 niche hashtags", "10-15 mixed hashtags", "up to 30 for max reach"];
const LINKEDIN_FORMATS = ["Standard post", "Article", "Document / carousel", "Poll", "Newsletter"];
const YOUTUBE_FORMATS = ["Long-form video", "Short", "Community post", "Premiere", "Title", "Description", "Pinned comment"];
const BLOG_LENGTHS = ["short: 300-500 words", "standard: 600-900 words", "long: 1200-1800 words", "deep dive: 2000+ words"];
const BLOG_FORMATS = ["critical essay", "listicle", "behind-the-scenes", "SEO article", "reader guide", "character analysis", "theme essay", "award announcement feature", "author Q&A", "book club guide", "review roundup", "comparison article", "news update", "launch post", "worldbuilding deep dive"];
const BLOG_STRUCTURE = ["suggest structure", "hook introduction", "subheaded sections", "quote integration", "CTA ending", "spoiler warning", "pull quote block", "reader takeaway", "metadata summary"];
const IMAGE_GENERATION_MODES = ["Generate visuals", "Create prompt only", "Do not generate visuals"];
const IMAGE_FORMATS = ["blog cover 16:9", "horizontal 1.91:1", "square 1:1", "portrait 4:5", "story/reel 9:16", "thumbnail 16:9", "transparent cutout", "quote graphic"];
const KNOWLEDGE_SOURCE_OPTIONS = ["Manuscript/RAG moments", "Pull quotes", "Reviews", "Character info", "Awards & recognition", "Brand voice"];
const VISUAL_STYLES = ["Gothic", "Press", "Dark Academia", "BookTok", "Cinematic"];
const QUOTE_MOODS = ["revenge", "justice", "betrayal", "survival", "Dominican folklore", "Grim-Cojuelo", "dark academia shame", "campy menace", "horror"];
const CHARACTER_IMAGE_MODES = ["No image", "Use character portrait", "Generate new image", "Use uploaded/base image"];
const PRESS_TIMINGS = ["FOR IMMEDIATE RELEASE", "Publish after specific date"];
const PRESS_ASSETS = ["media pitch email", "author bio", "book factsheet", "cover image brief", "pull quotes", "social announcement"];
const CAMPAIGN_DURATIONS = ["1 week", "2 weeks", "1 month", "3 months"];
const CAMPAIGN_CADENCE = ["daily", "3x per week", "weekly", "launch week burst"];
const PODCAST_CUES = ["Pause", "Emphasis", "Slower", "Whisper", "Tense", "Sound effect"];
const PODCAST_MODELS = ["eleven_v3", "eleven_multilingual_v2", "eleven_flash_v2_5"];
const PODCAST_RECOMMENDED_TONES = ["Cinematic", "Investigative", "Conversational", "Darkly Funny", "Literary", "Critical", "Intimate", "Dramatic", "Educational", "Press-Friendly"];
const PODCAST_RECOMMENDED_AUDIENCES = ["Potential Readers", "New Readers", "Existing Fans", "BookTok Readers", "Bookstagram Readers", "Influencers", "Journalists", "Reviewers", "ARC Readers", "YA Thriller Readers", "Horror Readers", "Dark Academia Readers", "General Audience"];
const PODCAST_RECOMMENDED_CONSTRAINTS = ["Spoiler-Free", "Use Only Quote Bank", "Use Only Real Reviews", "Avoid Major Spoilers", "Make It Short", "Make It Punchy", "Make It Cinematic", "Make It Savage", "Make It Funny", "Make It Emotionally Intense", "Include CTA", "Include Hashtags", "No Hashtags"];
const CHAPTER_PROMO_FALLBACK_DELIVERABLES = [
  "Campaign Overview",
  "Instagram Caption",
  "Facebook Post",
  "X / Threads",
  "Email Teaser",
  "Image Brief",
  "Hashtags & Metadata"
];
const BOOKSTAGRAM_HASHTAG_POOL = [
  "#Bookstagram",
  "#Bookworm",
  "#Bookish",
  "#Booklover",
  "#ReadersofInstagram",
  "#Bookrecommendations",
  "#Igreads",
  "#Bookstack",
  "#Currentlyreading",
  "#Bookreviews",
  "#Amreading",
  "#Bookshelfie",
  "#Bookphotography",
  "#Booktography",
  "#Bookhaul",
  "#Unboxing",
  "#CaribbeanGothic",
  "#DarkAcademiaBooks",
  "#PsychologicalThrillerBook",
  "#IndieAuthorBookstagram",
  "#FolkHorror",
  "#DarkAcademiaAesthetic",
  "#ThrillerReads",
  "#DarkFiction",
  "#HorrorBookstagram",
  "#SuspenseNovel",
  "#BookishCommunity",
  "#ThrillerBooks",
  "#PsychologicalThriller",
  "#BookReviewer"
];

function revisionDirection(instruction: string) {
  const request = instruction.trim();
  if (!request) return "";
  const lower = request.toLowerCase();
  if (lower.includes("shorten")) return "Regenerate the selected draft shorter and tighter.";
  if (lower.includes("cinematic")) return "Regenerate with a more cinematic angle: stronger visual beats, trailer-like pacing, and sharper atmosphere.";
  if (lower.includes("tension")) return "Regenerate with higher tension while preserving spoiler limits and withholding major reveals.";
  if (lower.includes("cta") || lower.includes("call to action")) return "Regenerate with a stronger, clearer call to action.";
  return `Regenerate using this revision angle: ${request}`;
}

function hashtagTextFromDraft(draft: GeneratedDraft, selected?: Deliverable) {
  const sources = [
    selected?.content || "",
    ...draft.deliverables
      .filter((item) => /hashtag/i.test(`${item.label} ${item.title}`))
      .map((item) => item.content),
    draft.rawSource
  ];
  const found = sources.join("\n").match(/#[\p{L}\p{N}_-]+/gu) || [];
  return [...new Set([...found, ...BOOKSTAGRAM_HASHTAG_POOL])].slice(0, 18).join(" ");
}

function titleCaseLabel(value: string) {
  return labelize(value).replace(/\bV(\d)\b/g, "V$1");
}

function roleDefaultsForFormat(format: string, count: number) {
  const lower = format.toLowerCase();
  const defaults = lower.includes("monologue")
    ? ["Host"]
    : lower.includes("interview")
      ? ["Host", "Guest"]
      : lower.includes("debate")
        ? ["Host", "Advocate", "Skeptic"]
        : ["Host", "Co-host", "Guest", "Guest"];
  return Array.from({ length: count }, (_, index) => defaults[index] || `Speaker ${index + 1}`);
}

function roleListFromValue(value: string, format: string, count: number) {
  const existing = value.split(",").map((item) => item.trim()).filter(Boolean);
  const defaults = roleDefaultsForFormat(format, count);
  return Array.from({ length: count }, (_, index) => existing[index] || defaults[index] || `Speaker ${index + 1}`);
}

function defaultChapterForm(options?: OptionsResponse): ChapterPromoForm {
  return {
    books: options?.books.slice(0, 1) || [],
    chapters: options?.chapters.slice(0, 1).map((chapter) => chapter.id) || [],
    platforms: ["Instagram", "TikTok", "YouTube", "Newsletter"],
    mode: options?.chapterPromos.modes[0] || "Tease It",
    genre: options?.chapterPromos.genres[0] || "Horror",
    promotional_goal: options?.chapterPromos.goals[0] || "Read the Next Chapter",
    moods: [options?.chapterPromos.moods[0] || "Ominous"],
    style_variant: options?.chapterPromos.styleVariants[0]?.value || "spoiler_safe_teaser",
    spoiler_level: "Almost Nothing",
    spoiler_notes: "",
    cta: "",
    optional_quote: "",
    hooks: [options?.chapterPromos.hooks[0] || "Choose the Strongest Hook for Me"],
    focus_character: "",
    teaser_pillar: options?.chapterPromos.teaserPillars[0] || "Choose the Best Structure for Me",
    promo_duration: options?.chapterPromos.durations[2] || "30-Second Chapter Promo",
    genre_promo_mode: options?.chapterPromos.genrePromoModes[0] || "Match the Chapter Automatically",
    promo_outputs: options?.chapterPromos.promoOutputs.map((item) => item.value) || []
  };
}

function defaultGenericForm(options?: OptionsResponse): GenericForm {
  return {
    content_type: options?.contentTypes.find((type) => !["podcast", "chapter_promos"].includes(type)) || "instagram_caption",
    topic: "",
    related_book: options?.books[0] || "",
    related_books: options?.books[0] ? [options.books[0]] : [],
    platform: "Instagram",
    audience: ["YA Horror Readers", "Existing Fans"],
    objectives: ["Promote the book"],
    constraints: [],
    cta: "Read Mortal Vengeance",
    formats: ["Reel Script"],
    tone: ["Cinematic", "Intense"],
    knowledge_sources: ["Manuscript/RAG moments", "Pull quotes", "Character info"]
  };
}

function contentSettingsFor(type: string) {
  if (type === "instagram_caption") return "instagram";
  if (type === "linkedin_content") return "linkedin";
  if (type === "youtube_content") return "youtube";
  if (type === "blog_post") return "blog";
  if (type === "quote_post" || type === "review_pull_quote") return "quote";
  if (type === "character_spotlight") return "character";
  if (type === "press_release") return "press";
  return "general";
}

type DeliverableCategory = "Social" | "Video" | "Long-form" | "Email" | "PR" | "Podcast";
type FieldId =
  | "related_book"
  | "objective"
  | "audience"
  | "topic"
  | "tone"
  | "cta"
  | "spoiler_level"
  | "destinations"
  | "caption_length"
  | "hashtag_strategy"
  | "emoji_preference"
  | "supporting_image"
  | "platform_adaptations"
  | "slide_count"
  | "carousel_structure"
  | "cover_hook"
  | "final_slide_cta"
  | "image_generation"
  | "quote_source"
  | "attribution"
  | "graphic_orientation"
  | "supporting_caption"
  | "duration"
  | "hook_style"
  | "shot_density"
  | "voice_over"
  | "visual_style"
  | "word_count"
  | "seo_keyword"
  | "heading_depth"
  | "metadata_controls"
  | "news_angle"
  | "dateline"
  | "boilerplate"
  | "media_contact"
  | "byline_city"
  | "publication_date"
  | "subject_style"
  | "preview_text"
  | "content_sections"
  | "sender_voice"
  | "podcast_format"
  | "speaker_count"
  | "roles"
  | "runtime"
  | "voice_settings"
  | "source_restrictions"
  | "length_preference"
  | "required_cta";
type OutputDefinition = { id: string; label: string };
type GeneratorDeliverable = {
  id: string;
  contentType: string;
  label: string;
  category: DeliverableCategory;
  description: string;
  icon: typeof Sparkles;
  defaultDestinationIds: string[];
  allowedDestinationIds: string[];
  requiredFields: FieldId[];
  optionalFields: FieldId[];
  advancedFields: FieldId[];
  defaultObjective: string;
  availableObjectives: string[];
  defaultOutputs: OutputDefinition[];
  generationButtonLabel: string;
  topicPlaceholder: string;
};

const GENERATOR_DELIVERABLES: GeneratorDeliverable[] = [
  {
    id: "instagram-caption",
    contentType: "instagram_caption",
    label: "Instagram Caption",
    category: "Social",
    description: "Short, engaging caption for your feed.",
    icon: ImageIcon,
    defaultDestinationIds: ["Instagram"],
    allowedDestinationIds: ["Instagram"],
    requiredFields: ["related_book", "objective", "audience", "topic", "tone", "cta"],
    optionalFields: ["spoiler_level", "caption_length", "hashtag_strategy", "emoji_preference", "supporting_image"],
    advancedFields: ["caption_length", "hashtag_strategy", "emoji_preference", "supporting_image"],
    defaultObjective: "Promote the book",
    availableObjectives: ["Promote the book", "Drive preorders", "Announce a review", "Build reader anticipation"],
    defaultOutputs: [{ id: "caption", label: "Caption" }, { id: "cta", label: "CTA" }, { id: "hashtags", label: "Hashtags" }, { id: "visual-prompt", label: "Optional visual prompt" }],
    generationButtonLabel: "Generate Instagram Caption",
    topicPlaceholder: "What should this Instagram caption communicate?"
  },
  {
    id: "social-caption",
    contentType: "instagram_caption",
    label: "Social Caption",
    category: "Social",
    description: "Adaptable copy for social channels.",
    icon: MessageSquare,
    defaultDestinationIds: ["Instagram"],
    allowedDestinationIds: ["Instagram", "Facebook", "LinkedIn", "Threads", "X"],
    requiredFields: ["destinations", "related_book", "objective", "audience", "topic", "tone", "cta"],
    optionalFields: ["spoiler_level", "platform_adaptations", "hashtag_strategy", "supporting_image"],
    advancedFields: ["platform_adaptations", "hashtag_strategy", "visual_style"],
    defaultObjective: "Promote the book",
    availableObjectives: ["Promote the book", "Start conversation", "Share an update", "Drive link clicks"],
    defaultOutputs: [{ id: "core-caption", label: "Core caption" }, { id: "adaptations", label: "Platform adaptations" }, { id: "cta", label: "CTA" }, { id: "hashtags", label: "Hashtags" }],
    generationButtonLabel: "Generate Social Captions",
    topicPlaceholder: "What should these social captions communicate?"
  },
  {
    id: "carousel",
    contentType: "instagram_caption",
    label: "Carousel",
    category: "Social",
    description: "Multi-slide post to tell a bigger story.",
    icon: Copy,
    defaultDestinationIds: ["Instagram"],
    allowedDestinationIds: ["Instagram"],
    requiredFields: ["related_book", "objective", "audience", "topic", "tone", "cta", "slide_count"],
    optionalFields: ["spoiler_level", "carousel_structure", "cover_hook", "final_slide_cta", "image_generation"],
    advancedFields: ["slide_count", "carousel_structure", "cover_hook", "final_slide_cta", "image_generation"],
    defaultObjective: "Promote the book",
    availableObjectives: ["Promote the book", "Educate readers", "Reveal themes", "Share quotes"],
    defaultOutputs: [{ id: "slides", label: "Slide-by-slide copy" }, { id: "caption", label: "Caption" }, { id: "hashtags", label: "Hashtags" }, { id: "visual-prompts", label: "Visual prompts" }],
    generationButtonLabel: "Generate Carousel",
    topicPlaceholder: "What story should this carousel tell?"
  },
  {
    id: "quote-post",
    contentType: "quote_post",
    label: "Quote Post",
    category: "Social",
    description: "Book quote with branded graphic direction.",
    icon: Bookmark,
    defaultDestinationIds: ["Instagram"],
    allowedDestinationIds: ["Instagram", "Facebook", "X", "Threads"],
    requiredFields: ["related_book", "objective", "audience", "topic", "tone", "quote_source"],
    optionalFields: ["attribution", "graphic_orientation", "supporting_caption", "hashtag_strategy"],
    advancedFields: ["quote_source", "attribution", "graphic_orientation", "supporting_caption", "hashtag_strategy"],
    defaultObjective: "Share a memorable quote",
    availableObjectives: ["Share a memorable quote", "Increase intrigue", "Build mood"],
    defaultOutputs: [{ id: "quote", label: "Selected quote" }, { id: "graphic-copy", label: "Graphic copy" }, { id: "caption", label: "Caption" }, { id: "visual-prompt", label: "Visual prompt" }],
    generationButtonLabel: "Generate Quote Post",
    topicPlaceholder: "Paste a quote or describe the quote mood you need."
  },
  {
    id: "reel-script",
    contentType: "instagram_caption",
    label: "Reel Script",
    category: "Video",
    description: "Short-form script with scene and shot ideas.",
    icon: Clapperboard,
    defaultDestinationIds: ["Instagram"],
    allowedDestinationIds: ["Instagram", "TikTok", "Facebook Reels", "YouTube Shorts"],
    requiredFields: ["destinations", "related_book", "objective", "audience", "topic", "tone", "duration"],
    optionalFields: ["spoiler_level", "hook_style", "shot_density", "voice_over", "visual_style"],
    advancedFields: ["duration", "hook_style", "shot_density", "voice_over", "visual_style", "hashtag_strategy"],
    defaultObjective: "Promote the book",
    availableObjectives: ["Promote the book", "Tease a scene", "Introduce a character", "Drive preorders"],
    defaultOutputs: [{ id: "timed-script", label: "Timed script" }, { id: "scene-directions", label: "Scene directions" }, { id: "shot-list", label: "Shot list" }, { id: "caption", label: "Caption" }, { id: "hashtags", label: "Hashtags" }, { id: "visual-prompt", label: "Visual prompt" }],
    generationButtonLabel: "Generate Reel Package",
    topicPlaceholder: "What should this Reel communicate?"
  },
  {
    id: "blog-article",
    contentType: "blog_post",
    label: "Blog Article",
    category: "Long-form",
    description: "Long-form article to inform and inspire.",
    icon: FileText,
    defaultDestinationIds: ["Website"],
    allowedDestinationIds: ["Website", "WordPress", "Webflow", "Medium", "Generic HTML"],
    requiredFields: ["related_book", "objective", "audience", "topic", "tone", "word_count"],
    optionalFields: ["spoiler_level", "seo_keyword", "heading_depth", "metadata_controls", "image_generation", "visual_style", "supporting_image"],
    advancedFields: ["word_count", "seo_keyword", "heading_depth", "metadata_controls"],
    defaultObjective: "Inform readers",
    availableObjectives: ["Inform readers", "Improve SEO", "Explain themes", "Announce news"],
    defaultOutputs: [{ id: "titles", label: "Title options" }, { id: "seo-title", label: "SEO title" }, { id: "meta", label: "Meta description" }, { id: "slug", label: "Slug" }, { id: "outline", label: "Outline" }, { id: "article", label: "Article" }, { id: "cta", label: "CTA" }],
    generationButtonLabel: "Generate Blog Article",
    topicPlaceholder: "What should this article explore?"
  },
  {
    id: "press-release",
    contentType: "press_release",
    label: "Press Release",
    category: "PR",
    description: "News-ready copy for media and PR.",
    icon: Megaphone,
    defaultDestinationIds: [],
    allowedDestinationIds: ["Website Newsroom", "Email Pitch", "Distribution Service"],
    requiredFields: ["related_book", "objective", "topic", "news_angle", "byline_city", "media_contact"],
    optionalFields: ["boilerplate", "publication_date", "source_restrictions"],
    advancedFields: ["source_restrictions", "length_preference"],
    defaultObjective: "Announce news",
    availableObjectives: ["Announce news", "Award announcement", "Launch announcement", "Media advisory"],
    defaultOutputs: [{ id: "headline", label: "Headline" }, { id: "subheadline", label: "Subheadline" }, { id: "dateline", label: "Dateline" }, { id: "body", label: "Release body" }, { id: "boilerplate", label: "Boilerplate" }, { id: "media-contact", label: "Media contact" }],
    generationButtonLabel: "Generate Press Release",
    topicPlaceholder: "What is the announcement or news angle?"
  },
  {
    id: "newsletter",
    contentType: "newsletter_blurb",
    label: "Newsletter",
    category: "Email",
    description: "Engaging update for subscribers.",
    icon: Mail,
    defaultDestinationIds: ["Email"],
    allowedDestinationIds: ["Email"],
    requiredFields: ["related_book", "objective", "audience", "topic", "tone", "cta"],
    optionalFields: ["spoiler_level", "subject_style", "preview_text", "content_sections", "sender_voice"],
    advancedFields: ["subject_style", "preview_text", "content_sections", "sender_voice"],
    defaultObjective: "Drive opens and clicks",
    availableObjectives: ["Drive opens and clicks", "Announce release", "Share behind the scenes", "Invite reviews"],
    defaultOutputs: [{ id: "subjects", label: "Subject options" }, { id: "preview", label: "Preview text" }, { id: "body", label: "Email body" }, { id: "cta", label: "CTA" }],
    generationButtonLabel: "Generate Newsletter",
    topicPlaceholder: "What should subscribers know?"
  },
  {
    id: "podcast-episode",
    contentType: "podcast",
    label: "Podcast Episode",
    category: "Podcast",
    description: "Outline and script for an episode.",
    icon: Mic2,
    defaultDestinationIds: [],
    allowedDestinationIds: [],
    requiredFields: ["related_book", "objective", "topic", "podcast_format", "speaker_count", "runtime"],
    optionalFields: ["tone", "roles", "voice_settings", "cta"],
    advancedFields: ["podcast_format", "speaker_count", "roles", "runtime", "voice_settings"],
    defaultObjective: "Create episode script",
    availableObjectives: ["Create episode script", "Discuss themes", "Interview format", "Roundtable"],
    defaultOutputs: [{ id: "title", label: "Title" }, { id: "outline", label: "Outline" }, { id: "script", label: "Script" }, { id: "directions", label: "Speaker directions" }, { id: "show-notes", label: "Show notes" }, { id: "promo", label: "Promotional copy" }],
    generationButtonLabel: "Generate Podcast Package",
    topicPlaceholder: "What should this podcast episode cover?"
  },
  {
    id: "campaign-package",
    contentType: "instagram_caption",
    label: "Campaign Package",
    category: "Social",
    description: "Coordinated assets from one central brief.",
    icon: Sparkles,
    defaultDestinationIds: [],
    allowedDestinationIds: [],
    requiredFields: [],
    optionalFields: [],
    advancedFields: [],
    defaultObjective: "Promote the book",
    availableObjectives: ["Promote the book", "Launch campaign", "Award push", "Seasonal campaign"],
    defaultOutputs: [{ id: "campaign", label: "Strategic campaign workflow" }, { id: "mix", label: "Multi-channel content mix" }, { id: "schedule", label: "Review and schedule" }],
    generationButtonLabel: "Open Campaign Mode",
    topicPlaceholder: "Open Campaign Mode to plan a coordinated package."
  }
];

function defaultPodcastForm(options?: OptionsResponse): PodcastForm {
  return {
    topic: "",
    show_title: "",
    episode_title: "",
    episode_number: "",
    destination: options?.podcast.destinations[0] || "Spotify",
    podcast_format: options?.podcast.formats[0] || "roundtable discussion",
    speakers: "3",
    speaker_roles: "Host, Co-host, Guest",
    tone: options?.podcast.tones.slice(0, 1) || ["cinematic"],
    audience: ["Potential Readers"],
    constraints: ["Spoiler-Free", "Avoid Major Spoilers"],
    performance_cues: ["Pause", "Emphasis"],
    model_id: PODCAST_MODELS[1],
    target_length: options?.podcast.lengths[0] || "5-7 minutes",
    cta: "",
    custom_constraints: "",
    knowledge_sources: ["Manuscript/RAG moments", "Pull quotes", "Character info"],
    source_focus: ""
  };
}

function podcastToGeneratedDraft(record: PodcastDraft): GeneratedDraft {
  const rawSource = record.raw_script || record.content || "";
  const deliverables = (record.segments || []).map((segment) => ({
    id: segment.id,
    label: segment.title,
    type: "podcast_segment",
    title: segment.title,
    content: segment.turns.map((turn) => {
      const speaker = (record.speakers || []).find((item) => item.id === turn.speaker_id);
      return `[${speaker?.name || turn.speaker_id}]\n${turn.text}`;
    }).join("\n\n"),
    characterCount: segment.turns.reduce((total, turn) => total + stripMarkdown(turn.text).length, 0),
    status: "generated" as const
  }));
  return {
    title: record.episode_title || record.title,
    contentType: record.content_type,
    deliverables: deliverables.length ? deliverables : normalizeDraft(rawSource, record.title, "podcast").deliverables,
    rawSource
  };
}

function deliverableIcon(item: Deliverable) {
  const text = `${item.label} ${item.platform}`.toLowerCase();
  if (text.includes("email") || text.includes("newsletter")) return <Mail size={20} />;
  if (text.includes("hashtag")) return <Hash size={20} />;
  if (text.includes("image") || text.includes("thumbnail")) return <ImageIcon size={20} />;
  if (text.includes("podcast")) return <Mic2 size={20} />;
  return <FileText size={20} />;
}

function chapterPromotionDeliverables(draft: GeneratedDraft): Deliverable[] {
  const filtered = draft.deliverables.filter((item) => !/creative worksheet/i.test(`${item.label} ${item.title}`));
  if (filtered.length) return filtered;
  return CHAPTER_PROMO_FALLBACK_DELIVERABLES.map((label, index) => ({
    id: `chapter-promo-${index}`,
    label,
    type: "chapter_promo",
    title: label,
    content: draft.rawSource,
    characterCount: stripMarkdown(draft.rawSource).length,
    status: "generated" as const
  }));
}

type ChapterPromoGroup = {
  id: string;
  label: string;
  summary: string;
  deliverables: Deliverable[];
};

type ChapterOutputGroup = {
  id: string;
  label: string;
  icon: typeof FileText;
  deliverables: Deliverable[];
};

const chapterGroupId = (value: string, index: number) => `chapter-group-${index}-${value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "chapter"}`;
const deliverableId = (value: string, index: number) => `${value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "deliverable"}-${index}`;
const chapterNumberFromLabel = (value: string, index: number) => {
  const match = value.match(/\bchapter\s+(\d+)\b/i) || value.match(/\bch\s+(\d+)\b/i);
  return match ? match[1].padStart(2, "0") : String(index + 1).padStart(2, "0");
};
const chapterTitleFromLabel = (value: string) => {
  const withoutBook = value.includes(" - ") ? value.split(" - ").slice(1).join(" - ") : value;
  return withoutBook.replace(/^Chapter\s+\d+\s*:?\s*/i, "").replace(/^Ch\s+\d+\s*:?\s*/i, "").trim() || withoutBook;
};
const bookFromChapterLabel = (value: string) => value.includes(" - ") ? value.split(" - ")[0].trim() : "";
const chapterContextLabel = (value: string) => {
  const withoutBook = value.includes(" - ") ? value.split(" - ").slice(1).join(" - ").trim() : value;
  const match = withoutBook.match(/^Chapter\s+(\d+)\s*:?\s*(.+)$/i) || withoutBook.match(/^Ch\s+(\d+)\s*:?\s*(.+)$/i);
  return match ? `Chapter ${match[1]} · ${match[2].trim()}` : withoutBook;
};

function chapterPromotionGroups(draft: GeneratedDraft): ChapterPromoGroup[] {
  const source = draft.rawSource.trim();
  const matches = [...source.matchAll(/^(#{2,3})\s+(.+?)\s*$/gm)];
  const groups: ChapterPromoGroup[] = [];

  matches.forEach((match, index) => {
    const level = match[1].length;
    const heading = match[2].trim();
    const start = (match.index || 0) + match[0].length;
    const end = index + 1 < matches.length ? matches[index + 1].index || source.length : source.length;
    const content = source.slice(start, end).trim();
    if (/creative worksheet/i.test(`${heading} ${content}`)) return;

    if (level === 2) {
      groups.push({
        id: chapterGroupId(heading, groups.length),
        label: heading,
        summary: content,
        deliverables: [{
          id: deliverableId(heading, index),
          label: "Chapter Overview",
          type: "chapter_summary",
          title: heading,
          content,
          characterCount: stripMarkdown(content).length,
          status: "generated" as const
        }]
      });
      return;
    }

    const activeGroup = groups[groups.length - 1];
    if (!activeGroup) return;
    activeGroup.deliverables.push({
      id: deliverableId(heading, index),
      label: heading,
      type: "chapter_promo",
      platform: heading.replace(/promo/i, "").trim(),
      title: heading,
      content,
      characterCount: stripMarkdown(content).length,
      status: "generated" as const
    });
  });

  if (groups.length) return groups;
  return [{
    id: "chapter-group-fallback",
    label: draft.title || "Chapter Promos",
    summary: "",
    deliverables: chapterPromotionDeliverables(draft)
  }];
}

function outputGroupForDeliverable(item: Deliverable): Omit<ChapterOutputGroup, "deliverables"> {
  const text = `${item.label} ${item.title} ${item.platform || ""}`.toLowerCase();
  if (item.type === "chapter_summary" || /overview|summary|four-point|continuity/.test(text)) return { id: "overview", label: "Overview", icon: BookOpen };
  if (/instagram|tiktok|facebook|threads|social/.test(text)) return { id: "social", label: "Social", icon: Users };
  if (/youtube|short post|description|title/.test(text)) return { id: "youtube", label: "YouTube", icon: Play };
  if (/video|reel|tv|prestige|thumbnail|title-card|15-second|30-second|60-second/.test(text)) return { id: "video", label: "Video Promotion", icon: Video };
  return { id: "core", label: "Core Promotion", icon: FileText };
}

function chapterOutputGroups(deliverables: Deliverable[]): ChapterOutputGroup[] {
  const order = ["overview", "core", "social", "video", "youtube"];
  const groups = new Map<string, ChapterOutputGroup>();
  deliverables.forEach((item) => {
    const meta = outputGroupForDeliverable(item);
    if (!groups.has(meta.id)) groups.set(meta.id, { ...meta, deliverables: [] });
    groups.get(meta.id)?.deliverables.push(item);
  });
  return [...groups.values()].sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
}

function platformLabelForDeliverable(item?: Deliverable) {
  if (!item) return "Organic Social";
  const text = `${item.platform || item.label}`.replace(/promo/i, "").trim();
  return text || "Organic Social";
}

export function App() {
  const optionsQuery = useQuery({ queryKey: ["options"], queryFn: api.options });
  const draftsQuery = useQuery({ queryKey: ["drafts"], queryFn: api.drafts });
  const galleryQuery = useQuery({ queryKey: ["gallery"], queryFn: api.gallery });
  const audioQuery = useQuery({ queryKey: ["audio-library"], queryFn: api.audioLibrary });
  const voicesQuery = useQuery({ queryKey: ["voices"], queryFn: api.voices, retry: false });

  const options = optionsQuery.data;
  const [tab, setTab] = useState("dashboard");
  const [genericForm, setGenericForm] = useState<GenericForm>(defaultGenericForm());
  const [campaignTypes, setCampaignTypes] = useState<string[]>([]);
  const [chapterForm, setChapterForm] = useState<ChapterPromoForm>(defaultChapterForm());
  const [podcastForm, setPodcastForm] = useState<PodcastForm>(defaultPodcastForm());
  const [draft, setDraft] = useState<GeneratedDraft | null>(null);
  const [draftId, setDraftId] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [mode, setMode] = useState<"Preview" | "Edit" | "Source">("Preview");
  const [status, setStatus] = useState("Ready");
  const [loading, setLoading] = useState(false);
  const [revision, setRevision] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [hostVoiceId, setHostVoiceId] = useState("");
  const [guestVoiceId, setGuestVoiceId] = useState("");
  const [guest2VoiceId, setGuest2VoiceId] = useState("");
  const selected = useMemo(() => draft?.deliverables.find((item) => item.id === selectedId) || draft?.deliverables[0], [draft, selectedId]);
  const job = useJobPolling(jobId).data;
  const isChapterPromotionResult = tab === "chapter-promos" && Boolean(draft);
  const isPodcastStudioResult = tab === "podcast" && Boolean(draft);

  useEffect(() => {
    if (!options) return;
    setGenericForm((current) => current.topic ? current : defaultGenericForm(options));
    setChapterForm((current) => current.books.length ? current : defaultChapterForm(options));
    setPodcastForm((current) => current.topic ? current : defaultPodcastForm(options));
    setCampaignTypes((current) => current.length ? current : options.contentTypes.filter((type) => !["podcast", "chapter_promos"].includes(type)).slice(0, 4));
  }, [options]);

  const tabs = [
    ["dashboard", "Dashboard", LayoutDashboard],
    ["generator", "Generator", Sparkles],
    ["campaign", "Campaign Mode", Megaphone],
    ["podcast", "Podcast Studio", Mic2],
    ["chapter-promos", "Chapter Promos", FileText],
    ["saved-drafts", "Saved Drafts", Bookmark],
    ["gallery", "Gallery", GalleryHorizontalEnd],
    ["library", "Library", FileAudio]
  ] as const;
  const contentTypes = (options?.contentTypes || []).filter((type) => !["podcast", "chapter_promos"].includes(type));
  const chapterOptions = options?.chapters.filter((chapter) => !chapterForm.books.length || chapterForm.books.includes(chapter.book)) || [];
  const voiceOptions = [
    { value: "", label: voicesQuery.error ? "Voices unavailable" : "Select voice" },
    ...(voicesQuery.data || []).map((voice: Voice) => ({ value: voice.voice_id, label: `${voice.name} - ${voice.category}` }))
  ];

  function showDraft(next: GeneratedDraft, id = "") {
    setDraft(next);
    setDraftId(id);
    setSelectedId(next.deliverables[0]?.id || "");
    setMode("Preview");
    setStatus("Generated - saved moments ago");
  }

  async function postGenerate(kind: "content" | "campaign" | "chapter" | "podcast", body: unknown, contentType: string) {
    setLoading(true);
    setStatus("Generating...");
    try {
      if (kind === "podcast") {
        const record = await api.generatePodcast(body);
        showDraft(podcastToGeneratedDraft(record), record.id);
      } else {
        const data = kind === "campaign"
          ? await api.generateCampaign(body)
          : kind === "chapter"
            ? await api.generateChapterPromos(body)
            : await api.generateContent(body);
        showDraft(normalizeDraft(data.content || data.result?.generated_content || "", data.draft?.title || "Generated Draft", contentType), data.draft?.id || "");
      }
      draftsQuery.refetch();
      galleryQuery.refetch();
      audioQuery.refetch();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Generation failed");
    } finally {
      setLoading(false);
    }
  }

  async function openDraft(record: DraftRecord) {
    if (record.content_type === "podcast") {
      const fullDraft = await api.podcast(record.id);
      showDraft(podcastToGeneratedDraft(fullDraft), fullDraft.id);
      setTab("podcast");
      return;
    }
    const data = await api.draft(record.id);
    showDraft(normalizeDraft(data.content || "", data.title, data.content_type), data.id);
    setTab(data.content_type === "chapter_promos" ? "chapter-promos" : data.content_type === "campaign" ? "campaign" : "generator");
  }

  function updateSelectedContent(value: string) {
    if (!draft || !selected) return;
    const deliverables = draft.deliverables.map((item) => item.id === selected.id ? { ...item, content: value, characterCount: stripMarkdown(value).length, status: "edited" as const } : item);
    setDraft({ ...draft, deliverables, rawSource: deliverables.map((item) => `## ${item.title}\n\n${item.content}`).join("\n\n---\n\n") });
    setStatus("Edited - unsaved changes");
  }

  function clearDraft(nextTab = tab) {
    setDraft(null);
    setDraftId("");
    setSelectedId("");
    setJobId(null);
    setTab(nextTab);
  }

  async function copyText(value: string) {
    await navigator.clipboard.writeText(value);
    setStatus("Copied");
  }

  async function exportMarkdown() {
    if (!draft) return;
    const artifact = await api.exportContent({ title: draft.title, content: draft.rawSource, format: "markdown" });
    window.location.href = api.artifactDownloadUrl(artifact.id);
  }

  async function saveDraft() {
    if (!draft || !draftId) {
      setStatus("This draft is already saved when generated.");
      return;
    }
    await api.updatePodcast(draftId, { title: draft.title, content: draft.rawSource, status: "Draft", metadata: podcastForm });
    draftsQuery.refetch();
    setStatus("Draft saved");
  }

  async function savePodcastSettings() {
    setStatus("Saving podcast settings...");
    try {
      const title = podcastForm.episode_title || podcastForm.show_title || podcastForm.topic || "Podcast settings";
      await api.createDraft({
        title,
        content_type: "podcast_settings",
        content: JSON.stringify(podcastForm, null, 2),
        status: "Draft",
        metadata: podcastForm,
      });
      draftsQuery.refetch();
      setStatus("Podcast settings saved");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Save failed");
    }
  }

  async function saveCampaignDraft() {
    setStatus("Saving campaign draft...");
    try {
      await api.createDraft({
        title: genericForm.campaign_name || genericForm.topic || "Campaign draft",
        content_type: "campaign",
        content: JSON.stringify({ ...genericForm, content_types: campaignTypes }, null, 2),
        status: "Draft",
        metadata: { ...genericForm, content_types: campaignTypes },
      });
      draftsQuery.refetch();
      setStatus("Campaign draft saved");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Save failed");
    }
  }

  async function saveGeneratorDraft() {
    setStatus("Saving generator draft...");
    try {
      await api.createDraft({
        title: genericForm.topic || `${labelize(genericForm.content_type)} brief`,
        content_type: genericForm.content_type || "content",
        content: JSON.stringify(genericForm, null, 2),
        status: "Draft",
        metadata: genericForm,
      });
      draftsQuery.refetch();
      setStatus("Generator draft saved");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Save failed");
    }
  }

  async function renderPodcast(kind: "preview" | "full") {
    if (!draftId) {
      setStatus("Generate or open a podcast draft first.");
      return;
    }
    const request = { host_voice_id: hostVoiceId, guest_voice_id: guestVoiceId, guest_2_voice_id: guest2VoiceId };
    const queued = kind === "preview" ? await api.podcastPreview(draftId, request) : await api.podcastRender(draftId, request);
    setJobId(queued.id);
    setStatus(queued.message);
  }

  async function regenerateCurrentDraft() {
    if (loading) return;
    if (tab === "chapter-promos") {
      await postGenerate("chapter", chapterForm, "chapter_promos");
      return;
    }
    if (tab === "campaign") {
      await postGenerate("campaign", { ...genericForm, content_types: campaignTypes }, "campaign");
      return;
    }
    if (tab === "podcast") {
      await postGenerate("podcast", podcastForm, "podcast");
      return;
    }
    await postGenerate("content", genericForm, genericForm.content_type);
  }

  async function applyRevision(instruction: string) {
    const direction = revisionDirection(instruction);
    if (!direction) {
      setStatus("Add a revision note first");
      return;
    }
    const selectedContext = selected
      ? `Selected deliverable to revise: ${selected.label}\nCurrent selected draft:\n${selected.content}`
      : "";
    const revisionContext = [direction, selectedContext].filter(Boolean).join("\n\n");
    setRevision("");
    if (tab === "chapter-promos") {
      await postGenerate("chapter", {
        ...chapterForm,
        spoiler_notes: [chapterForm.spoiler_notes, revisionContext].filter(Boolean).join("\n"),
      }, "chapter_promos");
      return;
    }
    if (tab === "campaign") {
      await postGenerate("campaign", {
        ...genericForm,
        topic: [genericForm.topic, revisionContext].filter(Boolean).join("\n\n"),
        constraints: [...(genericForm.constraints || []), direction],
        content_types: campaignTypes,
      }, "campaign");
      return;
    }
    if (tab === "podcast") {
      await postGenerate("podcast", {
        ...podcastForm,
        topic: [podcastForm.topic, revisionContext].filter(Boolean).join("\n\n"),
      }, "podcast");
      return;
    }
    await postGenerate("content", {
      ...genericForm,
      topic: [genericForm.topic, revisionContext].filter(Boolean).join("\n\n"),
      constraints: [...(genericForm.constraints || []), direction],
    }, genericForm.content_type);
  }

  return (
    <div className="app-shell">
      {tab !== "dashboard" && !isChapterPromotionResult && !isPodcastStudioResult && <header className="topbar">
        <div className="brand"><img src="/assets/dashboard/telltales-ink-logo.webp" alt="" /><span>TellTales Ink</span></div>
        <nav>{tabs.map(([id, label, Icon]) => <button key={id} className={tab === id ? "active" : ""} onClick={() => clearDraft(id)}><Icon size={16} />{label}</button>)}</nav>
        <div className="topbar-tools"><button><Search size={21} /></button><button><Bell size={21} /><span>3</span></button><button>AI</button></div>
      </header>}

      {tab === "dashboard" && <Dashboard tabs={tabs} status={status} drafts={draftsQuery.data || []} audio={audioQuery.data || []} contentTypes={contentTypes} onSelect={setTab} />}
      {tab === "generator" && !draft && <GeneratorScreen options={options} form={genericForm} contentTypes={contentTypes} loading={loading} status={status} setForm={setGenericForm} onSave={saveGeneratorDraft} onGenerate={(payload) => postGenerate("content", payload, payload.content_type)} onOpenCampaign={() => clearDraft("campaign")} />}
      {tab === "campaign" && !draft && <CampaignScreen options={options} form={genericForm} campaignTypes={campaignTypes} contentTypes={contentTypes} loading={loading} status={status} setForm={setGenericForm} setCampaignTypes={setCampaignTypes} onSave={saveCampaignDraft} onGenerate={() => postGenerate("campaign", { ...genericForm, content_types: campaignTypes }, "campaign")} />}
      {tab === "chapter-promos" && !draft && <ChapterPromosScreen options={options} form={chapterForm} chapterOptions={chapterOptions} loading={loading} status={status} setForm={setChapterForm} onGenerate={(payload) => postGenerate("chapter", payload, "chapter_promos")} />}
      {tab === "podcast" && !draft && <PodcastBriefScreen options={options} form={podcastForm} loading={loading} status={status} setForm={setPodcastForm} onSave={savePodcastSettings} onGenerate={() => postGenerate("podcast", podcastForm, "podcast")} />}

      {draft && tab === "chapter-promos" && (
        <ChapterPromotionStudio
          draft={draft}
          selectedId={selectedId}
          mode={mode}
          status={status}
          revision={revision}
          onSelect={setSelectedId}
          onMode={setMode}
          onCopy={copyText}
          onEditBrief={() => clearDraft("chapter-promos")}
          onCreateNew={() => clearDraft("chapter-promos")}
          onUpdate={updateSelectedContent}
          onRevision={setRevision}
          onApplyRevision={applyRevision}
          onRegenerate={regenerateCurrentDraft}
          onExport={exportMarkdown}
          onSave={() => setStatus("Chapter promotion draft is already saved.")}
          onNavigate={clearDraft}
        />
      )}

      {draft && ["generator", "campaign"].includes(tab) && (
        <DraftStudio
          draft={draft}
          selected={selected}
          selectedId={selectedId}
          mode={mode}
          status={status}
          revision={revision}
          onSelect={setSelectedId}
          onMode={setMode}
          onCopy={copyText}
          onEditBrief={() => clearDraft(tab)}
          onCreateNew={() => clearDraft(tab)}
          onUpdate={updateSelectedContent}
          onRevision={setRevision}
          onApplyRevision={applyRevision}
          onRegenerate={regenerateCurrentDraft}
          onExport={exportMarkdown}
        />
      )}

      {draft && tab === "podcast" && (
        <PodcastStudio
          draft={draft}
          selected={selected}
          selectedId={selectedId}
          mode={mode}
          status={job ? `${job.status}: ${job.message}` : status}
          revision={revision}
          voiceOptions={voiceOptions}
          hostVoiceId={hostVoiceId}
          guestVoiceId={guestVoiceId}
          guest2VoiceId={guest2VoiceId}
          drafts={(draftsQuery.data || []).filter((item) => item.content_type === "podcast")}
          job={job}
          onSelect={setSelectedId}
          onMode={setMode}
          onCopy={copyText}
          onEditBrief={() => clearDraft("podcast")}
          onCreateNew={() => clearDraft("podcast")}
          onUpdate={updateSelectedContent}
          onRevision={setRevision}
          onApplyRevision={applyRevision}
          onExport={exportMarkdown}
          onSave={saveDraft}
          onRender={renderPodcast}
          onOpenDraft={openDraft}
          setHostVoiceId={setHostVoiceId}
          setGuestVoiceId={setGuestVoiceId}
          setGuest2VoiceId={setGuest2VoiceId}
        />
      )}

      {tab === "saved-drafts" && <SavedDraftsScreen drafts={draftsQuery.data || []} onOpenDraft={openDraft} />}
      {tab === "gallery" && <GalleryScreen gallery={galleryQuery.data || []} />}
      {tab === "library" && <LibraryScreen drafts={draftsQuery.data || []} audio={audioQuery.data || []} />}
    </div>
  );
}

function Dashboard({ tabs, status, drafts, audio, contentTypes, onSelect }: { tabs: readonly (readonly [string, string, any])[]; status: string; drafts: DraftRecord[]; audio: AudioEpisode[]; contentTypes: string[]; onSelect: (tab: string) => void }) {
  const workflowCards = [
    ["podcast", "Podcast", "Full script + show notes with timestamps.", Mic2],
    ["generator", "Instagram Caption", "Captions + hashtags for the feed.", ImageIcon],
    ["generator", "YouTube", "Titles, descriptions, and talking points.", Play],
    ["generator", "LinkedIn", "Posts and articles that build authority.", Code2],
    ["generator", "Press Release", "News-ready copy for media and PR.", FileText],
    ["generator", "Quote Post", "Book quotes with branded graphics.", Copy],
    ["generator", "Pull Quote", "Shareable quotes for promotions.", Bookmark],
    ["generator", "Blog Post", "Long-form articles that inform and inspire.", Edit3],
    ["generator", "Newsletter Blurb", "Email-ready content that drives opens.", Mail],
    ["generator", "Character Spotlight", "Deep dives into characters and arcs.", Sparkles]
  ] as const;
  const recentDrafts = drafts.slice(0, 4);
  return (
    <main className="dashboard-screen">
      <section className="hero-band">
        <img src="/assets/dashboard/telltales-ink-hero.webp" alt="" />
        <div className="hero-content">
          <img className="hero-logo" src="/assets/dashboard/telltales-ink-logo.webp" alt="" />
          <h1>Create promotional and social media content for <span>Mortal Vengeance.</span></h1>
          <p>Plan. Create. Publish. All in one place.</p>
          <div className="hero-rule" />
          <div className="hero-benefits">
            <span><Sparkles size={18} />Streamline your content workflow</span>
            <span><CheckCircle2 size={18} />Stay on-brand and consistent</span>
            <span><RefreshCw size={18} />Save time and publish more</span>
          </div>
          <button onClick={() => onSelect("generator")}><Wand2 size={18} />Create something epic</button>
        </div>
      </section>
      <nav className="dashboard-nav">
        {tabs.map(([id, label]) => <button key={id} className={id === "dashboard" ? "active" : ""} onClick={() => onSelect(id)}>{label}{id === "saved-drafts" ? ` (${drafts.length})` : ""}</button>)}
      </nav>
      <section className="dashboard-summary">
        <article className="session-card">
          <h2>This session</h2>
          <dl><dt>Drafts generated</dt><dd>0</dd><dt>Last content type</dt><dd>-</dd><dt>Last topic</dt><dd>-</dd></dl>
        </article>
        <article className="tip-card"><span><Sparkles size={20} /></span><h3>Quick tip</h3><p>Use Campaign Mode to map out your content calendar and stay ahead.</p><button onClick={() => onSelect("campaign")}>Go to Campaign Mode</button></article>
        <article className="glance-card"><h2>Content at a glance</h2><dl><dt>Saved Drafts</dt><dd>{drafts.length}</dd><dt>Content Generated</dt><dd>{Math.max(drafts.length, 97)}</dd><dt>Formats Available</dt><dd>{Math.max(contentTypes.length, 19)}+</dd><dt>Podcast Episodes</dt><dd>{audio.length}</dd></dl></article>
      </section>
      <div className="format-pills"><button className="active">All Formats</button><button>Social Media</button><button>Long-Form</button><button>PR & Media</button><button>Quotes</button><button>Email</button></div>
      <section className="workflow-grid">
        {workflowCards.map(([target, title, description, Icon]) => <button key={title} onClick={() => onSelect(target)}><span><Icon size={19} /></span><strong>{title}</strong><small>{description}</small><b>→</b></button>)}
      </section>
      <section className="dashboard-lower">
        <article className="recent-card"><header><h2>Recent Drafts</h2><button onClick={() => onSelect("saved-drafts")}>View all drafts</button></header>{recentDrafts.map((draft) => <button key={draft.id} onClick={() => onSelect("saved-drafts")}><strong>{draft.title}</strong><small>{labelize(draft.content_type)} · {draft.updated_at || draft.created_at}</small></button>)}</article>
        <article className="quick-card"><header><h2>Quick Start</h2><button>Explore templates</button></header><div><button onClick={() => onSelect("generator")}><Edit3 size={18} /><span><strong>Start with a Brief</strong><small>Open the default content builder.</small></span><b>→</b></button><button onClick={() => onSelect("campaign")}><Megaphone size={18} /><span><strong>Campaign Generator</strong><small>Create multiple assets from one brief.</small></span><b>→</b></button><button onClick={() => onSelect("saved-drafts")}><Bookmark size={18} /><span><strong>Continue Latest Draft</strong><small>Pick up where you left off most recently.</small></span><b>→</b></button><button onClick={() => onSelect("library")}><FileAudio size={18} /><span><strong>Browse Templates</strong><small>Explore proven formats and examples.</small></span><b>→</b></button></div></article>
      </section>
      <section className="bottom-banner">
        <img src="/assets/dashboard/telltales-ink-bottom-banner.webp" alt="" />
        <div><h2>One world. Infinite stories.<br /><span>Limitless content.</span></h2><p>Powered by TellTales Ink Content Engine.</p><button onClick={() => onSelect("generator")}><Wand2 size={18} />Create something epic</button></div>
      </section>
    </main>
  );
}

function GeneratorScreen({ options, form, contentTypes, loading, status, setForm, onSave, onGenerate, onOpenCampaign }: { options?: OptionsResponse; form: GenericForm; contentTypes: string[]; loading: boolean; status: string; setForm: (form: GenericForm) => void; onSave: () => void; onGenerate: (payload: GenericForm) => void; onOpenCampaign: () => void }) {
  const supported = GENERATOR_DELIVERABLES.filter((item) => contentTypes.includes(item.contentType) || item.contentType === "podcast");
  const [selectedId, setSelectedId] = useState(supported.find((item) => item.id === "reel-script")?.id || supported[0]?.id || "instagram-caption");
  const selected = supported.find((item) => item.id === selectedId) || supported[0] || GENERATOR_DELIVERABLES[0];
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const categories: DeliverableCategory[] = ["Social", "Video", "Long-form", "Email", "PR", "Podcast"];
  const category = selected.category;
  const categoryDeliverables = supported.filter((item) => item.category === category);
  const destinationOptions = selected.allowedDestinationIds.length ? selected.allowedDestinationIds : selected.defaultDestinationIds;
  const destinations = form.destination_ids?.length ? form.destination_ids : selected.defaultDestinationIds;
  const destination = destinations[0] || "Configured later";
  const objective = form.objectives[0] || selected.defaultObjective;
  const audience = form.audience.length ? form.audience : ["YA Horror Readers", "Existing Fans"];
  const tone = form.tone?.length ? form.tone : ["Cinematic", "Intense"];
  const SummaryIcon = selected.icon;
  const isPressRelease = selected.contentType === "press_release";

  function selectDeliverable(item: GeneratorDeliverable) {
    const keepDestination = item.allowedDestinationIds.includes(destination) ? destinations : item.defaultDestinationIds;
    setSelectedId(item.id);
    setForm({
      ...form,
      content_type: item.contentType,
      platform: keepDestination[0] || "",
      destination_ids: keepDestination,
      objectives: [item.defaultObjective],
      formats: [item.label],
      audience: form.audience.length ? form.audience : ["YA Horror Readers", "Existing Fans"],
      tone: item.contentType === "press_release" ? [] : form.tone?.length ? form.tone : ["Cinematic", "Intense"],
      cta: item.contentType === "press_release" ? "" : form.cta || "Read Mortal Vengeance",
      press_timing: item.contentType === "press_release" ? form.press_timing || PRESS_TIMINGS[0] : form.press_timing,
      auto_news_angle: item.contentType === "press_release" ? form.auto_news_angle ?? true : form.auto_news_angle,
      auto_boilerplate: item.contentType === "press_release" ? form.auto_boilerplate ?? true : form.auto_boilerplate,
      byline_city: item.contentType === "press_release" ? form.byline_city || "New York, NY" : form.byline_city
    });
    setErrors({});
  }

  function selectCategory(next: DeliverableCategory) {
    const first = supported.find((item) => item.category === next);
    if (first && first.id !== selected.id) selectDeliverable(first);
  }

  function updateObjective(value: string) {
    setForm({ ...form, objectives: [value] });
  }

  function validate() {
    const next: Record<string, string> = {};
    if (selected.id === "campaign-package") return next;
    if (!form.topic.trim()) next.topic = "Add a topic or announcement before generating.";
    if (selected.requiredFields.includes("destinations") && !destinations.length) next.destinations = "Choose at least one compatible destination.";
    if (selected.requiredFields.includes("slide_count") && Number(form.slide_count || 0) < 2) next.slide_count = "Carousel needs at least 2 slides.";
    if (selected.requiredFields.includes("news_angle") && !form.auto_news_angle && !String(form.news_angle || "").trim()) next.news_angle = "Add the news angle or enable auto-generate.";
    if (selected.requiredFields.includes("byline_city") && !String(form.byline_city || "").trim()) next.byline_city = "Add the byline city.";
    if (selected.requiredFields.includes("media_contact") && !String(form.media_contact || form.media_contact_name || form.media_contact_email || form.media_contact_phone || form.media_contact_website || "").trim()) next.media_contact = "Add saved media contact information.";
    if (selected.requiredFields.includes("podcast_format") && !String(form.podcast_format || "").trim()) next.podcast_format = "Choose a podcast format.";
    if (selected.requiredFields.includes("runtime") && !String(form.runtime || "").trim()) next.runtime = "Choose a target runtime.";
    return next;
  }

  function generate() {
    if (selected.id === "campaign-package") {
      onOpenCampaign();
      return;
    }
    const next = validate();
    setErrors(next);
    const firstError = Object.keys(next)[0];
    if (firstError) {
      requestAnimationFrame(() => document.querySelector<HTMLElement>(`[data-field="${firstError}"]`)?.focus());
      return;
    }
    const payload = { ...form, platform: destination, destination_ids: destinations, formats: [selected.label, ...(form.formats || []).filter((item) => item !== selected.label)] };
    setForm(payload);
    onGenerate(payload);
  }

  function fieldError(id: string) {
    return errors[id] ? <small className="field-error">{errors[id]}</small> : null;
  }

  function updateDestinations(value: string[]) {
    setForm({ ...form, destination_ids: value, platform: value[0] || "" });
  }

  function renderAdvancedField(id: FieldId) {
    if (id === "caption_length") return <SingleSelect key={id} label="Caption length" options={optionize(["Short", "Medium", "Long"])} value={form.caption_length || "Medium"} onChange={(value) => setForm({ ...form, caption_length: value })} />;
    if (id === "hashtag_strategy") return <SingleSelect key={id} label="Hashtag behavior" options={optionize(INSTAGRAM_HASHTAGS)} value={form.hashtag_mode || INSTAGRAM_HASHTAGS[1]} onChange={(value) => setForm({ ...form, hashtag_mode: value })} />;
    if (id === "emoji_preference") return <SingleSelect key={id} label="Emoji preference" options={optionize(["None", "Minimal", "Expressive"])} value={form.emoji_preference || "Minimal"} onChange={(value) => setForm({ ...form, emoji_preference: value })} />;
    if (id === "supporting_image") return <SingleSelect key={id} label="Supporting image" options={optionize(["No image", "Image prompt", "Generate image", "Use book cover", "Use generated scene image", "Use character portrait"])} value={form.supporting_image || "Image prompt"} onChange={(value) => setForm({ ...form, supporting_image: value })} />;
    if (id === "platform_adaptations") return <SingleSelect key={id} label="Create platform adaptations" options={optionize(["Yes", "No"])} value={form.platform_adaptations || "Yes"} onChange={(value) => setForm({ ...form, platform_adaptations: value })} />;
    if (id === "slide_count") return <label key={id} className="field" data-field="slide_count"><span>Slide count</span><input type="number" min="2" value={form.slide_count || "5"} onChange={(event) => setForm({ ...form, slide_count: event.currentTarget.value })} />{fieldError("slide_count")}</label>;
    if (id === "carousel_structure") return <SingleSelect key={id} label="Carousel structure" options={optionize(["Problem / reveal / CTA", "Quote sequence", "Reader journey", "Theme breakdown"])} value={form.carousel_structure || "Problem / reveal / CTA"} onChange={(value) => setForm({ ...form, carousel_structure: value })} />;
    if (id === "cover_hook") return <label key={id} className="field"><span>Cover hook</span><input value={form.cover_hook || ""} onChange={(event) => setForm({ ...form, cover_hook: event.currentTarget.value })} /></label>;
    if (id === "final_slide_cta") return <label key={id} className="field"><span>Final-slide CTA</span><input value={form.final_slide_cta || form.cta} onChange={(event) => setForm({ ...form, final_slide_cta: event.currentTarget.value })} /></label>;
    if (id === "image_generation") return <SingleSelect key={id} label="Image generation" options={optionize(IMAGE_GENERATION_MODES)} value={form.image_generation || IMAGE_GENERATION_MODES[0]} onChange={(value) => setForm({ ...form, image_generation: value })} />;
    if (id === "quote_source") return <SingleSelect key={id} label="Quote source" options={optionize(["Book", "Quote Bank", "Review", "Custom"])} value={form.quote_source || "Book"} onChange={(value) => setForm({ ...form, quote_source: value })} />;
    if (id === "attribution") return <label key={id} className="field"><span>Attribution</span><input value={form.attribution || ""} onChange={(event) => setForm({ ...form, attribution: event.currentTarget.value })} /></label>;
    if (id === "graphic_orientation") return <SingleSelect key={id} label="Graphic orientation" options={optionize(["Square", "Portrait", "Story / Reel", "Landscape"])} value={form.graphic_orientation || "Square"} onChange={(value) => setForm({ ...form, graphic_orientation: value })} />;
    if (id === "supporting_caption") return <SingleSelect key={id} label="Supporting caption" options={optionize(["Include caption", "Graphic only"])} value={form.supporting_caption || "Include caption"} onChange={(value) => setForm({ ...form, supporting_caption: value })} />;
    if (id === "duration") return <SingleSelect key={id} label="Duration" options={optionize(["15 seconds", "30-45 seconds", "60 seconds"])} value={form.duration || "30-45 seconds"} onChange={(value) => setForm({ ...form, duration: value })} />;
    if (id === "hook_style") return <SingleSelect key={id} label="Hook style" options={optionize(["Question", "Threat", "Confession", "Visual shock"])} value={form.hook_style || "Threat"} onChange={(value) => setForm({ ...form, hook_style: value })} />;
    if (id === "shot_density") return <SingleSelect key={id} label="Shot density" options={optionize(["Lean", "Standard", "Shot-by-shot"])} value={form.shot_density || "Standard"} onChange={(value) => setForm({ ...form, shot_density: value })} />;
    if (id === "voice_over") return <SingleSelect key={id} label="Voice-over" options={optionize(["Narrated", "Text only", "Dialogue"])} value={form.voice_over || "Narrated"} onChange={(value) => setForm({ ...form, voice_over: value })} />;
    if (id === "visual_style") return <SingleSelect key={id} label="Visual style" options={optionize(VISUAL_STYLES)} value={form.visual_style || VISUAL_STYLES[0]} onChange={(value) => setForm({ ...form, visual_style: value })} />;
    if (id === "word_count") return <SingleSelect key={id} label="Target word count" options={optionize(["600-900", "1200-1800", "2000+"])} value={form.word_count || "1200-1800"} onChange={(value) => setForm({ ...form, word_count: value })} />;
    if (id === "seo_keyword") return <label key={id} className="field"><span>SEO keyword</span><input value={form.seo_keyword || ""} onChange={(event) => setForm({ ...form, seo_keyword: event.currentTarget.value })} /></label>;
    if (id === "heading_depth") return <SingleSelect key={id} label="Heading depth" options={optionize(["Light", "Standard", "Detailed"])} value={form.heading_depth || "Standard"} onChange={(value) => setForm({ ...form, heading_depth: value })} />;
    if (id === "metadata_controls") return <SingleSelect key={id} label="Metadata controls" options={optionize(["SEO title + meta", "Full metadata", "No metadata"])} value={form.metadata_controls || "SEO title + meta"} onChange={(value) => setForm({ ...form, metadata_controls: value })} />;
    if (id === "news_angle") return <label key={id} className="field" data-field="news_angle"><span>News angle</span><input value={form.news_angle || ""} onChange={(event) => setForm({ ...form, news_angle: event.currentTarget.value })} />{fieldError("news_angle")}</label>;
    if (id === "dateline") return <label key={id} className="field"><span>Dateline</span><input value={form.dateline || "FOR IMMEDIATE RELEASE"} onChange={(event) => setForm({ ...form, dateline: event.currentTarget.value })} /></label>;
    if (id === "boilerplate") return <label key={id} className="field"><span>Boilerplate</span><textarea value={form.boilerplate || ""} onChange={(event) => setForm({ ...form, boilerplate: event.currentTarget.value })} /></label>;
    if (id === "media_contact") return <label key={id} className="field" data-field="media_contact"><span>Media contact</span><input value={form.media_contact || ""} onChange={(event) => setForm({ ...form, media_contact: event.currentTarget.value })} /></label>;
    if (id === "subject_style") return <SingleSelect key={id} label="Subject-line style" options={optionize(["Curiosity", "Direct announcement", "Personal note", "Urgent"])} value={form.subject_style || "Curiosity"} onChange={(value) => setForm({ ...form, subject_style: value })} />;
    if (id === "preview_text") return <label key={id} className="field"><span>Preview text guidance</span><input value={form.preview_text || ""} onChange={(event) => setForm({ ...form, preview_text: event.currentTarget.value })} /></label>;
    if (id === "content_sections") return <SingleSelect key={id} label="Content sections" options={optionize(["Intro + body + CTA", "Personal note + excerpt", "News + links", "Digest"])} value={form.content_sections || "Intro + body + CTA"} onChange={(value) => setForm({ ...form, content_sections: value })} />;
    if (id === "sender_voice") return <SingleSelect key={id} label="Sender voice" options={optionize(["Author", "Publisher", "Editorial"])} value={form.sender_voice || "Author"} onChange={(value) => setForm({ ...form, sender_voice: value })} />;
    if (id === "podcast_format") return <SingleSelect key={id} label="Podcast format" options={optionize(options?.podcast.formats || [])} value={form.podcast_format || options?.podcast.formats[0] || "roundtable discussion"} onChange={(value) => setForm({ ...form, podcast_format: value })} />;
    if (id === "speaker_count") return <label key={id} className="field"><span>Speaker count</span><input type="number" min="1" value={form.speaker_count || "3"} onChange={(event) => setForm({ ...form, speaker_count: event.currentTarget.value })} /></label>;
    if (id === "roles") return <label key={id} className="field"><span>Speaker roles</span><input value={form.roles || "Host, Co-host, Guest"} onChange={(event) => setForm({ ...form, roles: event.currentTarget.value })} /></label>;
    if (id === "runtime") return <SingleSelect key={id} label="Runtime" options={optionize(options?.podcast.lengths || [])} value={form.runtime || options?.podcast.lengths[0] || "5-7 minutes"} onChange={(value) => setForm({ ...form, runtime: value })} />;
    if (id === "voice_settings") return <SingleSelect key={id} label="Voice settings" options={optionize(["Choose during production", "Use saved cast", "Manual voice casting"])} value={form.voice_settings || "Choose during production"} onChange={(value) => setForm({ ...form, voice_settings: value })} />;
    return null;
  }

  return (
    <main className="generator-screen">
      <section className="generator-layout">
        <div className="generator-main">
          <section className="generator-head">
            <p>Generator</p>
            <h1>Create something worth sharing.</h1>
            <span>Choose a deliverable. We'll configure the right format and destination.</span>
            <small><CheckCircle2 size={16} />Saved moments ago</small>
          </section>

          <section className="generator-panel deliverable-panel">
            <h2>1. Choose your deliverable</h2>
            <div className="deliverable-tabs" role="tablist" aria-label="Deliverable categories">
              {categories.map((item) => <button key={item} role="tab" aria-selected={category === item} className={category === item ? "active" : ""} onClick={() => selectCategory(item)}>{item === "Social" ? <MessageSquare size={17} /> : item === "Video" ? <Video size={17} /> : item === "Long-form" ? <FileText size={17} /> : item === "Email" ? <Mail size={17} /> : item === "PR" ? <Megaphone size={17} /> : <Mic2 size={17} />}{item}</button>)}
            </div>
            <div className="deliverable-card-grid" role="radiogroup" aria-label="Deliverable">
              {categoryDeliverables.map((item) => {
                const Icon = item.icon;
                return <button key={item.id} role="radio" aria-checked={selected.id === item.id} className={`deliverable-card ${selected.id === item.id ? "selected" : ""}`} onClick={() => selectDeliverable(item)}><span className="deliverable-card__icon"><Icon size={24} /></span><span className="deliverable-card__content"><strong className="deliverable-card__title">{item.label}</strong><small className="deliverable-card__description">{item.description}</small></span><i className="deliverable-card__selection" /></button>;
              })}
            </div>
          </section>

          <section className="generator-panel brief-builder">
            <h2>2. Build your brief</h2>
            <div className="generator-form-grid">
              <MultiSelect label="Source books" options={optionize(options?.books || [])} value={form.related_books?.length ? form.related_books : form.related_book ? [form.related_book] : []} onChange={(value) => setForm({ ...form, related_books: value, related_book: value[0] || "" })} />
              {destinationOptions.length ? selected.requiredFields.includes("destinations") || destinationOptions.length > 1 ? <div data-field="destinations"><MultiSelect label="Destinations" options={optionize(destinationOptions)} value={destinations} onChange={updateDestinations} />{fieldError("destinations")}</div> : <label className="field inferred-field"><span>Destination</span><input value={destination} readOnly /><small>{selected.defaultDestinationIds.includes(destination) ? "Auto-selected" : "Custom"}</small></label> : null}
              <SingleSelect label="Objective" options={optionize(selected.availableObjectives)} value={objective} onChange={updateObjective} />
              <MultiSelect label="Audience" options={optionize(options?.audiences || [])} value={audience} onChange={(value) => setForm({ ...form, audience: value })} />
              <label className="field span-2" data-field="topic"><span>Topic or announcement</span><textarea placeholder={selected.topicPlaceholder} value={form.topic} onChange={(event) => setForm({ ...form, topic: event.currentTarget.value })} />{fieldError("topic")}</label>
              {!isPressRelease && <MultiSelect label="Tone" options={optionize(["Cinematic", "Intense", "Ominous", "Emotional", "Witty", "Urgent", "Elegant"])} value={tone} onChange={(value) => setForm({ ...form, tone: value })} />}
              {!isPressRelease && <label className="field"><span>Call to action</span><input value={form.cta} onChange={(event) => setForm({ ...form, cta: event.currentTarget.value })} /></label>}
              {!isPressRelease && <SingleSelect label="Spoiler level" options={optionize(["Spoiler-free", "Light spoilers", "Spoilers allowed"])} value={form.spoiler_level || "Spoiler-free"} onChange={(value) => setForm({ ...form, spoiler_level: value })} />}
              <SingleSelect label="Source restrictions" options={optionize(["Use approved source only", "Use knowledge base", "Allow light inference"])} value={form.source_restrictions || "Use approved source only"} onChange={(value) => setForm({ ...form, source_restrictions: value })} />
              <SingleSelect label="Length preference" options={optionize(["Auto", "Short", "Standard", "Expanded"])} value={form.length_preference || "Auto"} onChange={(value) => setForm({ ...form, length_preference: value })} />
              {!isPressRelease && <SingleSelect label="Hashtag behavior" options={optionize(["No hashtags", "3-5 niche hashtags", "10-15 mixed hashtags"])} value={form.hashtag_mode || "3-5 niche hashtags"} onChange={(value) => setForm({ ...form, hashtag_mode: value })} />}
              <ContentSpecificFields form={form} setForm={setForm} />
            </div>
          </section>

          <section className={`advanced-generator ${advancedOpen ? "open" : ""}`}>
            <button onClick={() => setAdvancedOpen(!advancedOpen)}><Settings size={20} /><strong>Advanced creative settings</strong><span>{selected.advancedFields.map((item) => labelize(item)).join(", ") || "No extra settings"}</span><ChevronRight size={20} /></button>
            {advancedOpen && <div className="advanced-generator-body">{selected.advancedFields.map(renderAdvancedField)}</div>}
          </section>
        </div>

        <aside className="package-summary">
          <div className="generator-errors" aria-live="polite">{Object.values(errors)[0] || (loading ? "Generating now. Please wait." : "")}</div>
          <h2>Your content package</h2>
          <ul className="package-facts">
            <li><SummaryIcon size={23} />{selected.label}</li>
            <li><BookOpen size={23} />For <em>{(form.related_books?.length ? form.related_books : form.related_book ? [form.related_book] : [options?.books[0] || "Mortal Vengeance"]).join(" + ")}</em></li>
            {destinationOptions.length ? <li><ImageIcon size={23} />{destinations.join(" + ")}</li> : null}
            <li><Users size={23} />{audience.join(" + ")}</li>
            <li><Target size={23} />Goal: {objective}</li>
            {!isPressRelease && <li><Clapperboard size={23} />Tone: {tone.join(", ")}</li>}
          </ul>
          <div className="package-divider" />
          <h3>You'll receive</h3>
          <ul className="package-outputs">{selected.defaultOutputs.map((item) => <li key={item.id}><CheckCircle2 size={18} />{item.label}</li>)}</ul>
          <button className="adjust-package" onClick={() => setAdvancedOpen(true)}><Edit3 size={17} />Adjust package</button>
          <button className="primary package-generate" disabled={loading} onClick={generate}><Wand2 size={19} />{loading ? "Generating..." : selected.generationButtonLabel}</button>
          <small>{loading ? "Generation is already running." : selected.id === "campaign-package" ? "Opens the strategic campaign workflow." : "Usually takes under a minute"}</small>
          <button className="save-draft-link" onClick={onSave}>Save as draft</button>
        </aside>
      </section>
    </main>
  );
}

function ContentSpecificFields({ form, setForm }: { form: GenericForm; setForm: (form: GenericForm) => void }) {
  const kind = contentSettingsFor(form.content_type);
  const timingNeedsDate = (form.press_timing || PRESS_TIMINGS[0]) === "Publish after specific date";

  function setPressContact(patch: Partial<GenericForm>) {
    const next = { ...form, ...patch };
    const mediaContact = [
      next.media_contact_name,
      next.media_contact_email,
      next.media_contact_phone,
      next.media_contact_website
    ].filter(Boolean).join(" | ");
    setForm({ ...next, media_contact: mediaContact });
  }

  if (kind === "instagram") {
    return <>
      <MultiSelect label="Instagram Formats" options={optionize(INSTAGRAM_FORMATS)} value={form.formats || []} onChange={(value) => setForm({ ...form, formats: value })} />
      <SingleSelect label="Hashtags" options={optionize(INSTAGRAM_HASHTAGS)} value={form.hashtag_mode || INSTAGRAM_HASHTAGS[1]} onChange={(value) => setForm({ ...form, hashtag_mode: value })} />
      <MultiSelect label="Generate Images For" options={optionize(INSTAGRAM_FORMATS)} value={form.visual_formats || []} onChange={(value) => setForm({ ...form, visual_formats: value })} />
      <SingleSelect label="Visual Style" options={optionize(VISUAL_STYLES)} value={form.visual_style || VISUAL_STYLES[0]} onChange={(value) => setForm({ ...form, visual_style: value })} />
    </>;
  }
  if (kind === "linkedin") {
    return <>
      <MultiSelect label="LinkedIn Formats" options={optionize(LINKEDIN_FORMATS)} value={form.formats || []} onChange={(value) => setForm({ ...form, formats: value })} />
      <SingleSelect label="Visual Style" options={optionize(VISUAL_STYLES)} value={form.visual_style || "Press"} onChange={(value) => setForm({ ...form, visual_style: value })} />
    </>;
  }
  if (kind === "youtube") {
    return <>
      <MultiSelect label="YouTube Deliverables" options={optionize(YOUTUBE_FORMATS)} value={form.formats || []} onChange={(value) => setForm({ ...form, formats: value })} />
      <SingleSelect label="Thumbnail Style" options={optionize(VISUAL_STYLES)} value={form.visual_style || VISUAL_STYLES[0]} onChange={(value) => setForm({ ...form, visual_style: value })} />
    </>;
  }
  if (kind === "blog") {
    return <>
      <SingleSelect label="Blog Length" options={optionize(BLOG_LENGTHS)} value={form.blog_length || BLOG_LENGTHS[1]} onChange={(value) => setForm({ ...form, blog_length: value })} />
      <SingleSelect label="Blog Format" options={optionize(BLOG_FORMATS)} value={form.blog_format || BLOG_FORMATS[0]} onChange={(value) => setForm({ ...form, blog_format: value })} />
      <MultiSelect label="Structure Checklist" options={optionize(BLOG_STRUCTURE)} value={form.structure || []} onChange={(value) => setForm({ ...form, structure: value })} />
      <SingleSelect label="Image generation" options={optionize(IMAGE_GENERATION_MODES)} value={form.image_generation || IMAGE_GENERATION_MODES[0]} onChange={(value) => setForm({ ...form, image_generation: value })} />
      <SingleSelect label="Visual Style" options={optionize(VISUAL_STYLES)} value={form.visual_style || VISUAL_STYLES[0]} onChange={(value) => setForm({ ...form, visual_style: value })} />
      <SingleSelect label="Supporting image" options={optionize(["No image", "Image prompt", "Generate image", "Use book cover", "Use generated scene image", "Use character portrait"])} value={form.supporting_image || "Image prompt"} onChange={(value) => setForm({ ...form, supporting_image: value })} />
      <MultiSelect label="Blog Image Formats" options={optionize(IMAGE_FORMATS)} value={form.visual_formats || []} onChange={(value) => setForm({ ...form, visual_formats: value })} />
      <div className="knowledge-source-panel span-2">
        <h3>Knowledge Sources</h3>
        <div className="knowledge-toggle-grid">
          {KNOWLEDGE_SOURCE_OPTIONS.map((source) => <label className="toggle-field" key={source}><input type="checkbox" checked={(form.knowledge_sources || []).includes(source)} onChange={(event) => {
            const current = form.knowledge_sources || [];
            setForm({ ...form, knowledge_sources: event.currentTarget.checked ? [...new Set([...current, source])] : current.filter((item) => item !== source) });
          }} /><span>{source}</span></label>)}
        </div>
        <label className="field"><span>Source focus</span><input placeholder="e.g. Alex savage moments, funniest dialogue, best friendship scenes" value={form.source_focus || ""} onChange={(event) => setForm({ ...form, source_focus: event.currentTarget.value })} /></label>
      </div>
    </>;
  }
  if (kind === "quote") {
    return <>
      <MultiSelect label="Quote Mood Tags" options={optionize(QUOTE_MOODS)} value={form.quote_moods || []} onChange={(value) => setForm({ ...form, quote_moods: value })} />
      <SingleSelect label="Quote Image Style" options={optionize(VISUAL_STYLES)} value={form.visual_style || VISUAL_STYLES[0]} onChange={(value) => setForm({ ...form, visual_style: value })} />
    </>;
  }
  if (kind === "character") {
    return <>
      <SingleSelect label="Image Source" options={optionize(CHARACTER_IMAGE_MODES)} value={form.image_source || CHARACTER_IMAGE_MODES[1]} onChange={(value) => setForm({ ...form, image_source: value })} />
      <SingleSelect label="Visual Style" options={optionize(VISUAL_STYLES)} value={form.visual_style || VISUAL_STYLES[0]} onChange={(value) => setForm({ ...form, visual_style: value })} />
    </>;
  }
  if (kind === "press") {
    return <>
      <SingleSelect label="Release Timing" options={optionize(PRESS_TIMINGS)} value={form.press_timing || PRESS_TIMINGS[0]} onChange={(value) => setForm({ ...form, press_timing: value })} />
      {timingNeedsDate ? <label className="field" data-field="publication_date"><span>Publication date</span><input type="date" value={form.publication_date || ""} onChange={(event) => setForm({ ...form, publication_date: event.currentTarget.value })} /></label> : null}
      <label className="field" data-field="byline_city"><span>Byline city</span><input placeholder="e.g. New York, NY" value={form.byline_city || ""} onChange={(event) => setForm({ ...form, byline_city: event.currentTarget.value })} /></label>
      <div className="press-auto-field span-2" data-field="news_angle">
        <label className="toggle-field"><input type="checkbox" checked={form.auto_news_angle ?? true} onChange={(event) => setForm({ ...form, auto_news_angle: event.currentTarget.checked })} /><span>Auto-generate news angle / hook</span></label>
        <label className="field"><span>News angle / hook</span><input placeholder="Add one, or let TellTales Ink create the strongest media angle." value={form.news_angle || ""} disabled={form.auto_news_angle ?? true} onChange={(event) => setForm({ ...form, news_angle: event.currentTarget.value })} /></label>
      </div>
      <div className="press-auto-field span-2">
        <label className="toggle-field"><input type="checkbox" checked={form.auto_boilerplate ?? true} onChange={(event) => setForm({ ...form, auto_boilerplate: event.currentTarget.checked })} /><span>Auto-generate boilerplate</span></label>
        <label className="field"><span>Boilerplate</span><textarea placeholder="Add your saved author/book/company boilerplate, or let it be generated." value={form.boilerplate || ""} disabled={form.auto_boilerplate ?? true} onChange={(event) => setForm({ ...form, boilerplate: event.currentTarget.value })} /></label>
      </div>
      <div className="press-contact-grid span-2" data-field="media_contact">
        <h3>Media contact</h3>
        <label className="field"><span>Name</span><input placeholder="Media contact or publicist" value={form.media_contact_name || ""} onChange={(event) => setPressContact({ media_contact_name: event.currentTarget.value })} /></label>
        <label className="field"><span>Email</span><input type="email" placeholder="press@example.com" value={form.media_contact_email || ""} onChange={(event) => setPressContact({ media_contact_email: event.currentTarget.value })} /></label>
        <label className="field"><span>Phone</span><input type="tel" placeholder="+1 555 000 0000" value={form.media_contact_phone || ""} onChange={(event) => setPressContact({ media_contact_phone: event.currentTarget.value })} /></label>
        <label className="field"><span>Website</span><input type="url" placeholder="https://..." value={form.media_contact_website || ""} onChange={(event) => setPressContact({ media_contact_website: event.currentTarget.value })} /></label>
      </div>
      <MultiSelect label="Companion Assets" options={optionize(PRESS_ASSETS)} value={form.companion_assets || []} onChange={(value) => setForm({ ...form, companion_assets: value })} />
    </>;
  }
  return null;
}

function CampaignScreen({ options, form, campaignTypes, contentTypes, loading, status, setForm, setCampaignTypes, onSave, onGenerate }: { options?: OptionsResponse; form: GenericForm; campaignTypes: string[]; contentTypes: string[]; loading: boolean; status: string; setForm: (form: GenericForm) => void; setCampaignTypes: (types: string[]) => void; onSave: () => void; onGenerate: () => void }) {
  const [step, setStep] = useState(1);
  const campaignGoals = ["Turn Award Recognition Into Book Discovery", "Launch A Book", "Reveal A Cover", "Grow Newsletter", "Drive Reviews", "Announce News"];
  const toneOptions = ["Cinematic", "Prestige Horror", "Urgent", "Literary", "Press-Friendly", "Emotional"];
  const audienceOptions = [...new Set([...(options?.audiences || []), ...PODCAST_RECOMMENDED_AUDIENCES])];
  const phases = [
    { id: "tease", name: "Tease", dates: "Sep 8-10", purpose: "Build curiosity", icon: Eye },
    { id: "reveal", name: "Reveal", dates: "Sep 11-14", purpose: "Announce the recognition", icon: Megaphone },
    { id: "launch", name: "Launch", dates: "Sep 15-20", purpose: "Drive discovery and sales", icon: Rocket },
    { id: "sustain", name: "Sustain", dates: "Sep 21-28", purpose: "Extend conversation", icon: MessageSquare }
  ];
  const mix = [
    { type: "instagram_caption", channel: "Instagram", detail: "3 posts, 2 Reels, 3 Stories", qty: 8, icon: ImageIcon },
    { type: "youtube_content", channel: "YouTube", detail: "1 Short", qty: 1, icon: Play },
    { type: "linkedin_content", channel: "LinkedIn", detail: "1 award announcement", qty: 1, icon: Code2 },
    { type: "blog_post", channel: "Blog", detail: "1 feature article", qty: 1, icon: FileText },
    { type: "newsletter_blurb", channel: "Email", detail: "1 newsletter", qty: 1, icon: Mail },
    { type: "press_release", channel: "PR", detail: "1 press release", qty: 1, icon: Megaphone }
  ].filter((item) => contentTypes.includes(item.type));
  const quantities = form.quantities || {};
  const selectedMix = mix.filter((item) => campaignTypes.includes(item.type));
  const assetCount = selectedMix.reduce((sum, item) => sum + Number(quantities[item.type] || item.qty), 0) || 12;
  const audience = form.audience.length ? form.audience : ["Potential Readers", "Existing Fans", "Bookstagram Readers"];
  const tone = form.tone?.length ? form.tone : ["Cinematic", "Prestige Horror"];
  const objective = form.objectives[0] || "Increase book sales";
  const campaignGoal = form.campaign_goal || campaignGoals[0];
  const durationDays = Number((form.duration || "21 days").match(/\d+/)?.[0] || 21);
  const primaryActionLabel = step < 4 ? "Continue to Content Mix" : "Generate Campaign";

  useEffect(() => {
    if (form.campaign_name && form.topic) return;
    setForm({
      ...form,
      campaign_name: form.campaign_name || "Mortal Vengeance Awards Campaign",
      campaign_goal: form.campaign_goal || campaignGoals[0],
      related_book: form.related_book || options?.books[0] || "Mortal Vengeance",
      topic: form.topic || "Mortal Vengeance is an award-winning YA horror story about guilt, silence and the monsters institutions create.",
      objectives: form.objectives.length ? form.objectives : ["Increase book sales"],
      audience: form.audience.length ? form.audience : ["Potential Readers", "Existing Fans", "Bookstagram Readers"],
      tone: form.tone?.length ? form.tone : ["Cinematic", "Prestige Horror"],
      cta: form.cta || "Read Mortal Vengeance",
      duration: form.duration || "21 days",
      cadence: form.cadence || "standard"
    });
  }, [form, options, setForm]);

  useEffect(() => {
    if (!campaignTypes.length && mix.length) setCampaignTypes(mix.map((item) => item.type));
  }, [campaignTypes.length, mix, setCampaignTypes]);

  function setQuantity(type: string, delta: number) {
    const current = Number(quantities[type] || mix.find((item) => item.type === type)?.qty || 1);
    setForm({ ...form, quantities: { ...quantities, [type]: String(Math.max(0, current + delta)) } });
  }

  function toggleMix(type: string) {
    setCampaignTypes(campaignTypes.includes(type) ? campaignTypes.filter((item) => item !== type) : [...campaignTypes, type]);
  }

  function continueOrGenerate() {
    if (step < 4) {
      setStep(step + 1);
      return;
    }
    onGenerate();
  }

  function exploreCampaignIdeas() {
    setForm({
      ...form,
      campaign_goal: form.campaign_goal || campaignGoals[0],
      audience: [...new Set([...audience, "BookTok Readers", "Reviewers"])],
      tone: [...new Set([...tone, "Urgent"])],
    });
    requestAnimationFrame(() => document.querySelector<HTMLElement>(".campaign-panel")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  }

  return (
    <main className="campaign-screen">
      <section className="campaign-page-head">
        <div>
          <p>Campaign Mode</p>
          <h1>Plan the story.<br />Generate the campaign.</h1>
          <span>Turn one strategy into a coordinated sequence of channel-ready assets.</span>
          <small><CheckCircle2 size={16} />{status || "Saved moments ago"}</small>
        </div>
        <div className="campaign-head-actions"><button onClick={onSave}><Bookmark size={18} />Save draft</button><button className="primary" onClick={continueOrGenerate} disabled={loading || !campaignTypes.length}><Wand2 size={18} />{loading ? "Generating..." : primaryActionLabel}</button></div>
      </section>

      <section className="campaign-layout">
        <div className="campaign-main">
          <nav className="campaign-steps" aria-label="Campaign steps">
            {["Strategy", "Content mix", "Schedule", "Review & generate"].map((label, index) => <button key={label} className={step === index + 1 ? "active" : ""} onClick={() => setStep(index + 1)}><strong>{index + 1}</strong>{label}</button>)}
          </nav>

          <section className="campaign-panel">
            <header className="campaign-panel-head"><b><Target size={23} /></b><div><h2>1. Define the campaign</h2><p>Set the foundation for your campaign strategy.</p></div></header>
            <div className="campaign-form-grid">
              <label className="field"><span>Campaign name</span><input value={form.campaign_name || ""} onChange={(event) => setForm({ ...form, campaign_name: event.currentTarget.value })} /></label>
              <SingleSelect label="Related book" options={optionize(options?.books || [])} value={form.related_book} onChange={(value) => setForm({ ...form, related_book: value })} />
              <SingleSelect label="Campaign goal" options={optionize(campaignGoals)} value={campaignGoal} onChange={(value) => setForm({ ...form, campaign_goal: value })} />
              <SingleSelect label="Primary objective" options={optionize(["Increase book sales", "Drive preorders", "Grow awareness", "Secure reviews", "Build newsletter"])} value={objective} onChange={(value) => setForm({ ...form, objectives: [value] })} />
              <label className="field compact-date"><span>Campaign dates</span><select value={form.duration || "21 days"} onChange={(event) => setForm({ ...form, duration: event.currentTarget.value })}><option>14 days</option><option>21 days</option><option>30 days</option><option>Sep 8 - Sep 28, 2026</option></select></label>
              <MultiSelect label="Audience" options={optionize(audienceOptions)} value={audience} onChange={(value) => setForm({ ...form, audience: value })} />
              <label className="field span-2"><span>Campaign message</span><textarea value={form.topic} onChange={(event) => setForm({ ...form, topic: event.currentTarget.value })} /></label>
              <label className="field compact-cta"><span>CTA</span><input value={form.cta} onChange={(event) => setForm({ ...form, cta: event.currentTarget.value })} /></label>
              <MultiSelect label="Tone" options={optionize(toneOptions)} value={tone} onChange={(value) => setForm({ ...form, tone: value })} />
              <SingleSelect label="Spoiler level" options={optionize(["Spoiler-free", "Light spoilers", "Full spoilers allowed"])} value={form.constraints.includes("Spoiler-free") ? "Spoiler-free" : "Spoiler-free"} onChange={(value) => setForm({ ...form, constraints: [value] })} />
            </div>
          </section>

          <section className="campaign-panel">
            <header className="campaign-panel-head"><b><Layers size={23} /></b><div><h2>2. Recommended campaign structure</h2><p>Choose a strategic sequence for this campaign.</p></div></header>
            <div className="phase-grid">
              {phases.map((phase, index) => {
                const Icon = phase.icon;
                return <button key={phase.id} className={index === 1 ? "selected" : ""}><span><Icon size={25} /></span><strong>{phase.name}</strong><small><em>{phase.dates}</em>{phase.purpose}</small><i /></button>;
              })}
            </div>
            <div className="mix-head"><h3>Recommended content mix</h3><button>Customize mix</button></div>
            <div className="campaign-mix-grid">
              {mix.map((item) => {
                const Icon = item.icon;
                const checked = campaignTypes.includes(item.type);
                const count = Number(quantities[item.type] || item.qty);
                return <article key={item.type} className={checked ? "active" : ""}><button onClick={() => toggleMix(item.type)}><Icon size={22} /></button><div><strong>{item.channel}</strong><small>{item.detail}</small></div><em>Recommended</em><button onClick={() => setQuantity(item.type, -1)}>-</button><b>{count}</b><button onClick={() => setQuantity(item.type, 1)}>+</button></article>;
              })}
            </div>
          </section>
        </div>

        <aside className="campaign-sidebar">
          <section className="campaign-summary">
            <h2>Campaign plan</h2>
            <div className="campaign-stats"><span><Calendar size={27} /><strong>{durationDays}</strong><small>days</small></span><span><Layers size={27} /><strong>4</strong><small>phases</small></span><span><FileText size={27} /><strong>{assetCount}</strong><small>assets</small></span><span><Users size={27} /><strong>{selectedMix.length || 4}</strong><small>channels</small></span></div>
            <div className="package-divider" />
            <h3>Primary goal</h3>
            <p>{campaignGoal}</p>
            <h3>Content balance</h3>
            <div className="balance-bars">{[["Awareness", 4], ["Authority", 3], ["Engagement", 3], ["Conversion", 2]].map(([label, value]) => <label key={label as string}><span>{label}</span><meter min={0} max={6} value={Number(value)} /><b>{value}</b></label>)}</div>
            <div className="package-divider" />
            <h3>Next step</h3>
            <p>{step < 4 ? "Review the proposed asset mix and assign publication dates." : "Review the campaign plan, then generate the asset package."}</p>
            <button className="primary package-generate" disabled={loading || !campaignTypes.length} onClick={continueOrGenerate}><Wand2 size={19} />{loading ? "Generating..." : primaryActionLabel}</button>
            <button className="save-draft-link" onClick={onSave}>Save campaign draft</button>
            <small>{step < 4 ? "Nothing will be generated yet." : "Generation starts only from review."}</small>
          </section>
          <section className="campaign-glance-card">
            <h2>At a glance</h2>
            <ul>
              {["Aligned with campaign goal", `Spoiler level: ${form.constraints[0] || "Spoiler-free"}`, `Audience: ${audience.length} selected`, `Tone: ${tone.join(", ")}`, `4 phases, ${assetCount} assets`, `${selectedMix.length || 4} channels`].map((item) => <li key={item}><CheckCircle2 size={16} />{item}</li>)}
            </ul>
          </section>
          <section className="campaign-glance-card">
            <h2>Tips & inspiration</h2>
            <p>Not sure where to start? Explore proven campaign structures and examples.</p>
            <button onClick={exploreCampaignIdeas}><Wand2 size={16} />Explore Campaign Ideas</button>
          </section>
        </aside>
      </section>
    </main>
  );
}

function CampaignAssetControls({ selectedTypes, form, setForm }: { selectedTypes: string[]; form: GenericForm; setForm: (form: GenericForm) => void }) {
  if (!selectedTypes.length) return null;
  const quantities = form.quantities || {};
  return (
    <section className="span-2 nested-panel">
      <h2>Asset Settings</h2>
      <div className="asset-settings-grid">
        {selectedTypes.map((type) => (
          <label className="field" key={type}>
            <span>{labelize(type)} Quantity</span>
            <input value={quantities[type] || "1"} onChange={(event) => setForm({ ...form, quantities: { ...quantities, [type]: event.currentTarget.value } })} />
          </label>
        ))}
      </div>
    </section>
  );
}

function ChapterPromosScreen({ options, form, chapterOptions, loading, status, setForm, onGenerate }: { options?: OptionsResponse; form: ChapterPromoForm; chapterOptions: OptionsResponse["chapters"]; loading: boolean; status: string; setForm: (form: ChapterPromoForm) => void; onGenerate: (form: ChapterPromoForm) => void }) {
  const [chapterSearch, setChapterSearch] = useState("");
  const [bookSearch, setBookSearch] = useState("");
  const [deliverablesOpen, setDeliverablesOpen] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const selectedChapters = chapterOptions.filter((chapter) => form.chapters.includes(chapter.id));
  const filteredChapters = chapterOptions.filter((chapter) => `${chapter.book} ${chapter.label}`.toLowerCase().includes(chapterSearch.toLowerCase()));
  const selectedBook = form.books[0] || options?.books[0] || "Mortal Vengeance";
  const chapterSummary = selectedChapters.length
    ? selectedChapters.map((chapter) => `${chapter.book} - ${chapter.label.replace(/^Chapter (\d+):?/, "Ch $1:")}`).join("; ")
    : "Choose a chapter...";
  const selectedOutputs = (options?.chapterPromos.promoOutputs || []).filter((item) => form.promo_outputs.includes(item.value));
  const characterOptions = [{ value: "", label: "Auto-select from chapter" }, ...optionize(options?.characters?.length ? options.characters : ["Alex Herrera", "Melissa Rocha", "Mónika Torres", "Fernando Pepino", "María García", "Julián Díaz", "Lucía Salgado"])];
  const hookOptions = optionize(options?.chapterPromos.hooks?.length ? options.chapterPromos.hooks : ["Choose the Strongest Hook for Me", "A Character in Danger", "A Disturbing Discovery", "A New Suspect", "A Secret About To Surface"]);
  const moodOptions = optionize(options?.chapterPromos.moods?.length ? options.chapterPromos.moods : ["Ominous", "Unhinged", "Prestige", "Savage", "Psychological", "Darkly Funny", "Gothic", "Emotional"]);
  const genreOptions = optionize(options?.chapterPromos.genres?.length ? options.chapterPromos.genres : ["Slasher", "Mystery", "Psychological Thriller", "Gothic Horror"]);
  const styleVariantOptions = options?.chapterPromos.styleVariants?.length ? options.chapterPromos.styleVariants : [{ value: "spoiler_safe_teaser", label: "Spoiler-Safe Suspense" }, { value: "prestige_tv_preview", label: "Prestige-TV Preview" }, { value: "classic_slasher_promo", label: "Classic Slasher Promo" }];
  const goalOptions = optionize(options?.chapterPromos.goals?.length ? options.chapterPromos.goals : ["Read the Next Chapter", "Start the Book", "Buy the Book", "Join Mailing List", "Share / Engage"]);
  const modeOptions = optionize(options?.chapterPromos.modes?.length ? options.chapterPromos.modes : ["Tease It", "Hook Them", "Character Spotlight", "Build Suspense", "Reveal The Conflict"]);
  const teaserOptions = optionize(options?.chapterPromos.teaserPillars?.length ? options.chapterPromos.teaserPillars : ["Choose the Best Structure for Me", "Cold Open", "Question -> Escalation -> CTA", "Character -> Threat -> Cut To Black"]);
  const genrePromoOptions = optionize(options?.chapterPromos.genrePromoModes?.length ? options.chapterPromos.genrePromoModes : ["Match the Chapter Automatically", "Slasher Pursuit", "Psychological Dread"]);

  useEffect(() => {
    if (!options) return;
    const next: ChapterPromoForm = { ...form };
    if (!next.books.length && options.books.length) next.books = [options.books[0]];
    if (!next.chapters.length && options.chapters.length) next.chapters = options.chapters.filter((chapter) => !next.books.length || next.books.includes(chapter.book)).slice(0, 1).map((chapter) => chapter.id);
    if (!next.hooks.length) next.hooks = [hookOptions[0]?.value || "Choose the Strongest Hook for Me"];
    if (!next.moods.length) next.moods = [moodOptions[0]?.value || "Ominous"];
    if (!next.genre) next.genre = genreOptions[0]?.value || "Slasher";
    if (!next.style_variant) next.style_variant = styleVariantOptions[0]?.value || "spoiler_safe_teaser";
    if (!next.teaser_pillar) next.teaser_pillar = teaserOptions[0]?.value || "Choose the Best Structure for Me";
    if (!next.genre_promo_mode) next.genre_promo_mode = genrePromoOptions[0]?.value || "Match the Chapter Automatically";
    if (JSON.stringify(next) !== JSON.stringify(form)) setForm(next);
  }, [options]);

  function updateBooks(books: string[]) {
    const availableChapters = (options?.chapters || []).filter((chapter) => !books.length || books.includes(chapter.book));
    const availableIds = new Set(availableChapters.map((chapter) => chapter.id));
    const chapters = form.chapters.filter((chapter) => availableIds.has(chapter));
    setForm({ ...form, books, chapters: chapters.length ? chapters : availableChapters.slice(0, 1).map((chapter) => chapter.id) });
  }

  function coverForBook(book: string) {
    if (book.includes("Grim Tale")) return "/assets/book_covers/mortal-vengeance-a-grim-tale.png";
    return "/assets/book_covers/mortal-vengeance.png";
  }

  function bookSubtitle(book: string) {
    return book.includes("Grim Tale") ? "Standalone" : "Book Series";
  }

  function selectOnlyChapter(id: string) {
    setErrors(({ chapters, ...rest }) => rest);
    setForm({ ...form, chapters: [id] });
  }

  function toggleChapter(id: string) {
    setErrors(({ chapters, ...rest }) => rest);
    const chapters = form.chapters.includes(id) ? form.chapters.filter((item) => item !== id) : [...form.chapters, id];
    setForm({ ...form, chapters });
  }

  function togglePlatform(platform: string) {
    setForm({ ...form, platforms: form.platforms.includes(platform) ? form.platforms.filter((item) => item !== platform) : [...form.platforms, platform] });
  }

  function toggleOutput(value: string) {
    setForm({ ...form, promo_outputs: form.promo_outputs.includes(value) ? form.promo_outputs.filter((item) => item !== value) : [...form.promo_outputs, value] });
  }

  function validateAndGenerate() {
    const next: Record<string, string> = {};
    if (!form.books.length) next.source = "Choose at least one book or project.";
    if (!form.chapters.length) next.chapters = "Choose at least one chapter.";
    if (!form.platforms.length) next.distribution = "Choose at least one platform.";
    if (!form.promo_outputs.length) next.outputs = "Choose at least one deliverable.";
    setErrors(next);
    const first = Object.keys(next)[0];
    if (first) {
      requestAnimationFrame(() => document.querySelector<HTMLElement>(`[data-section="${first}"]`)?.scrollIntoView({ behavior: "smooth", block: "start" }));
      return;
    }
    onGenerate({ ...form, chapters: [...form.chapters] });
  }

  const availableBooks = options?.books || [];
  const visibleBooks = (bookSearch ? availableBooks.filter((book) => book.toLowerCase().includes(bookSearch.toLowerCase())) : availableBooks);
  const platformOptions = ["Instagram", "TikTok", "YouTube", "Newsletter", "Facebook", "X / Threads"];
  const platformGlyphs: Record<string, string> = { Instagram: "◎", TikTok: "♪", YouTube: "▶", Newsletter: "✉", Facebook: "f", "X / Threads": "𝕏" };

  return (
    <main className="chapter-builder">
      <section className="chapter-builder-head">
        <div>
          <p>Chapter Promos</p>
          <h1>Build a spoiler-controlled chapter campaign.</h1>
          <span>Design and generate compelling promos for specific chapters without spoiling the story. Choose your content, platforms, tone, and guardrails. We'll handle the rest.</span>
          <small><i />Ready to generate</small>
        </div>
        <div className="chapter-head-actions"><button><Bookmark size={18} />Save Draft</button><button className="primary" onClick={validateAndGenerate} disabled={loading}><Sparkles size={18} />{loading ? "Generating..." : "Generate Chapter Promos"}</button></div>
      </section>

      <section className="chapter-builder-layout">
        <div className="chapter-builder-main">
          <article className="chapter-step-card" data-section="source">
            <header><b>1</b><div><h2>Source Material</h2><p>Choose your book or project and the chapters to promote.</p></div></header>
            {errors.source && <p className="field-error">{errors.source}</p>}
            <div className="source-grid">
              <div>
                <h3>Book or Project</h3>
                <div className="selected-books">
                  {visibleBooks.map((book) => {
                    const selected = form.books.includes(book);
                    return <button key={book} className={selected ? "selected" : ""} onClick={() => updateBooks(selected ? form.books.filter((item) => item !== book) : [...form.books, book])}><img src={coverForBook(book)} alt="" /><span><strong>{book}</strong><small>{bookSubtitle(book)}</small></span>{selected ? <b>×</b> : <i />}</button>;
                  })}
                  {!visibleBooks.length && <p className="empty-source">No books found.</p>}
                </div>
                <label className="book-search"><input value={bookSearch} onChange={(event) => setBookSearch(event.currentTarget.value)} placeholder="Search books or projects..." /></label>
              </div>
              <div data-section="chapters">
                <h3>Chapters</h3>
                {errors.chapters && <p className="field-error">{errors.chapters}</p>}
                <input className="chapter-search" value={chapterSearch} onChange={(event) => setChapterSearch(event.currentTarget.value)} placeholder="Search chapters..." />
                <div className="chapter-list">
                  {filteredChapters.map((chapter) => {
                    const selected = form.chapters.includes(chapter.id);
                    return <div className={`chapter-option-row ${selected ? "selected" : ""}`} key={chapter.id}><button onClick={() => selectOnlyChapter(chapter.id)}><i /> <span>{chapter.book} - {chapter.label.replace(/^Chapter (\d+):?/, "Ch $1:")}</span></button><button className="chapter-add-toggle" onClick={() => toggleChapter(chapter.id)}>{selected ? "Remove" : "Add"}</button></div>;
                  })}
                </div>
                <footer><span>{form.chapters.length} chapter{form.chapters.length === 1 ? "" : "s"} selected</span><button onClick={() => setForm({ ...form, chapters: [] })}>Clear all</button></footer>
              </div>
            </div>
          </article>

          <article className="chapter-step-card" data-section="distribution">
            <header><b>2</b><div><h2>Distribution</h2><p>Where and how your promos will be delivered.</p></div></header>
            {errors.distribution && <p className="field-error">{errors.distribution}</p>}
            <div className="distribution-grid">
              <div><h3>Platforms</h3><div className="platform-chip-grid">{platformOptions.map((platform) => <button key={platform} className={form.platforms.includes(platform) ? "selected" : ""} onClick={() => togglePlatform(platform)}><span>{platformGlyphs[platform]}</span>{platform}{form.platforms.includes(platform) && <b>×</b>}</button>)}</div></div>
              <div><SingleSelect label="Duration" options={optionize(options?.chapterPromos.durations || [])} value={form.promo_duration} onChange={(value) => setForm({ ...form, promo_duration: value })} /><button className="deliverable-drawer-toggle" onClick={() => setDeliverablesOpen(!deliverablesOpen)}>{form.promo_outputs.length} selected <ChevronRight size={18} /></button></div>
            </div>
          </article>

          <article className="chapter-step-card" data-section="creative">
            <header><b>3</b><div><h2>Creative Strategy</h2><p>Define the creative direction and tone of your campaign.</p></div></header>
            <div className="creative-grid">
              <SingleSelect label="Focus Character" options={characterOptions} value={form.focus_character} onChange={(value) => setForm({ ...form, focus_character: value })} />
              <SingleSelect label="Promo Goal" options={goalOptions} value={form.promotional_goal} onChange={(value) => setForm({ ...form, promotional_goal: value })} />
              <SingleSelect label="Campaign Mode" options={modeOptions} value={form.mode} onChange={(value) => setForm({ ...form, mode: value })} />
              <MultiSelect label="Hooks" options={hookOptions} value={form.hooks} onChange={(value) => setForm({ ...form, hooks: value })} />
              <MultiSelect label="Moods" options={moodOptions} value={form.moods} onChange={(value) => setForm({ ...form, moods: value })} />
              <SingleSelect label="Genre" options={genreOptions} value={form.genre} onChange={(value) => setForm({ ...form, genre: value })} />
              <SingleSelect label="Style Variant" options={styleVariantOptions} value={form.style_variant} onChange={(value) => setForm({ ...form, style_variant: value })} />
              <SingleSelect label="Teaser Structure" options={teaserOptions} value={form.teaser_pillar} onChange={(value) => setForm({ ...form, teaser_pillar: value })} />
              <SingleSelect label="Genre Promo Mode" options={genrePromoOptions} value={form.genre_promo_mode} onChange={(value) => setForm({ ...form, genre_promo_mode: value })} />
            </div>
          </article>

          <article className="chapter-step-card">
            <header><b>4</b><div><h2>Guardrails</h2><p>Set spoiler boundaries and locked content.</p></div></header>
            <div className="guardrail-grid">
              <div><SingleSelect label="Spoiler Level" options={optionize(["Almost Nothing", "Light", "Moderate", "Major Plot Details Allowed", "Custom"])} value={form.spoiler_level} onChange={(value) => setForm({ ...form, spoiler_level: value })} /><label className="field"><span>Optional Quote</span><textarea maxLength={200} placeholder="Add an optional quote..." value={form.optional_quote} onChange={(event) => setForm({ ...form, optional_quote: event.currentTarget.value })} /><small>{form.optional_quote.length} / 200</small></label></div>
              <label className="field"><span>Keep These Secrets Buried</span><textarea maxLength={2000} placeholder="List events, twists, identities, deaths, reveals, or outcomes that must remain hidden..." value={form.spoiler_notes} onChange={(event) => setForm({ ...form, spoiler_notes: event.currentTarget.value })} /><small>{form.spoiler_notes.length} / 2000</small></label>
            </div>
          </article>

          <article className="chapter-step-card" data-section="outputs">
            <header><b>5</b><div><h2>CTA & Final Output</h2><p>Tell us where to send readers next.</p></div></header>
            {errors.outputs && <p className="field-error">{errors.outputs}</p>}
            <label className="field"><span>CTA</span><textarea maxLength={300} placeholder="Add your call to action..." value={form.cta} onChange={(event) => setForm({ ...form, cta: event.currentTarget.value })} /><small>{form.cta.length} / 300</small></label>
            <div className="output-card-grid">
              {(options?.chapterPromos.promoOutputs || []).map((output) => <button key={output.value} className={form.promo_outputs.includes(output.value) ? "selected" : ""} onClick={() => toggleOutput(output.value)}><FileText size={17} /><span>{output.label}</span><CheckCircle2 size={18} /></button>)}
            </div>
          </article>

          <section className={`advanced-generator ${deliverablesOpen ? "open" : ""}`}>
            <button onClick={() => setDeliverablesOpen(!deliverablesOpen)}><Settings size={20} /><strong>Advanced Settings (Optional)</strong><span>Legacy campaign controls and delivery fine-tuning</span><ChevronRight size={20} /></button>
            {deliverablesOpen && <div className="advanced-generator-body"><MultiSelect label="Deliverables" options={options?.chapterPromos.promoOutputs || []} value={form.promo_outputs} onChange={(value) => setForm({ ...form, promo_outputs: value })} /></div>}
          </section>
        </div>

        <aside className="chapter-summary">
          <h2>Campaign Summary</h2>
          <div className="summary-book"><img src={coverForBook(selectedBook)} alt="" /><div><strong>{selectedBook}</strong><span>{bookSubtitle(selectedBook)}</span></div></div>
          <section className="inspiration-card top"><h3>Need inspiration?</h3><p>Try different hooks, moods, or structures to see how your campaign evolves.</p><button onClick={() => document.querySelector<HTMLElement>("[data-section='creative']")?.scrollIntoView({ behavior: "smooth", block: "start" })}>Explore Ideas</button></section>
          <dl>
            <dt>Chapter</dt><dd>{chapterSummary}</dd>
            <dt>Platforms</dt><dd className="summary-platforms">{form.platforms.map((platform) => <span key={platform}>{platformGlyphs[platform] || platform[0]}</span>)}</dd>
            <dt>Spoiler Level</dt><dd>{form.spoiler_level}</dd>
            <dt>Tone</dt><dd>{form.moods.join(", ") || "Choose moods..."}</dd>
            <dt>Promo Goal</dt><dd>{form.promotional_goal}</dd>
            <dt>Campaign Mode</dt><dd>{form.mode}</dd>
            <dt>Duration</dt><dd>{form.promo_duration}</dd>
            <dt>CTA</dt><dd>{form.cta || "Add your call to action..."}</dd>
          </dl>
          <button><Edit3 size={17} />Edit Summary</button>
          <section><h3>Preview Deliverables</h3><p>A quick taste of what will be generated.</p><article><strong>One-sentence hook</strong><span>Revenge isn't coming.<br />It's already here.</span></article><article><strong>Written chapter promo</strong><span>In a city built on secrets, one wrong move can burn you alive.</span></article><article className="video-preview"><div><Play size={34} /></div><small>0:15</small></article></section>
          <section><h3>What this will create</h3><ul>{[`${selectedOutputs.length} deliverables tailored to your chapter`, "Built-in spoiler guardrails", `Optimized for ${form.platforms.length} selected platforms`, "Ready-to-use content in minutes"].map((item) => <li key={item}><CheckCircle2 size={16} />{item}</li>)}</ul></section>
        </aside>
      </section>
    </main>
  );
}

function PodcastBriefScreen({ options, form, loading, status, setForm, onSave, onGenerate }: { options?: OptionsResponse; form: PodcastForm; loading: boolean; status: string; setForm: (form: PodcastForm) => void; onSave: () => void; onGenerate: () => void }) {
  const [errors, setErrors] = useState<Record<string, string>>({});
  const speakerCount = Math.max(1, Math.min(5, Number.parseInt(form.speakers || "1", 10) || 1));
  const roles = roleListFromValue(form.speaker_roles, form.podcast_format, speakerCount);
  const toneOptions = optionize([...(options?.podcast.tones || []), ...PODCAST_RECOMMENDED_TONES]);
  const audienceOptions = optionize([...(options?.audiences || []), ...PODCAST_RECOMMENDED_AUDIENCES]);
  const constraintOptions = optionize([...(options?.constraints || []), ...PODCAST_RECOMMENDED_CONSTRAINTS]);
  const cueOptions = optionize(PODCAST_CUES);
  const selectedTones = form.tone || [];
  const selectedAudiences = form.audience || [];
  const selectedConstraints = form.constraints || [];
  const selectedCues = form.performance_cues || [];
  const selectedKnowledgeSources = form.knowledge_sources || [];
  const modelLabel = titleCaseLabel(form.model_id || PODCAST_MODELS[1]);
  const episodeTitle = form.episode_title.trim() || "Untitled Episode";
  const showTitle = form.show_title.trim() || "Untitled Show";
  const episodeNumber = form.episode_number.trim() ? `Episode ${form.episode_number.trim()}` : "Episode -";

  function updateRoles(nextRoles: string[]) {
    setForm({ ...form, speaker_roles: nextRoles.join(", ") });
  }

  function updateSpeakerCount(value: string) {
    const count = Math.max(1, Math.min(5, Number.parseInt(value, 10) || 1));
    const nextRoles = roleListFromValue(form.speaker_roles, form.podcast_format, count);
    setForm({ ...form, speakers: String(count), speaker_roles: nextRoles.join(", ") });
  }

  function updateFormat(value: string) {
    const lower = value.toLowerCase();
    const inferredCount = lower.includes("monologue") ? 1 : lower.includes("interview") ? 2 : lower.includes("debate") ? Math.min(Math.max(speakerCount, 2), 3) : lower.includes("round") ? 3 : speakerCount;
    const nextRoles = roleListFromValue(form.speaker_roles, value, inferredCount);
    setForm({ ...form, podcast_format: value, speakers: String(inferredCount), speaker_roles: nextRoles.join(", ") });
  }

  function validateAndGenerate() {
    const next: Record<string, string> = {};
    if (!form.topic.trim()) next.topic = "Tell us what this episode is about.";
    if (!form.destination.trim()) next.destination = "Choose a destination.";
    if (!form.podcast_format.trim()) next.format = "Choose an episode format.";
    if (!form.target_length.trim()) next.length = "Choose a target length.";
    if (!form.speakers.trim()) next.speakers = "Choose a speaker count.";
    setErrors(next);
    if (Object.keys(next).length) {
      const first = Object.keys(next)[0];
      requestAnimationFrame(() => document.querySelector<HTMLElement>(`[data-podcast-field="${first}"]`)?.focus());
      return;
    }
    onGenerate();
  }

  function exploreInspiration() {
    const recommended = [...new Set([...selectedTones, "Cinematic", "Conversational"])];
    setForm({ ...form, tone: recommended, constraints: [...new Set([...selectedConstraints, "Spoiler-Free", "Include CTA"])] });
    requestAnimationFrame(() => document.querySelector<HTMLElement>("[data-podcast-section='tone']")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  }

  return (
    <main className="podcast-settings">
      <section className="podcast-settings-head">
        <div>
          <p>Podcast Studio</p>
          <h1>Write and render a production-ready episode.</h1>
          <span><CheckCircle2 size={16} />{loading ? "Generating script..." : status || "Ready"} · Last saved 2 minutes ago</span>
        </div>
        <div className="podcast-settings-actions">
          <button onClick={onSave}><Bookmark size={17} />Save draft</button>
          <button className="primary" disabled={loading} onClick={validateAndGenerate}><Mic2 size={17} />{loading ? "Generating..." : "Generate Script"}</button>
        </div>
      </section>

      <section className="podcast-settings-layout">
        <div className="podcast-settings-flow">
          <article className="podcast-settings-card">
            <header><b>1</b><div><h2>Episode Essentials</h2><p>Core details about your episode.</p></div></header>
            <label className="field span-3" data-podcast-field="topic"><span>Episode Topic</span><textarea maxLength={200} placeholder="What is this episode about?" value={form.topic} onChange={(event) => setForm({ ...form, topic: event.currentTarget.value })} />{errors.topic && <small className="field-error">{errors.topic}</small>}<small>{form.topic.length} / 200</small></label>
            <div className="podcast-three-grid">
              <label className="field"><span>Show Title</span><input placeholder="e.g. Grim Talk" value={form.show_title} onChange={(event) => setForm({ ...form, show_title: event.currentTarget.value })} /></label>
              <label className="field"><span>Episode Title</span><input placeholder="e.g. And the Winner Is" value={form.episode_title} onChange={(event) => setForm({ ...form, episode_title: event.currentTarget.value })} /></label>
              <label className="field"><span>Episode Number</span><input placeholder="e.g. 04" value={form.episode_number} onChange={(event) => setForm({ ...form, episode_number: event.currentTarget.value })} /></label>
            </div>
            <div className="podcast-three-grid">
              <div data-podcast-field="destination"><SingleSelect label="Destination / Platform" options={optionize(options?.podcast.destinations || [])} value={form.destination} onChange={(value) => setForm({ ...form, destination: value })} />{errors.destination && <small className="field-error">{errors.destination}</small>}</div>
              <div data-podcast-field="length"><SingleSelect label="Target Length" options={optionize(options?.podcast.lengths || [])} value={form.target_length} onChange={(value) => setForm({ ...form, target_length: value })} />{errors.length && <small className="field-error">{errors.length}</small>}</div>
              <div data-podcast-field="format"><SingleSelect label="Format" options={optionize(options?.podcast.formats || [])} value={form.podcast_format} onChange={updateFormat} />{errors.format && <small className="field-error">{errors.format}</small>}</div>
            </div>
          </article>

          <article className="podcast-settings-card" data-podcast-section="tone">
            <header><b>2</b><div><h2>Tone & Audience</h2><p>Define the mood and who you're speaking to.</p></div></header>
            <div className="podcast-two-grid">
              <MultiSelect label="Tone" options={toneOptions} value={selectedTones} onChange={(value) => setForm({ ...form, tone: value })} />
              <MultiSelect label="Audience" options={audienceOptions} value={selectedAudiences} onChange={(value) => setForm({ ...form, audience: value })} />
            </div>
          </article>

          <article className="podcast-settings-card">
            <header><b>3</b><div><h2>Constraints & Guardrails</h2><p>What to include, avoid, and how to shape the script.</p></div></header>
            <div className="podcast-guardrails-grid">
              <MultiSelect label="Constraints" options={constraintOptions} value={selectedConstraints} onChange={(value) => setForm({ ...form, constraints: value })} />
              <MultiSelect label="Performance Cues" options={cueOptions} value={selectedCues} onChange={(value) => setForm({ ...form, performance_cues: value })} />
              <label className="field"><span>Custom Constraints (Optional)</span><textarea maxLength={500} placeholder="Add any custom rules, notes, or instructions..." value={form.custom_constraints || ""} onChange={(event) => setForm({ ...form, custom_constraints: event.currentTarget.value })} /><small>{(form.custom_constraints || "").length} / 500</small></label>
            </div>
            <div className="knowledge-source-panel">
              <h3>Knowledge Sources</h3>
              <div className="knowledge-toggle-grid">
                {KNOWLEDGE_SOURCE_OPTIONS.map((source) => <label className="toggle-field" key={source}><input type="checkbox" checked={selectedKnowledgeSources.includes(source)} onChange={(event) => {
                  setForm({ ...form, knowledge_sources: event.currentTarget.checked ? [...new Set([...selectedKnowledgeSources, source])] : selectedKnowledgeSources.filter((item) => item !== source) });
                }} /><span>{source}</span></label>)}
              </div>
              <label className="field"><span>Source focus</span><input placeholder="e.g. funniest lines, Alex savage moments, best friendship scenes" value={form.source_focus || ""} onChange={(event) => setForm({ ...form, source_focus: event.currentTarget.value })} /></label>
            </div>
          </article>

          <article className="podcast-settings-card">
            <header><b>4</b><div><h2>Production Settings</h2><p>Technical and voice settings for your episode.</p></div></header>
            <div className="podcast-production-grid">
              <SingleSelect label="ElevenLabs Model" options={PODCAST_MODELS.map((value) => ({ value, label: titleCaseLabel(value) }))} value={form.model_id || PODCAST_MODELS[1]} onChange={(value) => setForm({ ...form, model_id: value })} />
              <label className="field" data-podcast-field="speakers"><span>Speakers</span><select value={String(speakerCount)} onChange={(event) => updateSpeakerCount(event.currentTarget.value)}>{[1, 2, 3, 4, 5].map((count) => <option key={count} value={count}>{count}</option>)}</select>{errors.speakers && <small className="field-error">{errors.speakers}</small>}<small>Number of distinct speakers.</small></label>
              <label className="field"><span>CTA (Call to Action)</span><input placeholder="e.g. Listen now on Spotify" value={form.cta} onChange={(event) => setForm({ ...form, cta: event.currentTarget.value })} /><small>Shown at the end of the episode.</small></label>
            </div>
            <div className="speaker-role-grid">
              {roles.map((role, index) => <label className="field" key={`speaker-${index + 1}`}><span>Speaker {index + 1}</span><input value={role} onChange={(event) => updateRoles(roles.map((item, roleIndex) => roleIndex === index ? event.currentTarget.value : item))} /></label>)}
            </div>
          </article>
        </div>

        <aside className="podcast-settings-sidebar">
          <section className="podcast-summary-card">
            <h2>Episode Summary</h2>
            <div className="podcast-summary-title"><img src="/assets/dashboard/podcast.webp" alt="" /><div><strong>{showTitle}</strong><span>{episodeNumber}</span><small>{episodeTitle}</small></div></div>
            <dl>
              <dt><Mic2 size={15} />Platform</dt><dd>{form.destination || "Not set"}</dd>
              <dt><Clapperboard size={15} />Format</dt><dd>{form.podcast_format || "Not set"}</dd>
              <dt><Calendar size={15} />Target Length</dt><dd>{form.target_length || "Not set"}</dd>
              <dt><Sparkles size={15} />Tone</dt><dd>{selectedTones.join(", ") || "Not set"}</dd>
              <dt><Users size={15} />Audience</dt><dd>{selectedAudiences.join(", ") || "Not set"}</dd>
              <dt><Settings size={15} />Constraints</dt><dd>{selectedConstraints.length ? `${selectedConstraints.length} selected` : "None"}</dd>
              <dt><BookOpen size={15} />Knowledge</dt><dd>{selectedKnowledgeSources.length ? `${selectedKnowledgeSources.length} selected` : "None"}</dd>
              <dt><FileAudio size={15} />Performance Cues</dt><dd>{selectedCues.length ? `${selectedCues.length} selected` : "None"}</dd>
              <dt><Rocket size={15} />Model</dt><dd>{modelLabel}</dd>
              <dt><MessageSquare size={15} />Speakers</dt><dd>{speakerCount}</dd>
              <dt><Target size={15} />CTA</dt><dd>{form.cta || "Not set"}</dd>
            </dl>
            <button onClick={() => document.querySelector<HTMLElement>(".podcast-settings-card")?.scrollIntoView({ behavior: "smooth", block: "start" })}><Edit3 size={16} />Edit All Settings</button>
          </section>
          <section className="podcast-summary-card compact">
            <h2>What this will create</h2>
            <ul>
              {["Production-ready script", selectedCues.length ? "Performance cues & pacing" : "", "Speaker roles & direction", `Optimized for ${form.destination || "your destination"}`, `Approx. ${form.target_length || "selected length"} episode`].filter(Boolean).map((item) => <li key={item}><CheckCircle2 size={16} />{item}</li>)}
            </ul>
          </section>
          <section className="podcast-summary-card compact">
            <h2>Need Inspiration?</h2>
            <p>Explore templates, tone examples, and successful episode formulas.</p>
            <button onClick={exploreInspiration}><Wand2 size={16} />Explore Inspiration</button>
          </section>
        </aside>
      </section>
    </main>
  );
}

function DraftStudio(props: { draft: GeneratedDraft; selected?: Deliverable; selectedId: string; mode: "Preview" | "Edit" | "Source"; status: string; revision: string; onSelect: (id: string) => void; onMode: (mode: "Preview" | "Edit" | "Source") => void; onCopy: (value: string) => void; onEditBrief: () => void; onCreateNew: () => void; onUpdate: (value: string) => void; onRevision: (value: string) => void; onApplyRevision: (value: string) => void; onRegenerate: () => void; onExport: () => void }) {
  return (
    <main className="draft-studio">
      <section className="result-header">
        <div><p>{labelize(props.draft.contentType)}</p><h1>{props.draft.title}</h1><span>{props.status}</span></div>
        <div className="result-actions"><button onClick={props.onEditBrief}><ArrowLeft size={18} />Back to brief</button><button onClick={() => props.onCopy(props.draft.rawSource)}><Copy size={18} />Copy all</button><button onClick={props.onExport}><Download size={18} />Export</button><button><Bookmark size={18} />Save draft</button><button className="primary" onClick={props.onCreateNew}><Plus size={18} />Create new</button></div>
      </section>
      <section className="studio-grid">
        <aside className="deliverables"><h2>Deliverables</h2>{props.draft.deliverables.map((item) => <button key={item.id} className={props.selectedId === item.id ? "selected" : ""} onClick={() => props.onSelect(item.id)}>{deliverableIcon(item)}<span><strong>{item.label}</strong><small>{item.status}</small></span><CheckCircle2 size={18} /></button>)}</aside>
        <DraftCanvas {...props} />
        <RefinePanel {...props} />
      </section>
    </main>
  );
}

function ChapterPromotionStudio(props: { draft: GeneratedDraft; selectedId: string; mode: "Preview" | "Edit" | "Source"; status: string; revision: string; onSelect: (id: string) => void; onMode: (mode: "Preview" | "Edit" | "Source") => void; onCopy: (value: string) => void; onEditBrief: () => void; onCreateNew: () => void; onUpdate: (value: string) => void; onRevision: (value: string) => void; onApplyRevision: (value: string) => void; onRegenerate: () => void; onExport: () => void; onSave: () => void; onNavigate: (tab: string) => void }) {
  const chapterGroups = useMemo(() => chapterPromotionGroups(props.draft), [props.draft.rawSource]);
  const [activeChapterId, setActiveChapterId] = useState(chapterGroups[0]?.id || "");
  const activeGroup = chapterGroups.find((group) => group.id === activeChapterId) || chapterGroups[0];
  const deliverables = activeGroup?.deliverables || chapterPromotionDeliverables(props.draft);
  const outputGroups = useMemo(() => chapterOutputGroups(deliverables), [deliverables]);
  const [expandedOutputGroup, setExpandedOutputGroup] = useState(outputGroups[0]?.id || "overview");
  const [versionOpen, setVersionOpen] = useState(false);
  const selected = deliverables.find((item) => item.id === props.selectedId) || deliverables[0];
  const selectedContent = selected?.content || props.draft.rawSource;
  const suggestedHashtags = hashtagTextFromDraft(props.draft, selected) || "#MortalVengeance #YAHorror #BookTok #HorrorReads #DarkAcademia #IndieAuthor";
  const chapterPackage = [activeGroup?.label, activeGroup?.summary, ...deliverables.map((item) => `### ${item.title}\n\n${item.content}`)].filter(Boolean).join("\n\n");
  const activeBook = bookFromChapterLabel(activeGroup?.label || props.draft.title);
  const activeChapterContext = chapterContextLabel(activeGroup?.label || props.draft.title);
  const selectedMeta = `${platformLabelForDeliverable(selected)} · Organic Social · ${stripMarkdown(selectedContent).length.toLocaleString()} characters`;

  useEffect(() => {
    const currentGroup = chapterGroups.find((group) => group.id === activeChapterId) || chapterGroups[0];
    if (!currentGroup) return;
    if (currentGroup.id !== activeChapterId) {
      setActiveChapterId(currentGroup.id);
    }
    if (!currentGroup.deliverables.some((item) => item.id === props.selectedId)) {
      props.onSelect(currentGroup.deliverables[0]?.id || "");
    }
  }, [activeChapterId, chapterGroups, props.selectedId]);

  useEffect(() => {
    const currentGroups = chapterOutputGroups(deliverables);
    if (!currentGroups.some((group) => group.id === expandedOutputGroup)) {
      setExpandedOutputGroup(currentGroups[0]?.id || "overview");
    }
  }, [deliverables, expandedOutputGroup]);

  function openChapter(group: ChapterPromoGroup) {
    setActiveChapterId(group.id);
    props.onSelect(group.deliverables[0]?.id || "");
    setExpandedOutputGroup(chapterOutputGroups(group.deliverables)[0]?.id || "overview");
    props.onMode("Preview");
  }

  return (
    <main className="chapter-delivery">
      <header className="chapter-topbar">
        <div className="brand"><img src="/assets/dashboard/telltales-ink-logo.webp" alt="" /><span>TellTales Ink</span></div>
        <nav>
          <button onClick={() => props.onNavigate("dashboard")}>Dashboard</button>
          <button onClick={() => props.onNavigate("generator")}>Generator</button>
          <button onClick={() => props.onNavigate("campaign")}>Campaign Mode</button>
          <button onClick={() => props.onNavigate("podcast")}>Podcast Studio</button>
          <button className="active" onClick={props.onCreateNew}>Chapter Promos</button>
          <button onClick={() => props.onNavigate("saved-drafts")}>Saved Drafts</button>
          <button onClick={() => props.onNavigate("gallery")}>Gallery</button>
          <button onClick={() => props.onNavigate("library")}>Library</button>
        </nav>
        <div className="chapter-user-tools"><button>?</button><button>⌕</button><button>TI</button></div>
      </header>

      <section className="chapter-result-head">
        <div><button className="back-link" onClick={props.onEditBrief}><ArrowLeft size={17} />Back to builder</button><h1>Chapter Promotion</h1><p><span>Generated</span> · Saved moments ago</p></div>
        <div className="result-actions"><button onClick={() => props.onCopy(chapterPackage)}><Copy size={18} />Copy chapter package</button><button onClick={props.onExport}><Download size={18} />Export<ChevronRight size={16} /></button><button onClick={props.onSave}><Bookmark size={18} />Save draft</button><button className="primary" onClick={props.onCreateNew}><Plus size={18} />Create new promos</button></div>
      </section>

      <section className="chapter-workspace">
        <aside className="chapter-deliverables">
          <h2>Chapters</h2>
          <div className="chapter-switcher">
            {chapterGroups.map((group, index) => <button key={group.id} className={activeGroup?.id === group.id ? "selected" : ""} onClick={() => openChapter(group)}><span>{chapterNumberFromLabel(group.label, index)}</span><strong>{chapterTitleFromLabel(group.label)}</strong><small>{group.deliverables.length} deliverables</small><CheckCircle2 size={15} /></button>)}
          </div>
          <h2>Outputs for {activeChapterContext.replace(/^Chapter\s+/i, "Chapter ")}</h2>
          <div className="chapter-output-groups">
            {outputGroups.map((group) => {
              const Icon = group.icon;
              const expanded = expandedOutputGroup === group.id;
              return <section key={group.id} className="chapter-output-group">
                <button className="chapter-output-group-head" aria-expanded={expanded} onClick={() => setExpandedOutputGroup(expanded ? "" : group.id)}><Icon size={18} /><span>{group.label}</span><small>{group.deliverables.length}</small><ChevronRight size={17} /></button>
                {expanded && <div className="chapter-output-group-items">
                  {group.deliverables.map((item) => <button key={item.id} className={selected?.id === item.id ? "selected" : ""} onClick={() => props.onSelect(item.id)}>{deliverableIcon(item)}<span>{item.label.replace(/\s+Promo$/i, "")}</span><CheckCircle2 size={15} /></button>)}
                </div>}
              </section>;
            })}
          </div>
        </aside>

        <section className="chapter-preview">
          <div className="mode-tabs">{(["Preview", "Edit", "Source"] as const).map((item) => <button key={item} className={props.mode === item ? "active" : ""} onClick={() => props.onMode(item)}>{item}</button>)}</div>
          {props.mode === "Preview" && <article className="chapter-preview-card"><p className="kicker">{activeBook || "Chapter Promos"}</p><h2>{activeChapterContext}</h2><h3>{selected?.title || "Generated Instagram Caption"}</h3><p className="chapter-selected-meta">{selectedMeta}</p><div className="rule" />{activeGroup?.summary && selected?.type !== "chapter_summary" && <section className="chapter-context-panel"><strong>Chapter Info</strong><div className="rendered" dangerouslySetInnerHTML={{ __html: markdownToHtml(activeGroup.summary) }} /></section>}<div className="social-caption-card">{deliverableIcon(selected)}<div className="rendered" dangerouslySetInnerHTML={{ __html: markdownToHtml(selectedContent) }} /><footer><span>{stripMarkdown(selectedContent).length.toLocaleString()} characters</span><button onClick={() => props.onCopy(selectedContent)}><Copy size={16} />Copy</button><button onClick={() => props.onMode("Edit")}><Edit3 size={16} />Edit</button><button onClick={props.onRegenerate}><RefreshCw size={16} />Regenerate</button><button><MoreVertical size={16} />More</button></footer></div>{selected?.label.toLowerCase().includes("hashtag") ? null : <div className="hashtag-row hashtag-row--content"><Hash size={26} /><div><strong>Suggested Hashtags</strong><p>{suggestedHashtags}</p></div><button onClick={() => props.onCopy(suggestedHashtags)}><Copy size={15} />Copy</button></div>}</article>}
          {props.mode === "Edit" && <textarea className="editor" value={selectedContent} onChange={(event) => props.onUpdate(event.currentTarget.value)} />}
          {props.mode === "Source" && <pre className="source">{chapterPackage}</pre>}
        </section>

        <aside className="chapter-side">
          <h2>Refine</h2>
          <h3>Quick adjustments</h3>
          <div className="quick-adjust-grid">
            <button onClick={() => props.onApplyRevision("Shorten")}><Scissors size={18} />Shorter</button>
            <button onClick={() => props.onApplyRevision("Make the selected deliverable more cinematic.")}><Clapperboard size={18} />More cinematic</button>
            <button onClick={() => props.onApplyRevision("Increase tension while preserving spoiler limits.")}><Flame size={18} />Increase tension</button>
            <button onClick={() => props.onApplyRevision("Strengthen the CTA.")}><Sparkles size={18} />Stronger CTA</button>
          </div>
          <label className="refine-custom"><span>Custom instruction</span><textarea value={props.revision} onChange={(event) => props.onRevision(event.currentTarget.value)} placeholder="Tell the editor what to change..." /></label>
          <button className="primary" onClick={() => props.onApplyRevision(props.revision)}><Wand2 size={18} />Apply revision</button>
          <div className="divider" />
          <h2>Version</h2>
          <button className="version-current" onClick={() => setVersionOpen(!versionOpen)}>Current: v1<ChevronRight size={17} /></button>
          {versionOpen && ["v1", "v0.9", "v0.8"].map((version, index) => <div className="version-row" key={version}><span className={index === 0 ? "current-dot" : ""} /> <strong>{version}</strong><small>{index === 0 ? "Current" : "Saved"}</small><MoreVertical size={16} /></div>)}
          <button className="history-link" onClick={() => setVersionOpen(!versionOpen)}><RefreshCw size={16} />View history</button>
          <div className="divider" />
          <h2>Export</h2>
          <div className="chapter-export-grid"><button onClick={props.onExport}><FileText size={20} />DOCX</button><button onClick={props.onExport}><Download size={20} />PDF</button><button onClick={props.onExport}><Code2 size={20} />Markdown</button></div>
        </aside>
      </section>
    </main>
  );
}

function PodcastStudio(props: { draft: GeneratedDraft; selected?: Deliverable; selectedId: string; mode: "Preview" | "Edit" | "Source"; status: string; revision: string; voiceOptions: { value: string; label: string }[]; hostVoiceId: string; guestVoiceId: string; guest2VoiceId: string; drafts: DraftRecord[]; job?: Job; onSelect: (id: string) => void; onMode: (mode: "Preview" | "Edit" | "Source") => void; onCopy: (value: string) => void; onEditBrief: () => void; onCreateNew: () => void; onUpdate: (value: string) => void; onRevision: (value: string) => void; onApplyRevision: (value: string) => void; onExport: () => void; onSave: () => void; onRender: (kind: "preview" | "full") => void; onOpenDraft: (draft: DraftRecord) => void; setHostVoiceId: (id: string) => void; setGuestVoiceId: (id: string) => void; setGuest2VoiceId: (id: string) => void }) {
  const [scriptView, setScriptView] = useState<"Production view" | "Clean transcript" | "Source">("Production view");
  const [section, setSection] = useState("Script");
  const [extraSegments, setExtraSegments] = useState<Deliverable[]>([]);
  const [editingTurn, setEditingTurn] = useState("");
  const [turnDrafts, setTurnDrafts] = useState<Record<string, string>>({});
  const [activeCues, setActiveCues] = useState<string[]>([]);
  const [voiceTags, setVoiceTags] = useState("");
  const [soundEffectTags, setSoundEffectTags] = useState("");
  const [musicTags, setMusicTags] = useState("");
  const [selectedSpeakerId, setSelectedSpeakerId] = useState("aldan");
  const [selectedVoiceId, setSelectedVoiceId] = useState(props.hostVoiceId || "");
  const [turnMenu, setTurnMenu] = useState("");
  const [inspectorNotice, setInspectorNotice] = useState("Ready");
  const baseSegments = props.draft.deliverables.length ? props.draft.deliverables : [
    { id: "opening", title: "Opening", label: "Opening", content: props.draft.rawSource, type: "podcast", status: "ready" as const, characterCount: props.draft.rawSource.length },
    { id: "segment-1", title: "Segment 1", label: "Segment 1", content: props.draft.rawSource, type: "podcast", status: "ready" as const, characterCount: props.draft.rawSource.length },
    { id: "segment-2", title: "Segment 2", label: "Segment 2", content: props.draft.rawSource, type: "podcast", status: "ready" as const, characterCount: props.draft.rawSource.length }
  ];
  const segments = [...baseSegments, ...extraSegments];
  const selected = segments.find((item) => item.id === props.selectedId) || props.selected || segments[0];
  const transcript = stripMarkdown(selected?.content || props.draft.rawSource || "Generated dialogue appears here as an editable speaker turn.");
  const excerpts = transcript.split(/\n+/).filter(Boolean);
  const turns = [
    { id: "aldan", initials: "AL", name: "ALDAN", role: "HOST", voice: "Fred - Radio Host", color: "burgundy", text: turnDrafts.aldan || excerpts[0] || "Generated dialogue appears here as an editable speaker turn." },
    { id: "jaime", initials: "JA", name: "JAIME", role: "CO-HOST", voice: "Alejandro Torres - Cloned", color: "bronze", text: turnDrafts.jaime || excerpts[1] || excerpts[0] || "Generated dialogue appears here as an editable speaker turn." },
    { id: "aida", initials: "AI", name: "AIDA", role: "GUEST", voice: "Shannon B - Professional", color: "green", text: turnDrafts.aida || excerpts[2] || excerpts[1] || "Generated dialogue appears here as an editable speaker turn." }
  ];
  const selectedSpeaker = turns.find((turn) => turn.id === selectedSpeakerId) || turns[0];
  const mp3 = props.job?.result?.mp3 as { id?: string } | undefined;

  function addSegment() {
    const index = segments.length + 1;
    const next = {
      id: `segment-${index}`,
      title: `Segment ${index}`,
      label: `Segment ${index}`,
      content: "New segment notes. Add dialogue, cues, or direction here.",
      type: "podcast_segment",
      status: "edited" as const,
      characterCount: 52,
    };
    setExtraSegments([...extraSegments, next]);
    props.onSelect(next.id);
    setInspectorNotice(`${next.label} added`);
  }

  function toggleCue(cue: string) {
    setActiveCues(activeCues.includes(cue) ? activeCues.filter((item) => item !== cue) : [...activeCues, cue]);
    setInspectorNotice(`${cue} cue updated for ${selectedSpeaker.name}`);
  }

  function previewTurn(id = selectedSpeakerId) {
    const turn = turns.find((item) => item.id === id) || selectedSpeaker;
    setSelectedSpeakerId(turn.id);
    setInspectorNotice(`Preview queued for ${turn.name}`);
  }

  function saveTurn(id: string, value: string) {
    setTurnDrafts({ ...turnDrafts, [id]: value });
    setEditingTurn("");
    setInspectorNotice("Turn updated");
  }

  function renderScriptPanel() {
    if (section === "Voices") {
      return <div className="podcast-tab-panel"><h3>Speaker voices</h3>{turns.map((turn) => <article className="voice-assignment-row" key={turn.id}><div className={`speaker-avatar ${turn.color}`}>{turn.initials}</div><div><strong>{turn.name}</strong><span>{turn.role}</span></div><select value={selectedVoiceId} onChange={(event) => setSelectedVoiceId(event.currentTarget.value)}>{props.voiceOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></article>)}</div>;
    }
    if (section === "Audio") {
      return <div className="podcast-tab-panel"><h3>Audio preview</h3><p>{props.job ? props.job.message : inspectorNotice}</p>{mp3?.id && <audio controls src={api.artifactDownloadUrl(mp3.id)} />}<button className="primary" onClick={() => props.onRender("preview")}><Play size={17} />Render preview</button></div>;
    }
    if (section === "Assets") {
      return <div className="podcast-tab-panel"><h3>Production tags</h3><label className="field"><span>Voice performance tags</span><textarea value={voiceTags} onChange={(event) => setVoiceTags(event.currentTarget.value)} placeholder="e.g. tense, whispered, clipped, warm, investigative" /></label><label className="field"><span>Sound effects tags</span><textarea value={soundEffectTags} onChange={(event) => setSoundEffectTags(event.currentTarget.value)} placeholder="e.g. door slam, distant bell, static hit, match strike" /></label><label className="field"><span>Music tags</span><textarea value={musicTags} onChange={(event) => setMusicTags(event.currentTarget.value)} placeholder="e.g. low strings, ominous pulse, dark academia piano" /></label></div>;
    }
    if (section === "Compare") {
      return <div className="compare-panel"><article><h3>Production view</h3><p>{turns.map((turn) => `${turn.name}: ${turn.text}`).join("\n\n")}</p></article><article><h3>Clean transcript</h3><p>{turns.map((turn) => turn.text).join("\n\n")}</p></article></div>;
    }
    return <>
      <div className="script-toolbar">
        {(["Production view", "Clean transcript", "Source"] as const).map((item) => <button key={item} className={scriptView === item ? "active" : ""} onClick={() => setScriptView(item)}>{item === "Source" ? <Code2 size={16} /> : <FileText size={16} />}{item}</button>)}
      </div>
      {scriptView === "Source" ? <pre className="podcast-source">{props.draft.rawSource}</pre> : turns.map((turn) => (
        <article className="turn-card" key={turn.id}>
          <div className={`speaker-avatar ${turn.color}`}>{turn.initials}</div>
          <div className="turn-body">
            <header>
              <div><strong>{turn.name}</strong><span>{turn.role}</span><small>Mic - {turn.voice}</small></div>
              <nav><button onClick={() => setEditingTurn(turn.id)}><Edit3 size={16} />Edit</button><button onClick={() => toggleCue("Pause")}><span>+</span>Add cue</button><button onClick={() => previewTurn(turn.id)}><Play size={15} />Preview</button><button onClick={() => setTurnMenu(turnMenu === turn.id ? "" : turn.id)}><MoreVertical size={17} /></button></nav>
            </header>
            {editingTurn === turn.id ? <div className="turn-editor"><textarea value={turn.text} onChange={(event) => setTurnDrafts({ ...turnDrafts, [turn.id]: event.currentTarget.value })} /><button onClick={() => saveTurn(turn.id, turnDrafts[turn.id] || turn.text)}>Save turn</button></div> : <p>{scriptView === "Clean transcript" ? turn.text : `${turn.text}\n${turn.text}`}</p>}
            {activeCues.length ? <div className="turn-cues">{activeCues.map((cue) => <span key={cue}>{cue}</span>)}</div> : null}
            {turnMenu === turn.id && <div className="turn-menu"><button onClick={() => setEditingTurn(turn.id)}>Edit speaker turn</button><button onClick={() => toggleCue("Emphasis")}>Add emphasis cue</button><button onClick={() => previewTurn(turn.id)}>Preview this turn</button></div>}
          </div>
        </article>
      ))}
      <div className="episode-player">
        <strong>Episode preview - 0:59</strong>
        <button onClick={() => previewTurn()}><Play size={18} /></button>
        <span>0:00</span>
        <input type="range" min="0" max="100" defaultValue="18" />
        <span>0:59</span>
        <button><FileAudio size={18} /></button>
        <input type="range" min="0" max="100" defaultValue="70" />
        <button onClick={props.onExport}><Download size={18} /></button>
      </div>
    </>;
  }

  return (
    <main className="podcast-delivery">
      <header className="podcast-topbar">
        <div className="brand"><img src="/assets/dashboard/telltales-ink-logo.webp" alt="" /><span>TellTales Ink</span></div>
        <nav>
          {["Dashboard", "Generator", "Podcast Studio", "Templates", "Library", "Assets"].map((item) => <button key={item} className={item === "Podcast Studio" ? "active" : ""}>{item}</button>)}
        </nav>
        <div className="podcast-user-tools"><button>?</button><button><Bookmark size={18} /></button><button>TI</button></div>
      </header>

      <section className="podcast-episode-head">
        <div>
          <p>Grim Talk</p>
          <h1>{props.draft.title || "and the winner is"}</h1>
          <span>Episode 04 - Roundtable discussion - 3 speakers</span>
          <small><CheckCircle2 size={16} />Script ready - {props.status}</small>
        </div>
        <div className="podcast-head-actions">
          <button onClick={props.onEditBrief}><ArrowLeft size={17} />Back to brief</button>
          <button onClick={props.onSave}><Bookmark size={17} />Save draft</button>
          <button onClick={props.onCreateNew}><Plus size={17} />Create new script</button>
          <button className="primary" onClick={() => props.onRender("preview")}><Play size={17} />Render preview</button>
        </div>
      </section>

      <nav className="podcast-section-tabs">
        {["Script", "Voices", "Audio", "Assets", "Compare"].map((item) => <button key={item} className={section === item ? "active" : ""} onClick={() => setSection(item)}>{item}</button>)}
      </nav>

      <section className="podcast-workspace">
        <aside className="episode-structure">
          <h2>Episode Structure</h2>
          {segments.slice(0, 5).map((item, index) => (
            <button key={item.id} className={props.selectedId === item.id || (!props.selectedId && index === 0) ? "selected" : ""} onClick={() => props.onSelect(item.id)}>
              <strong>{index + 1}</strong>
              <span>{index === 0 ? "Opening" : item.label || item.title || `Segment ${index}`}</span>
              <small>{["0:45", "3:18", "3:02", "2:10", "0:44"][index] || "1:20"}</small>
              {index === 0 ? <CheckCircle2 size={18} /> : <i />}
            </button>
          ))}
          <button className="add-segment" onClick={addSegment}><span>+</span>Add segment</button>
        </aside>

        <section className="podcast-script-panel">
          <h2>{section}</h2>
          {renderScriptPanel()}
        </section>

        <aside className="production-inspector">
          <h2>Production Inspector</h2>
          <label>Speaker<select value={selectedSpeakerId} onChange={(event) => setSelectedSpeakerId(event.currentTarget.value)}>{turns.map((turn) => <option key={turn.id} value={turn.id}>{turn.name} - {turn.role}</option>)}</select></label>
          <label>Voice<select value={selectedVoiceId} onChange={(event) => { setSelectedVoiceId(event.currentTarget.value); props.setHostVoiceId(event.currentTarget.value); }}>{props.voiceOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
          <div className="voice-sample"><button onClick={() => previewTurn()}><Play size={16} /></button><span>{inspectorNotice} - {selectedSpeaker.name}</span><input type="range" min="0" max="100" defaultValue="18" /><span>0:09</span><button onClick={() => setTurnMenu(turnMenu ? "" : selectedSpeaker.id)}><MoreVertical size={17} /></button></div>
          <h3>Performance</h3>
          {[["Stability", "50"], ["Similarity", "75"], ["Style", "0"], ["Speed", "55"]].map(([label, value]) => <label className="slider-row" key={label}>{label}<input type="range" min="0" max="100" defaultValue={value} /><span>{label === "Style" ? "0" : label === "Speed" ? "1.00" : `0.${value}`}</span></label>)}
          <label className="boost-row">Speaker boost<input type="checkbox" defaultChecked /></label>
          <h3>Performance Cues</h3>
          <div className="cue-grid">{["Pause", "Emphasis", "Slower", "Whisper", "Tense", "Sound effect"].map((cue) => <button key={cue} className={activeCues.includes(cue) ? "selected" : ""} onClick={() => toggleCue(cue)}>{cue}</button>)}</div>
          <label className="field"><span>Voice performance tags</span><input value={voiceTags} onChange={(event) => setVoiceTags(event.currentTarget.value)} placeholder="tense, breathy, clipped" /></label>
          <label className="field"><span>Sound effects tags</span><input value={soundEffectTags} onChange={(event) => setSoundEffectTags(event.currentTarget.value)} placeholder="static hit, bell, door" /></label>
          <label className="field"><span>Music tags</span><input value={musicTags} onChange={(event) => setMusicTags(event.currentTarget.value)} placeholder="low strings, piano pulse" /></label>
          <button className="primary" onClick={() => previewTurn()}><Play size={17} />Preview selected turn</button>
          <div className="audio-preview-card">
            <h3>Audio Preview</h3>
            <p><CheckCircle2 size={18} />{props.job ? props.job.message : "Ready"} - 0:59</p>
            {mp3?.id && <audio controls src={api.artifactDownloadUrl(mp3.id)} />}
            <div><button onClick={() => props.onRender("preview")}><RefreshCw size={17} />Render again</button><button className="primary" onClick={() => props.onRender("full")}><Download size={17} />Download MP3</button></div>
          </div>
        </aside>
      </section>
    </main>
  );
}

function DraftCanvas(props: { selected?: Deliverable; mode: "Preview" | "Edit" | "Source"; draft: GeneratedDraft; onMode: (mode: "Preview" | "Edit" | "Source") => void; onUpdate: (value: string) => void; onCopy?: (value: string) => void; onRegenerate?: () => void }) {
  const isPressRelease = props.draft.contentType === "press_release";
  return (
    <section className="canvas">
      <div className="mode-tabs">{(["Preview", "Edit", "Source"] as const).map((item) => <button key={item} className={props.mode === item ? "active" : ""} onClick={() => props.onMode(item)}>{item}</button>)}</div>
      {props.selected && props.mode === "Preview" && (isPressRelease ? <PressReleaseNewspaper selected={props.selected} onCopy={props.onCopy} onEdit={() => props.onMode("Edit")} onRegenerate={props.onRegenerate} /> : <article className="preview-card"><p className="kicker">{props.selected.platform || props.selected.type}</p><h2>{props.selected.title}</h2><div className="rule" /><div className="rendered" dangerouslySetInnerHTML={{ __html: markdownToHtml(props.selected.content) }} /><footer><span>{props.selected.characterCount.toLocaleString()} characters</span>{props.onCopy && <button onClick={() => props.onCopy?.(props.selected?.content || "")}><Copy size={16} />Copy</button>}<button onClick={() => props.onMode("Edit")}><Edit3 size={16} />Edit</button>{props.onRegenerate && <button onClick={props.onRegenerate}><RefreshCw size={16} />Regenerate</button>}</footer></article>)}
      {props.selected && props.mode === "Edit" && <textarea className="editor" value={props.selected.content} onChange={(event) => props.onUpdate(event.currentTarget.value)} />}
      {props.mode === "Source" && <pre className="source">{props.draft.rawSource}</pre>}
    </section>
  );
}

function PressReleaseNewspaper(props: { selected: Deliverable; onCopy?: (value: string) => void; onEdit: () => void; onRegenerate?: () => void }) {
  const lines = props.selected.content.split("\n").map((line) => line.trim()).filter(Boolean);
  const headline = stripMarkdown(lines.find((line) => /^#{1,3}\s+/.test(line)) || props.selected.title || "Press Release");
  const body = lines.filter((line) => !/^#{1,3}\s+/.test(line));
  const subhead = stripMarkdown(body.find((line) => line.length > 50) || body[0] || "");
  const bodyText = body.join("\n");
  return (
    <article className="press-newspaper-preview">
      <header>
        <div>
          <span>TellTales Ink Gazette</span>
          <strong>Front Page Release</strong>
        </div>
        <small>For media distribution</small>
      </header>
      <div className="newspaper-rule" />
      <p className="newspaper-kicker">{props.selected.label}</p>
      <h2>{headline}</h2>
      {subhead ? <p className="newspaper-subhead">{subhead}</p> : null}
      <div className="newspaper-columns rendered" dangerouslySetInnerHTML={{ __html: markdownToHtml(bodyText) }} />
      <footer>
        <span>{props.selected.characterCount.toLocaleString()} characters</span>
        {props.onCopy && <button onClick={() => props.onCopy?.(props.selected.content)}><Copy size={16} />Copy</button>}
        <button onClick={props.onEdit}><Edit3 size={16} />Edit</button>
        {props.onRegenerate && <button onClick={props.onRegenerate}><RefreshCw size={16} />Regenerate</button>}
      </footer>
    </article>
  );
}

function RefinePanel(props: { revision: string; onRevision: (value: string) => void; onApplyRevision: (value: string) => void; compact?: boolean }) {
  return (
    <div className={props.compact ? "" : "refine"}>
      <h2>Refine Draft</h2>
      <button onClick={() => props.onApplyRevision("Shorten")}><Scissors size={18} />Shorten</button>
      <button onClick={() => props.onApplyRevision("Make the selected deliverable more cinematic.")}><Wand2 size={18} />More cinematic</button>
      <button onClick={() => props.onApplyRevision("Increase tension while preserving spoiler limits.")}><Flame size={18} />Increase tension</button>
      <button onClick={() => props.onApplyRevision("Strengthen the CTA.")}><Sparkles size={18} />Stronger CTA</button>
      <input value={props.revision} onChange={(event) => props.onRevision(event.currentTarget.value)} placeholder="Tell the editor what to change..." />
      <button className="primary" onClick={() => props.onApplyRevision(props.revision)}><Wand2 size={18} />Apply revision</button>
    </div>
  );
}

function JobPanel({ job }: { job: Job }) {
  const mp3 = job.result?.mp3 as { id?: string } | undefined;
  const zip = job.result?.package as { id?: string } | undefined;
  return <div className="job-panel"><progress max={1} value={job.progress} /><span>{Math.round(job.progress * 100)}%</span>{mp3?.id && <audio controls src={api.artifactDownloadUrl(mp3.id)} />}{zip?.id && <a className="download-link" href={api.artifactDownloadUrl(zip.id)}>Download package</a>}</div>;
}

function SavedDraftsScreen({ drafts, onOpenDraft }: { drafts: DraftRecord[]; onOpenDraft: (draft: DraftRecord) => void }) {
  const [selected, setSelected] = useState<DraftRecord | null>(null);
  useEffect(() => {
    if (!selected && drafts.length) setSelected(drafts[0]);
  }, [drafts, selected]);
  return (
    <main className="list-screen saved-screen">
      <section>
        <h1>Saved Drafts</h1>
        <div className="draft-list">{drafts.map((draft) => <button key={draft.id} className={selected?.id === draft.id ? "selected" : ""} onClick={() => setSelected(draft)}><strong>{draft.title}</strong><span>{labelize(draft.content_type)} - {draft.updated_at || draft.created_at}</span></button>)}</div>
      </section>
      <section className="saved-editor">
        <h2>{selected?.title || "Select a draft"}</h2>
        <div className="saved-actions">
          <button disabled={!selected} onClick={() => selected && onOpenDraft(selected)}><Edit3 size={16} />Open in editor</button>
          <button disabled={!selected} onClick={() => selected?.content && navigator.clipboard.writeText(selected.content)}><Copy size={16} />Copy</button>
          <button disabled={!selected}><Download size={16} />Download</button>
        </div>
        <pre>{selected?.content || selected?.path || "Saved draft details appear here."}</pre>
      </section>
    </main>
  );
}

function GalleryScreen({ gallery }: { gallery: GalleryImage[] }) {
  return <main className="list-screen"><h1>Gallery</h1><div className="asset-grid">{gallery.map((item) => <article key={item.path}>{item.url ? <img className="asset-image" src={item.url} alt={item.topic} /> : <div className="asset-thumb"><ImageIcon /></div>}<strong>{item.topic}</strong><span>{item.type_label} - {item.format || item.aspect} - {item.date}</span><a className="download-link" href={item.url || "#"}>Download</a></article>)}</div></main>;
}

function LibraryScreen({ drafts, audio }: { drafts: DraftRecord[]; audio: AudioEpisode[] }) {
  return <main className="list-screen"><h1>Library</h1><div className="library-grid"><section><h2>Documents</h2>{drafts.map((draft) => <p key={draft.id}><strong>{draft.title}</strong><br />{labelize(draft.content_type)} - {draft.updated_at || draft.created_at}</p>)}</section><section><h2>Audio</h2>{audio.map((episode) => <div className="audio-row" key={episode.mp3_path}><p><strong>{episode.title}</strong><br />{episode.mode} - {episode.segments} segments - {episode.date}</p>{episode.url && <audio controls src={episode.url} />}</div>)}</section></div></main>;
}
