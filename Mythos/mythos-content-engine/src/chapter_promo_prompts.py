"""Genre-aware chapter teaser + summary prompt builders for the Chapter Promos feature.

The Chapter Promos panel reads a real chapter from a manuscript and produces either a
SWBST chapter summary or platform teasers. These builders encode teaser craft (present
tense, no clichés, unresolved endings, genre-matched tone) and a selectable structural
"pillar", and they keep every output grounded in the actual chapter text passed in.
"""

from __future__ import annotations


# Mode of the panel: creative campaign intensity vs. an objective chapter summary.
CHAPTER_PROMO_MODES = ["Tease It", "Promote It", "Give Me Both", "Chapter Summary"]

CHAPTER_PROMO_MODE_INSTRUCTIONS = {
    "Tease It": "Create curiosity through atmosphere, implication, and unanswered questions.",
    "Promote It": "Sell the chapter as a major story event with stronger stakes and more context.",
    "Give Me Both": "Generate the restrained teaser and the fuller promotional package.",
    "Chapter Summary": "Summarize what happens for planning, continuity, and analysis.",
    # Backward-compatible value from older saved drafts.
    "Teaser / Promo": "Create spoiler-controlled promotional assets for the chapter.",
}

PROMO_HOOK_TYPES = [
    "Choose the Strongest Hook for Me",
    "A Character in Danger",
    "A Disturbing Discovery",
    "A New Suspect",
    "A Major Confrontation",
    "A Secret About to Surface",
    "An Impossible Decision",
    "A Relationship Breaking Apart",
    "A Body-Count Threat",
    "A Historical Revelation",
    "Institutional Hypocrisy",
    "Custom Hook",
]

PROMOTIONAL_GOALS = [
    "Read the Next Chapter",
    "Start the Book",
    "Return to the Series",
    "Discuss a Revelation",
    "Fear for a Character",
    "Question Every Suspect",
    "Preorder or Buy the Book",
    "Custom Goal",
]

PROMOTIONAL_MOODS = [
    "Ominous",
    "Unhinged",
    "Prestige",
    "Savage",
    "Psychological",
    "Darkly Funny",
    "Gothic",
    "Emotional",
    "Custom Mood",
]

PROMOTIONAL_MOOD_INSTRUCTIONS = {
    "Ominous": "Something terrible is approaching.",
    "Unhinged": "The characters, and possibly the campaign, are losing control.",
    "Prestige": "Elegant, cinematic, and quietly threatening.",
    "Savage": "Fast, bloody, and unapologetically slasher.",
    "Psychological": "Paranoid, manipulative, and difficult to trust.",
    "Darkly Funny": "Terrible things happen; priorities remain absurd.",
    "Gothic": "Atmospheric, haunted by history, and steeped in place.",
    "Emotional": "Character consequences take precedence over spectacle.",
    "Custom Mood": "Follow the user's custom mood notes.",
}

PROMO_OUTPUT_FORMATS = {
    "one_sentence_hook": "One-sentence hook",
    "excerpt_teaser": "50-150-word excerpt teaser",
    "written_promo": "Written chapter promo",
    "social_15": "15-second social promo",
    "tv_30": "30-second TV-style promo",
    "prestige_60": "60-second prestige promo",
    "caption_options": "Three caption options",
    "tagline_options": "Three tagline options",
    "title_card": "Thumbnail / title-card concept",
}

DEFAULT_PROMO_OUTPUTS = [
    "one_sentence_hook",
    "excerpt_teaser",
    "written_promo",
    "caption_options",
    "tagline_options",
    "title_card",
]

PROMO_DURATIONS = [
    "Written Campaign Only",
    "15-Second Social Sting",
    "30-Second Chapter Promo",
    "60-Second Prestige Preview",
]

PROMO_DURATION_GUIDANCE = {
    "15-Second Social Sting": (
        "Structure any video promo as 0-3s immediate hook, 3-8s two complications, "
        "8-12s mystery/horror/emotional sting, 12-15s chapter title and CTA. "
        "Use about 25-45 spoken words."
    ),
    "30-Second Chapter Promo": (
        "Structure any video promo as 0-5s opening situation, 5-12s central conflict, "
        "12-22s accelerating fragments, 22-26s final shock or question, 26-30s title card. "
        "Use about 55-80 spoken words."
    ),
    "60-Second Prestige Preview": (
        "Structure any video promo as 0-10s atmospheric opening, 10-25s narrative setup, "
        "25-45s conflicts and suspicions, 45-53s climactic montage, 53-60s sting and title card. "
        "Use about 100-140 spoken words with room for silence."
    ),
    "Written Campaign Only": "Do not force a timed video script unless the requested outputs explicitly include one.",
    # Backward-compatible values from older saved drafts.
    "Auto-match requested outputs": "Match timing to the requested outputs and platform.",
    "15-second social promo": "Use the 15-second social sting structure.",
    "30-second standard promo": "Use the 30-second chapter promo structure.",
    "60-second prestige promo": "Use the 60-second prestige preview structure.",
}

GENRE_PROMO_MODES = {
    "Match the Chapter Automatically": (
        "Choose the promotional mode that honestly matches the chapter's strongest actual hook."
    ),
    "Slasher Pursuit": (
        "Emphasize immediate danger, pursuit or isolation, a recognizable weapon or killer motif, "
        "rapid editing, and a possible victim. Do not reveal who dies, the complete murder, or the killer."
    ),
    "Psychological Dread": (
        "Emphasize contradictory memories, manipulation, mistrust, fragmented dialogue, unreliable evidence, "
        "and a character questioning what they witnessed."
    ),
    "Caribbean Gothic": (
        "Emphasize landscape as pressure or witness, oppressive heat/rain/river/vegetation/darkness/decay, "
        "historical guilt, family secrets, and culturally specific imagery only when grounded in the chapter."
    ),
    "Dark Academia": (
        "Emphasize institutional power, reputation, grades, discipline, archives, rituals, corruption under "
        "respectability, and intellectual language colliding with bodily danger."
    ),
    "Satirical Horror": (
        "Emphasize absurd priorities during crisis, institutional spin, privilege, hypocrisy, image management, "
        "and a punchline that increases rather than dissolves the tension."
    ),
    "Historical Mystery": (
        "Emphasize a past event with present consequences, a photograph/letter/name/location/testimony, "
        "parallel timelines, and one carefully withheld connection."
    ),
    "Prestige Ensemble Drama": (
        "Emphasize overlapping character motives, emotional fallout, controlled escalation, and a cinematic finish."
    ),
    "Custom Direction": "Follow the user's supplied creative direction.",
    # Backward-compatible value from older saved drafts.
    "Auto-match chapter": "Choose the promotional mode that honestly matches the chapter's strongest actual hook.",
}
GENRE_PROMO_MODE_OPTIONS = list(GENRE_PROMO_MODES.keys())

STYLE_VARIANTS = {
    "chapter_promos": [
        ("spoiler_safe_teaser", "Spoiler-Safe Suspense"),
        ("prestige_tv_preview", "Prestige-TV Preview"),
        ("classic_slasher_promo", "Classic Slasher Promo"),
        ("psychological_spiral", "Psychological Spiral"),
        ("caribbean_gothic_dread", "Caribbean Gothic Dread"),
        ("dark_academia_blood", "Dark Academia with Blood on the Syllabus"),
        ("satirical_chaos", "Satirical Chaos"),
        ("previously_on", "Previously on Mortal Vengeance"),
        ("next_chapter_preview", "Next Chapter Preview"),
        ("custom_direction", "Custom Direction"),
    ],
}

STYLE_VARIANT_INSTRUCTIONS = {
    "spoiler_safe_teaser": "Promote the chapter with tension, mood, and stakes without revealing major spoilers.",
    "prestige_tv_preview": "Make the chapter feel like the next episode of a premium serialized drama: controlled, cinematic, and sharp.",
    "classic_slasher_promo": "Use pursuit, isolation, danger, and a clean sting without revealing deaths or killer identity.",
    "psychological_spiral": "Use mistrust, contradiction, manipulation, and uncertainty as the central engine.",
    "caribbean_gothic_dread": "Let place, history, weather, family pressure, and atmosphere carry the threat.",
    "dark_academia_blood": "Use institutional polish, academic ritual, reputation, and bodily danger in collision.",
    "satirical_chaos": "Let absurd priorities and institutional spin intensify the horror instead of undercutting it.",
    "previously_on": "Write like a serialized recap-preview hybrid: remind readers what pressure is already in motion, then open the next door.",
    "next_chapter_preview": "Position the chapter as the next urgent installment, with a clean hook and unresolved ending.",
    "custom_direction": "Follow the user's custom creative direction.",
    # Backward-compatible variant keys.
    "chapter_hook": "Open with a sharp hook that makes the chapter feel urgent, specific, and worth reading.",
    "quote_led_promo": "Use a short chapter quote or quote-style line as the anchor, then add brief promotional context.",
    "social_countdown": "Write copy suitable for a launch countdown, chapter reveal, or serialized reading campaign.",
}


# Genre -> tone instruction injected into the system preamble of every call.
GENRE_TONES = {
    "Slasher": (
        "Fast, dangerous, physical. Use isolation, pursuit, weapon motifs, possible victims, and abrupt reversals."
    ),
    "Mystery": (
        "Curious, clue-driven, tense. Use contradictions, suspects, hidden motives, and unanswered questions."
    ),
    "Psychological Thriller": (
        "Paranoid, manipulative, intimate. Use mistrust, unreliable evidence, memory, and pressure."
    ),
    "Suspense": (
        "Controlled, escalating, uneasy. Let the audience feel the danger approaching before it arrives."
    ),
    "Mystery / Thriller / Suspense": (
        "Cryptic, tense, urgent, atmospheric. Raise questions instead of giving answers. "
        "Use secrets, investigations, hidden motives, and unreliable details."
    ),
    "Horror": (
        "Visceral, eerie, threatening. Create dread, a shiver, or an impulse to look away. "
        "Use isolation, wrongness, and physical menace."
    ),
    "Caribbean Gothic": (
        "Atmospheric, place-haunted, historically pressured. Let landscape, weather, memory, family, and silence carry threat."
    ),
    "Dark Academia": (
        "Polished, intellectual, institutional, and dangerous. Let reputation, ritual, archives, grades, and power hide violence."
    ),
    "Satire": (
        "Sharp, socially aware, and darkly funny. Let absurd priorities expose the horror underneath."
    ),
    "Historical Mystery": (
        "Elegant, investigative, and time-layered. Use documents, testimony, photographs, old names, and present consequences."
    ),
    "Custom Blend": (
        "Follow the user's specified blend while keeping the chapter's actual content and spoiler limits intact."
    ),
    "Action / Fantasy / Sci-Fi": (
        "Bold, high-energy, cinematic. Promise momentum and danger. "
        "Use urgency, escalation, and impossible odds."
    ),
    "Romance / Drama": (
        "Emotional, intimate, yearning. Highlight internal conflict and longing. "
        "Use the gap between what characters want and what they risk."
    ),
    "Comedy": (
        "Witty, fast, personality-driven. Let the protagonist's voice carry the chaos. "
        "Make it relatable and slightly absurd."
    ),
    "Literary Fiction": (
        "Precise, layered, emotionally resonant. Favor mood over event. "
        "Let a single detail carry the weight."
    ),
}
GENRE_OPTIONS = list(GENRE_TONES.keys())


# Teaser "pillar" -> structural anchor guidance. The selected pillar shapes how every
# platform teaser opens, so the output commits to one clean format instead of mixing all.
TEASER_PILLARS = {
    "Choose the Best Structure for Me": (
        "Select the structure that best fits the chapter's actual hook and platform. Do not mix every structure at once."
    ),
    "Plot + Consequence": (
        "State what happens and immediately attach what it could cost. Keep it clean, specific, and unresolved."
    ),
    "Question + Threat": (
        "Frame the central mystery through immediate danger: what is unknown, who is exposed, and what pressure is closing in."
    ),
    "Character + Impossible Choice": (
        "Center one person trapped between two terrible outcomes. Make the choice feel urgent without revealing the decision."
    ),
    "Clue + Contradiction": (
        "Lead with evidence, then show the detail that does not fit. Make the contradiction the reason to keep reading."
    ),
    "Atmosphere + Intrusion": (
        "Establish the chapter's world, then let one wrong thing enter it: a sound, object, person, threat, or image."
    ),
    "Quote + Sting": (
        "Build around one unforgettable line, then cut to a final image, question, or threat before the answer arrives."
    ),
    "Satire + Slasher Reversal": (
        "Start with vanity, denial, branding, or institutional absurdity; end with danger sharp enough to draw blood."
    ),
    "In Which… (Literary/Serialized)": (
        "Frame the teaser as a wry, serialized chapter blurb built around an \"In which …\" "
        "line — e.g. \"In which a favor becomes a debt no one can pay.\" Understated, "
        "literary, a little knowing."
    ),
    "One-Sentence Plot + Consequence": (
        "Anchor on a single sentence that states the chapter's central action and the "
        "consequence looming over it — the action and its price, nothing more."
    ),
    "Juicy Quote / Loaded Line": (
        "Lead with one loaded line of dialogue or prose taken EXACTLY from the chapter text, "
        "in quotation marks, chosen because it implies far more than it states. Build a few "
        "lines of context around it without explaining it."
    ),
    "Emotional Pivot": (
        "Center the teaser on the chapter's emotional turn — the moment a character's feeling "
        "shifts, breaks, or hardens. Lead with interior stakes, not plot mechanics."
    ),
    "Quick Formula: Character + Conflict + Consequence": (
        "Name the character, the conflict they face, and what is at stake if they fail — in "
        "that order, tight and concrete."
    ),
}
TEASER_PILLAR_OPTIONS = [
    "Choose the Best Structure for Me",
    "Plot + Consequence",
    "Question + Threat",
    "Character + Impossible Choice",
    "Clue + Contradiction",
    "Atmosphere + Intrusion",
    "Quote + Sting",
    "Satire + Slasher Reversal",
]


# Sensible default genre per book (stand-in for a per-novel genre config). Users can override.
DEFAULT_GENRE_BY_BOOK = {
    "Mortal Vengeance": "Horror",
    "Mortal Vengeance: A Grim Tale": "Horror",
    "Mortal Vengeance II: To Reel or Not Too Real?": "Mystery / Thriller / Suspense",
}


def genre_tone(genre: str) -> str:
    return GENRE_TONES.get(genre, GENRE_TONES["Mystery / Thriller / Suspense"])


def pillar_guidance(pillar: str) -> str:
    return TEASER_PILLARS.get(pillar, TEASER_PILLARS["Plot + Consequence"])


def default_genre_for_book(book: str) -> str:
    return DEFAULT_GENRE_BY_BOOK.get(book, "Mystery / Thriller / Suspense")


def style_variant_options() -> dict[str, str]:
    return dict(STYLE_VARIANTS["chapter_promos"])


def style_variant_instruction(variant: str) -> str:
    return STYLE_VARIANT_INSTRUCTIONS.get(variant, STYLE_VARIANT_INSTRUCTIONS["spoiler_safe_teaser"])


def campaign_mode_instruction(mode: str) -> str:
    return CHAPTER_PROMO_MODE_INSTRUCTIONS.get(mode, CHAPTER_PROMO_MODE_INSTRUCTIONS["Tease It"])


def promotional_mood_instruction(mood: str) -> str:
    return PROMOTIONAL_MOOD_INSTRUCTIONS.get(mood, mood or "Match the chapter mood.")


def promo_output_options() -> dict[str, str]:
    return dict(PROMO_OUTPUT_FORMATS)


def genre_promo_mode_instruction(mode: str) -> str:
    return GENRE_PROMO_MODES.get(mode, GENRE_PROMO_MODES["Match the Chapter Automatically"])


def promo_duration_instruction(duration: str) -> str:
    return PROMO_DURATION_GUIDANCE.get(duration, "Match timing to the requested outputs and platform.")


def promo_output_labels(selected) -> list[str]:
    keys = [str(item).strip() for item in (selected or []) if str(item).strip()]
    if not keys:
        keys = DEFAULT_PROMO_OUTPUTS
    return [PROMO_OUTPUT_FORMATS.get(key, key) for key in keys]


def _format_requested_outputs(selected) -> str:
    labels = promo_output_labels(selected)
    return "\n".join(f"- {label}" for label in labels)


def normalize_promo_outputs(selected, promo_duration: str = "") -> list[str]:
    keys = [str(item).strip() for item in (selected or []) if str(item).strip()]
    if not keys:
        keys = list(DEFAULT_PROMO_OUTPUTS)
    duration_output = {
        "15-Second Social Sting": "social_15",
        "30-Second Chapter Promo": "tv_30",
        "60-Second Prestige Preview": "prestige_60",
        "15-second social promo": "social_15",
        "30-second standard promo": "tv_30",
        "60-second prestige promo": "prestige_60",
    }.get(str(promo_duration or "").strip())
    if duration_output and duration_output not in keys:
        keys.append(duration_output)
    if str(promo_duration or "").strip() == "Written Campaign Only":
        keys = [key for key in keys if key not in {"social_15", "tv_30", "prestige_60"}]
    return keys


def _system_preamble(novel_title: str, genre: str) -> str:
    return (
        f'You are a chapter teaser and promotional-copy specialist for the novel "{novel_title}".\n'
        f"GENRE CONTEXT: {genre}\n\n"
        "\"Tell Tales Ink\" is the internal name of the tool generating this content, not a publisher "
        "or narrator. Never write \"Tell Tales Ink\" into the generated copy itself.\n\n"
        "TEASER BEST PRACTICES — ALWAYS FOLLOW:\n"
        "- Write in present tense (\"Elena discovers\", not \"Elena discovered\").\n"
        "- Never use clichés such as \"Little did they know\", \"Nothing would ever be the same\", "
        "\"Secrets will be revealed\", or \"A shocking twist awaits\".\n"
        "- Never explain the resolution. Stop before the answer — create an information gap, do not close it.\n"
        "- Stay SPOILER-FREE: do not reveal the chapter's ending, major twists, or the book's big reveals.\n"
        "- Match mood to the genre exactly, and focus on stakes — what could be gained, lost, exposed, or destroyed.\n"
        "- Ground every reference in the provided chapter text. Do not invent plot, characters, or events. "
        "Use exact wording only when quoting directly from the chapter.\n"
        "- A summary gives information. A teaser creates a question. A promo creates an event.\n\n"
        f"GENRE-SPECIFIC TONE:\n{genre_tone(genre)}\n"
    )


def build_summary_prompt(novel_title: str, chapter_label: str, chapter_text: str, genre: str) -> str:
    """Objective SWBST chapter summary (used standalone and as teaser grounding)."""
    return (
        _system_preamble(novel_title, genre)
        + f'\nWrite a chapter summary for "{chapter_label}" of the novel "{novel_title}".\n\n'
        "CORE SUMMARY QUESTIONS:\n"
        "- Main Goal: what does the central character want at the beginning?\n"
        "- Key Conflict: what obstacle, confrontation, or complication blocks it?\n"
        "- Turning Point: what discovery, decision, or event changes the direction?\n"
        "- Resolution / Cliffhanger: where does the chapter leave the characters?\n"
        "- Crucial Information: what clues, relationships, secrets, world-building, or mythology are introduced?\n\n"
        "For complex genre fiction, track plot momentum, character development, historical/thematic material, "
        "mystery clues and red herrings, suspect status, alliances/betrayals, injuries/disappearances/deaths, "
        "and contribution to the series mythology when they appear.\n\n"
        "OUTPUT FORMAT (markdown):\n"
        "### Chapter Summary\n"
        "[100-250 words. Follow chronology, use active language, and prioritize cause and effect.]\n\n"
        "### Four-Point Record\n"
        "- Plot Beat: [what mechanically advances the story]\n"
        "- Atmosphere and Genre Anchor: [dominant horror/thriller/Gothic/historical/academic/satirical elements]\n"
        "- Clues, Secrets, and Red Herrings: [evidence, contradictions, suspect movement]\n"
        "- Character and Casualty Status: [who changes, allies/betrays, becomes isolated, is injured, disappears, or dies]\n\n"
        "### Continuity Notes\n"
        "[dates, objects, injuries, unanswered questions, survivor status, body count, mythology notes — only if present]\n\n"
        "RULES: stay objective (no personal commentary), present tense throughout, focus on decisions, "
        "consequences, and what changes (not description), 150–250 words total. Keep the book's major "
        "twists and ending unspoiled; chapter-level beats are fine.\n\n"
        f'CHAPTER TEXT:\n"""\n{chapter_text}\n"""\n\n'
        "Output only the summary in the format above. No preamble or meta-commentary."
    )


_PLATFORM_STRUCTURES = {
    "Instagram": (
        "Write an Instagram teaser caption.\n"
        "STRUCTURE:\n"
        "1. Opening hook (1–2 lines max) using the teaser engine above.\n"
        "2. 2–4 lines of atmospheric build — present tense, no plot recap.\n"
        "3. One line that drops off right before the answer.\n"
        "4. A direct CTA, e.g. \"Read this chapter now.\" or a \"Would you …?\" question.\n"
        "5. 6–8 targeted hashtags on a new line.\n"
        "TARGET LENGTH: 250–350 characters for the hook/body; hashtags separate."
    ),
    "TikTok": (
        "Write TikTok / Reels / Shorts promo copy.\n"
        "STRUCTURE:\n"
        "### Hook Text\n"
        "A first-frame line under 12 words.\n"
        "### Voiceover\n"
        "15-45 spoken words depending on the selected promo length. Use one hook, one escalation, one sting.\n"
        "### Visual Beats\n"
        "3-5 quick shots or text beats using concrete chapter details.\n"
        "### Caption\n"
        "Short caption with CTA and 4-6 tags. No plot recap."
    ),
    "Blog": (
        "Write a blog chapter teaser.\n"
        "STRUCTURE:\n"
        "- Opening hook (1 short paragraph, max 3 sentences) using the teaser pillar above.\n"
        "- 2–3 short paragraphs building atmosphere and stakes — no full plot summary.\n"
        "- End on an unresolved tension or unanswered question.\n"
        "- One CTA sentence (e.g. \"Read this chapter here.\").\n"
        "TARGET LENGTH: 150–250 words. It should read like a controlled leak — just enough to make "
        "the reader feel they need the chapter."
    ),
    "Blog or Website": (
        "Write a website-ready chapter teaser.\n"
        "STRUCTURE:\n"
        "- A headline that sells the chapter's hook.\n"
        "- A 50-150-word teaser that builds stakes without resolving them.\n"
        "- A short CTA line.\n"
        "- Optional title-card or feature-image direction if requested."
    ),
    "Newsletter": (
        "Write a newsletter chapter promo.\n"
        "STRUCTURE:\n"
        "### Subject Line Options\n"
        "Three subject lines, each with a different angle.\n"
        "### Preview Text\n"
        "One line under 90 characters.\n"
        "### Newsletter Blurb\n"
        "80-150 words: hook, selective context, stakes, unresolved final beat, CTA."
    ),
    "LinkedIn": (
        "Write a LinkedIn chapter teaser. LinkedIn readers respond to stakes, emotional truth, and human "
        "cost — even in fiction. Frame the teaser around the character's internal conflict or moral dilemma.\n"
        "STRUCTURE:\n"
        "- A 1-line hook that speaks to a universal emotional experience.\n"
        "- 2–3 short paragraphs: what the character faces, what is at stake, why it matters.\n"
        "- End on an unresolved question or decision point.\n"
        "- Optional: 1 sentence connecting the theme to something readers recognize in their own lives.\n"
        "- A CTA, then 4–6 hashtags.\n"
        "TARGET LENGTH: 200–300 words. Grounded and a little more reflective than Instagram."
    ),
    "YouTube": (
        "Write YouTube promotional content as three clearly separated markdown sections:\n"
        "### Title\n"
        "One compelling title under 70 characters — tension, curiosity, or escalation; no clickbait.\n"
        "### Description\n"
        "3–4 short paragraphs: (1) chapter hook using the teaser pillar; (2) atmosphere and stakes, no "
        "spoilers or resolution; (3) CTA with a link placeholder and a subscribe line.\n"
        "### Short Post\n"
        "4–6 punchy lines for the Community tab / Shorts caption. Staccato rhythm. End on a line that "
        "makes the viewer stop scrolling."
    ),
    "Multi-Platform Campaign": (
        "Write a compact cross-platform chapter campaign.\n"
        "STRUCTURE:\n"
        "### Campaign Hook\n"
        "One sentence that captures the chapter's central promise.\n"
        "### Instagram / Threads\n"
        "Short social caption with hashtags.\n"
        "### TikTok / Reels\n"
        "First-frame hook, short voiceover, and visual beats.\n"
        "### Newsletter\n"
        "Subject line, preview text, and 80-120-word blurb.\n"
        "### YouTube\n"
        "Title, description opener, and Shorts caption.\n"
        "### Taglines\n"
        "Three options with distinct personalities."
    ),
}


def build_teaser_prompt(
    platform: str,
    novel_title: str,
    chapter_label: str,
    chapter_text: str,
    summary: str,
    genre: str,
    teaser_pillar: str,
    spoiler_level: str = "Spoiler-free",
    promo_objective: str = "",
    tone: str = "",
    cta: str = "",
    optional_quote: str = "",
    style_variant: str = "spoiler_safe_teaser",
    primary_hook: str = "Choose the Strongest Hook for Me",
    promo_outputs=None,
    campaign_mode: str = "Tease It",
    featured_character: str = "",
    promo_duration: str = "",
    genre_promo_mode: str = "",
    visual_motif: str = "",
    sound_motif: str = "",
    music_style: str = "",
    editing_rhythm: str = "",
    color_lighting: str = "",
    final_tagline: str = "",
    spoiler_notes: str = "",
) -> str:
    """Dedicated, platform-specific teaser prompt grounded in the real chapter text."""
    structure = _PLATFORM_STRUCTURES.get(platform, _PLATFORM_STRUCTURES["Instagram"])
    quote_instruction = (
        f'\nOPTIONAL USER-PROVIDED QUOTE / EXCERPT:\n"{optional_quote.strip()}"\n'
        "Use this as an anchor only if it fits the requested spoiler level. Preserve exact wording if quoted.\n"
        if optional_quote and optional_quote.strip()
        else ""
    )
    return (
        _system_preamble(novel_title, genre)
        + f"\nTEASER ENGINE: {teaser_pillar}\n{pillar_guidance(teaser_pillar)}\n\n"
        + f"CAMPAIGN MODE: {campaign_mode or 'Tease It'}\n"
        + f"{campaign_mode_instruction(campaign_mode or 'Tease It')}\n\n"
        + f"WHAT SHOULD SELL THIS CHAPTER: {primary_hook or 'Choose the Strongest Hook for Me'}\n"
        + "If auto-selecting, choose the chapter's most compelling threat, discovery, confrontation, secret, decision, reversal, emotional rupture, atmospheric event, or body-count question. Choose the hook that best represents the chapter, not merely the loudest scene.\n\n"
        + f"CREATIVE DIRECTION: {genre_promo_mode or 'Match the Chapter Automatically'}\n"
        + f"{genre_promo_mode_instruction(genre_promo_mode or 'Match the Chapter Automatically')}\n\n"
        + f"PROMO DURATION: {promo_duration or 'Auto-match requested outputs'}\n"
        + f"{promo_duration_instruction(promo_duration or 'Auto-match requested outputs')}\n\n"
        + f"STYLE VARIANT: {style_variant_options().get(style_variant, style_variant)}\n"
        + f"{style_variant_instruction(style_variant)}\n\n"
        + f"SPOILER LEVEL: {spoiler_level or 'Spoiler-free'}\n"
        + f"PROMO OBJECTIVE: {promo_objective or 'Drive chapter reads without spoiling the chapter.'}\n"
        + f"PROMOTIONAL MOOD: {tone or genre_tone(genre)}\n"
        + f"{promotional_mood_instruction(tone)}\n"
        + f"CTA: {cta or 'Read this chapter now.'}\n"
        + f"FEATURED CHARACTER: {featured_character or 'Auto-select from the chapter.'}\n"
        + f"FINAL TAGLINE: {final_tagline or 'Create one if useful; keep it short and unresolved.'}\n"
        + "CREATIVE DIRECTION:\n"
        + f"- Visual motif: {visual_motif or 'Use a concrete recurring motif from the chapter.'}\n"
        + f"- Sound motif: {sound_motif or 'Use restrained sound design tied to the scene.'}\n"
        + f"- Music style: {music_style or 'Match the chapter mood; do not overstate the genre.'}\n"
        + f"- Editing rhythm: {editing_rhythm or 'Escalate cleanly from context to complication to sting.'}\n"
        + f"- Color and lighting: {color_lighting or 'Reflect the actual setting, mood, and platform.'}\n"
        + (f"SPOILER NOTES / LIMITS:\n{spoiler_notes.strip()}\n" if spoiler_notes and spoiler_notes.strip() else "")
        + quote_instruction
        + f'TASK: {structure}\n\n'
        + f'This teaser is for "{chapter_label}" of the novel "{novel_title}".\n\n'
        + "INTERNAL GROUNDING (chapter SWBST summary — for accuracy; do not quote it verbatim):\n"
        + f"{summary}\n\n"
        + f'CHAPTER TEXT (the source of truth — base everything on this):\n"""\n{chapter_text}\n"""\n\n'
        + "SPOILER-CONTROL SYSTEM:\n"
        + "- Safe: appears near the beginning or is already known. Can be shown directly.\n"
        + "- Suggestive: can be implied without revealing the outcome. Ideal promo material.\n"
        + "- Sensitive: changes how the chapter is understood. Use fragments only.\n"
        + "- Forbidden: major death, killer identity, decisive betrayal, final revelation, or the chapter's answer. Do not include.\n"
        + "A strong promo may show the setup of danger without confirming the outcome, present someone as suspicious without declaring guilt, and reveal that evidence exists without explaining what it proves.\n\n"
        + "PROMO CRAFT RULES:\n"
        + "- Focus on one central hook and escalate from context to complication to unresolved final beat.\n"
        + "- Use specific details: object, place, threat, line, sound, weather, institutional detail, or motif.\n"
        + "- End at maximum uncertainty, before the answer, attack, decision, identity reveal, or consequence.\n"
        + "- Do not use generic claims like \"everything changes\" or \"secrets will be revealed\".\n"
        + "- Dialogue must make sense alone, reveal conflict, sound like the character, create curiosity, and conceal more than it explains.\n"
        + "- Visual direction should use a few recurring motifs, not a random aesthetic montage. Sound direction should use restraint.\n\n"
        + "OUTPUT REQUIREMENTS:\n"
        + _format_requested_outputs(promo_outputs)
        + "\n\n"
        + "QUALITY CHECK BEFORE FINALIZING:\n"
        + "- Represents the actual chapter.\n"
        + "- Establishes character, conflict, or atmosphere immediately.\n"
        + "- Escalates instead of listing events.\n"
        + "- Preserves major revelations.\n"
        + "- Reflects the correct genre balance.\n"
        + "- Ends on the strongest unresolved beat.\n"
        + "- Makes the audience want the chapter, not feel as though it has already read it.\n\n"
        + "Output only the final copy for this platform. No meta-commentary or labels beyond any "
        + "section headers the structure requires."
    )
