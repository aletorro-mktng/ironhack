"""Genre-aware chapter teaser + summary prompt builders for the Chapter Promos feature.

The Chapter Promos panel reads a real chapter from a manuscript and produces either a
SWBST chapter summary or platform teasers. These builders encode teaser craft (present
tense, no clichés, unresolved endings, genre-matched tone) and a selectable structural
"pillar", and they keep every output grounded in the actual chapter text passed in.
"""

from __future__ import annotations


# Mode of the panel: tight platform teasers vs. an objective chapter summary.
CHAPTER_PROMO_MODES = ["Teaser / Promo", "Chapter Summary"]


# Genre -> tone instruction injected into the system preamble of every call.
GENRE_TONES = {
    "Mystery / Thriller / Suspense": (
        "Cryptic, tense, urgent, atmospheric. Raise questions instead of giving answers. "
        "Use secrets, investigations, hidden motives, and unreliable details."
    ),
    "Horror": (
        "Visceral, eerie, threatening. Create dread, a shiver, or an impulse to look away. "
        "Use isolation, wrongness, and physical menace."
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
TEASER_PILLAR_OPTIONS = list(TEASER_PILLARS.keys())


# Sensible default genre per book (stand-in for a per-novel genre config). Users can override.
DEFAULT_GENRE_BY_BOOK = {
    "Mortal Vengeance": "Horror",
    "Mortal Vengeance: A Grim Tale": "Horror",
    "Mortal Vengeance II: To Reel or Not Too Real?": "Mystery / Thriller / Suspense",
}


def genre_tone(genre: str) -> str:
    return GENRE_TONES.get(genre, GENRE_TONES["Mystery / Thriller / Suspense"])


def pillar_guidance(pillar: str) -> str:
    return TEASER_PILLARS.get(pillar, TEASER_PILLARS["One-Sentence Plot + Consequence"])


def default_genre_for_book(book: str) -> str:
    return DEFAULT_GENRE_BY_BOOK.get(book, "Mystery / Thriller / Suspense")


def _system_preamble(novel_title: str, genre: str) -> str:
    return (
        f'You are a chapter teaser and promotional-copy specialist for the novel "{novel_title}".\n'
        f"GENRE CONTEXT: {genre}\n\n"
        "TEASER BEST PRACTICES — ALWAYS FOLLOW:\n"
        "- Write in present tense (\"Elena discovers\", not \"Elena discovered\").\n"
        "- Never use clichés such as \"Little did they know\", \"Nothing would ever be the same\", "
        "\"Secrets will be revealed\", or \"A shocking twist awaits\".\n"
        "- Never explain the resolution. Stop before the answer — create an information gap, do not close it.\n"
        "- Stay SPOILER-FREE: do not reveal the chapter's ending, major twists, or the book's big reveals.\n"
        "- Match mood to the genre exactly, and focus on stakes — what could be gained, lost, exposed, or destroyed.\n"
        "- Ground every reference in the provided chapter text. Do not invent plot, characters, or events. "
        "Use exact wording only when quoting directly from the chapter.\n\n"
        f"GENRE-SPECIFIC TONE:\n{genre_tone(genre)}\n"
    )


def build_summary_prompt(novel_title: str, chapter_label: str, chapter_text: str, genre: str) -> str:
    """Objective SWBST chapter summary (used standalone and as teaser grounding)."""
    return (
        _system_preamble(novel_title, genre)
        + f'\nWrite a chapter summary for "{chapter_label}" of the novel "{novel_title}".\n\n'
        "Use the SWBST framework:\n"
        "- SOMEBODY: who drives this chapter?\n"
        "- WANTED: what does that character want or need?\n"
        "- BUT: what obstacle, complication, or conflict gets in the way?\n"
        "- SO: how does the character respond or adapt?\n"
        "- THEN: what consequence or shift pushes the story into the next chapter?\n\n"
        "OUTPUT FORMAT (markdown):\n"
        "### Chapter Summary\n"
        "[1–2 paragraph SWBST narrative — objective, present tense, cause-and-effect focused]\n\n"
        "### Key Characters\n"
        "[bullet list — one line each on each character's role in this chapter]\n\n"
        "### Narrative Movement\n"
        "[one sentence: what changed by the end of this chapter — emotionally, plot-wise, or relationally]\n\n"
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
        "1. Opening hook (1–2 lines max) using the teaser pillar above.\n"
        "2. 2–4 lines of atmospheric build — present tense, no plot recap.\n"
        "3. One line that drops off right before the answer.\n"
        "4. A direct CTA, e.g. \"Read this chapter now.\" or a \"Would you …?\" question.\n"
        "5. 6–8 targeted hashtags on a new line.\n"
        "TARGET LENGTH: 250–350 characters for the hook/body; hashtags separate."
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
}


def build_teaser_prompt(
    platform: str,
    novel_title: str,
    chapter_label: str,
    chapter_text: str,
    summary: str,
    genre: str,
    teaser_pillar: str,
) -> str:
    """Dedicated, platform-specific teaser prompt grounded in the real chapter text."""
    structure = _PLATFORM_STRUCTURES.get(platform, _PLATFORM_STRUCTURES["Instagram"])
    return (
        _system_preamble(novel_title, genre)
        + f"\nPRIMARY TEASER PILLAR: {teaser_pillar}\n{pillar_guidance(teaser_pillar)}\n\n"
        + f'TASK: {structure}\n\n'
        + f'This teaser is for "{chapter_label}" of the novel "{novel_title}".\n\n'
        + "INTERNAL GROUNDING (chapter SWBST summary — for accuracy; do not quote it verbatim):\n"
        + f"{summary}\n\n"
        + f'CHAPTER TEXT (the source of truth — base everything on this):\n"""\n{chapter_text}\n"""\n\n'
        + "Output only the final copy for this platform. No meta-commentary or labels beyond any "
        + "section headers the structure requires."
    )
