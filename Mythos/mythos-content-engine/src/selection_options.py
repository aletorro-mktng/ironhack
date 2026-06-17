QUOTE_MOOD_TAGS = [
    "funny",
    "snarky",
    "sad",
    "tragic",
    "scary",
    "brutal",
    "roast",
    "savage",
    "romantic",
    "emotional",
    "dramatic",
    "iconic",
    "villainous",
    "grief",
    "revenge",
    "justice",
    "friendship",
    "betrayal",
    "survival",
    "media-satire",
    "Dominican-folklore",
    "Dominican folklore",
    "Grim-Cojuelo",
    "LGBT",
    "Toxic Masculinity",
    "Class",
    "Guilt",
    "Mental Health",
    "Bullying",
    "Religious guilt",
    "Institutional cruelty",
    "Media rot",
    "Campy menace",
    "Dark academia shame",
    "Grief masked as jokes",
    "Horror",
]

QUOTE_BOOK_OPTIONS = [
    "All books",
    "Mortal Vengeance",
    "Mortal Vengeance: A Grim Tale",
]

CHARACTER_TAGS = [
    "Alex Herrera",
    "Melissa Rocha",
    "Mónika Torres",
    "Mario Stinga",
    "Manuel Freites",
    "Fernando Pepino",
    "Enrique Hartling",
    "María García",
    "Julián Díaz",
    "Lucía Salgado",
    "Marcos",
    "Profesora Lourdes",
    "Lieutenant Ricardo García",
    "The Grim Cojuelo",
    "Doña Silvia",
    "Padre Ángel",
    "Sister María Gracia",
    "Padre Ignacio",
    "Don Ramón",
    "Doña Patricia",
    "Elías Sarraff",
    "Other / unnamed character",
    "Doctora Mateo",
    "Principal Davis Beltrán",
    "Humberto",
    "Doña Laura",
]

AUDIENCE_OPTIONS = [
    "potential readers",
    "new readers",
    "existing fans",
    "BookTok readers",
    "Bookstagram readers",
    "influencers",
    "journalists",
    "reviewers",
    "ARC readers",
    "YA thriller readers",
    "horror readers",
    "dark academia readers",
    "general audience",
]

# Audience split into sub-groups for faster scanning.
AUDIENCE_PLATFORM_OPTIONS = [
    "BookTok readers",
    "Bookstagram readers",
    "influencers",
]

AUDIENCE_ROLE_OPTIONS = [
    "journalists",
    "reviewers",
    "ARC readers",
]

AUDIENCE_READER_OPTIONS = [
    "potential readers",
    "new readers",
    "existing fans",
    "YA thriller readers",
    "horror readers",
    "dark academia readers",
    "general audience",
]

SOCIAL_OBJECTIVES = [
    "reach",
    "engagement",
    "comments",
    "shares",
    "saves",
    "tags",
    "follows",
    "clicks",
    "profile visits",
    "newsletter signups",
    "ARC signups",
    "preorders",
    "book sales",
    "reviews",
    "UGC / fan responses",
]

CONSTRAINT_OPTIONS = [
    "spoiler-free",
    "use only quote bank",
    "use only real reviews",
    "avoid major spoilers",
    "make it short",
    "make it punchy",
    "make it cinematic",
    "make it savage",
    "make it funny",
    "make it emotionally intense",
    "include CTA",
    "include hashtags",
    "no hashtags",
]

# Constraints split into two scannable groups.
CONSTRAINT_ACCURACY_OPTIONS = [
    "spoiler-free",
    "avoid major spoilers",
    "use only quote bank",
    "use only real reviews",
    "do not mention awards unless verified",
]

CONSTRAINT_TONE_OPTIONS = [
    "make it short",
    "make it punchy",
    "make it cinematic",
    "make it savage",
    "make it funny",
    "make it emotionally intense",
    "include CTA",
    "include hashtags",
    "no hashtags",
]

# Pairs that pull in opposite directions — surfaced as a soft warning.
CONSTRAINT_CONFLICT_PAIRS = [
    ("include hashtags", "no hashtags"),
    ("make it short", "make it cinematic"),
    ("make it funny", "make it emotionally intense"),
    ("make it funny", "make it savage"),
]

PLATFORM_OPTIONS = [
    "Instagram",
    "Bookstagram",
    "Goodreads",
    "YouTube",
    "LinkedIn",
    "newsletter",
    "blog",
    "press kit",
    "landing page",
]

PODCAST_DESTINATION_OPTIONS = [
    "Spotify",
    "Apple Podcasts",
    "YouTube",
    "YouTube Music",
    "Amazon Music",
    "Audible",
    "RSS feed",
    "download",
    "press kit",
    "website embed",
    "other",
]

PODCAST_FORMAT_OPTIONS = [
    "one-person monologue",
    "two-person interview",
    "roundtable discussion",
    "critical essay / critique",
    "news-style segment",
    "behind-the-scenes author commentary",
    "character or lore deep dive",
    "review and analysis",
]

PODCAST_TONE_OPTIONS = [
    "cinematic",
    "investigative",
    "conversational",
    "darkly funny",
    "literary",
    "critical",
    "intimate",
    "dramatic",
    "educational",
    "press-friendly",
]

PODCAST_LENGTH_OPTIONS = [
    "short: 3-5 minutes",
    "standard: 8-12 minutes",
    "deep dive: 20-30 minutes",
    "long-form: 45-60 minutes",
]

ELEVENLABS_MODEL_OPTIONS = [
    "eleven_v3",
    "eleven_multilingual_v2",
    "eleven_flash_v2_5",
]

PRESS_RELEASE_TIMING_OPTIONS = [
    "FOR IMMEDIATE RELEASE",
    "EMBARGOED UNTIL (set embargo date below)",
]

YOUTUBE_CONTENT_FORMAT_OPTIONS = [
    "Long-form video",
    "Short",
    "Community post",
    "Premiere",
    "Title",
    "Description",
    "Pinned comment",
]

LINKEDIN_CONTENT_FORMAT_OPTIONS = [
    "Standard post",
    "Article",
    "Document / carousel",
    "Poll",
    "Newsletter",
]

# Email-appropriate objectives for newsletter_blurb (UX-10).
NEWSLETTER_OBJECTIVE_OPTIONS = [
    "open rate",
    "click-through",
    "reply",
    "forwards",
    "conversions",
    "list growth",
    "preorders",
    "reviews",
]

NEWSLETTER_STRUCTURE_OPTIONS = [
    "announcement",
    "roundup / digest",
    "personal note",
    "behind-the-scenes",
]

INSTAGRAM_FORMAT_OPTIONS = [
    "Feed post",
    "Reel",
    "Story",
    "Carousel",
]

INSTAGRAM_HASHTAG_OPTIONS = [
    "no hashtags",
    "3-5 niche hashtags",
    "10-15 mixed hashtags",
    "up to 30 for max reach",
]

BLOG_LENGTH_OPTIONS = [
    "short: 300-500 words",
    "standard: 600-900 words",
    "long: 1200-1800 words",
    "deep dive: 2000+ words",
]

PRESS_RELEASE_DESTINATION_OPTIONS = [
    "newswire / PR distribution",
    "press kit",
    "media email list",
    "company newsroom / blog",
    "journalist outreach",
    "landing page",
    "other",
]
