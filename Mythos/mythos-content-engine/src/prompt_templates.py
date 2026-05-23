from pathlib import Path


TEMPLATE_FOLDER = Path("templates")


SUPPORTED_CONTENT_TYPES = {
    "instagram_caption": "instagram_caption.md",
    "tiktok_post": "tiktok_post.md",
    "blog_post": "blog_post.md",
    "newsletter_blurb": "newsletter_blurb.md",
    "character_spotlight": "character_spotlight.md",
    "review_pull_quote": "review_pull_quote.md",
}


def list_supported_content_types():
    """
    Return all supported content type keys.
    """
    return list(SUPPORTED_CONTENT_TYPES.keys())


def load_template(content_type):
    """
    Load a prompt template by content type.
    """
    if content_type not in SUPPORTED_CONTENT_TYPES:
        supported = ", ".join(SUPPORTED_CONTENT_TYPES.keys())
        raise ValueError(
            f"Unsupported content type: {content_type}. "
            f"Supported types are: {supported}"
        )

    template_file = TEMPLATE_FOLDER / SUPPORTED_CONTENT_TYPES[content_type]

    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")

    return template_file.read_text(encoding="utf-8")


def build_prompt(template_text, topic, knowledge_context):
    """
    Combine template, user topic, and knowledge base context into one complete prompt.
    """
    return template_text.format(
        topic=topic,
        knowledge_context=knowledge_context
    )


if __name__ == "__main__":
    print("Supported content types:")

    for content_type in list_supported_content_types():
        print(f"- {content_type}")

    test_template = load_template("instagram_caption")

    print("\nInstagram template preview:")
    print(test_template[:500])