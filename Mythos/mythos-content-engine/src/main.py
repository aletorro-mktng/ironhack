from prompt_templates import list_supported_content_types
from content_pipeline import run_pipeline
from selection_options import (
    QUOTE_MOOD_TAGS,
    QUOTE_BOOK_OPTIONS,
    CHARACTER_TAGS,
    AUDIENCE_OPTIONS,
    SOCIAL_OBJECTIVES,
    CONSTRAINT_OPTIONS,
    PLATFORM_OPTIONS,
    PODCAST_DESTINATION_OPTIONS,
    PODCAST_FORMAT_OPTIONS,
    PODCAST_TONE_OPTIONS,
    PODCAST_LENGTH_OPTIONS,
    ELEVENLABS_MODEL_OPTIONS,
)


def choose_one(options, label):
    """
    Let the user choose one option or add a custom one.
    """
    print(f"\n{label}:\n")

    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")

    print("0. Other / add custom")

    while True:
        choice = input(f"\nSelect {label.lower()} by number: ").strip()

        if choice == "0":
            custom = input(f"Type custom {label.lower()}: ").strip()
            return custom if custom else "Not specified"

        if choice.isdigit():
            choice_index = int(choice) - 1

            if 0 <= choice_index < len(options):
                return options[choice_index]

        print("Invalid selection. Please choose a valid number.")


def choose_many(options, label):
    """
    Let the user choose multiple options by number.
    Select 0 to add a custom option.
    Example input: 1, 3, 7
    """
    print(f"\n{label}:\n")

    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")

    print("0. Other / add custom")

    raw_choices = input(
        f"\nSelect one or more {label.lower()} by number, separated by commas:\n> "
    ).strip()

    selected = []

    if raw_choices:
        for item in raw_choices.split(","):
            item = item.strip()

            if item == "0":
                custom = input(f"Type custom {label.lower()}: ").strip()
                if custom:
                    selected.append(custom)

            elif item.isdigit():
                choice_index = int(item) - 1

                if 0 <= choice_index < len(options):
                    selected.append(options[choice_index])

    if not selected:
        return "Not specified"

    return ", ".join(dict.fromkeys(selected))


def display_content_types():
    """
    Show available content types to the user.
    """
    content_types = list_supported_content_types()

    print("\nAvailable content types:\n")

    for index, content_type in enumerate(content_types, start=1):
        print(f"{index}. {content_type}")

    return content_types


def get_user_content_type(content_types):
    """
    Ask the user to select a content type by number.
    """
    while True:
        choice = input("\nSelect a content type by number: ").strip()

        if choice.isdigit():
            choice_index = int(choice) - 1

            if 0 <= choice_index < len(content_types):
                return content_types[choice_index]

        print("Invalid selection. Please choose a valid number.")


def collect_brief(content_type):
    """
    Collect a structured social-media brief from the user.
    """
    topic = input("\nWhat should this content be about?\n> ").strip()

    related_book = choose_one(
        QUOTE_BOOK_OPTIONS,
        "Related book/source (optional)"
    )

    destination_options = PODCAST_DESTINATION_OPTIONS if content_type == "podcast" else PLATFORM_OPTIONS
    platform_label = "Podcast destination" if content_type == "podcast" else "Platform"

    platform = choose_one(
        destination_options,
        platform_label
    )

    social_objectives = choose_many(
        SOCIAL_OBJECTIVES,
        "Social objectives"
    )

    audience = choose_many(
        AUDIENCE_OPTIONS,
        "Audience"
    )

    cta = input("\nDesired CTA or next step?\n> ").strip()

    constraints = choose_many(
        CONSTRAINT_OPTIONS,
        "Constraints"
    )

    quote_book = "Not applicable"
    mood_tags = "Not applicable"
    character_tags = "Not applicable"
    podcast_format = "Not applicable"
    podcast_speakers = "Not applicable"
    podcast_speaker_roles = "Not applicable"
    podcast_tone = "Not applicable"
    podcast_length = "Not applicable"
    elevenlabs_model = "Not applicable"
    elevenlabs_voice_ids = "Not applicable"

    if content_type == "quote_post":
        quote_book = choose_one(
            QUOTE_BOOK_OPTIONS,
            "Quote book/source"
        )

        mood_tags = choose_many(
            QUOTE_MOOD_TAGS,
            "Quote mood/category tags"
        )

        character_tags = choose_many(
            CHARACTER_TAGS,
            "Character tags"
        )

    if content_type == "podcast":
        podcast_format = choose_one(
            PODCAST_FORMAT_OPTIONS,
            "Podcast format"
        )

        podcast_speakers = input(
            "\nHow many speakers should the podcast have?\n> "
        ).strip() or "Not specified"

        podcast_speaker_roles = input(
            "\nSpeaker roles/names? Example: Host, Author, Critic\n> "
        ).strip() or "Not specified"

        podcast_tone = choose_many(
            PODCAST_TONE_OPTIONS,
            "Podcast tone"
        )

        podcast_length = choose_one(
            PODCAST_LENGTH_OPTIONS,
            "Podcast target length"
        )

        elevenlabs_model = choose_one(
            ELEVENLABS_MODEL_OPTIONS,
            "ElevenLabs model"
        )

        elevenlabs_voice_ids = input(
            "\nElevenLabs voice IDs by speaker? Example: Host=voice_id, Guest=voice_id. Press Enter if undecided.\n> "
        ).strip() or "Not specified"

    brief_parts = [
        f"Topic: {topic or 'Not specified'}",
        f"Related book/source: {related_book}",
        f"Platform: {platform}",
        f"Social objectives: {social_objectives}",
        f"Audience: {audience}",
        f"CTA: {cta or 'Not specified'}",
        f"Constraints: {constraints}",
        f"Requested quote book/source: {quote_book}",
        f"Requested quote mood/category tags: {mood_tags}",
        f"Requested character tags: {character_tags}",
        f"Podcast format: {podcast_format}",
        f"Podcast speaker count: {podcast_speakers}",
        f"Podcast speaker roles/names: {podcast_speaker_roles}",
        f"Podcast tone: {podcast_tone}",
        f"Podcast target length: {podcast_length}",
        f"ElevenLabs model: {elevenlabs_model}",
        f"ElevenLabs voice IDs: {elevenlabs_voice_ids}",
    ]

    return "\n".join(brief_parts)


def main():
    """
    CLI entry point for Mythos Content Engine.
    """
    print("\nMythos Content Engine")
    print("=====================")
    print("Generate brand-specific social content using filtered Mortal Vengeance knowledge.")

    content_types = display_content_types()
    selected_content_type = get_user_content_type(content_types)

    structured_brief = collect_brief(selected_content_type)

    print("\nFiltering relevant context and generating content...\n")

    result = run_pipeline(
        content_type=selected_content_type,
        topic=structured_brief
    )

    print("Content generated successfully.")
    print(f"Filtered context saved to: {result['filtered_context_path']}")
    print(f"Generation prompt saved to: {result['prompt_path']}")
    print(f"Draft saved to: {result['draft_path']}")

    print("\nGenerated content preview:\n")
    print(result["generated_content"])


if __name__ == "__main__":
    main()
