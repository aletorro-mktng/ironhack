from prompt_templates import list_supported_content_types
from content_pipeline import run_pipeline


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


def main():
    """
    CLI entry point for Mythos Content Engine.
    """
    print("\nMythos Content Engine")
    print("=====================")
    print("Generate brand-specific content using the Mortal Vengeance knowledge base.")

    content_types = display_content_types()
    selected_content_type = get_user_content_type(content_types)

    topic = input("\nWhat should this content be about?\n> ").strip()

    if not topic:
        print("No topic provided. Exiting.")
        return

    print("\nGenerating content...\n")

    result = run_pipeline(
        content_type=selected_content_type,
        topic=topic
    )

    print("Content generated successfully.")
    print(f"Prompt saved to: {result['prompt_path']}")
    print(f"Draft saved to: {result['draft_path']}")

    print("\nGenerated content preview:\n")
    print(result["generated_content"])


if __name__ == "__main__":
    main()