from datetime import datetime
from pathlib import Path

from context_filter import select_relevant_context
from prompt_templates import load_template, build_prompt
from llm_integration import generate_text


OUTPUT_DIR = Path("outputs")


def save_output(content: str, content_type: str, label: str = "draft") -> Path:
    """
    Save generated content, prompts, or filtered context to the outputs folder.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{content_type}_{label}_{timestamp}.md"
    output_path = OUTPUT_DIR / filename

    output_path.write_text(content, encoding="utf-8")

    return output_path


def create_generation_prompt(content_type: str, topic: str, filtered_context: str) -> str:
    """
    Build the final generation prompt using only filtered relevant context.
    """
    template_text = load_template(content_type)

    final_prompt = build_prompt(
        template_text=template_text,
        topic=topic,
        knowledge_context=filtered_context
    )

    return final_prompt


def run_pipeline(content_type: str, topic: str) -> dict:
    """
    Run the two-stage content pipeline:

    1. Filter relevant context with an LLM call.
    2. Generate final content with a separate LLM call.
    """
    filtered_context = select_relevant_context(
        content_type=content_type,
        topic=topic
    )

    filtered_context_path = save_output(
        content=filtered_context,
        content_type=content_type,
        label="filtered_context"
    )

    generation_prompt = create_generation_prompt(
        content_type=content_type,
        topic=topic,
        filtered_context=filtered_context
    )

    prompt_path = save_output(
        content=generation_prompt,
        content_type=content_type,
        label="generation_prompt"
    )

    generated_content = generate_text(generation_prompt)

    draft_path = save_output(
        content=generated_content,
        content_type=content_type,
        label="draft"
    )

    return {
        "filtered_context_path": filtered_context_path,
        "prompt_path": prompt_path,
        "draft_path": draft_path,
        "generated_content": generated_content
    }


if __name__ == "__main__":
    test_content_type = "instagram_caption"
    test_topic = "Create a launch caption for Mortal Vengeance II: To Reel or Not Too Real?"

    result = run_pipeline(test_content_type, test_topic)

    print("Two-stage pipeline ran successfully.")
    print(f"Filtered context saved to: {result['filtered_context_path']}")
    print(f"Generation prompt saved to: {result['prompt_path']}")
    print(f"Draft saved to: {result['draft_path']}")

    print("\nGenerated content preview:\n")
    print(result["generated_content"])