from datetime import datetime
from pathlib import Path

from knowledge_base import build_knowledge_context
from prompt_templates import load_template, build_prompt
from llm_integration import generate_text


OUTPUT_DIR = Path("outputs")


def save_output(content: str, content_type: str, label: str = "draft") -> Path:
    """
    Save generated content or prompt text to the outputs folder.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{content_type}_{label}_{timestamp}.md"
    output_path = OUTPUT_DIR / filename

    output_path.write_text(content, encoding="utf-8")

    return output_path


def create_content_prompt(content_type: str, topic: str) -> str:
    """
    Build a complete prompt using the knowledge base and selected template.
    """
    context = build_knowledge_context()
    template_text = load_template(content_type)

    full_prompt = build_prompt(
        template_text=template_text,
        topic=topic,
        knowledge_context=context["full_context"]
    )

    return full_prompt


def run_pipeline(content_type: str, topic: str) -> dict:
    """
    Run the full content pipeline:
    knowledge base + template + topic → prompt → LLM output → saved files.
    """
    prompt = create_content_prompt(content_type, topic)

    prompt_path = save_output(
        content=prompt,
        content_type=content_type,
        label="prompt"
    )

    generated_content = generate_text(prompt)

    draft_path = save_output(
        content=generated_content,
        content_type=content_type,
        label="draft"
    )

    return {
        "prompt_path": prompt_path,
        "draft_path": draft_path,
        "generated_content": generated_content
    }


if __name__ == "__main__":
    test_content_type = "instagram_caption"
    test_topic = "Create a launch caption for Mortal Vengeance II: To Reel or Not Too Real?"

    result = run_pipeline(test_content_type, test_topic)

    print("Pipeline ran successfully.")
    print(f"Saved prompt to: {result['prompt_path']}")
    print(f"Saved draft to: {result['draft_path']}")
    print("\nGenerated content preview:\n")
    print(result["generated_content"])