from datetime import datetime
from pathlib import Path

from llm_integration import generate_text
from content_pipeline import run_pipeline


OUTPUT_DIR = Path("outputs")


def save_comparison(content: str) -> Path:
    """
    Save the uniqueness comparison to the outputs folder.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"uniqueness_comparison_{timestamp}.md"

    output_path.write_text(content, encoding="utf-8")

    return output_path


def run_uniqueness_comparison() -> dict:
    """
    Compare generic AI content against Mythos Content Engine output.
    """
    topic = "Create an Instagram launch caption for Mortal Vengeance II: To Reel or Not Too Real?"

    generic_prompt = f"""
Create an Instagram caption for a thriller book.

Topic:
{topic}

Keep it exciting and include hashtags.
"""

    generic_output = generate_text(generic_prompt)

    mythos_result = run_pipeline(
        content_type="instagram_caption",
        topic=topic
    )

    mythos_output = mythos_result["generated_content"]

    comparison = f"""# Uniqueness Comparison

## Test Topic

{topic}

---

## Generic AI Output

{generic_output}

---

## Mythos Content Engine Output

{mythos_output}

---

## Comparison Notes

The generic output uses a broad prompt with no brand knowledge, no series context, no content playbook, no review rules, and no Mortal Vengeance-specific positioning.

The Mythos Content Engine output is expected to be more specific because it uses:

- Mortal Vengeance brand voice
- Series-specific language and positioning
- The primary knowledge base
- The secondary research layer
- Platform-specific Instagram rules
- Anti-generic language constraints
- The content playbook’s strategy rules
- The project’s prompt template system

This comparison demonstrates that the system does not simply produce generic AI content. It generates brand-aligned content shaped by curated project knowledge and reusable prompt templates.
"""

    comparison_path = save_comparison(comparison)

    return {
        "comparison_path": comparison_path,
        "generic_output": generic_output,
        "mythos_output": mythos_output
    }


if __name__ == "__main__":
    result = run_uniqueness_comparison()

    print("Uniqueness comparison generated successfully.")
    print(f"Saved comparison to: {result['comparison_path']}")

    print("\nGeneric output preview:\n")
    print(result["generic_output"])

    print("\nMythos output preview:\n")
    print(result["mythos_output"])