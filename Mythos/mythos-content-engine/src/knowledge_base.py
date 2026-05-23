"""Knowledge base access helpers."""

from pathlib import Path


def read_markdown_files(root: str) -> dict[str, str]:
    """Return markdown file contents keyed by relative path."""
    base_path = Path(root)
    return {
        str(path.relative_to(base_path)): path.read_text(encoding="utf-8")
        for path in sorted(base_path.rglob("*.md"))
    }
from document_processor import load_knowledge_base


def format_documents_as_context(documents, section_title):
    """
    Convert a list of loaded markdown documents into one readable context block.
    """
    context_parts = [f"# {section_title}\n"]

    for document in documents:
        context_parts.append(f"## {document['title']}\n")
        context_parts.append(document["content"])
        context_parts.append("\n")

    return "\n".join(context_parts)


def build_knowledge_context():
    """
    Load primary and secondary knowledge base files and format them for prompt use.
    """
    knowledge_base = load_knowledge_base()

    primary_context = format_documents_as_context(
        knowledge_base["primary"],
        "Primary Knowledge Base: Brand and Story Context"
    )

    secondary_context = format_documents_as_context(
        knowledge_base["secondary"],
        "Secondary Research Layer: Market and Platform Context"
    )

    full_context = f"""
{primary_context}

---

{secondary_context}
"""

    return {
        "primary_context": primary_context,
        "secondary_context": secondary_context,
        "full_context": full_context
    }


if __name__ == "__main__":
    context = build_knowledge_context()

    print("Knowledge context built successfully.")
    print("\nPrimary context preview:")
    print(context["primary_context"][:500])

    print("\nSecondary context preview:")
    print(context["secondary_context"][:500])