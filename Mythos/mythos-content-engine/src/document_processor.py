"""Document loading and normalization utilities."""


def load_document(path: str) -> str:
    """Read a UTF-8 text document from disk."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()
from pathlib import Path


def load_markdown_file(file_path):
    """
    Load a single markdown file and return its metadata and content.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix != ".md":
        raise ValueError(f"File is not a markdown file: {file_path}")

    content = path.read_text(encoding="utf-8")

    return {
        "title": path.stem,
        "path": str(path),
        "content": content
    }


def load_markdown_folder(folder_path):
    """
    Load all markdown files from a folder.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    documents = []

    for file_path in sorted(folder.glob("*.md")):
        document = load_markdown_file(file_path)
        documents.append(document)

    return documents


def load_knowledge_base(
    primary_path="knowledge_base/primary",
    secondary_path="knowledge_base/secondary",
    publishing_path="knowledge_base/publishing"
):
    """
    Load primary, secondary, and (when present) publishing knowledge base documents.
    """
    primary_documents = load_markdown_folder(primary_path)
    secondary_documents = load_markdown_folder(secondary_path)

    knowledge_base = {
        "primary": primary_documents,
        "secondary": secondary_documents
    }

    if Path(publishing_path).exists():
        knowledge_base["publishing"] = load_markdown_folder(publishing_path)

    return knowledge_base


if __name__ == "__main__":
    knowledge_base = load_knowledge_base()

    print("Knowledge base loaded successfully.")
    print(f"Primary documents: {len(knowledge_base['primary'])}")
    print(f"Secondary documents: {len(knowledge_base['secondary'])}")

    print("\nPrimary files:")
    for document in knowledge_base["primary"]:
        print(f"- {document['title']}")

    print("\nSecondary files:")
    for document in knowledge_base["secondary"]:
        print(f"- {document['title']}")