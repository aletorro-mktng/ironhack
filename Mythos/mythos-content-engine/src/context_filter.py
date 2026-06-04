import re

from document_processor import load_knowledge_base
from llm_integration import generate_text


MAX_CHUNK_CHARS = 1800
TOP_CHUNKS = 10


STOPWORDS = {
    "the", "and", "or", "for", "with", "from", "into", "that", "this",
    "about", "using", "only", "when", "what", "which", "should", "would",
    "create", "generate", "write", "post", "content", "quote", "quotes",
    "a", "an", "to", "of", "in", "on", "by", "as", "is", "are", "be"
}


def tokenize(text: str) -> set[str]:
    """
    Convert text into useful lowercase search tokens.
    """
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", text.lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 2}


def split_text_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split markdown text into manageable chunks.
    Tries to preserve markdown sections when possible.
    """
    sections = re.split(r"(?=^#{1,4}\s+)", text, flags=re.MULTILINE)

    chunks = []

    for section in sections:
        section = section.strip()

        if not section:
            continue

        if len(section) <= max_chars:
            chunks.append(section)
        else:
            for start in range(0, len(section), max_chars):
                chunks.append(section[start:start + max_chars])

    return chunks


def build_candidate_chunks(content_type: str, topic: str) -> list[dict]:
    """
    Load the knowledge base, split documents into chunks, and rank them
    using lightweight keyword matching before the LLM filtering step.
    """
    knowledge_base = load_knowledge_base()

    keywords = tokenize(topic + " " + content_type.replace("_", " "))

    candidates = []

    for layer_name, documents in knowledge_base.items():
        for document in documents:
            title = document["title"]

            if content_type == "quote_post" and title != "quote_bank":
                continue

            path = document["path"]
            chunks = split_text_into_chunks(document["content"])

            for index, chunk in enumerate(chunks, start=1):
                chunk_tokens = tokenize(chunk)
                score = len(keywords.intersection(chunk_tokens))

                candidates.append({
                    "layer": layer_name,
                    "title": title,
                    "path": path,
                    "chunk_index": index,
                    "score": score,
                    "content": chunk
                })

    ranked_candidates = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True
    )

    useful_candidates = [item for item in ranked_candidates if item["score"] > 0]

    if not useful_candidates:
        useful_candidates = ranked_candidates[:TOP_CHUNKS]

    return useful_candidates[:TOP_CHUNKS]


def format_candidate_chunks(candidates: list[dict]) -> str:
    """
    Format candidate chunks for the relevance-filtering prompt.
    """
    formatted_chunks = []

    for candidate in candidates:
        header = (
            f"## Source: {candidate['title']} "
            f"| Layer: {candidate['layer']} "
            f"| Chunk: {candidate['chunk_index']}"
        )

        formatted_chunks.append(f"{header}\n\n{candidate['content']}")

    return "\n\n---\n\n".join(formatted_chunks)


def build_relevance_filter_prompt(content_type: str, topic: str, candidate_context: str) -> str:
    """
    Build the first-stage LLM prompt.
    This prompt filters the knowledge base before final generation.
    """
    return f"""
You are the context filtering layer for Mythos Content Engine.

Your task is NOT to write the final content.

Your task is to select and summarize only the context that is relevant to the user's request.

Content type:
{content_type}

User request:
{topic}

Candidate knowledge base context:
{candidate_context}

Instructions:

1. Keep only information relevant to the request.
2. Preserve exact real review quotes if they are relevant.
3. Preserve exact book quotes if they are relevant.
4. Keep source names when using review excerpts.
5. Identify any spoiler restrictions.
6. Identify any factual or canon constraints.
7. Do not invent facts, reviews, quotes, awards, ratings, or book details.
8. Do not generate the final marketing content.
9. If no relevant quote or review exists, say so clearly.
10. For quote_post requests, use only the Mortal Vengeance Quote Bank.
11. For quote_post requests, if the requested book/source has no matching entry in the quote bank, say no approved quote is currently available for that book/source.

Return the filtered context using this structure:

# Filtered Context

## Relevant Brand / Book Context

## Relevant Quotes or Review Excerpts

## Relevant Audience / Strategy Notes

## Spoiler or Accuracy Constraints

## Do Not Use / Avoid
"""


def select_relevant_context(content_type: str, topic: str) -> str:
    """
    First LLM call:
    Filter the knowledge base into a smaller, relevant context brief.
    """
    candidates = build_candidate_chunks(content_type, topic)
    candidate_context = format_candidate_chunks(candidates)

    relevance_prompt = build_relevance_filter_prompt(
        content_type=content_type,
        topic=topic,
        candidate_context=candidate_context
    )

    filtered_context = generate_text(relevance_prompt)

    return filtered_context


if __name__ == "__main__":
    test_context = select_relevant_context(
        content_type="instagram_caption",
        topic="Create a launch caption for Mortal Vengeance II."
    )

    print(test_context)
