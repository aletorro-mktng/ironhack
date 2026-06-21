"""Semantic (embedding-based) relevance scoring for manuscript passage retrieval.

Companion to ``quote_semantic.py``. Where that module embeds the curated quote
bank, this one embeds passage chunks from the full novel manuscripts so the
engine can pull the most relevant *book* passages directly — not only quotes that
someone already added to the quote bank. The two layers augment each other:
quote-bank RAG keeps surfacing vetted, attributed quotes, while manuscript RAG
opens up the entire text of the novels to semantic discovery.

Chunks are keyed by a stable content hash. Vectors are embedded once with
``text-embedding-3-small`` (already an installed dependency) and cached on disk;
new chunks are embedded and merged into the cache, existing ones are reused, so
the manuscript corpus is only ever embedded a single time.

Everything is best-effort: if there is no API key, the network is down, numpy is
missing, or anything raises, ``query_relevance`` returns an empty mapping and the
caller falls straight through to the existing deterministic keyword retrieval.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from llm_integration import EMBEDDING_MODEL, embed_texts, embeddings_available


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "knowledge_base" / ".embeddings_cache"
CACHE_PATH = CACHE_DIR / "manuscripts.npz"
CACHE_VERSION = "1"


def chunk_id(chunk: str) -> str:
    """Return a stable content-hash id for a manuscript passage chunk."""
    return hashlib.sha256((chunk or "").encode("utf-8")).hexdigest()[:24]


def _load_cache():
    if not CACHE_PATH.exists():
        return None
    try:
        import numpy as np

        data = np.load(CACHE_PATH, allow_pickle=True)
        if str(data["model"]) != EMBEDDING_MODEL or str(data["version"]) != CACHE_VERSION:
            return None
        return {
            "ids": [str(x) for x in data["ids"]],
            "vectors": data["vectors"],
        }
    except Exception:
        return None


def _save_cache(ids, vectors) -> None:
    try:
        import numpy as np

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            CACHE_PATH,
            ids=np.array(ids, dtype=object),
            vectors=vectors,
            model=np.array(EMBEDDING_MODEL),
            version=np.array(CACHE_VERSION),
        )
    except Exception:
        pass


def _ensure_index(chunks):
    """Build/refresh the {chunk_id: normalized_vector} index.

    Only chunks whose hash is not already cached are embedded; the cache then
    accumulates every chunk seen across runs and chunkings, so each unique
    manuscript passage is embedded exactly once.
    """
    import numpy as np

    ids = [chunk_id(chunk) for chunk in chunks]

    cache = _load_cache()
    cached: dict[str, object] = {}
    if cache:
        for i, cid in enumerate(cache["ids"]):
            cached[cid] = cache["vectors"][i]

    seen: set[str] = set()
    to_embed = []
    for i, cid in enumerate(ids):
        if cid not in cached and cid not in seen:
            seen.add(cid)
            to_embed.append(i)

    if to_embed:
        embedded = embed_texts([chunks[i] for i in to_embed])
        for j, i in enumerate(to_embed):
            cached[ids[i]] = embedded[j]
        merged_ids = list(cached.keys())
        merged_vectors = np.asarray([cached[cid] for cid in merged_ids], dtype=np.float32)
        _save_cache(merged_ids, merged_vectors)

    return {cid: cached[cid] for cid in ids if cid in cached}


def query_relevance(query: str, chunks: list[str]) -> dict[str, float]:
    """Return {chunk_id: cosine_similarity in [0, 1]} for the query vs each chunk.

    Returns an empty dict on any failure so the caller can fall back to keyword
    retrieval. Vectors are L2-normalized, so cosine is a dot product.
    """
    try:
        import numpy as np

        if not embeddings_available() or not chunks or not str(query or "").strip():
            return {}
        index = _ensure_index(chunks)
        if not index:
            return {}
        query_vector = embed_texts([query])[0]
        scores: dict[str, float] = {}
        for chunk in chunks:
            cid = chunk_id(chunk)
            vector = index.get(cid)
            if vector is not None:
                scores[cid] = max(0.0, float(np.dot(vector, query_vector)))
        return scores
    except Exception:
        return {}


if __name__ == "__main__":
    # Pre-warm the manuscript embedding cache so the first real query is fast.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from context_filter import (
        BOOK_MANUSCRIPT_FILES,
        PRIVATE_MANUSCRIPT_DIR,
        split_text_into_chunks,
    )

    if not embeddings_available():
        print("No OPENAI_API_KEY found; skipping manuscript embedding pre-warm.")
        raise SystemExit(0)

    total = 0
    for book, filenames in BOOK_MANUSCRIPT_FILES.items():
        for filename in filenames:
            path = PRIVATE_MANUSCRIPT_DIR / filename
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            # Warm both chunk sizes used by the two manuscript candidate builders.
            for max_chars in (2400, 2600):
                chunks = split_text_into_chunks(text, max_chars=max_chars)
                _ensure_index(chunks)
                total += len(chunks)
            print(f"  embedded {book} ({filename})")
            break
    print(f"Manuscript embedding cache warmed ({total} chunk slots). -> {CACHE_PATH}")
