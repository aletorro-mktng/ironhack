"""Semantic (embedding-based) relevance scoring for quote-bank retrieval.

Lightweight RAG layer for quote_post: embeds the quote-bank entries once with
OpenAI ``text-embedding-3-small`` (already an installed dependency), caches the
vectors on disk keyed by each entry's stable ``## Quote XXX-N`` id plus a content
hash, and scores entries against a query by cosine similarity.

Everything is best-effort: if there is no API key, the network is down, numpy is
missing, or anything raises, ``query_relevance`` returns an empty mapping and the
caller falls straight through to the existing deterministic keyword retrieval.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from llm_integration import EMBEDDING_MODEL, embed_texts, embeddings_available


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "knowledge_base" / ".embeddings_cache"
CACHE_PATH = CACHE_DIR / "quote_bank.npz"
CACHE_VERSION = "1"

_QUOTE_ID_RE = re.compile(r"##\s+Quote\s+([A-Za-z]+-\d+)")


def quote_entry_id(entry: str) -> str:
    """Return the stable ``XXX-N`` id from a quote-bank entry, or a content hash."""
    match = _QUOTE_ID_RE.search(entry or "")
    if match:
        return match.group(1)
    return "h:" + hashlib.sha256((entry or "").encode("utf-8")).hexdigest()[:16]


def _entry_hash(entry: str) -> str:
    return hashlib.sha256((entry or "").encode("utf-8")).hexdigest()


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
            "hashes": [str(x) for x in data["hashes"]],
            "vectors": data["vectors"],
        }
    except Exception:
        return None


def _save_cache(ids, hashes, vectors) -> None:
    try:
        import numpy as np

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            CACHE_PATH,
            ids=np.array(ids, dtype=object),
            hashes=np.array(hashes, dtype=object),
            vectors=vectors,
            model=np.array(EMBEDDING_MODEL),
            version=np.array(CACHE_VERSION),
        )
    except Exception:
        pass


def _ensure_index(entries):
    """Build/refresh the {entry_id: normalized_vector} index, re-embedding only changes."""
    import numpy as np

    ids = [quote_entry_id(entry) for entry in entries]
    hashes = [_entry_hash(entry) for entry in entries]

    cache = _load_cache()
    cached: dict[tuple[str, str], object] = {}
    if cache:
        for i, eid in enumerate(cache["ids"]):
            cached[(eid, cache["hashes"][i])] = cache["vectors"][i]

    to_embed = [i for i in range(len(entries)) if (ids[i], hashes[i]) not in cached]
    fresh: dict[int, object] = {}
    if to_embed:
        embedded = embed_texts([entries[i] for i in to_embed])
        for j, i in enumerate(to_embed):
            fresh[i] = embedded[j]

    index: dict[str, object] = {}
    out_ids: list[str] = []
    out_hashes: list[str] = []
    out_vectors: list[object] = []
    for i in range(len(entries)):
        vector = fresh.get(i)
        if vector is None:
            vector = cached.get((ids[i], hashes[i]))
        if vector is None:
            continue
        index[ids[i]] = vector
        out_ids.append(ids[i])
        out_hashes.append(hashes[i])
        out_vectors.append(vector)

    if fresh and out_vectors:
        _save_cache(out_ids, out_hashes, np.asarray(out_vectors, dtype=np.float32))
    return index


def query_relevance(query: str, entries: list[str]) -> dict[str, float]:
    """Return {entry_id: cosine_similarity in [0, 1]} for the query vs each entry.

    Returns an empty dict on any failure so the caller can fall back to keyword
    retrieval. Vectors are L2-normalized, so cosine is a dot product.
    """
    try:
        import numpy as np

        if not embeddings_available() or not entries or not str(query or "").strip():
            return {}
        index = _ensure_index(entries)
        if not index:
            return {}
        query_vector = embed_texts([query])[0]
        scores: dict[str, float] = {}
        for entry in entries:
            vector = index.get(quote_entry_id(entry))
            if vector is not None:
                scores[quote_entry_id(entry)] = max(0.0, float(np.dot(vector, query_vector)))
        return scores
    except Exception:
        return {}
