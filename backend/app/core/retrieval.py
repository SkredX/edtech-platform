"""
Retrieval shared by /chat and /clone. Pinecone ANN search already does the
heavy lifting (it's a single vector query, cheap and fast), so the "two
stage" optimization here is mostly about enriching what we send to the LLM
afterwards using metadata that's already attached to each match — no extra
API or DB calls required.

`score_chunks_locally` is kept as a standalone utility too: it's the same
TF-IDF + metadata-keyword scoring logic used by the original CloneGen
Colab notebook, useful if you ever want to re-rank a larger candidate set
pulled back from Pinecone (e.g. top_k=20) down to the best few before
building the LLM context block, without any additional embedding calls.

`dedupe_chunks` and `compress_chunk_text` are the two additions that cut
Groq prompt tokens further, both pure local CS/math, zero extra API calls:
  - dedupe_chunks: shingled-text Jaccard similarity to drop near-duplicate
    chunks (common when a PDF chunker overlaps page boundaries) before they
    ever reach the LLM prompt.
  - compress_chunk_text: per-chunk extractive summarization — score each
    sentence in a chunk against the query via TF-IDF cosine similarity and
    keep only the top-scoring sentences, instead of sending the chunk's
    full raw text. This is classic extractive summarization (Luhn-style
    sentence scoring), not an LLM call.
"""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.embeddings import embed_query
from app.core.pinecone_client import query_chunks


def score_chunks_locally(query: str, chunk_metadata: list[dict], top_k: int = 8) -> list[dict]:
    """Pure TF-IDF re-ranking over already-retrieved chunk metadata. Zero API cost."""
    if not chunk_metadata:
        return []

    enriched = [
        " ".join(
            [
                c.get("text", ""),
                c.get("semantic_topic", ""),
                c.get("semantic_topic", ""),  # light upweight via repetition
                c.get("context_summary", ""),
                " ".join(c.get("tfidf_keywords", []) or []),
                " ".join(c.get("entities", []) or []),
            ]
        )
        for c in chunk_metadata
    ]
    corpus = enriched + [query]

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return chunk_metadata[:top_k]

    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = sorted(zip(sims, chunk_metadata), key=lambda x: -x[0])
    return [c for _, c in ranked[:top_k]]


def _shingles(text: str, k: int = 5) -> set[str]:
    """Word-level k-shingles, used for cheap near-duplicate detection."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def dedupe_chunks(chunks: list[dict], similarity_threshold: float = 0.8) -> list[dict]:
    """
    Drops near-duplicate chunks via Jaccard similarity over word shingles —
    no embeddings, no API calls, just set arithmetic. Keeps the
    first-seen (i.e. highest-ranked) copy of each near-duplicate group, so
    callers should dedupe AFTER ranking/reranking, not before.

    Why this matters for cost: overlapping PDF chunking + KMeans
    clustering can surface 2-3 chunks that are 90% the same paragraph for
    a single query. Sending all of them to Groq burns prompt tokens on
    redundant text without adding any signal.
    """
    if not chunks:
        return []

    kept: list[dict] = []
    kept_shingles: list[set[str]] = []

    for c in chunks:
        sh = _shingles(c.get("text", ""))
        is_dup = False
        for existing_sh in kept_shingles:
            if not sh or not existing_sh:
                continue
            union = sh | existing_sh
            if not union:
                continue
            jaccard = len(sh & existing_sh) / len(union)
            if jaccard >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(c)
            kept_shingles.append(sh)

    return kept


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def compress_chunk_text(
    query: str, text: str, max_sentences: int = 4, min_sentences_to_compress: int = 6
) -> str:
    """
    Extractive compression: scores each sentence in `text` by TF-IDF cosine
    similarity to the query and keeps only the top `max_sentences`,
    re-ordered back into original document order (so the kept sentences
    still read coherently). Pure local math — no LLM call.

    Short chunks are left untouched (`min_sentences_to_compress` guard):
    compressing an already-short chunk risks cutting context that the
    answer genuinely needs, for a token saving too small to matter.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) < min_sentences_to_compress:
        return text

    corpus = sentences + [query]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return text

    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    top_idx = sorted(sims.argsort()[::-1][:max_sentences])  # re-sort by original position
    return " ".join(sentences[i] for i in top_idx)


def retrieve_context(
    tenant_id: str,
    query: str,
    top_k: int = 5,
    rerank: bool = False,
    document_name: str | None = None,
    document_names: list[str] | None = None,
    dedupe: bool = True,
) -> tuple[list[dict], float]:
    """
    Returns (chunks, top_confidence_score). Confidence feeds the chatbot's
    fallback guardrail (see chatbot/guardrails.py).

    If `rerank=True`, pulls a wider candidate set from Pinecone (top_k * 3,
    capped at 20) and re-ranks locally via TF-IDF before trimming back down
    to top_k — useful when Pinecone's pure vector similarity surfaces
    near-duplicates from the same cluster.

    If `document_names` is given, retrieval is scoped to chunks from any
    of those ingested documents (Pinecone metadata filter on
    `source_document` with `$in`) instead of the tenant's whole corpus —
    this is what the chat page's chapter picker uses when one or more
    chapters are selected, so a query against a handful of chapters
    doesn't compete with the rest of the unrelated corpus. `document_name`
    (singular) is kept as a back-compat alias for a one-item selection.

    If `dedupe=True` (default), near-duplicate chunks are dropped after
    ranking (see `dedupe_chunks`) and backfilled from the remaining
    candidate pool so the caller still gets up to `top_k` distinct chunks
    where the pool allows it, instead of silently returning fewer.
    """
    q_vec = embed_query(query)
    # Pull extra candidates whenever we might need to backfill after dedup,
    # not just when reranking — otherwise dedup can silently starve the
    # caller down to fewer than top_k chunks.
    pull_k = min(max(top_k * 3, top_k + 10), 20) if (rerank or dedupe) else top_k

    names = document_names or ([document_name] if document_name else None)
    metadata_filter = {"source_document": {"$in": names}} if names else None

    result = query_chunks(tenant_id, q_vec, top_k=pull_k, metadata_filter=metadata_filter)
    matches = result.get("matches", []) if isinstance(result, dict) else result.matches

    if not matches:
        return [], 0.0

    confidence = matches[0]["score"] if isinstance(matches[0], dict) else matches[0].score
    chunks = [m["metadata"] if isinstance(m, dict) else m.metadata for m in matches]
    scores = [m["score"] if isinstance(m, dict) else m.score for m in matches]
    for c, s in zip(chunks, scores):
        # Stashed under a private-ish key so it travels with the chunk
        # through rerank/dedupe without touching the tuple's public shape —
        # existing callers that do `chunks, _ = retrieve_context(...)` are
        # unaffected, and chatbot/router.py can read it for the spread-based
        # escalation check without an extra Pinecone round trip.
        c["_retrieval_score"] = float(s)

    if rerank:
        chunks = score_chunks_locally(query, chunks, top_k=len(chunks))

    if dedupe:
        chunks = dedupe_chunks(chunks)

    chunks = chunks[:top_k]

    return chunks, float(confidence)


def extract_query_terms(text: str) -> set[str]:
    """Lightweight tokenizer used when building cache keys / debugging relevance."""
    stopwords = {
        "a", "an", "the", "is", "are", "which", "what", "of", "in", "on",
        "for", "to", "and", "or", "with", "that", "this", "by", "as", "be",
    }
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {w for w in words if w not in stopwords}
