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


def retrieve_context(
    tenant_id: str,
    query: str,
    top_k: int = 5,
    rerank: bool = False,
    document_name: str | None = None,
) -> tuple[list[dict], float]:
    """
    Returns (chunks, top_confidence_score). Confidence feeds the chatbot's
    fallback guardrail (see chatbot/guardrails.py).

    If `rerank=True`, pulls a wider candidate set from Pinecone (top_k * 3,
    capped at 20) and re-ranks locally via TF-IDF before trimming back down
    to top_k — useful when Pinecone's pure vector similarity surfaces
    near-duplicates from the same cluster.

    If `document_name` is given, retrieval is scoped to chunks from that
    one ingested document only (Pinecone metadata filter on
    `source_document`) instead of the tenant's whole corpus — this is
    what the chat page's chapter picker uses, so a query against a
    180-chunk chapter doesn't compete with the other 360 chunks from
    unrelated chapters.
    """
    q_vec = embed_query(query)
    pull_k = min(top_k * 3, 20) if rerank else top_k
    metadata_filter = {"source_document": {"$eq": document_name}} if document_name else None

    result = query_chunks(tenant_id, q_vec, top_k=pull_k, metadata_filter=metadata_filter)
    matches = result.get("matches", []) if isinstance(result, dict) else result.matches

    if not matches:
        return [], 0.0

    confidence = matches[0]["score"] if isinstance(matches[0], dict) else matches[0].score
    chunks = [m["metadata"] if isinstance(m, dict) else m.metadata for m in matches]

    if rerank:
        chunks = score_chunks_locally(query, chunks, top_k=top_k)
    else:
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
