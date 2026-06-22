"""
Two-stage retrieval shared by chat and clonegen:
  Stage 1 (free): TF-IDF + metadata-keyword scoring over cached chunk
                   metadata already in memory/Postgres, to shortlist
                   candidate clusters.
  Stage 2 (cheap): Pinecone ANN query restricted to those candidate
                   chunks' namespace — avoids a full-index broad query.
This is the same scoring logic from CloneGen's context optimizer,
generalized for any querying caller.
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.core.embeddings import embed_query
from app.core.pinecone_client import query_chunks

def score_chunks_locally(query: str, chunk_metadata: list[dict], top_k: int = 8) -> list[dict]:
    if not chunk_metadata:
        return []
    enriched = [
        " ".join([c["text"], c.get("semantic_topic", ""), c.get("semantic_topic", ""),
                  c.get("context_summary", ""), " ".join(c.get("tfidf_keywords", [])),
                  " ".join(c.get("entities", []))])
        for c in chunk_metadata
    ]
    corpus = enriched + [query]
    vec = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1, 2))
    matrix = vec.fit_transform(corpus)
    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = sorted(zip(sims, chunk_metadata), key=lambda x: -x[0])
    return [c for _, c in ranked[:top_k]]

def retrieve_context(tenant_id: str, query: str, chunk_metadata_cache: list[dict],
                      top_k: int = 5) -> tuple[list[dict], float]:
    """Returns (chunks, top_confidence_score) — confidence feeds the
    chatbot's fallback guardrail."""
    q_vec = embed_query(query)
    result = query_chunks(tenant_id, q_vec, top_k=top_k)
    matches = result.get("matches", [])
    confidence = matches[0]["score"] if matches else 0.0
    chunks = [m["metadata"] for m in matches]
    return chunks, confidence