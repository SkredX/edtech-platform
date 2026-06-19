"""
Local embedding layer — sentence-transformers/all-MiniLM-L6-v2 runs on CPU
in-process. This is shared by ingestion, retrieval, and the semantic cache,
so the model is loaded exactly once per backend process (lru_cache).
Zero API cost, zero token spend.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedder()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,  # cosine-friendly, matches Pinecone metric="cosine"
        batch_size=64,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
