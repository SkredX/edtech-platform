"""
Embedding layer — calls Pinecone's hosted Inference API (llama-text-embed-v2)
instead of running sentence-transformers/PyTorch in-process.

Why: this app targets Render's free tier (512 MB RAM). Loading PyTorch +
sentence-transformers in-process routinely uses 600 MB–1 GB on its own,
which OOMs there. Pinecone Inference is free up to 5 M tokens/month on the
Starter plan, needs no extra account (we already need Pinecone), and removes
~1.5 GB of dependencies (torch + friends) from the Docker image, which also
speeds up Render's build.

llama-text-embed-v2 outputs 1024-dimensional vectors by default.
pinecone_client.py's EMBEDDING_DIM must match this value.
"""
from functools import lru_cache

from pinecone import Pinecone

from app.config import settings

_BATCH_SIZE = 90  # Pinecone Inference per-call input limit is 96; stay under it


@lru_cache(maxsize=1)
def _client() -> Pinecone:
    return Pinecone(api_key=settings.PINECONE_API_KEY)


def _embed(texts: list[str], input_type: str) -> list[list[float]]:
    if not texts:
        return []
    pc = _client()
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        result = pc.inference.embed(
            model=settings.EMBEDDING_MODEL,
            inputs=batch,
            parameters={
                "input_type": input_type,  # "passage" for documents, "query" for searches
                "truncate": "END",
            },
        )
        out.extend(item["values"] for item in result)
    return out


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Use for document chunks going INTO the index."""
    return _embed(texts, input_type="passage")


def embed_query(query: str) -> list[float]:
    """Use for search queries (chat/cache lookups) going against the index."""
    return _embed([query], input_type="query")[0]