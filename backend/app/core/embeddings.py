"""
Embedding layer — calls Pinecone's hosted Inference API (llama-text-embed-v2)
instead of running sentence-transformers/PyTorch in-process.

Why: this app targets Render's free tier (512 MB RAM). Loading PyTorch +
sentence-transformers in-process routinely uses 600 MB-1 GB on its own,
which OOMs there. Pinecone Inference is free up to 5M tokens/month on the
Starter plan, needs no extra account (we already need Pinecone), and
removes ~1.5 GB of dependencies (torch + friends) from the Docker image,
which also speeds up Render's build.

Trade-off: every embed call is now a network round-trip instead of local
CPU inference, so ingestion of very large documents will be slower than
before. For typical chatbot/clonegen usage (a handful of chunks per
request) this is unnoticeable.
"""
from functools import lru_cache

from pinecone import Pinecone

from app.config import settings

EMBEDDING_DIM = 384  # must match pinecone_client.py's index dimension
_BATCH_SIZE = 90     # Pinecone Inference's per-call input limit is 96; stay under it


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
                "dimension": EMBEDDING_DIM,
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
