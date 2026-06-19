"""
Single shared Pinecone index for the whole platform. Multi-tenancy is
enforced via `namespace=tenant_id` on every read/write — this is what
keeps one institute's content from ever leaking into another's retrieval
results, without needing separate indexes (which would cost more).

Updated for pinecone-client v9 (ground-up rewrite, June 2026):
  - `pc.has_index(name=...)` replaces manually scanning `list_indexes()`
  - `pc.Index(host=...)` replaces `pc.Index(name_string)` — Pinecone's
    docs now explicitly discourage targeting an index by name for data
    operations, so we look the host up once via `describe_index` and
    cache the handle.
"""
from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec

from app.config import settings

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


@lru_cache(maxsize=1)
def _client() -> Pinecone:
    return Pinecone(api_key=settings.PINECONE_API_KEY)


@lru_cache(maxsize=1)
def get_index():
    pc = _client()
    if not pc.has_index(name=settings.PINECONE_INDEX):
        pc.create_index(
            name=settings.PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index_config = pc.describe_index(name=settings.PINECONE_INDEX)
    return pc.Index(host=index_config.host)


def upsert_chunks(tenant_id: str, records: list[dict]) -> None:
    """records: list of {"id", "values", "metadata"} dicts, already embedded."""
    if not records:
        return
    index = get_index()
    # Batch upserts at 100 vectors/call to stay well under Pinecone's payload limits
    for i in range(0, len(records), 100):
        index.upsert(vectors=records[i : i + 100], namespace=tenant_id)


def query_chunks(
    tenant_id: str,
    vector: list[float],
    top_k: int = 5,
    metadata_filter: dict | None = None,
):
    index = get_index()
    return index.query(
        namespace=tenant_id,
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter,
    )


def delete_document(tenant_id: str, document_name: str) -> None:
    """Remove all chunks for a given source document (e.g. on re-ingest)."""
    index = get_index()
    index.delete(namespace=tenant_id, filter={"source_document": {"$eq": document_name}})
