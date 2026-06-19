from pinecone import Pinecone, ServerlessSpec
from app.config import settings

_pc = Pinecone(api_key=settings.PINECONE_API_KEY)

def get_index():
    existing = [idx.name for idx in _pc.list_indexes()]
    if settings.PINECONE_INDEX not in existing:
        _pc.create_index(
            name=settings.PINECONE_INDEX,
            dimension=384,  # MiniLM-L6-v2 output dim
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return _pc.Index(settings.PINECONE_INDEX)

def upsert_chunks(tenant_id: str, records: list[dict]):
    """records already contain id, values, metadata — namespace = tenant_id
    enforces multi-tenant isolation at the Pinecone level."""
    index = get_index()
    index.upsert(vectors=records, namespace=tenant_id)

def query_chunks(tenant_id: str, vector: list[float], top_k: int = 5,
                  cluster_filter: dict | None = None):
    index = get_index()
    return index.query(
        namespace=tenant_id,
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=cluster_filter,
    )