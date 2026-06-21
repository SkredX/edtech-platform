"""
Orchestrates the full ingestion pipeline and exposes it as a FastAPI route.
This is the only place where bulk Groq usage happens (centroid summaries),
and it's a one-time cost per document — separate from the per-request
budgets in /chat and /clone.
"""
import hashlib
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.core.document_registry import list_documents, register_document
from app.core.pinecone_client import delete_document, upsert_chunks
from app.ingestion.chunker import chunk_pdf
from app.ingestion.clusterer import cluster_and_summarize
from app.tenants.auth import get_tenant_id

router = APIRouter(prefix="/ingest", tags=["ingestion"])


def _make_chunk_id(document_name: str, idx: int, text: str) -> str:
    base = f"{document_name}-{idx}-{text[:50]}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def ingest_document(tenant_id: str, pdf_bytes: bytes, filename: str) -> dict:
    raw_chunks = chunk_pdf(pdf_bytes)
    enriched, embeddings = cluster_and_summarize(raw_chunks)
    vectors = embeddings.tolist()

    # Replace any previous ingest of a document with the same filename,
    # so re-uploading doesn't create duplicate/stale vectors.
    delete_document(tenant_id, filename)

    records = []
    for i, c in enumerate(enriched):
        records.append(
            {
                "id": _make_chunk_id(filename, i, c["text"]),
                "values": vectors[i],
                "metadata": {
                    "source_document": filename,
                    "chunk_index": i,
                    "source_page": c.get("source_page"),
                    "text": c["text"],
                    "tfidf_keywords": c.get("tfidf_keywords", []),
                    "entities": c.get("entities", []),
                    "cluster_id": c.get("cluster_id"),
                    "is_cluster_centroid": c.get("is_cluster_centroid", False),
                    "semantic_topic": c.get("semantic_topic", "Unknown"),
                    "context_summary": c.get("context_summary", ""),
                    "char_count": c.get("char_count", len(c["text"])),
                },
            }
        )

    upsert_chunks(tenant_id, records)
    register_document(tenant_id, filename, chunk_count=len(records))

    n_clusters = len({c["cluster_id"] for c in enriched})
    return {
        "document_name": filename,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunks_ingested": len(records),
        "clusters": n_clusters,
        "groq_calls_used": n_clusters,  # one call per cluster centroid
    }


class IngestResponse(BaseModel):
    document_name: str
    ingested_at_utc: str
    chunks_ingested: int
    clusters: int
    groq_calls_used: int


@router.post("", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = await run_in_threadpool(ingest_document, tenant_id, pdf_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return result


class DocumentInfo(BaseModel):
    document_name: str
    grade: str | None
    subject_code: str | None
    subject_label: str | None
    chapter: str | None
    label: str
    chunk_count: int


@router.get("/documents", response_model=list[DocumentInfo])
def get_documents(tenant_id: str = Depends(get_tenant_id)):
    """Powers the chapter picker on the chat page — lets a student narrow
    retrieval to one ingested document instead of searching the whole
    tenant corpus."""
    return list_documents(tenant_id)
