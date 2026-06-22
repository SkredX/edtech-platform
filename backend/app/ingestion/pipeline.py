# backend/app/ingestion/pipeline.py
"""
Orchestrates the full ingestion pipeline and exposes it as a FastAPI route.
This is the only place where bulk Groq usage happens (centroid summaries),
and it's a one-time cost per document — separate from the per-request
budgets in /chat and /clone.

Background-task variant (POST /ingest/async):
  • Returns a job_id immediately.
  • Runs ingest_document() in a FastAPI BackgroundTasks thread.
  • Writes per-step progress to Redis (key: ingest:job:{job_id}, TTL 1 h).
  • GET /ingest/status/{job_id} lets the UI poll for progress + final result.

The original blocking POST /ingest is kept for backward compatibility.
No new API calls are introduced — only Redis reads/writes which are already
used by the semantic cache.
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone

import redis
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.config import settings
from app.core.document_registry import backfill_registry_from_pinecone, list_documents, register_document
from app.core.pinecone_client import delete_document, upsert_chunks
from app.ingestion.chunker import chunk_pdf
from app.ingestion.clusterer import cluster_and_summarize
from app.tenants.auth import get_tenant_id

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# ── Redis job store ───────────────────────────────────────────────────────────
_r = redis.from_url(settings.REDIS_URL, decode_responses=True)
JOB_KEY_TMPL = "ingest:job:{job_id}"
JOB_TTL = 3600  # 1 hour — plenty of time for the UI to poll and display results


def _job_key(job_id: str) -> str:
    return JOB_KEY_TMPL.format(job_id=job_id)


def _write_job(job_id: str, payload: dict) -> None:
    _r.setex(_job_key(job_id), JOB_TTL, json.dumps(payload))


def _read_job(job_id: str) -> dict | None:
    raw = _r.get(_job_key(job_id))
    return json.loads(raw) if raw else None


# ── Core ingestion logic (unchanged) ─────────────────────────────────────────

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


# ── Background worker with progress reporting ─────────────────────────────────

def _ingest_with_progress(job_id: str, tenant_id: str, pdf_bytes: bytes, filename: str) -> None:
    """
    Runs in a background thread (via BackgroundTasks).
    Writes progress updates to Redis so the polling endpoint can surface them.

    Steps and their approximate % weights:
      10%  — chunking (CPU, fast)
      50%  — embedding + clustering (CPU + Pinecone embed call, slowest step)
      80%  — Groq centroid labelling (network, scales with cluster count)
      95%  — Pinecone upsert + registry write
     100%  — done
    """
    def _progress(step: str, pct: int, detail: str = "") -> None:
        job = _read_job(job_id) or {}
        job.update({"status": "running", "step": step, "progress": pct, "detail": detail})
        _write_job(job_id, job)

    try:
        _progress("Chunking PDF", 5)
        raw_chunks = chunk_pdf(pdf_bytes)
        _progress("Chunking PDF", 10, f"{len(raw_chunks)} chunks extracted")

        _progress("Embedding & clustering", 15, "Computing embeddings — this is the longest step")
        # cluster_and_summarize internally: embed → KMeans → Groq batch calls
        # We monkey-patch nothing; just report before/after each sub-phase by
        # wrapping at this level. Fine-grained intra-function hooks would couple
        # us too tightly to clusterer internals.
        enriched, embeddings = cluster_and_summarize(raw_chunks)
        n_clusters = len({c["cluster_id"] for c in enriched})
        _progress("Groq topic labelling", 80, f"{n_clusters} topic clusters labelled")

        vectors = embeddings.tolist()
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

        _progress("Uploading to Pinecone", 90, f"Upserting {len(records)} vectors")
        upsert_chunks(tenant_id, records)
        register_document(tenant_id, filename, chunk_count=len(records))

        # Build cluster topic preview for the UI: one entry per unique cluster.
        seen: set[int] = set()
        topics: list[dict] = []
        for c in enriched:
            cid = c["cluster_id"]
            if cid not in seen:
                seen.add(cid)
                topics.append(
                    {
                        "cluster_id": cid,
                        "topic": c.get("semantic_topic", "Unknown"),
                        "summary": c.get("context_summary", ""),
                        "chunk_count": sum(1 for x in enriched if x["cluster_id"] == cid),
                    }
                )
        topics.sort(key=lambda t: t["cluster_id"])

        _write_job(
            job_id,
            {
                "status": "done",
                "step": "Complete",
                "progress": 100,
                "detail": "",
                "result": {
                    "document_name": filename,
                    "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
                    "chunks_ingested": len(records),
                    "clusters": n_clusters,
                    "groq_calls_used": n_clusters,
                    "topics": topics,
                },
            },
        )

    except Exception as exc:  # noqa: BLE001
        _write_job(
            job_id,
            {
                "status": "error",
                "step": "Failed",
                "progress": 0,
                "detail": str(exc),
                "result": None,
            },
        )


# ── API routes ────────────────────────────────────────────────────────────────

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
    """Original blocking endpoint — kept for backward compatibility."""
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


class AsyncIngestStartResponse(BaseModel):
    job_id: str


@router.post("/async", response_model=AsyncIngestStartResponse)
async def ingest_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Non-blocking upload: accepts the PDF, stores job state in Redis, and
    returns a job_id immediately. The actual ingestion runs in a background
    thread. Poll GET /ingest/status/{job_id} for progress.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    job_id = str(uuid.uuid4())
    # Write initial state before kicking off background task so the first poll
    # always finds a record (no race window between response and first write).
    _write_job(
        job_id,
        {
            "status": "running",
            "step": "Queued",
            "progress": 0,
            "detail": f"Starting ingestion of {file.filename}",
            "result": None,
        },
    )

    background_tasks.add_task(
        _ingest_with_progress, job_id, tenant_id, pdf_bytes, file.filename
    )
    return {"job_id": job_id}


class JobStatusResponse(BaseModel):
    job_id: str
    status: str          # "running" | "done" | "error"
    step: str
    progress: int        # 0–100
    detail: str
    result: dict | None  # populated only when status == "done"


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_ingest_status(job_id: str, tenant_id: str = Depends(get_tenant_id)):
    """
    Polling endpoint for the Teacher UI progress bar.
    Returns current progress (0-100), step label, and — when done — the full
    result including per-cluster topic previews.
    """
    job = _read_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    return {"job_id": job_id, **job}


# ── Existing document list + backfill routes (unchanged) ─────────────────────

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
    """Powers the chapter picker on the chat page."""
    return list_documents(tenant_id)


class BackfillResponse(BaseModel):
    backfilled_documents: list[str]
    already_registered: list[str]
    total_distinct_documents_in_pinecone: int


@router.post("/backfill-registry", response_model=BackfillResponse)
def backfill_registry(tenant_id: str = Depends(get_tenant_id)):
    """
    One-time (but safely repeatable) resync — see original docstring in the
    previous version of this file for the full explanation.
    """
    return backfill_registry_from_pinecone(tenant_id)