"""
Tracks which documents have been ingested per tenant, and parses their
filenames into structured grade/subject/chapter metadata for the chapter
picker on the chat page.

Why Redis and not a Pinecone metadata scan: Pinecone doesn't offer a
"list distinct metadata values" query — getting the set of document names
would mean pulling back a large/unbounded number of vectors and dedup'ing
client-side, which is slow and gets slower as a tenant's corpus grows.
A small Redis hash (one field per document) is instant and is naturally
kept in sync with ingestion, since registration happens right after a
successful upsert in the same request.

Expected filename convention (institute-specific, not a hard requirement —
files that don't match are still registered, just with chapter=None so
they don't crash anything, they just won't get a parsed label):
    LP_NEET_11B_Cell the unit of life_without solutions.pdf
                 ^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              grade+   chapter name
              subject
                code
"""
import json
import re

import redis

from app.config import settings

_r = redis.from_url(settings.REDIS_URL, decode_responses=True)

REGISTRY_KEY_TMPL = "doc_registry:{tenant_id}"  # redis HASH: filename -> json(doc info)

# Extend as your institute's naming convention covers more subjects.
_SUBJECT_LABELS = {
    "B": "Biology",
    "P": "Physics",
    "C": "Chemistry",
    "M": "Mathematics",
}

_FILENAME_RE = re.compile(
    r"^LP_NEET_(?P<grade>\d{2})(?P<subject>[A-Za-z])_(?P<chapter>.+?)_without[ _]solutions\.pdf$",
    re.IGNORECASE,
)


def parse_document_name(filename: str) -> dict:
    """
    Returns {grade, subject_code, subject_label, chapter, label} on a
    successful parse, or all-None fields (except the raw filename) if the
    filename doesn't match the expected convention — callers should treat
    that as "ungrouped" rather than failing.
    """
    m = _FILENAME_RE.match(filename.strip())
    if not m:
        return {
            "grade": None,
            "subject_code": None,
            "subject_label": None,
            "chapter": None,
            "label": filename,
        }

    grade = m.group("grade")
    subject_code = m.group("subject").upper()
    chapter = m.group("chapter").strip()
    subject_label = _SUBJECT_LABELS.get(subject_code, subject_code)

    return {
        "grade": grade,
        "subject_code": subject_code,
        "subject_label": subject_label,
        "chapter": chapter,
        "label": f"{grade}{subject_code} · {chapter}",
    }


def register_document(tenant_id: str, filename: str, chunk_count: int) -> None:
    parsed = parse_document_name(filename)
    entry = {**parsed, "document_name": filename, "chunk_count": chunk_count}
    _r.hset(REGISTRY_KEY_TMPL.format(tenant_id=tenant_id), filename, json.dumps(entry))


def list_documents(tenant_id: str) -> list[dict]:
    raw = _r.hgetall(REGISTRY_KEY_TMPL.format(tenant_id=tenant_id))
    docs = [json.loads(v) for v in raw.values()]
    docs.sort(key=lambda d: (d.get("grade") or "", d.get("subject_code") or "", d.get("chapter") or d["document_name"]))
    return docs


def remove_document(tenant_id: str, filename: str) -> None:
    _r.hdel(REGISTRY_KEY_TMPL.format(tenant_id=tenant_id), filename)


def backfill_registry_from_pinecone(tenant_id: str) -> dict:
    """
    Rebuilds missing registry entries by reading ground truth directly out
    of Pinecone, for documents that were ingested before this Redis
    registry existed (or any time Redis itself was flushed/redeployed
    without its data persisting) and therefore never got a
    `register_document()` call — those documents' chunks are fully present
    and searchable in Pinecone, just invisible to the chapter picker, which
    only ever reads Redis.

    Cost profile: there is no "list distinct metadata values" query in
    Pinecone (see the module docstring above), so recovering every
    filename requires reading `source_document` off every vector at least
    once. This is done via `index.list()` (ID-only, paginated, cheap) to
    get every vector ID in the tenant's namespace, then `index.fetch()` in
    batches of 100 IDs per call to pull metadata — for a corpus of ~900
    chunks that's ~9 fetch calls total, a one-time bounded cost, not
    proportional to query volume. No embedding model call and no Groq
    call happen anywhere in this function, so it costs nothing on the
    paid-API budget the rest of this upgrade is trying to protect.

    Idempotent: documents already in the registry are left completely
    untouched, so this is safe to run repeatedly (e.g. as an admin "resync"
    button after every bulk upload, or on a schedule).
    """
    from app.core.pinecone_client import get_index

    index = get_index()
    existing = {d["document_name"] for d in list_documents(tenant_id)}

    chunk_counts: dict[str, int] = {}

    def _fetch_batch(ids: list[str]) -> None:
        if not ids:
            return
        fetched = index.fetch(ids=ids, namespace=tenant_id)
        records = fetched.get("vectors", {}) if isinstance(fetched, dict) else fetched.vectors
        for rec in records.values():
            meta = (rec.get("metadata", {}) if isinstance(rec, dict) else rec.metadata) or {}
            doc_name = meta.get("source_document")
            if doc_name:
                chunk_counts[doc_name] = chunk_counts.get(doc_name, 0) + 1

    batch: list[str] = []
    for page in index.list(namespace=tenant_id):
        ids = page if isinstance(page, list) else getattr(page, "ids", page)
        for vec_id in ids:
            batch.append(vec_id)
            if len(batch) >= 100:
                _fetch_batch(batch)
                batch = []
    _fetch_batch(batch)

    backfilled = []
    for doc_name, count in chunk_counts.items():
        if doc_name in existing:
            continue
        register_document(tenant_id, doc_name, chunk_count=count)
        backfilled.append(doc_name)

    return {
        "backfilled_documents": sorted(backfilled),
        "already_registered": sorted(existing),
        "total_distinct_documents_in_pinecone": len(chunk_counts),
    }
