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
