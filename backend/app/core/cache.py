"""
Semantic cache shared by /chat and /clone. Before any Groq call, we check
whether a sufficiently similar query has already been answered (cosine
similarity over the local embedding — no API call involved) and reuse the
cached response instead. This is the layer that lets the chatbot and
CloneGen share API budget instead of each burning its own.

Keys are namespaced per tenant AND per scope ("chat" / "clone") so one
institute's cached answers never leak into another's, and a chat answer
is never returned for a clonegen request or vice versa.

--- OPTIMISATION vs previous version ---

Previously, get_cached() called embed_query() to get the query vector for
similarity comparison, then set_cached() called embed_query() AGAIN on the
exact same string to store it.  That was two Pinecone Inference calls per
cache-miss chat turn — one to check, one to write — even though the vector
is mathematically identical.

Now get_cached() returns the query vector it computed alongside the
response (or None), and the router passes it directly into set_cached()
so the embedding is computed exactly once per cache interaction.

API call reduction per chat turn (cache miss):
  OLD: 2 Pinecone embed calls  (get_cached + set_cached both embed)
  NEW: 1 Pinecone embed call   (get_cached embeds once, passes vector to set_cached)

API call reduction per chat turn (cache hit):
  Unchanged — get_cached still embeds once to do the similarity check, and
  set_cached is never called, so this path was already optimal.

The public signatures of get_cached and set_cached change slightly:
  get_cached now returns (response | None, query_vector)
  set_cached now accepts an optional pre_computed_vector kwarg

Callers that ignore the vector still compile — set_cached re-embeds if
pre_computed_vector is None, preserving backward compatibility with any
future caller that doesn't thread the vector through.
"""
import hashlib
import json
from typing import Any

import numpy as np
import redis

from app.config import settings
from app.core.embeddings import embed_query

_r = redis.from_url(settings.REDIS_URL, decode_responses=True)

CACHE_PREFIX = "semcache"
INDEX_KEY_TMPL = "semcache:index:{tenant_id}:{scope}"  # redis SET of member keys


def _cosine(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)) + 1e-9
    return float(np.dot(a_arr, b_arr) / denom)


def _entry_key(tenant_id: str, scope: str, query: str) -> str:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"{CACHE_PREFIX}:{tenant_id}:{scope}:{digest}"


def get_cached(
    tenant_id: str,
    query: str,
    scope: str,
) -> tuple[dict | None, list[float]]:
    """
    Returns (cached_response_or_None, query_embedding_vector).

    The caller MUST always receive and thread the returned vector into
    set_cached() on a cache miss — this is what eliminates the second
    embed_query call.  The vector is always returned (never None) so the
    caller can use it unconditionally without an isinstance check.

    Previous callers that treated the return value as a single dict/None
    will need updating — see chatbot/router.py and clonegen/router.py for
    the updated call-sites.
    """
    index_key = INDEX_KEY_TMPL.format(tenant_id=tenant_id, scope=scope)
    member_keys = _r.smembers(index_key)

    # Always embed — we need the vector whether this is a hit or miss.
    q_vec = embed_query(query)

    if not member_keys:
        return None, q_vec

    best_sim, best_response = 0.0, None

    for key in member_keys:
        raw = _r.get(key)
        if raw is None:
            # Expired entry — clean the stale index reference.
            _r.srem(index_key, key)
            continue
        cached = json.loads(raw)
        sim = _cosine(q_vec, cached["vector"])
        if sim > best_sim:
            best_sim, best_response = sim, cached["response"]

    if best_sim >= settings.CACHE_SIMILARITY_THRESHOLD:
        return best_response, q_vec

    return None, q_vec


def set_cached(
    tenant_id: str,
    query: str,
    scope: str,
    response: dict[str, Any],
    ttl_seconds: int = 86400,
    pre_computed_vector: list[float] | None = None,
) -> None:
    """
    Stores a query-response pair in the semantic cache.

    Pass `pre_computed_vector` (the second element returned by get_cached)
    to avoid re-embedding the query string.  If omitted or None, the
    function falls back to calling embed_query() itself — this preserves
    backward compatibility but wastes one Pinecone Inference call.
    """
    q_vec = pre_computed_vector if pre_computed_vector is not None else embed_query(query)
    entry_key = _entry_key(tenant_id, scope, query)
    index_key = INDEX_KEY_TMPL.format(tenant_id=tenant_id, scope=scope)

    _r.setex(entry_key, ttl_seconds, json.dumps({"vector": q_vec, "response": response}))
    _r.sadd(index_key, entry_key)
    _r.expire(index_key, ttl_seconds)