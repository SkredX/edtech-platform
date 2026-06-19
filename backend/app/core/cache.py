"""
Semantic cache shared by /chat and /clone. Before any Groq call, we check
whether a sufficiently similar query has already been answered (cosine
similarity over the local embedding — no API call involved) and reuse the
cached response instead. This is the layer that lets the chatbot and
CloneGen share API budget instead of each burning its own.

Keys are namespaced per tenant AND per scope ("chat" / "clone") so one
institute's cached answers never leak into another's, and a chat answer
is never returned for a clonegen request or vice versa.
"""
import hashlib
import json

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


def get_cached(tenant_id: str, query: str, scope: str) -> dict | None:
    index_key = INDEX_KEY_TMPL.format(tenant_id=tenant_id, scope=scope)
    member_keys = _r.smembers(index_key)
    if not member_keys:
        return None

    q_vec = embed_query(query)
    best_sim, best_response = 0.0, None

    for key in member_keys:
        raw = _r.get(key)
        if raw is None:
            # Expired entry — clean the stale index reference
            _r.srem(index_key, key)
            continue
        cached = json.loads(raw)
        sim = _cosine(q_vec, cached["vector"])
        if sim > best_sim:
            best_sim, best_response = sim, cached["response"]

    if best_sim >= settings.CACHE_SIMILARITY_THRESHOLD:
        return best_response
    return None


def set_cached(tenant_id: str, query: str, scope: str, response: dict, ttl_seconds: int = 86400) -> None:
    q_vec = embed_query(query)
    entry_key = _entry_key(tenant_id, scope, query)
    index_key = INDEX_KEY_TMPL.format(tenant_id=tenant_id, scope=scope)

    _r.setex(entry_key, ttl_seconds, json.dumps({"vector": q_vec, "response": response}))
    _r.sadd(index_key, entry_key)
    _r.expire(index_key, ttl_seconds)
