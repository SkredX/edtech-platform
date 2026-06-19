"""
Semantic cache shared by /chat and /clone. Before any Groq call, we check
whether a sufficiently similar query has already been answered (cosine sim
over the local embedding, not an API call) and reuse the cached response.
This is what lets the chatbot and CloneGen share API budget instead of
each burning their own.
"""
import json, redis, numpy as np
from app.config import settings
from app.core.embeddings import embed_query

_r = redis.from_url(settings.REDIS_URL, decode_responses=True)
CACHE_PREFIX = "semcache:"

def _cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def get_cached(query: str, scope: str) -> dict | None:
    """scope = 'chat' or 'clone' — keeps the two response shapes separate
    while still sharing the same Redis instance and lookup logic."""
    q_vec = embed_query(query)
    keys = _r.keys(f"{CACHE_PREFIX}{scope}:*")
    for key in keys:
        cached = json.loads(_r.get(key))
        sim = _cosine(q_vec, cached["vector"])
        if sim >= settings.CACHE_SIMILARITY_THRESHOLD:
            return cached["response"]
    return None

def set_cached(query: str, scope: str, response: dict, ttl_seconds: int = 86400):
    q_vec = embed_query(query)
    key = f"{CACHE_PREFIX}{scope}:{hash(query)}"
    _r.setex(key, ttl_seconds, json.dumps({"vector": q_vec, "response": response}))