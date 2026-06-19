"""
Minimal tenant auth: each institute gets an API key, mapped to a tenant_id
that's used as the Pinecone namespace and the cache key prefix. Swap this
for proper JWT/OAuth later — the contract (`get_tenant_id` returning a
trusted string) is what the rest of the app depends on, so the internals
can change without touching routers.
"""
from functools import lru_cache

from fastapi import Header, HTTPException

from app.config import settings


@lru_cache(maxsize=1)
def _key_map() -> dict[str, str]:
    mapping = {}
    for pair in settings.TENANT_API_KEYS.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        key, tenant_id = pair.split(":", 1)
        mapping[key.strip()] = tenant_id.strip()
    return mapping


def get_tenant_id(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    mapping = _key_map()
    tenant_id = mapping.get(x_api_key)
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
    return tenant_id
