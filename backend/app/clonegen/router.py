from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.clonegen.generator import call_groq_clones
from app.clonegen.parser import extract_core_topic, parse_seed_question
from app.config import settings
from app.core.cache import get_cached, set_cached
from app.core.retrieval import retrieve_context
from app.tenants.auth import get_tenant_id

router = APIRouter(prefix="/clone", tags=["clonegen"])


class CloneRequest(BaseModel):
    raw_seed: str
    manual_correct: str | None = None
    n_clones: int = 3


class CloneOption(BaseModel):
    question: str
    options: dict[str, str]
    correct: str
    explanation: str


class CloneResponse(BaseModel):
    clones: list[CloneOption] | None = None
    error: str | None = None
    from_cache: bool = False


@router.post("", response_model=CloneResponse)
def generate_clones(req: CloneRequest, tenant_id: str = Depends(get_tenant_id)):
    parsed = parse_seed_question(req.raw_seed)
    if not parsed["options"]:
        return {"error": "Could not detect options (A)-(D) in the seed question."}

    correct_letter = parsed["marked_correct"] or req.manual_correct
    if correct_letter not in parsed["options"]:
        return {"error": "Correct answer not detected. Mark it inline (e.g. 'Answer: C') or provide manual_correct."}

    n = max(1, min(10, req.n_clones))
    cache_key = f"{req.raw_seed}|{correct_letter}|n={n}"

    # get_cached now returns (response_or_None, query_vector).
    # Thread the vector into set_cached to avoid a second embed_query call.
    cached_response, q_vec = get_cached(tenant_id, cache_key, scope="clone")
    if cached_response is not None:
        return {**cached_response, "from_cache": True}

    query = extract_core_topic(parsed["stem"], parsed["options"])
    chunks, _ = retrieve_context(tenant_id, query, top_k=6, rerank=True)

    context_block = "\n\n".join(
        f"[{c.get('semantic_topic', 'Context')}] {c.get('text', '')}" for c in chunks
    )

    result = call_groq_clones(
        api_key=settings.GROQ_API_KEY,
        seed_stem=parsed["stem"],
        seed_correct=parsed["options"][correct_letter],
        tag=parsed["tag"],
        context=context_block,
        n=n,
    )

    if "error" not in result:
        # Pass the pre-computed vector — set_cached won't re-embed.
        set_cached(tenant_id, cache_key, scope="clone", response=result, pre_computed_vector=q_vec)

    return {**result, "from_cache": False}