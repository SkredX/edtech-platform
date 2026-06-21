from fastapi import APIRouter, Depends
from groq import Groq
from pydantic import BaseModel

from app.chatbot.guardrails import escalation_response, should_escalate
from app.config import settings
from app.core.cache import get_cached, set_cached
from app.core.retrieval import retrieve_context
from app.tenants.auth import get_tenant_id
import logging

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/chat", tags=["chatbot"])
_groq = Groq(api_key=settings.GROQ_API_KEY)

SYSTEM_PROMPT = """You are a study assistant for this institute only. Answer
ONLY using the provided context chunks. If the answer is not in the context,
say the material does not cover this and offer to escalate to a teacher.
Always cite the source page/topic in your answer."""


class ChatRequest(BaseModel):
    message: str
    document_name: str | None = None  # deprecated single-chapter filter, kept for back-compat
    document_names: list[str] | None = None  # set when the chapter picker scopes the query to one or more chapters


class ChatResponse(BaseModel):
    answer: str
    escalate: bool
    from_cache: bool


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, tenant_id: str = Depends(get_tenant_id)):
    # Normalize both the legacy single-doc field and the new multi-doc field
    # into one list so retrieval only has one shape to deal with.
    doc_names = req.document_names or ([req.document_name] if req.document_name else None)

    # Cache key includes the document filter (sorted so selection order
    # doesn't fragment the cache) — a chapter-scoped answer and an
    # unscoped answer to the same question aren't necessarily the same.
    doc_key = ",".join(sorted(doc_names)) if doc_names else ""
    cache_key = f"{req.message}|docs={doc_key}"
    cached = get_cached(tenant_id, cache_key, scope="chat")
    if cached:
        return {**cached, "from_cache": True}

    chunks, confidence = retrieve_context(
        tenant_id, req.message, top_k=5, document_names=doc_names
    )
    logger.info(
        "chat query tenant=%s docs=%s confidence=%.4f threshold=%.4f matched_chunks=%d",
        tenant_id, doc_names, confidence, settings.CHAT_CONFIDENCE_THRESHOLD, len(chunks),
    )

    if not chunks or should_escalate(confidence):
        return escalation_response()

    context_block = "\n\n".join(
        f"[{c.get('semantic_topic', 'Context')} | p.{c.get('source_page', '?')}] {c.get('text', '')}"
        for c in chunks
    )

    resp = _groq.chat.completions.create(
        model=settings.GROQ_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {req.message}"},
        ],
        temperature=0.2,
        max_completion_tokens=500,
    )
    answer = resp.choices[0].message.content

    result = {"answer": answer, "escalate": False}
    set_cached(tenant_id, cache_key, scope="chat", response=result)
    return {**result, "from_cache": False}
