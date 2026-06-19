from fastapi import APIRouter, Depends
from groq import Groq
from pydantic import BaseModel

from app.chatbot.guardrails import escalation_response, should_escalate
from app.config import settings
from app.core.cache import get_cached, set_cached
from app.core.retrieval import retrieve_context
from app.tenants.auth import get_tenant_id

router = APIRouter(prefix="/chat", tags=["chatbot"])
_groq = Groq(api_key=settings.GROQ_API_KEY)

SYSTEM_PROMPT = """You are a study assistant for this institute only. Answer
ONLY using the provided context chunks. If the answer is not in the context,
say the material does not cover this and offer to escalate to a teacher.
Always cite the source page/topic in your answer."""


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    escalate: bool
    from_cache: bool


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, tenant_id: str = Depends(get_tenant_id)):
    cached = get_cached(tenant_id, req.message, scope="chat")
    if cached:
        return {**cached, "from_cache": True}

    chunks, confidence = retrieve_context(tenant_id, req.message, top_k=5)

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
    set_cached(tenant_id, req.message, scope="chat", response=result)
    return {**result, "from_cache": False}
