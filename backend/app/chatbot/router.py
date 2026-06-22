import logging

from fastapi import APIRouter, Depends
from groq import Groq
from pydantic import BaseModel

from app.chatbot.guardrails import escalation_response, should_escalate
from app.config import settings
from app.core.cache import get_cached, set_cached
from app.core.retrieval import compress_chunk_text, retrieve_context
from app.tenants.auth import get_tenant_id

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/chat", tags=["chatbot"])
_groq = Groq(api_key=settings.GROQ_API_KEY)

# Structure is mandatory, not a suggestion — a weak/optional instruction
# here is what produced the wall-of-text answers this prompt replaces.
# Concrete Markdown rules + a worked example get far more consistent
# formatting out of small/fast Groq models than abstract instructions like
# "format your answer nicely" do.
SYSTEM_PROMPT = """You are a study assistant for this institute only. Answer \
ONLY using the provided context chunks. If the answer is not in the context, \
say the material does not cover this and offer to escalate to a teacher.

Formatting rules — follow these for every answer, no exceptions:
1. Open with a one-sentence direct answer to the question.
2. If the answer has more than one part (a definition with several \
features, a comparison, a list of examples, steps), break it into short \
Markdown bullet points (`- `). Never write more than 2 sentences of prose \
back-to-back — convert to bullets instead.
3. For comparisons between two or more things, use a "**Term**" bolded \
sub-heading for each thing being compared, with its own bullet list \
underneath — never interleave both things in the same paragraph.
4. Bold (`**term**`) the key vocabulary the student is expected to learn, \
the first time it appears only.
5. Keep the whole answer under 180 words unless the question explicitly \
asks for a long-form explanation.
6. End with exactly one line in this exact format, nothing after it: \
`Sources: "<topic> | p.<page>"` (comma-separate if multiple).
7. Do not repeat the student's question back to them. Do not restate \
previous turns. Answer only the current question.

Example of the expected shape:
**Photosynthesis** is the process plants use to convert light energy into \
chemical energy.
- Occurs mainly in the **chloroplasts** of leaf cells.
- Requires sunlight, water, and carbon dioxide as inputs.
- Produces glucose and oxygen as outputs.
Sources: "Photosynthesis Overview | p.12\""""


class ChatRequest(BaseModel):
    message: str
    document_name: str | None = None  # deprecated single-chapter filter, kept for back-compat
    document_names: list[str] | None = None  # set when the chapter picker scopes the query


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

    # get_cached now returns (response_or_None, query_vector).
    # The query vector is always computed here (one Pinecone embed call) and
    # passed into set_cached on a miss, so set_cached never needs to re-embed
    # the same string — saving one Pinecone Inference call per cache-miss turn.
    cached_response, q_vec = get_cached(tenant_id, cache_key, scope="chat")
    if cached_response is not None:
        return {**cached_response, "from_cache": True}

    # rerank=True: pulls a wider local candidate pool and re-scores it with
    # TF-IDF before picking the final top_k — costs zero extra API calls
    # (Pinecone query stays a single round trip) but materially improves
    # which 5 chunks make it into the Groq prompt.
    chunks, confidence = retrieve_context(
        tenant_id, req.message, top_k=5, rerank=True, document_names=doc_names
    )
    all_scores = [c.get("_retrieval_score", confidence) for c in chunks] or [confidence]

    logger.info(
        "chat query tenant=%s docs=%s confidence=%.4f threshold=%.4f matched_chunks=%d",
        tenant_id, doc_names, confidence, settings.CHAT_CONFIDENCE_THRESHOLD, len(chunks),
    )

    if not chunks or should_escalate(confidence, all_scores):
        return escalation_response()

    # Extractive compression (local TF-IDF sentence scoring, see
    # core/retrieval.py) shrinks each chunk to its most query-relevant
    # sentences before it goes into the prompt — cuts Groq input tokens on
    # long chunks without an extra API call.
    context_block = "\n\n".join(
        f"[{c.get('semantic_topic', 'Context')} | p.{c.get('source_page', '?')}] "
        f"{compress_chunk_text(req.message, c.get('text', ''))}"
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

    # Pass the pre-computed query vector so set_cached doesn't embed again.
    set_cached(tenant_id, cache_key, scope="chat", response=result, pre_computed_vector=q_vec)

    return {**result, "from_cache": False}