"""
Confidence guardrail for the chatbot: if Pinecone's top match score is
below threshold, the question almost certainly isn't covered by the
institute's ingested material — answering anyway risks hallucination, so
we escalate to a human instead of spending a Groq call on a low-confidence
answer. This also saves cost: no LLM call happens on a guardrail trip.
"""
from app.config import settings

ESCALATION_MESSAGE = (
    "This doesn't seem to be covered in your course material. "
    "Want me to forward this question to your teacher?"
)


def should_escalate(confidence: float) -> bool:
    return confidence < settings.CHAT_CONFIDENCE_THRESHOLD


def escalation_response() -> dict:
    return {
        "answer": ESCALATION_MESSAGE,
        "escalate": True,
        "from_cache": False,
    }
