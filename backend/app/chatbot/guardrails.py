"""
Confidence guardrail for the chatbot: if Pinecone's top match score is
below threshold, the question almost certainly isn't covered by the
institute's ingested material — answering anyway risks hallucination, so
we escalate to a human instead of spending a Groq call on a low-confidence
answer. This also saves cost: no LLM call happens on a guardrail trip.

`should_escalate` optionally also looks at the SPREAD of scores across all
retrieved chunks (max - mean), not just the top score in isolation — pure
local arithmetic, no extra API/embedding call since the scores are already
returned by the Pinecone query that already happened. A large spread (top
chunk clearly stands out) is evidence the corpus genuinely has a strong
match even if the absolute score is on the lower side for the embedding
model in use; a flat, low spread (nothing stands out) is stronger evidence
of "not covered" than the top score alone. This is passed in as an
optional argument so existing callers that only have a single confidence
float keep working unchanged.
"""
from app.config import settings

ESCALATION_MESSAGE = (
    "This doesn't seem to be covered in your course material. "
    "Want me to forward this question to your teacher?"
)


def should_escalate(confidence: float, all_scores: list[float] | None = None) -> bool:
    if confidence < settings.CHAT_CONFIDENCE_THRESHOLD:
        return False if _has_standout_match(confidence, all_scores) else True
    return False


def _has_standout_match(top_score: float, all_scores: list[float] | None) -> bool:
    """
    Rescues a borderline top score from escalation if it clearly stands
    out above the rest of the retrieved set (spread-based signal). Only
    kicks in within a narrow band just under the threshold, so it can't
    turn into a way to wave through genuinely irrelevant queries — it's a
    fixed margin below the cutoff, plus a real gap above the rest of the
    pack required to qualify.
    """
    if not all_scores or len(all_scores) < 2:
        return False
    margin_below_threshold = settings.CHAT_CONFIDENCE_THRESHOLD - top_score
    if margin_below_threshold > 0.05:
        return False  # too far under the threshold for a spread argument to apply

    rest = sorted(all_scores, reverse=True)[1:]
    mean_rest = sum(rest) / len(rest)
    spread = top_score - mean_rest
    return spread >= 0.12


def escalation_response() -> dict:
    return {
        "answer": ESCALATION_MESSAGE,
        "escalate": True,
        "from_cache": False,
    }
