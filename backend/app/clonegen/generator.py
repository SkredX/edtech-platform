"""
Batched Groq clone generation — one API call produces n clones, regardless
of how many the user requests, keeping cost flat per generation request
rather than scaling with n. Direct port of the Colab CloneGen notebook's
Module 4 (the corrected, batched version), adapted to return plain dicts
instead of HTML since rendering now happens in the Next.js frontend.
"""
import json
import time

from app.config import settings

CLONE_PROMPT_TEMPLATE = """You are an expert NEET/JEE question setter writing original
exam-quality multiple-choice questions for a question bank.

SEED QUESTION (for concept reference only — do NOT reuse its wording or its scenario):
"{seed_stem}"
Seed correct answer: {seed_correct}
Subject tag: {tag}

SOURCE CONTEXT (verified material — every fact you use must be grounded in this):
\"\"\"
{context}
\"\"\"

TASK: Write {n} DISTINCT cloned questions that test the exact same underlying
concept as the seed question, following these rules strictly:

1. Each clone must test the SAME core concept/principle as the seed, but must NOT
   be a synonym-swap or word-rearrangement of the seed. Change the framing: use a
   different scenario, a different specific example from the context, a different
   angle of questioning (e.g. "which of these is NOT...", an application scenario,
   a comparison, a cause-effect framing), while still requiring the same conceptual
   understanding to answer correctly.
2. All three distractors per question must be genuinely plausible — drawn from
   related concepts, sibling terms, or commonly confused ideas actually present in
   the source context, not arbitrary filler.
3. Each clone must be conceptually distinct from the others in this batch (no two
   clones should be trivial rewordings of each other either).
4. Ground every explanation in the provided context — cite the relevant fact, don't
   just restate the answer.

Respond with ONLY a JSON object, no markdown fences, no commentary, in exactly this shape:
{{
  "clones": [
    {{
      "question": "...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct": "A",
      "explanation": "..."
    }}
  ]
}}
The "clones" array must contain exactly {n} items."""

MAX_CONTEXT_CHARS = 2400


def call_groq_clones(
    api_key: str,
    seed_stem: str,
    seed_correct: str,
    tag: str,
    context: str,
    n: int,
    max_retries: int = 3,
) -> dict:
    from groq import Groq

    client = Groq(api_key=api_key)
    prompt = CLONE_PROMPT_TEMPLATE.format(
        seed_stem=seed_stem,
        seed_correct=seed_correct,
        tag=tag or "General",
        context=context[:MAX_CONTEXT_CHARS],
        n=n,
    )

    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.GROQ_CLONE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_completion_tokens=400 * max(n, 1) + 200,
                response_format={"type": "json_object"},
            )
            raw_text = resp.choices[0].message.content.strip()
            parsed = json.loads(raw_text)
            clones = parsed.get("clones", [])
            if not clones:
                raise ValueError("Model returned an empty clone list.")

            cleaned = []
            for c in clones:
                opts = c.get("options", {})
                if not all(k in opts for k in "ABCD") or c.get("correct") not in "ABCD":
                    continue  # skip malformed entries rather than failing the whole batch
                cleaned.append(
                    {
                        "question": c["question"],
                        "options": opts,
                        "correct": c["correct"],
                        "explanation": c.get("explanation", ""),
                    }
                )

            if not cleaned:
                raise ValueError("All returned clones were malformed.")
            return {"clones": cleaned, "api_calls_made": 1}

        except json.JSONDecodeError as e:
            last_err = f"JSON parse failure: {e}"
        except Exception as e:
            last_err = str(e)
        time.sleep(2 ** attempt)

    return {"error": f"Groq generation failed after {max_retries} attempts: {last_err}"}
