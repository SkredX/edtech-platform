"""
Parses a raw, natively formatted seed MCQ string into stem / tag / options,
and extracts a lightweight query surface for retrieval. Pure regex/string
logic — zero API cost. Direct port of the Colab CloneGen notebook's
Module 2, unchanged in behavior.
"""
import re

_TAG_RE = re.compile(r"\[([^\[\]]+)\]")
_OPTION_RE = re.compile(r"\(?\*?\s*([A-Da-d])\)\s*([^()]+?)(?=\s*\(?\*?\s*[A-Da-d]\)|\s*$)")
_ANSWER_RE = re.compile(r"(?:answer|ans)\s*[:\-]?\s*\(?([A-Da-d])\)?", re.IGNORECASE)

_STOPWORDS = {
    "a", "an", "the", "is", "are", "which", "what", "of", "in", "on",
    "for", "to", "and", "or", "with", "that", "this", "by", "as", "be",
}


def parse_seed_question(raw: str) -> dict:
    """
    Parses a string like:
      "A specialised membranous structure ... is: [NEET 2025]
       (A) Cristae (B) Endoplasmic Reticulum (C) Mesosome (D) Chromatophores"
    Returns: {stem, tag, options: {A:.., B:.., C:.., D:..}, marked_correct or None}

    The correct answer is recovered from an inline marker if present
    (e.g. "*C) ..." or trailing "Answer: C"); otherwise the caller should
    fall back to an explicit `manual_correct` field, since the raw option
    list alone doesn't reliably encode which option is right.
    """
    text = raw.strip()

    tag_match = _TAG_RE.search(text)
    tag = tag_match.group(1).strip() if tag_match else ""

    text_wo_tag = _TAG_RE.sub("", text).strip()

    marked_correct = None
    ans_match = _ANSWER_RE.search(text_wo_tag)
    if ans_match:
        marked_correct = ans_match.group(1).upper()
        text_wo_tag = _ANSWER_RE.sub("", text_wo_tag).strip()

    first_opt = re.search(r"\(?\*?\s*A\)\s*", text_wo_tag)
    if first_opt:
        stem = text_wo_tag[: first_opt.start()].strip()
        opt_block = text_wo_tag[first_opt.start():]
    else:
        stem, opt_block = text_wo_tag, ""

    options: dict[str, str] = {}
    for m in _OPTION_RE.finditer(opt_block):
        letter = m.group(1).upper()
        value = m.group(2).strip().rstrip(",.;")
        if "*" in opt_block[max(0, m.start() - 2): m.start()]:
            marked_correct = letter
        options[letter] = value

    return {
        "stem": stem.rstrip(": ").strip(),
        "tag": tag,
        "options": options,
        "marked_correct": marked_correct,
    }


def extract_core_topic(stem: str, options: dict) -> str:
    """
    Lightweight, deterministic topic extraction (no API call): takes the
    stem's content words plus all option terms as the query surface used
    for retrieval against the vector index / TF-IDF re-ranker.
    """
    words = re.findall(r"[A-Za-z]{3,}", stem.lower())
    keywords = [w for w in words if w not in _STOPWORDS]
    option_terms = list(options.values())
    return " ".join(keywords[:25] + option_terms)
