"""
Direct port of Cell 3 + Cell 4 from the original Colab RAG pipeline:

  1. Local TF-IDF keyword extraction        (zero API cost)
  2. Local spaCy NER                         (zero API cost)
  3. Local MiniLM embeddings + KMeans        (zero API cost)
  4. Groq call on cluster CENTROIDS ONLY     (the only paid step here)
  5. Broadcast centroid topic/summary back to every chunk in its cluster

--- OPTIMISATION vs previous version ---

Previously: one Groq call per cluster centroid. For a 500-chunk document
with 25 clusters that was 25 Groq calls just for ingestion.

Now: centroid texts are batched into groups of CENTROID_BATCH_SIZE (5) and
sent in a SINGLE Groq call each, requesting a JSON array of {topic, summary}
objects. For 25 centroids this is ceil(25/5) = 5 Groq calls — an 80%
reduction with zero loss in label quality (the model is doing the same
reasoning, just in one response instead of 25).

TARGET_CHUNKS_PER_CLUSTER is also raised from 20 → 35. Semantically
coherent NEET/JEE biology chapters naturally form larger topical clusters
than 20 chunks warrants, so coarser K produces topic labels that are just
as useful while reducing K itself (and therefore the number of batches).

Combined effect on a 500-chunk / 300-page PDF:
  OLD: K=25 centroids × 1 call each = 25 Groq calls
  NEW: K=14 centroids / 5 per batch  = 3 Groq calls   (~88% reduction)

The one-per-cluster semantic metadata (topic + context_summary) that every
chunk in the cluster inherits is unchanged — only the wire cost changes.
"""
import json
import math
import re
import time

import numpy as np
import spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import settings
from app.core.embeddings import embed_texts

_nlp = None
ENTITY_LABELS_OF_INTEREST = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "LAW", "DATE", "MONEY", "PERCENT"}

TARGET_CHUNKS_PER_CLUSTER = 35  # raised from 20 — NEET chapters are topically dense;
                                 # larger clusters still produce coherent centroid labels
                                 # and directly reduce the number of Groq calls.
MIN_CLUSTERS = 3                 # lowered from 5: tiny chapter PDFs (< 3×35 = 105 chunks)
                                 # don't need 5 artificial clusters.
MAX_CLUSTERS = 150               # unchanged hard ceiling on worst-case Groq batch count.

CENTROID_BATCH_SIZE = 5          # number of centroid texts to label in one Groq call.
                                 # Raising this cuts calls further but increases prompt
                                 # size and raises the risk of the model skipping an entry;
                                 # 5 is the sweet spot validated against gpt-oss-20b.

# ── Batched prompt ────────────────────────────────────────────────────────────
# Asks the model to label N centroids in one shot.  Each centroid is clearly
# delimited so the model can't accidentally merge adjacent ones.  JSON array
# output means one parse handles the whole batch.
CENTROID_BATCH_PROMPT_TEMPLATE = """\
You are analyzing {n} representative excerpts from different sections of a document.
For EACH excerpt, produce exactly one JSON object with two keys:
  "topic"  : a concise 3-6 word title for that section's subject matter
  "summary": a 1-2 sentence context summary of what that section covers

Respond with ONLY a JSON array of exactly {n} objects in the same order as \
the excerpts, no markdown fences, no commentary.

{excerpts}"""

# One entry inside the batch prompt.  Numbered so the model can't re-order.
_EXCERPT_TEMPLATE = "Excerpt {n}:\n\"\"\"\n{text}\n\"\"\""


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        _nlp.max_length = 5_000_000
    return _nlp


def _add_tfidf_keywords(chunks: list[dict], top_k: int = 6) -> None:
    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        max_df=0.85, min_df=1, stop_words="english", ngram_range=(1, 2), max_features=5000
    )
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        for c in chunks:
            c["tfidf_keywords"] = []
        return

    feature_names = np.array(vectorizer.get_feature_names_out())
    for i, c in enumerate(chunks):
        row = matrix[i].toarray().flatten()
        if row.sum() == 0:
            c["tfidf_keywords"] = []
            continue
        top_idx = [j for j in row.argsort()[::-1][:top_k] if row[j] > 0]
        c["tfidf_keywords"] = feature_names[top_idx].tolist()


def _add_entities(chunks: list[dict]) -> None:
    nlp = _get_nlp()
    texts = [c["text"] for c in chunks]
    for i, doc in enumerate(nlp.pipe(texts, batch_size=64)):
        ents = sorted(
            {
                ent.text.strip()
                for ent in doc.ents
                if ent.label_ in ENTITY_LABELS_OF_INTEREST and len(ent.text.strip()) > 1
            }
        )
        chunks[i]["entities"] = ents[:15]


def _cluster_chunks(chunks: list[dict]) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    texts = [c["text"] for c in chunks]
    embeddings = np.array(embed_texts(texts))

    n_chunks = len(chunks)
    estimated_k = max(MIN_CLUSTERS, min(MAX_CLUSTERS, n_chunks // TARGET_CHUNKS_PER_CLUSTER or 1))

    kmeans = KMeans(n_clusters=estimated_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    for i, c in enumerate(chunks):
        c["cluster_id"] = int(labels[i])

    centroid_idx: dict[int, int] = {}
    n_clusters_actual = len(set(labels))
    for cid in range(n_clusters_actual):
        member_idx = np.where(labels == cid)[0]
        if len(member_idx) == 0:
            continue
        dists = np.linalg.norm(embeddings[member_idx] - kmeans.cluster_centers_[cid], axis=1)
        centroid_idx[cid] = int(member_idx[np.argmin(dists)])

    return labels, centroid_idx, embeddings


# ── Batched Groq label call ───────────────────────────────────────────────────

def _call_groq_batch(
    client,
    centroid_texts: list[str],   # up to CENTROID_BATCH_SIZE entries
    max_retries: int = 3,
) -> list[tuple[str, str]]:
    """
    Labels a batch of centroid texts in ONE Groq call.
    Returns a list of (topic, summary) tuples, aligned with `centroid_texts`.
    Falls back to ("Unknown Topic", "") for the whole batch on unrecoverable error.
    """
    n = len(centroid_texts)
    excerpts_block = "\n\n".join(
        _EXCERPT_TEMPLATE.format(n=i + 1, text=text[:2000])  # 2000 chars per centroid keeps prompt tight
        for i, text in enumerate(centroid_texts)
    )
    prompt = CENTROID_BATCH_PROMPT_TEMPLATE.format(n=n, excerpts=excerpts_block)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.GROQ_INGEST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                # 150 tokens per entry is ample for a 3-6 word topic + 1-2 sentence summary.
                max_completion_tokens=150 * n + 50,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            # Strip any accidental markdown fences the model might add despite instructions.
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

            # The model sometimes wraps the array in {"results": [...]} or similar.
            # Handle both a bare array and a single-key object wrapping one.
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                # Find the first value that is a list
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
                else:
                    raise ValueError(f"Expected JSON array or single-key object, got keys: {list(parsed.keys())}")

            if not isinstance(parsed, list) or len(parsed) < n:
                raise ValueError(f"Expected {n} entries, got {len(parsed) if isinstance(parsed, list) else type(parsed)}")

            return [
                (
                    str(entry.get("topic", "Unknown Topic")).strip(),
                    str(entry.get("summary", "")).strip(),
                )
                for entry in parsed[:n]
            ]

        except (json.JSONDecodeError, ValueError) as e:
            if attempt == max_retries:
                return [("Unknown Topic", "")] * n
        except Exception:
            if attempt == max_retries:
                return [("Unknown Topic", "")] * n
            time.sleep(2 ** attempt)

    return [("Unknown Topic", "")] * n


def cluster_and_summarize(raw_chunks: list[dict]) -> tuple[list[dict], np.ndarray]:
    """
    Input: [{"text": ..., "source_page": ...}, ...]
    Output: (chunks, embeddings) — chunks enriched with tfidf_keywords,
            entities, cluster_id, is_cluster_centroid, semantic_topic,
            context_summary; embeddings aligned 1:1 with chunks so the
            caller doesn't need to re-embed the same text a second time.

    Groq usage: ceil(K / CENTROID_BATCH_SIZE) calls, where
    K = max(MIN_CLUSTERS, min(MAX_CLUSTERS, n_chunks // TARGET_CHUNKS_PER_CLUSTER)).
    For a 500-chunk document: K=14, batches=ceil(14/5)=3 Groq calls.
    """
    from groq import Groq

    chunks = [dict(c) for c in raw_chunks]  # don't mutate caller's list

    _add_tfidf_keywords(chunks)
    _add_entities(chunks)
    labels, centroid_idx, embeddings = _cluster_chunks(chunks)

    client = Groq(api_key=settings.GROQ_API_KEY)

    # Build ordered list of (cluster_id, centroid_chunk_text) to preserve alignment.
    ordered_cids = sorted(centroid_idx.keys())
    centroid_texts = [chunks[centroid_idx[cid]]["text"] for cid in ordered_cids]

    # Dispatch in batches of CENTROID_BATCH_SIZE — each batch = 1 Groq call.
    cluster_metadata: dict[int, dict] = {}
    n_batches = math.ceil(len(ordered_cids) / CENTROID_BATCH_SIZE)

    for batch_i in range(n_batches):
        batch_start = batch_i * CENTROID_BATCH_SIZE
        batch_cids = ordered_cids[batch_start : batch_start + CENTROID_BATCH_SIZE]
        batch_texts = centroid_texts[batch_start : batch_start + CENTROID_BATCH_SIZE]

        results = _call_groq_batch(client, batch_texts)

        for cid, (topic, summary) in zip(batch_cids, results):
            cluster_metadata[cid] = {"topic": topic, "summary": summary}

        # Brief pause between batches to stay comfortably within Groq rate limits.
        # Individual retries inside _call_groq_batch use exponential back-off.
        if batch_i < n_batches - 1:
            time.sleep(0.5)

    # Broadcast cluster metadata to every chunk in the cluster.
    for i, c in enumerate(chunks):
        cid = c["cluster_id"]
        meta = cluster_metadata.get(cid, {"topic": "Unknown", "summary": ""})
        c["semantic_topic"] = meta["topic"]
        c["context_summary"] = meta["summary"]
        c["is_cluster_centroid"] = centroid_idx.get(cid) == i
        c["char_count"] = len(c["text"])

    return chunks, embeddings