"""
Direct port of Cell 3 + Cell 4 from the original Colab RAG pipeline:

  1. Local TF-IDF keyword extraction        (zero API cost)
  2. Local spaCy NER                         (zero API cost)
  3. Local MiniLM embeddings + KMeans        (zero API cost)
  4. Groq call on cluster CENTROIDS ONLY     (the only paid step here)
  5. Broadcast centroid topic/summary back to every chunk in its cluster

This is the one-time, per-document cost — separate from the per-request
chat/clonegen budget. For a 300-page doc with ~1500 chunks clustered into
~60 clusters, this is ~60 Groq calls total, not 1500.
"""
import json
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

TARGET_CHUNKS_PER_CLUSTER = 20
MIN_CLUSTERS = 5
MAX_CLUSTERS = 150  # hard ceiling caps worst-case Groq calls per ingest

CENTROID_PROMPT_TEMPLATE = """You are analyzing a representative excerpt from a larger document section.
Based ONLY on the text below, respond in strict JSON with exactly these two keys:
- "topic": a concise 3-6 word title for this section's subject matter
- "summary": a 1-2 sentence context summary of what this section covers

Text excerpt:
\"\"\"
{chunk_text}
\"\"\"

Respond with ONLY the JSON object, no markdown formatting, no extra commentary."""


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
    estimated_k = max(MIN_CLUSTERS, min(min(MAX_CLUSTERS, n_chunks), n_chunks // TARGET_CHUNKS_PER_CLUSTER or 1))

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


def _call_groq_for_centroid(client, chunk_text: str, max_retries: int = 3) -> tuple[str, str]:
    prompt = CENTROID_PROMPT_TEMPLATE.format(chunk_text=chunk_text[:3000])

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=settings.GROQ_INGEST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_completion_tokens=200,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
            parsed = json.loads(raw)
            return parsed.get("topic", "Unknown Topic").strip(), parsed.get("summary", "").strip()
        except json.JSONDecodeError:
            if attempt == max_retries:
                return "Unparsed Topic", ""
        except Exception:
            if attempt == max_retries:
                return "API Error", ""
            time.sleep(2 ** attempt)
    return "Unknown Topic", ""


def cluster_and_summarize(raw_chunks: list[dict]) -> tuple[list[dict], np.ndarray]:
    """
    Input: [{"text": ..., "source_page": ...}, ...]
    Output: (chunks, embeddings) — chunks enriched with tfidf_keywords,
            entities, cluster_id, is_cluster_centroid, semantic_topic,
            context_summary; embeddings aligned 1:1 with chunks so the
            caller doesn't need to re-embed the same text a second time.
    Only one Groq call per cluster (centroid), not per chunk.
    """
    from groq import Groq

    chunks = [dict(c) for c in raw_chunks]  # don't mutate caller's list

    _add_tfidf_keywords(chunks)
    _add_entities(chunks)
    labels, centroid_idx, embeddings = _cluster_chunks(chunks)

    client = Groq(api_key=settings.GROQ_API_KEY)
    cluster_metadata: dict[int, dict] = {}
    for cid, idx in sorted(centroid_idx.items()):
        topic, summary = _call_groq_for_centroid(client, chunks[idx]["text"])
        cluster_metadata[cid] = {"topic": topic, "summary": summary}
        time.sleep(0.3)  # gentle pacing for Groq rate limits

    for i, c in enumerate(chunks):
        cid = c["cluster_id"]
        meta = cluster_metadata.get(cid, {"topic": "Unknown", "summary": ""})
        c["semantic_topic"] = meta["topic"]
        c["context_summary"] = meta["summary"]
        c["is_cluster_centroid"] = centroid_idx.get(cid) == i
        c["char_count"] = len(c["text"])

    return chunks, embeddings
