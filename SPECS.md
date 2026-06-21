# SkredX EdTech RAG Platform — Technical Specifications

> **Scope:** Full stack specification covering every technology, library, algorithm, and design decision in the platform, with precise assignment to the layer or module where each is used.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Backend Tech Stack](#3-backend-tech-stack)
4. [Frontend Tech Stack](#4-frontend-tech-stack)
5. [Infrastructure & Deployment](#5-infrastructure--deployment)
6. [Data Pipeline — Document Ingestion](#6-data-pipeline--document-ingestion)
7. [Retrieval System](#7-retrieval-system)
8. [Chatbot — Doubt Assistant](#8-chatbot--doubt-assistant)
9. [CloneGen — MCQ Generation](#9-clonegen--mcq-generation)
10. [Semantic Cache](#10-semantic-cache)
11. [Document Registry & Chapter Picker](#11-document-registry--chapter-picker)
12. [Multi-Tenancy & Auth](#12-multi-tenancy--auth)
13. [Cost & API Call Optimisation](#13-cost--api-call-optimisation)
14. [Platform Features & Benefits](#14-platform-features--benefits)
15. [Configuration Reference](#15-configuration-reference)

---

## 1. System Overview

SkredX is a curriculum-aligned RAG (Retrieval-Augmented Generation) platform built for NEET/JEE exam preparation. It has three user-facing capabilities:

**Doubt Assistant (`/chat`)** — A student pastes or types a doubt. The system retrieves the most relevant passages from the institute's own uploaded course material and generates a structured answer grounded exclusively in that material. If the question is not covered, it escalates to a teacher rather than hallucinating.

**CloneGen (`/admin/clonegen`)** — A teacher pastes a seed MCQ from any question bank. The system parses it, retrieves semantically relevant context from the ingested material, and generates N conceptually equivalent but distinct clone questions in a single LLM call, grounded in verified course content.

**Document Ingestion (`/admin/ingest`)** — A teacher uploads a PDF. The system chunks it, enriches every chunk with TF-IDF keywords and named entities, clusters chunks by semantic similarity, generates one topic label and summary per cluster (not per chunk), embeds everything, and upserts into Pinecone. All expensive steps are one-time per document.

---

## 2. Architecture Diagram

```
Student / Teacher Browser
        │
        │  HTTPS
        ▼
┌───────────────────────────────────┐
│   Next.js 14 Frontend (Vercel)    │
│   /          Landing page         │
│   /chat       Doubt Assistant     │
│   /admin      Teacher Dashboard   │
│   /admin/ingest   PDF Upload      │
│   /admin/clonegen MCQ Generator   │
└──────────────┬────────────────────┘
               │  REST (X-API-Key header)
               ▼
┌───────────────────────────────────┐
│   FastAPI Backend (Render / Docker)│
│                                   │
│  POST /ingest        Ingestion    │
│  GET  /ingest/documents  Registry │
│  POST /ingest/backfill-registry   │
│  POST /chat          Chatbot      │
│  POST /clone         CloneGen     │
│  GET  /health        Healthcheck  │
└──────┬──────────────┬─────────────┘
       │              │
       ▼              ▼
┌──────────┐   ┌──────────────────────┐
│  Redis   │   │  Pinecone (Serverless)│
│          │   │                      │
│ Semantic │   │  Vector index        │
│  cache   │   │  (cosine, dim=384)   │
│          │   │                      │
│ Document │   │  Namespaced per      │
│ registry │   │  tenant_id           │
└──────────┘   └──────────────────────┘
                         │
               ┌─────────┴──────────┐
               │  Pinecone Inference │
               │  llama-text-embed-v2│
               │  (hosted embedding) │
               └────────────────────┘
                         │ (only for LLM calls)
               ┌─────────┴──────────┐
               │  Groq API          │
               │  (LLM completions) │
               └────────────────────┘
```

---

## 3. Backend Tech Stack

### Runtime & Framework

| Technology | Version | Role |
|---|---|---|
| Python | 3.12 | Primary backend language |
| FastAPI | 0.137.2 | REST API framework; async-native, auto OpenAPI docs |
| Uvicorn | 0.49.0 | ASGI server (production: `uvicorn[standard]` with `httptools` + `uvloop`) |
| Pydantic | 2.13.4 | Request/response schema validation and serialisation |
| pydantic-settings | 2.14.2 | Typed settings from environment variables / `.env` file |

FastAPI is used in `app/main.py` to compose three sub-routers — ingestion, chatbot, and clonegen — each in their own package (`app/ingestion/`, `app/chatbot/`, `app/clonegen/`). CORS middleware is configured to allow the Vercel frontend origin.

---

### Vector Database — Pinecone

| Property | Value |
|---|---|
| Package | `pinecone==9.1.0` (v9, ground-up SDK rewrite) |
| Index type | Serverless (AWS `us-east-1`) |
| Distance metric | Cosine similarity |
| Vector dimension | 384 (matches `llama-text-embed-v2` at 384-dim output) |
| Multi-tenancy | Enforced via `namespace=tenant_id` on every read/write |

**Where used:**

- `app/core/pinecone_client.py` — Singleton `get_index()` (LRU-cached), `upsert_chunks()` (batched at 100 vectors/call), `query_chunks()` (ANN search with optional metadata filter), `delete_document()` (pre-re-ingest cleanup).
- `app/core/embeddings.py` — `Pinecone.inference.embed()` for both passage and query embeddings (hosted model, no local GPU/RAM needed).
- `app/core/document_registry.py` — `index.list()` + `index.fetch()` used in `backfill_registry_from_pinecone()` to recover pre-existing chapters.

The single shared index uses Pinecone's **namespace** feature as a zero-cost multi-tenancy boundary: one institute's chunks are completely isolated from another's without needing separate indexes.

---

### Embeddings — Pinecone Inference (`llama-text-embed-v2`)

| Property | Value |
|---|---|
| Model | `llama-text-embed-v2` (Pinecone-hosted) |
| Output dimension | 384 |
| Input types | `"passage"` (for document chunks), `"query"` (for search queries) |
| Batch size | 90 inputs per call (Pinecone limit is 96; 90 adds a safety margin) |
| Cost | Free up to 5M tokens/month on the Pinecone Starter plan |

**Why hosted inference instead of local sentence-transformers:** The platform targets Render's free tier (512 MB RAM). Loading PyTorch + sentence-transformers in-process uses 600 MB–1 GB on its own, which causes OOM crashes. Using Pinecone's hosted inference removes the entire PyTorch dependency tree (~1.5 GB of packages) from the Docker image, drops the image build time significantly, and keeps RAM usage within budget — with no change to the mathematical output since the same underlying architecture is used.

**Where used:** `app/core/embeddings.py` — `embed_texts()` for document chunks during ingestion, `embed_query()` for search queries at chat/clonegen time and for semantic cache lookups.

---

### LLM — Groq API

| Property | Value |
|---|---|
| Package | `groq==1.4.0` |
| Chat model (Doubt Assistant) | `openai/gpt-oss-20b` — fast, low latency, cost-efficient |
| Clone model (CloneGen) | `openai/gpt-oss-120b` — higher quality for nuanced question generation |
| Ingest model (centroid summaries) | `openai/gpt-oss-20b` — compact JSON responses only |

**Where used:**

- `app/chatbot/router.py` — One call per unanswered (cache-miss) student question.
- `app/clonegen/generator.py` — One call per CloneGen request, regardless of how many clones are requested (batched in a single prompt).
- `app/ingestion/clusterer.py` — One call per semantic cluster centroid during ingestion (not per chunk).

Groq is chosen for its exceptionally low latency (token generation speed), which is critical for a student-facing doubt assistant where perceived response time directly affects usability.

---

### In-Memory Store — Redis

| Property | Value |
|---|---|
| Package | `redis==8.0.0` |
| Image (Docker) | `redis:7-alpine` |
| Connection | `redis.from_url(REDIS_URL, decode_responses=True)` |
| Data structures used | `HASH` (document registry), `SET` + `STRING` (semantic cache index + entries) |

Redis serves two completely separate purposes in this platform:

**1. Semantic Cache** (`app/core/cache.py`)
Keys are namespaced as `semcache:{tenant_id}:{scope}:{sha256_digest}`. An index `SET` per tenant+scope tracks all live entry keys (enabling full-scan cosine matching). Entries are stored as JSON blobs containing both the response and the query embedding vector, with a 24-hour TTL. Before any Groq call, `get_cached()` embeds the current query and computes cosine similarity against every cached query vector; if the best match exceeds `CACHE_SIMILARITY_THRESHOLD` (default 0.92), the cached response is returned with zero LLM cost.

**2. Document Registry** (`app/core/document_registry.py`)
A `HASH` per tenant (`doc_registry:{tenant_id}`) maps each ingested filename to a JSON object containing parsed metadata (grade, subject, chapter name, human-readable label, chunk count). This is what powers the chapter picker on the `/chat` page. Redis is chosen here over Pinecone because Pinecone has no "list distinct metadata values" query — recovering the full set of document names would require fetching every vector and deduplicating client-side, which is slow and gets slower with corpus size. A Redis hash lookup is O(1) and always instant.

---

### NLP / ML — Local, Zero-API-Cost Libraries

These libraries run entirely in-process during ingestion. No API calls, no network round-trips, no per-use cost.

| Library | Version | Algorithm / Use |
|---|---|---|
| scikit-learn | 1.9.0 | `TfidfVectorizer` (keyword extraction, re-ranking, extractive compression), `KMeans` (semantic clustering), `cosine_similarity` |
| spaCy | 3.8.14 | Named entity recognition (`en_core_web_sm` model, `ner` pipeline only) |
| NumPy | 1.26.4 | Embedding arithmetic, centroid distance calculation, Jaccard shingle sets, cosine similarity in the semantic cache |

**Where used:** `app/ingestion/clusterer.py` (TF-IDF keywords, spaCy NER, KMeans clustering), `app/core/retrieval.py` (TF-IDF re-ranking, shingled deduplication, extractive sentence compression), `app/core/cache.py` (NumPy cosine similarity for semantic cache matching).

---

### PDF Processing

| Library | Version | Role |
|---|---|---|
| PyPDF2 | 3.0.1 | Page-by-page text extraction from uploaded PDFs |
| langchain-text-splitters | 0.3.6 | `RecursiveCharacterTextSplitter` — hierarchical chunking with configurable size and overlap |

**Where used:** `app/ingestion/chunker.py`. Only the text-splitter package from LangChain is imported (not the full `langchain` dependency tree), keeping the Docker image lean. Chunk size is 1,000 characters with 150-character overlap (~15%), preserving cross-boundary concept continuity without excessive redundancy.

---

## 4. Frontend Tech Stack

### Framework & Runtime

| Technology | Version | Role |
|---|---|---|
| Next.js | 14.2.15 | React framework; App Router, SSR/SSG, file-based routing |
| React | 18.3.1 | UI component model |
| TypeScript | 5.6.2 | Static typing across all frontend files |
| Node.js | (Vercel-managed) | Build and runtime host |

---

### Styling

| Technology | Version | Role |
|---|---|---|
| Tailwind CSS | 3.4.10 | Utility-first CSS framework |
| tailwindcss-animate | 1.0.7 | Animation utilities |
| class-variance-authority | 0.7.0 | Typed variant prop system for UI components |
| clsx + tailwind-merge | 2.1.1 / 2.5.2 | Conditional class merging without specificity conflicts |

---

### UI Components & Libraries

| Technology | Version | Role |
|---|---|---|
| Radix UI (`@radix-ui/react-label`, `@radix-ui/react-slot`) | ^2.1 / ^1.1 | Accessible headless primitives |
| Lucide React | 0.446.0 | Icon set |
| Sonner | 1.5.0 | Toast notification system |
| @splinetool/react-spline + runtime | 4.0.0 / 1.9.48 | Interactive 3D robot on the landing page |

---

### Markdown Rendering

The `MarkdownMessage` component (`components/ui/markdown-message.tsx`) is a **custom, zero-dependency Markdown renderer** written specifically for this platform. It handles the exact Markdown subset that the Groq system prompt instructs the model to produce:

- `**bold**` → `<strong>`
- `*italic*` → `<em>`
- `` `inline code` `` → `<code>`
- `- bullet` → `<ul><li>`
- `1. numbered` → `<ol><li>`
- `### heading` → `<h4>`
- `Sources: …` footer — stripped from body and rendered as a styled citation line

The full parse-and-render runs inside a single `useMemo` call. This replaced `react-markdown@10` + `remark-gfm`, which ran a full unified/remark AST pipeline (tokenizer → parser → transformer → compiler) on every render, causing unnecessary re-render cycles that manifested as excess API calls in the Groq dashboard.

---

### Pages & Routing

| Route | File | Description |
|---|---|---|
| `/` | `app/page.tsx` | Landing page — 3D Spline robot, CTA buttons |
| `/chat` | `app/chat/page.tsx` | Doubt Assistant — chapter picker + chat interface |
| `/admin` | `app/admin/page.tsx` | Teacher Dashboard — links to ingest and clonegen |
| `/admin/ingest` | `app/admin/ingest/page.tsx` | PDF upload + chapter registry management |
| `/admin/clonegen` | `app/admin/clonegen/page.tsx` | Seed MCQ input + clone output display |

---

## 5. Infrastructure & Deployment

### Backend — Render (Free Tier)

The backend is containerised with Docker and deployed to Render's free web service tier.

| Property | Value |
|---|---|
| RAM limit | 512 MB |
| Idle behaviour | Spins down after 15 minutes of inactivity; cold start ~30–60s |
| Deployment method | Render Blueprint (`render.yaml`) — single-command deploy |
| Health check | `GET /health` → `{"status": "ok"}` |

The entire architecture is designed around the 512 MB RAM ceiling. Removing PyTorch/sentence-transformers (replaced by Pinecone hosted inference), importing only `langchain-text-splitters` (not full LangChain), and using spaCy's small model (`en_core_web_sm`) are all direct consequences of this constraint.

### Frontend — Vercel

Next.js 14 deploys to Vercel with zero configuration beyond setting `NEXT_PUBLIC_API_URL` to the Render backend URL.

### Redis — Managed (e.g. Upstash) or Docker

In local development, Redis runs as a Docker Compose service (`redis:7-alpine`) with a named volume for persistence. In production, a managed Redis provider (Upstash free tier works) is configured via the `REDIS_URL` environment variable.

### Local Development — Docker Compose

`docker-compose.yml` defines two services: `redis` and `backend`. The backend mounts `./backend` as a volume for hot-reload during development. Comment out the volume for production-style immutable builds.

---

## 6. Data Pipeline — Document Ingestion

**File:** `app/ingestion/` (`chunker.py`, `clusterer.py`, `pipeline.py`)

This is a one-time cost per document. The pipeline runs entirely sequentially and produces no ongoing per-request cost.

### Step 1 — PDF Text Extraction (zero API cost)

`PyPDF2.PdfReader` extracts text page-by-page. Each page is tagged with its page number so the `source_page` metadata travels with every downstream chunk to the final Pinecone vector, enabling the chatbot to cite page numbers in answers.

### Step 2 — Recursive Character Chunking (zero API cost)

`RecursiveCharacterTextSplitter` splits page text using a hierarchy of separators (`\n\n`, `\n`, `. `, ` `, `""`) so chunk boundaries fall at natural language boundaries rather than mid-sentence or mid-word. Parameters: 1,000-character chunk size, 150-character overlap.

### Step 3 — TF-IDF Keyword Extraction (zero API cost)

`TfidfVectorizer` (scikit-learn) is fitted across the entire document's chunk corpus. For each chunk, the top 6 TF-IDF-weighted unigrams and bigrams are stored as `tfidf_keywords` in the chunk's Pinecone metadata. These keywords later augment the re-ranking signal at retrieval time: the re-ranker upweights chunks whose pre-computed keywords overlap with the student's query.

### Step 4 — Named Entity Recognition (zero API cost)

spaCy's `en_core_web_sm` pipeline (NER component only; parser and lemmatizer disabled for speed) tags each chunk with up to 15 named entities (`PERSON`, `ORG`, `GPE`, `PRODUCT`, `EVENT`, `LAW`, `DATE`, `MONEY`, `PERCENT`). Stored as `entities` in Pinecone metadata. Used as additional re-ranking signal.

### Step 5 — Semantic Clustering with KMeans (zero API cost)

Chunks are embedded in batches of 90 using `llama-text-embed-v2` via Pinecone Inference. The resulting 384-dimensional embedding matrix is clustered using scikit-learn's `KMeans`.

Number of clusters K is computed as:

```
K = clamp(
      floor(n_chunks / TARGET_CHUNKS_PER_CLUSTER),
      MIN_CLUSTERS=5,
      MAX_CLUSTERS=150
    )
```

Where `TARGET_CHUNKS_PER_CLUSTER = 20`. For a 300-page document producing ~1,500 chunks, K ≈ 75 clusters. The `MAX_CLUSTERS = 150` hard ceiling caps the worst-case Groq call count per ingest, protecting against runaway costs on very large documents.

For each cluster, the **centroid chunk** is identified as the chunk whose embedding is geometrically closest to the KMeans centroid vector (L2 distance).

### Step 6 — LLM Semantic Labelling (one Groq call per cluster, not per chunk)

Only the centroid chunk of each cluster is sent to Groq (`openai/gpt-oss-20b`), with a JSON-mode prompt requesting:

```json
{ "topic": "3–6 word section title", "summary": "1–2 sentence summary" }
```

The topic and summary are then **broadcast** to every chunk in that cluster. This means a 1,500-chunk document gets 75 Groq calls during ingestion — not 1,500. Every non-centroid chunk gets its topic and summary metadata for free, purely by cluster membership.

### Step 7 — Pinecone Upsert

Every enriched chunk is upserted as a Pinecone vector with the following metadata payload:

| Metadata field | Source | Purpose |
|---|---|---|
| `source_document` | Filename | Chapter filter in retrieval |
| `chunk_index` | Sequential | Ordering reference |
| `source_page` | PyPDF2 page number | Citation in answers |
| `text` | Chunk text | Sent to Groq as context |
| `tfidf_keywords` | Step 3 | Re-ranking signal |
| `entities` | Step 4 | Re-ranking signal |
| `cluster_id` | Step 5 | Cluster membership |
| `is_cluster_centroid` | Step 5 | Centroid flag |
| `semantic_topic` | Step 6 | Citation label in answers |
| `context_summary` | Step 6 | Cluster-level context |
| `char_count` | Derived | Diagnostic |

Upserts are batched at 100 vectors per Pinecone call.

---

## 7. Retrieval System

**File:** `app/core/retrieval.py`

The retrieval pipeline is a multi-stage process that maximises answer quality while minimising tokens sent to the LLM. Every stage after the initial Pinecone query is pure local computation.

### Stage 1 — Query Embedding

The student's question is embedded with `embed_query()` (Pinecone Inference, `input_type="query"`). This produces a 384-dimensional query vector.

### Stage 2 — Approximate Nearest Neighbour Search (Pinecone)

A single `index.query()` call retrieves the top-K most similar chunks from the tenant's namespace. When the chapter picker has one or more chapters selected, a metadata filter `{"source_document": {"$in": [...]}}` is passed, scoping the ANN search to those chapters only. This reduces the effective search space from the entire corpus (e.g. 540 chunks) to the selected chapters (e.g. 180 chunks), improving both precision and speed.

When reranking or deduplication is enabled, a wider candidate pool is pulled (up to `min(top_k * 3 + 10, 20)`) to give the downstream stages more to work with before trimming back to `top_k`.

### Stage 3 — TF-IDF Re-Ranking (local, zero API cost)

Each candidate chunk's text is concatenated with its `semantic_topic` (repeated once for light upweighting), `context_summary`, `tfidf_keywords`, and `entities` into a single "enriched" document string. These enriched strings, plus the query, are vectorised together with `TfidfVectorizer` (4,000 features, unigrams and bigrams). Cosine similarity between the query TF-IDF vector and each enriched chunk vector produces a re-ranking score. This hybrid ANN + TF-IDF approach consistently surfaces more relevant chunks than ANN alone when queries use vocabulary that differs slightly from the exact words in the chunk.

### Stage 4 — Near-Duplicate Removal (local, zero API cost)

`dedupe_chunks()` computes word-level 5-shingles for each candidate chunk and drops any chunk whose Jaccard similarity with a higher-ranked chunk exceeds 0.80. This prevents overlapping PDF chunking from sending the same paragraph to Groq twice under slightly different chunk boundaries. After deduplication, the pool is trimmed to `top_k=5` final chunks.

### Stage 5 — Extractive Sentence Compression (local, zero API cost)

`compress_chunk_text()` applies Luhn-style extractive summarisation to each of the 5 final chunks individually. Each chunk's sentences are scored by TF-IDF cosine similarity against the query; the top 4 sentences are kept in original document order. Chunks with fewer than 6 sentences are left untouched (not worth compressing). This cuts Groq prompt token usage on long chunks without any LLM call, and gives the model a tighter, more query-focused context to draw from.

### Confidence Guardrail

The top Pinecone match score is used as a `confidence` value. If `confidence < CHAT_CONFIDENCE_THRESHOLD` (default 0.40), the system checks a spread-based rescue condition before escalating: if the top score clearly stands out above the mean of the rest (spread ≥ 0.12) and is within 0.05 of the threshold, it is not escalated. This prevents borderline but genuinely relevant answers from being dropped, while still escalating truly out-of-scope questions. No LLM call is made on an escalated question.

---

## 8. Chatbot — Doubt Assistant

**Files:** `app/chatbot/router.py`, `app/chatbot/guardrails.py`

### Request Flow

1. `POST /chat` receives `{message, document_names?}`.
2. Cache check: `get_cached()` embeds the message and cosine-scans Redis. Cache hit → immediate response, zero Groq cost.
3. Retrieval: `retrieve_context()` with `rerank=True`, `top_k=5`, optional chapter filter.
4. Guardrail: `should_escalate()` checks confidence + spread. Escalation → immediate human-handoff response, zero Groq cost.
5. Context block: 5 compressed chunks formatted as `[Topic | p.N] sentence sentence …`.
6. Single Groq call (`openai/gpt-oss-20b`, temperature 0.2, max 500 tokens).
7. Response cached in Redis. Response returned to frontend.

### System Prompt Design

The system prompt uses a worked example and explicit Markdown formatting rules (not abstract instructions like "format nicely") because concrete rules produce far more consistent formatting from small/fast models. Rules enforced:
- Open with a one-sentence direct answer.
- Use bullet points for multi-part answers; never more than 2 consecutive prose sentences.
- Use `**Term**` bolded sub-headings for comparisons.
- Bold key vocabulary on first appearance only.
- Stay under 180 words unless the question explicitly requests long-form.
- End with exactly one `Sources: "topic | p.N"` line.

---

## 9. CloneGen — MCQ Generation

**Files:** `app/clonegen/parser.py`, `app/clonegen/generator.py`, `app/clonegen/router.py`

### Seed Parsing (zero API cost)

`parse_seed_question()` uses regex to extract from a raw pasted question string:
- **Stem** — the question body before option (A)
- **Tag** — content inside `[brackets]` (e.g. `[NEET 2025]`)
- **Options** — `(A)` through `(D)` via pattern matching
- **Correct answer** — detected from inline markers (`*C) ...`) or a trailing `Answer: C` line

`extract_core_topic()` derives a retrieval query from the stem's content words plus all option texts, stripping stopwords. This becomes the query for Pinecone retrieval.

### Clone Generation (one Groq call for N clones)

All N requested clones are generated in a single Groq call (`openai/gpt-oss-120b`, `json_object` response format, temperature 0.6 for creative diversity). The prompt instructs the model to:
- Test the same core concept as the seed, not synonym-swap it.
- Change the framing: different scenario, different angle (NOT/comparison/application), not a restatement.
- Ground all three distractors in concepts actually present in the retrieved context.
- Ensure each clone is conceptually distinct from the others in the batch.

The `MAX_CONTEXT_CHARS = 2,400` cap on the context block prevents token bloat when retrieval returns very long chunks.

---

## 10. Semantic Cache

**File:** `app/core/cache.py`

The semantic cache is a custom implementation on top of Redis that provides **semantic** (meaning-based) cache hits, not just exact-string hits.

### How It Works

On every `set_cached()` call, the query is embedded (Pinecone Inference) and stored alongside the response in Redis as `{"vector": [...384 floats...], "response": {...}}` with a 24-hour TTL. The entry key is indexed in a Redis `SET` per tenant+scope.

On every `get_cached()` call, the incoming query is embedded and its vector is cosine-compared (NumPy dot product) against every stored vector in the index `SET`. If the highest similarity score ≥ `CACHE_SIMILARITY_THRESHOLD` (default 0.92), the cached response is returned.

### Effect

A student asking "What is photosynthesis?" and another asking "Explain photosynthesis to me" will hit the same cache entry if their embedding cosine similarity exceeds 0.92 — one Groq call serves both. The cache is **namespaced per tenant and per scope** (`chat` vs `clone`) so answers never cross-pollinate between institutes or between the two features.

---

## 11. Document Registry & Chapter Picker

**Files:** `app/core/document_registry.py`, `frontend/app/chat/page.tsx`

### Filename Parsing

Every ingested filename is parsed by `parse_document_name()` using a regex that handles the institute's naming convention:

```
LP_NEET_{grade}{subject}_{chapter}[_{year}]_without [detailed] solution[s][ (N)].pdf
```

Examples and their parsed output:

| Filename | Chapter | Label |
|---|---|---|
| `LP_NEET_11B_Cell the unit of life_without solutions.pdf` | `Cell the unit of life` | `11B · Cell the unit of life` |
| `LP_NEET_11B_Animal Kingdom_26-27_Without detailed solution (1).pdf` | `Animal Kingdom` | `11B · Animal Kingdom` |
| `LP_NEET_12B_Evolution_26-27_without detailed solutions.pdf` | `Evolution` | `12B · Evolution` |

Files not matching the convention are still registered; they just display their raw filename as the label.

### Auto-Backfill on Chat Page Load

When the `/chat` page mounts, it silently POSTs to `/ingest/backfill-registry` before fetching the document list. This recovers chapters that were uploaded before the Redis registry existed (or after a Redis flush) by scanning Pinecone metadata directly. The backfill is idempotent (already-registered documents are untouched) and costs zero Groq or embedding calls (Pinecone `list()` + `fetch()` only). A `useRef` guard prevents double-firing in React Strict Mode.

---

## 12. Multi-Tenancy & Auth

**File:** `app/tenants/auth.py`

Each institute is assigned an API key that maps to a `tenant_id` string. The mapping is stored as an environment variable (`TENANT_API_KEYS="key1:tenant1,key2:tenant2"`). Every API request must include an `X-API-Key` header; `get_tenant_id()` is a FastAPI `Depends()` dependency injected into all router endpoints.

The `tenant_id` is used as the Pinecone namespace and the Redis key prefix everywhere in the system, ensuring complete data isolation between institutes at both the vector store and cache layers without needing separate infrastructure per tenant.

---

## 13. Cost & API Call Optimisation

This section documents every deliberate design decision made to minimise paid API usage. The platform is engineered to be production-viable on free-tier infrastructure.

### Groq Call Budget

| Operation | Calls per event | Notes |
|---|---|---|
| Ingestion (per document) | 1 per cluster (K ≈ n_chunks / 20) | Hard-capped at 150 calls max |
| Doubt Assistant (per question) | 1 (cache miss) or 0 (cache hit) | Guardrail escalation costs 0 |
| CloneGen (per request) | 1 (cache miss) or 0 (cache hit) | All N clones in one call |

### Embedding Call Budget

| Operation | Calls | Notes |
|---|---|---|
| Ingestion | ceil(n_chunks / 90) | Pinecone free tier: 5M tokens/month |
| Chat query | 1 (always, even on cache hit) | Needed to check cache before knowing if it's a hit |
| CloneGen query | 1 (always) | Same reason |
| Cache entry write | 1 per new answer stored | Amortised over all future cache hits |

### Optimisation Techniques

**1. Cluster-then-label (not label-every-chunk)**
The biggest single cost reduction. Groq is called once per semantic cluster centroid. The topic and summary propagate to all other chunks in the cluster for free. For a 1,500-chunk document (75 clusters), this is 75 Groq calls instead of 1,500 — a 20× reduction.

**2. Semantic cache with cosine similarity matching**
Semantically equivalent questions share one cached response. Cache TTL is 24 hours. Both the chatbot and CloneGen share the same cache infrastructure.

**3. Confidence guardrail before LLM call**
Out-of-scope questions are detected by Pinecone retrieval confidence before any Groq call is made. Escalated questions cost 0 Groq tokens.

**4. TF-IDF re-ranking (no second embedding call)**
After the initial ANN search, re-ranking is done with TF-IDF (pure local math) rather than a second embedding or cross-encoder call. Same quality improvement, zero API cost.

**5. Extractive sentence compression (no summarisation call)**
Chunks are compressed by scoring sentences against the query with TF-IDF cosine similarity (Luhn-style extractive summarisation), not a separate LLM summarisation call. Reduces Groq prompt tokens per chat request.

**6. Jaccard deduplication (no second ANN query)**
Near-duplicate chunks from overlapping PDF boundaries are dropped using word-shingle Jaccard similarity — set arithmetic on strings, no API or embedding cost.

**7. Batched clone generation**
N clones in one Groq call using a structured JSON-mode prompt, not N separate calls. Cost is flat per CloneGen request, regardless of how many clones are requested (up to 10).

**8. Hosted embeddings (Pinecone Inference)**
Eliminates PyTorch and sentence-transformers from the runtime environment. No GPU required, no per-inference cost (within the 5M token/month free tier), no RAM spike from loading a local model.

**9. `lru_cache` on Pinecone client and index handles**
The Pinecone SDK client and `Index` object are instantiated once per process lifetime, not per request. Eliminates repeated SDK initialisation and `describe_index` calls.

**10. Chapter-scoped retrieval (Pinecone metadata filter)**
When a student selects chapters in the chapter picker, the Pinecone `$in` metadata filter restricts the ANN search to those chapters' vectors. Searching 180 vectors instead of 540 is faster and produces higher-precision results without any additional cost.

**11. Lightweight frontend Markdown renderer**
Replacing `react-markdown` + `remark-gfm` with a custom `useMemo`-based parser eliminates the full unified/remark AST pipeline (tokenizer → parser → transformer → compiler) from running on every render cycle, preventing parent-component re-renders that were causing excess POST calls to the backend.

**12. Backfill without embedding or LLM calls**
`backfill_registry_from_pinecone()` recovers missing chapter registry entries using only Pinecone's `list()` (ID-only, paginated) + `fetch()` (metadata-only) operations — zero Groq or embedding calls, bounded cost proportional to corpus size (not query volume).

---

## 14. Platform Features & Benefits

### For Students

**Curriculum-aligned answers only.** The system will not answer from general knowledge. Every answer is grounded exclusively in the institute's uploaded course material. If the material doesn't cover the question, the system says so and escalates to a teacher — it never hallucinates.

**Chapter-scoped search.** The chapter picker on the Doubt Assistant lets students narrow their question to one or more specific chapters, giving higher-precision answers and faster retrieval. Chapters are listed automatically from what has been ingested.

**Structured, readable answers.** Every answer follows a consistent format: a direct one-sentence answer, bullet points for multi-part content, bolded key vocabulary, and a page citation. This is enforced structurally at the prompt level, not just hoped for.

**Page citations.** Every answer includes `Sources: "Topic | p.N"` with the exact page number from the source PDF, so students can verify answers and read more.

**Sub-second responses for repeated questions.** The semantic cache returns identical or semantically equivalent answers from Redis without making any LLM call, making repeated or similar questions essentially instant.

### For Teachers

**300+ page PDF support.** The ingestion pipeline handles full-length course textbooks, not just short excerpts. The chunking, clustering, and embedding pipeline scales linearly.

**One-upload, always-available.** After a PDF is uploaded once, it is available to all students (in that tenant) indefinitely. The teacher does not need to be present for students to use the Doubt Assistant.

**CloneGen question bank expansion.** A teacher pastes any existing MCQ and gets N conceptually equivalent, distractor-grounded, exam-quality clone questions instantly — grounded in the same course material the students are studying. This dramatically accelerates question bank creation for NEET/JEE preparation.

**Re-ingest without duplication.** Uploading a PDF with the same filename as an existing document automatically deletes the old vectors before upserting the new ones. No stale or duplicate data accumulates.

**Admin dashboard.** `/admin` provides a teacher-facing interface for PDF upload, chapter registry management, and CloneGen, separate from the student-facing `/chat` interface.

### Architectural & Operational

**Multi-tenant isolation.** Multiple institutes can run on the same infrastructure. Data is isolated at the Pinecone namespace level and the Redis key prefix level. One institute cannot access another's documents, cache entries, or chapter registry.

**Free-tier viable.** The entire production stack (Render free web service + Pinecone Starter + Redis free tier on Upstash + Vercel Hobby) costs $0/month for moderate usage volumes, with a clear and gradual upgrade path as usage grows.

**Stateless backend.** The FastAPI backend holds no in-process state beyond `lru_cache`d SDK handles. All state lives in Redis and Pinecone. This means horizontal scaling (adding Render instances) requires no coordination.

**Idempotent backfill.** The chapter registry can be rebuilt from Pinecone at any time with no side effects. Restarting Redis, flushing the cache, or redeploying the backend does not lose any course content — Pinecone is the source of truth.

**Graceful degradation.** If the chapter registry fetch fails on the `/chat` page, the Doubt Assistant still works — it just searches the whole corpus instead of a selected chapter. If the semantic cache is unavailable, requests fall through to Groq. No user-visible error for infrastructure hiccups.

**Observable.** Every chat request logs `tenant_id`, selected documents, Pinecone confidence score, threshold, and matched chunk count to Uvicorn's error logger. This makes it straightforward to tune `CHAT_CONFIDENCE_THRESHOLD` against real traffic by watching the logs.

---

## 15. Configuration Reference

All configuration is via environment variables, typed through `pydantic-settings` in `app/config.py`.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key for all LLM calls |
| `PINECONE_API_KEY` | *(required)* | Pinecone API key for vector DB and hosted inference |
| `PINECONE_INDEX` | `edtech-rag` | Name of the Pinecone serverless index |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `EMBEDDING_MODEL` | `llama-text-embed-v2` | Pinecone-hosted embedding model name |
| `GROQ_CHAT_MODEL` | `openai/gpt-oss-20b` | Model for Doubt Assistant responses |
| `GROQ_CLONE_MODEL` | `openai/gpt-oss-120b` | Model for CloneGen question generation |
| `GROQ_INGEST_MODEL` | `openai/gpt-oss-20b` | Model for cluster centroid labelling |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity threshold for semantic cache hits |
| `CHAT_CONFIDENCE_THRESHOLD` | `0.40` | Pinecone score below which a question is escalated |
| `TENANT_API_KEYS` | `devkey123:demo-tenant` | Comma-separated `key:tenant_id` pairs |

### Threshold Tuning Guide

`CHAT_CONFIDENCE_THRESHOLD` should be tuned per deployment by inspecting the `confidence=` values logged by the chatbot on questions you know are and aren't in the ingested material. Start at 0.40 and raise it if the assistant is answering out-of-scope questions confidently; lower it if it is escalating questions that are clearly in the material.

`CACHE_SIMILARITY_THRESHOLD` at 0.92 means only very close semantic matches reuse a cached answer. Lower this (e.g. to 0.88) to get more cache hits at the cost of occasionally returning a slightly mismatched cached answer for a related but distinct question.
