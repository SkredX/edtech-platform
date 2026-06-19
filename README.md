# RAG Chatbot + CloneGen Platform

Unified backend (FastAPI) serving a document-ingestion pipeline, a RAG chatbot,
and CloneGen MCQ generation, all sharing one Pinecone index and one semantic
cache so API/token spend isn't duplicated across features. Frontend is Next.js
+ shadcn/Tailwind/TypeScript.

## Quick start (local dev)

### 1. Backend

```bash
cd backend
cp .env.example .env
# Fill in GROQ_API_KEY and PINECONE_API_KEY in .env
```

Option A — Docker (recommended, brings up Redis too):

```bash
cd ..
docker compose up --build
```

Option B — bare metal:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Run Redis separately, e.g.: docker run -p 6379:6379 redis:7-alpine
uvicorn app.main:app --reload --port 8000
```

Backend will be live at `http://localhost:8000`. Check `http://localhost:8000/health`.

### 2. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Frontend will be live at `http://localhost:3000`.

### 3. Authenticate requests

The backend expects an `X-API-Key` header on every request, mapped to a
`tenant_id` via `TENANT_API_KEYS` in `backend/.env` (format: `key:tenant_id`,
comma-separated for multiple institutes). The frontend reads the key from
`localStorage` under `speedlabs_api_key` — set it once in your browser console
during local dev:

```js
localStorage.setItem("speedlabs_api_key", "devkey123");
```

(`devkey123` is the default dev key in `.env.example`, mapped to `demo-tenant`.)

## Using the platform

1. **Ingest material**: go to `/admin/ingest`, upload a PDF. This chunks the
   document, extracts keywords/entities locally, clusters chunks with KMeans,
   and calls Groq once per cluster centroid (not per chunk) to generate a
   topic + summary, broadcast back to every chunk in that cluster.
2. **Chat**: go to `/chat`. Each question is checked against the semantic
   cache first (no API call on a cache hit), then retrieved from Pinecone,
   then answered by Groq grounded in the retrieved context. Low-confidence
   matches escalate to a teacher instead of risking a hallucinated answer.
3. **Generate clones**: go to `/admin/clonegen`, paste a seed MCQ (mark the
   correct option inline with `Answer: C` or select it from the dropdown).
   One batched Groq call returns however many clones you asked for (1–10).

## Cost-control mechanisms (why this stays cheap)

- **Ingestion**: only cluster centroids hit Groq — a 300-page PDF with ~1500
  chunks typically becomes ~60 Groq calls, not 1500.
- **Chat + CloneGen share one semantic cache**: a question answered once by
  the chatbot won't trigger a fresh Groq call if CloneGen (or another
  student) asks something semantically equivalent.
- **CloneGen batches n clones into one call** instead of one call per clone.
- **All embeddings are local** (MiniLM via sentence-transformers) — never
  billed, never rate-limited by a provider.

## Deployment

- **Frontend → Vercel**: point `NEXT_PUBLIC_API_URL` at your deployed backend.
- **Backend → Render / Railway / Fly.io** (not Vercel serverless — the
  embedding model and spaCy need a persistent process and memory footprint
  beyond serverless function limits).
- **Redis → Upstash** (free tier) for the semantic cache.
- **Pinecone → Serverless index**, one namespace per tenant for isolation.

## Repo structure

```
edtech-rag-platform/
├── backend/            FastAPI app — ingestion, chat, clonegen, shared core
├── frontend/            Next.js app — chat UI, admin dashboards, landing page
└── docker-compose.yml   Local dev: redis + backend
```
