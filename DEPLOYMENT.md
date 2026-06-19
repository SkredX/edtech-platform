# Deployment guide

Stack: code on GitHub → frontend on Vercel → backend on Render → vectors on Pinecone.
No Docker install needed on your laptop — Render builds the image on their servers.

## 0. Push to GitHub

The repo already has a git history and an `origin` pointing at
`https://github.com/SkredX/edtech-platform.git`. From the repo root:

```
git add -A
git commit -m "Fix dependency pins, Pinecone v9 API, Groq model names"
git push origin main
```

## 1. Pinecone

1. Create a free account at pinecone.io, create an API key.
2. You do **not** need to create the index by hand — `pinecone_client.py`
   calls `pc.has_index()` and creates it (384 dims, cosine, AWS us-east-1
   serverless) on first run if it's missing.
3. Keep the API key handy for step 3.

## 2. Groq

Create a free API key at console.groq.com. The model names in this repo
(`openai/gpt-oss-20b`, `openai/gpt-oss-120b`) are Groq's current models —
the previous `llama-3.1-8b-instant` / `llama-3.3-70b-versatile` pins were
deprecated on 2026-06-17, which is why they were changed.

## 3. Backend on Render

1. render.com → New → Blueprint → point it at your GitHub repo. It will
   read `backend/render.yaml` and pre-fill the service.
   - If you'd rather click through manually instead: New → Web Service →
     select the repo → Root Directory `backend` → Runtime: Docker.
2. Fill in the environment variables Render leaves blank (`sync: false`
   in the blueprint): `GROQ_API_KEY`, `PINECONE_API_KEY`, `REDIS_URL`,
   `TENANT_API_KEYS` (e.g. `mykey123:my-institute`).
3. **Redis**: Render's free tier no longer ships a usable free Redis of
   meaningful size. Use Upstash instead — it has a real free tier (REST +
   standard Redis protocol, 256MB+) at upstash.com. Create a database,
   copy the `rediss://` connection string into `REDIS_URL`.
4. Deploy. First build takes a while (spaCy model download + ML deps).

### Render's free tier RAM — now actually fine

The previous version of this guide flagged a real risk: loading
`sentence-transformers` (PyTorch) + spaCy in the same process typically
needs 600 MB-1 GB, over Render's 512 MB free-tier ceiling. That's fixed
now — `core/embeddings.py` calls Pinecone's hosted Inference API
(`llama-text-embed-v2`) instead of running an embedding model locally,
so PyTorch is no longer a dependency at all. What's left running
in-process is spaCy's small NER model (`en_core_web_sm`, no parser/
lemmatizer) plus scikit-learn's KMeans/TF-IDF — both comfortably under
512 MB even during ingestion.

Trade-off to know about: embedding calls are now a network round-trip to
Pinecone instead of local CPU inference, so very large document
ingestion will be somewhat slower. For normal chatbot/clonegen request
volume this isn't noticeable. Pinecone's Starter (free) plan includes 5M
embedding tokens/month, which is generous for a single-institute demo or
small deployment — watch it if you're ingesting very large document sets.

The free tier still spins the service down after 15 minutes idle, with a
30-60s cold start on the next request — fine for a demo, not for
production traffic with that constraint.

## 4. Frontend on Vercel

1. vercel.com → New Project → import the same GitHub repo.
2. Root Directory: `frontend`. Vercel auto-detects Next.js, no other
   config needed.
3. Set the environment variable used by `frontend/lib/api.ts` (check
   `frontend/.env.local.example` for the exact name — it's your Render
   backend's public URL, e.g. `https://edtech-rag-backend.onrender.com`)
   plus the dev API key matching one entry in `TENANT_API_KEYS`.
4. Deploy.

## 5. Smoke test

```
curl https://<your-render-url>/health
```

should return `{"status": "ok"}`. Then hit your Vercel URL and try the
ingest page with a small PDF before testing chat/clonegen.

## What changed in this pass (staying on Render's free tier)

- `backend/app/core/embeddings.py` — rewritten to call Pinecone's hosted
  Inference API (`llama-text-embed-v2`) instead of running
  `sentence-transformers`/PyTorch locally. This is the actual fix for
  the RAM problem, not just a config tweak.
- `backend/requirements.txt` — removed `sentence-transformers` (and the
  ~1.5 GB of PyTorch it pulls in) accordingly.
- `backend/app/config.py`, `backend/.env.example`, `backend/render.yaml`
  — `EMBEDDING_MODEL` now points at the Pinecone-hosted model name
  instead of a local sentence-transformers path.

## What changed in the previous pass

- `backend/requirements.txt` — repinned every dependency to current
  stable (Pinecone v5→v9, Groq pre-1.0→1.4, spaCy 3.7 yanked→3.8.14,
  etc.), and swapped `langchain`/`langchain-community` for the much
  lighter `langchain-text-splitters` (only thing actually used).
- `backend/app/core/pinecone_client.py` — rewritten for v9: `has_index()`
  instead of manually scanning `list_indexes()`, and `Index(host=...)`
  instead of the now-discouraged `Index(name_string)` pattern.
- `backend/app/config.py` + `backend/.env.example` — Groq model names
  updated to the non-deprecated replacements.
- `backend/app/ingestion/chunker.py` — import path updated for the new
  splitter package.
- `backend/render.yaml` — new, lets you deploy via Render's Blueprint
  flow instead of manual field entry.
- This file — new.

Everything else from the previous handoff (cache tenant-scoping fix,
guardrails module, multi-tenant auth, all frontend components) was
already complete and didn't need changes.
