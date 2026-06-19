from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.chatbot.router import router as chatbot_router
from app.clonegen.router import router as clonegen_router
from app.ingestion.pipeline import router as ingestion_router

app = FastAPI(
    title="EdTech RAG Platform",
    description="Unified backend: document ingestion, RAG chatbot, and CloneGen MCQ generation.",
    version="1.0.0",
)

# Tighten allow_origins to your actual Vercel domain(s) in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion_router)
app.include_router(chatbot_router)
app.include_router(clonegen_router)


@app.get("/health")
def health():
    return {"status": "ok"}
