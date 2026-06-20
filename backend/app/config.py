from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GROQ_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX: str = "edtech-rag"
    REDIS_URL: str = "redis://localhost:6379"

    EMBEDDING_MODEL: str = "llama-text-embed-v2"  # Pinecone-hosted, not run locally
    # llama-3.1-8b-instant / llama-3.3-70b-versatile were deprecated by Groq
    # on 2026-06-17. These are their recommended replacements.
    GROQ_CHAT_MODEL: str = "openai/gpt-oss-20b"          # fast, cheap — chatbot
    GROQ_CLONE_MODEL: str = "openai/gpt-oss-120b"        # quality — clonegen
    GROQ_INGEST_MODEL: str = "openai/gpt-oss-20b"        # centroid topic/summary only

    # Both thresholds are cosine similarities against Pinecone's hosted
    # llama-text-embed-v2 model. They were originally tuned for the local
    # all-MiniLM-L6-v2 model (since replaced — see core/embeddings.py) and
    # that model's similarity scores don't transfer 1:1: a "this is clearly
    # the right chunk" match can score meaningfully lower here. 0.40 below
    # is a reasonable starting point, not a verified-correct number — watch
    # the "chat query ... confidence=" lines in Render's logs (added in
    # chatbot/router.py) against questions you know ARE/AREN'T in your
    # ingested material, and raise/lower this until escalation behavior
    # matches reality. CACHE_SIMILARITY_THRESHOLD (semantic cache "is this
    # the same question as one we've already answered") may need the same
    # retuning — if caching never seems to hit, this is why.
    CACHE_SIMILARITY_THRESHOLD: float = 0.92
    CHAT_CONFIDENCE_THRESHOLD: float = 0.40

    # "key:tenant_id,key2:tenant_id2" — parsed into a dict in tenants/auth.py
    TENANT_API_KEYS: str = "devkey123:demo-tenant"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
