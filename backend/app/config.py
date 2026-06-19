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

    CACHE_SIMILARITY_THRESHOLD: float = 0.92
    CHAT_CONFIDENCE_THRESHOLD: float = 0.70

    # "key:tenant_id,key2:tenant_id2" — parsed into a dict in tenants/auth.py
    TENANT_API_KEYS: str = "devkey123:demo-tenant"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
