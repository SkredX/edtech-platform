from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROQ_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX: str = "edtech-rag"
    REDIS_URL: str = "redis://localhost:6379"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GROQ_CHAT_MODEL: str = "llama-3.1-8b-instant"      # fast, cheap — chatbot
    GROQ_CLONE_MODEL: str = "llama-3.3-70b-versatile"  # quality — clonegen
    CACHE_SIMILARITY_THRESHOLD: float = 0.92
    CHAT_CONFIDENCE_THRESHOLD: float = 0.70

    class Config:
        env_file = ".env"

settings = Settings()