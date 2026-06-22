from sentence_transformers import SentenceTransformer
from functools import lru_cache
from app.config import settings

@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer(settings.EMBEDDING_MODEL)

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True, batch_size=64).tolist()

def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]