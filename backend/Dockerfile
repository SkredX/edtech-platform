FROM python:3.11-slim

WORKDIR /app

# System deps needed by sentence-transformers / spaCy build chain
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy model used by ingestion/clusterer.py for local NER (zero API cost)
RUN python -m spacy download en_core_web_sm

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
