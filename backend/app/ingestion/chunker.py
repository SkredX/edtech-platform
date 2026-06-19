"""
PDF → page text → chunks. Direct port of Cell 3 (steps 1-2) from the
original Colab RAG pipeline. No API calls in this module.
"""
import io

from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

CHUNK_SIZE = 1000      # characters per chunk — balances context vs. embedding/API cost
CHUNK_OVERLAP = 150    # ~15% overlap preserves continuity of complex/cross-boundary concepts

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def extract_pdf_text(pdf_bytes: bytes) -> list[dict]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page_number": i + 1, "text": text})
    return pages


def chunk_pdf(pdf_bytes: bytes) -> list[dict]:
    """Returns raw chunks: [{"text": ..., "source_page": ...}, ...]"""
    pages = extract_pdf_text(pdf_bytes)

    raw_chunks = []
    for page in pages:
        if not page["text"].strip():
            continue
        for piece in _splitter.split_text(page["text"]):
            raw_chunks.append({"text": piece, "source_page": page["page_number"]})

    if not raw_chunks:
        raise ValueError(
            "No text could be extracted/chunked from this PDF "
            "(it may be scanned/image-based and need OCR)."
        )
    return raw_chunks
