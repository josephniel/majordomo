"""Local, provider-independent text embeddings.

Uses fastembed (ONNX, runs on the host) so semantic memory recall stays
LLM-agnostic — no embedding API calls to any provider. The model is loaded
lazily on first use and cached process-wide. Embedding is CPU-bound and
synchronous; callers should run it via asyncio.to_thread.

Model choice: multilingual (not the English-only bge-small) because the
operator's memory content mixes English and Tagalog/Taglish. Same 384
dimensions, so schema.sql needs no change. Rows embedded by an older model
are tagged via memory_entries.embedding_model and excluded from the vector
arm of recall until re-embedded (`cli.py memory reembed`).
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

log = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384  # keep in sync with schema.sql memory_entries.embedding vector(DIM)

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from fastembed import TextEmbedding  # lazy: heavy import
                log.info("loading local embedding model %s (first use)", MODEL_NAME)
                _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def embed(text: str) -> list[float]:
    """Return the embedding vector for `text` (empty list for empty input)."""
    text = (text or "").strip()
    if not text:
        return []
    vec = next(iter(_get_model().embed([text])))
    return [float(x) for x in vec]


def to_pgvector(vec: list[float]) -> Optional[str]:
    """Format a vector as a pgvector literal ('[f1,f2,...]'), or None if empty."""
    if not vec:
        return None
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
