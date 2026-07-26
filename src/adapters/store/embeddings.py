"""Local, provider-independent text embeddings.

Uses fastembed (ONNX, runs on the host) so semantic memory recall stays
LLM-agnostic — no embedding API calls to any provider. The model is loaded
lazily on first use and cached process-wide. Embedding is CPU-bound and
synchronous; callers should run it via asyncio.to_thread.

Asymmetric by default
---------------------
Retrieval here is asymmetric: a SHORT query ("workout schedule") is matched
against a LONGER stored fact ("The user trains at Anytime Fitness on Monday,
Wednesday and Friday evenings"). Retrieval-trained models encode the two
sides differently — `embed_query` and `embed_passage` apply whatever prefix
the chosen model expects. Using one symmetric `embed()` for both sides, as
this module previously did, throws that away.

Model choice (measured, not assumed)
------------------------------------
Ranked by `./manage eval-recall` over evals/recall_cases.yaml, vector arm
only, 20 cases:

    model                                        dim   recall@4     MRR
    paraphrase-multilingual-MiniLM-L12-v2 (old)  384      90.0%   0.768
    BAAI/bge-small-en-v1.5                       384      90.0%   0.806
    BAAI/bge-base-en-v1.5                        768      95.0%   0.885
    mixedbread-ai/mxbai-embed-large-v1          1024     100.0%   0.975
    intfloat/multilingual-e5-large              1024      90.0%   0.863

The incumbent was a *sentence-similarity* model being used for *retrieval* —
different training objective, and it showed: it scored the user's email
address higher than their employer for the query "where does the user work".

Cost of the default: ~20ms per embed vs ~3ms (irrelevant next to an LLM call)
and ~640MB resident. On a host also running a local 12B model, if that RAM
matters, set EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 — 768-dim, ~220MB, 95%.
Any dimension change is handled automatically (see below).

English-only, deliberately: the multilingual incumbent was chosen for Taglish
content, but the assistant now replies in English only, so the multilingual
tax buys nothing. EMBEDDING_MODEL=intfloat/multilingual-e5-large reverses
that if it ever matters again.

Changing the model
------------------
Vectors are tagged with the model that produced them
(`memory_entries.embedding_model`), and recall's vector arm only trusts
current-model vectors — so a model change degrades gracefully (FTS/trigram
still match) rather than returning garbage. If the DIMENSION also changed,
init_schema migrates the column and clears the stale vectors. Either way,
restore full semantic recall with:

    ./manage cli <persona> -- memory reembed
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "").strip() or DEFAULT_MODEL

# Dimensions for the models we've evaluated. Kept as a literal table so that
# importing this module stays cheap — resolving the dimension through
# fastembed would pull in onnxruntime at import time, which is exactly the
# cost the lazy model load below exists to avoid. Unknown models fall back to
# a (heavy) lookup, so a model outside this table still works.
_KNOWN_DIMS = {
    "mixedbread-ai/mxbai-embed-large-v1": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/multilingual-e5-large": 1024,
    "snowflake/snowflake-arctic-embed-l": 1024,
    "thenlper/gte-large": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


def _resolve_dim(model_name: str) -> int:
    known = _KNOWN_DIMS.get(model_name)
    if known:
        return known
    from fastembed import TextEmbedding  # heavy; only for unlisted models
    for spec in TextEmbedding.list_supported_models():
        if spec["model"] == model_name:
            log.info("resolved embedding dim for %s via fastembed", model_name)
            return int(spec["dim"])
    raise ValueError(
        f"EMBEDDING_MODEL={model_name!r} is not a supported fastembed model. "
        f"Known-good options: {', '.join(sorted(_KNOWN_DIMS))}"
    )


# Drives the vector(N) column width in schema.sql and docs.py. Never hardcode
# the number anywhere else — a mismatch between this and the DDL is a silent
# insert failure at runtime.
DIM = _resolve_dim(MODEL_NAME)

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


def embed_query(text: str) -> list[float]:
    """Embed a SEARCH QUERY (short, interrogative). Applies the model's query
    prefix where it has one."""
    text = (text or "").strip()
    if not text:
        return []
    vec = next(iter(_get_model().query_embed([text])))
    return [float(x) for x in vec]


def embed_passage(text: str) -> list[float]:
    """Embed a STORED PASSAGE (a fact, a document chunk)."""
    text = (text or "").strip()
    if not text:
        return []
    vec = next(iter(_get_model().passage_embed([text])))
    return [float(x) for x in vec]


def embed(text: str) -> list[float]:
    """Back-compat alias for `embed_passage`.

    Retained because callers that store content (save_entry, chunk ingest)
    read naturally as plain `embed`. Anything on the SEARCH side must call
    `embed_query` instead — passing a query through here silently costs
    retrieval quality rather than failing.
    """
    return embed_passage(text)


def to_pgvector(vec: list[float]) -> Optional[str]:
    """Format a vector as a pgvector literal ('[f1,f2,...]'), or None if empty."""
    if not vec:
        return None
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
