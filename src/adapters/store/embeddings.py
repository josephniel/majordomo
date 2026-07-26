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

Who chooses the model
---------------------
Whoever constructs the Embedder — in practice the composition root, from
resolved config. This module used to read EMBEDDING_MODEL from os.environ at
import time, which never worked: the composition root imports this package
before it loads the instance config, so the value was frozen from the ambient
shell and the documented setting was silently inert. Nothing failed — you
just always got the default. Reading config at import time is reading it
before it exists.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

DEFAULT_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

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
        f"embedding model {model_name!r} is not a supported fastembed model. "
        f"Known-good options: {', '.join(sorted(_KNOWN_DIMS))}"
    )


class Embedder:
    """Turns text into vectors with one specific model.

    Construct it from config and hand it to whatever stores text. Which model
    produced a vector is persisted alongside it
    (`memory_entries.embedding_model`) and determines the width of the vector
    column, so it is a property of the store, not of the process.

    ONE PER PROCESS, created by the entry point and passed down. The loaded
    model is ~640MB resident and lives on the instance, so two Embedders on
    the same model means two copies of it in RAM. That cost is deliberately
    attached to constructing a second one rather than hidden behind a
    module-level cache: sharing is the caller's decision, made visible by
    passing the same object to the memory store and the document store.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model_name = (model or "").strip() or DEFAULT_MODEL
        self._dim: int | None = None
        self._loaded = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"Embedder({self.model_name!r})"

    def __eq__(self, other: object) -> bool:
        # Compared when checking that two stores against one database agree
        # on their model — see the shared-database guard in runtime/.
        return isinstance(other, Embedder) and other.model_name == self.model_name

    def __hash__(self) -> int:
        return hash(self.model_name)

    @property
    def dim(self) -> int:
        """Width of the vector(N) column in schema.sql and docs.py. Never
        hardcode the number anywhere else — a mismatch between this and the
        DDL is a silent insert failure at runtime.

        Resolved on demand: for a listed model it's a dict lookup, and for an
        unlisted one it pulls in fastembed, which is exactly the cost the lazy
        model load below exists to avoid paying up front.
        """
        if self._dim is None:
            self._dim = _resolve_dim(self.model_name)
        return self._dim

    def _model(self):
        if self._loaded is None:
            with self._lock:
                if self._loaded is None:
                    from fastembed import TextEmbedding  # lazy: heavy import
                    log.info(
                        "loading local embedding model %s (first use)", self.model_name
                    )
                    self._loaded = TextEmbedding(model_name=self.model_name)
        return self._loaded

    def embed_query(self, text: str) -> list[float]:
        """Embed a SEARCH QUERY (short, interrogative). Applies the model's
        query prefix where it has one.
        """
        text = (text or "").strip()
        if not text:
            return []
        vec = next(iter(self._model().query_embed([text])))
        return [float(x) for x in vec]

    def embed_passage(self, text: str) -> list[float]:
        """Embed a STORED PASSAGE (a fact, a document chunk)."""
        text = (text or "").strip()
        if not text:
            return []
        vec = next(iter(self._model().passage_embed([text])))
        return [float(x) for x in vec]

    def embed(self, text: str) -> list[float]:
        """Alias for `embed_passage`.

        Retained because callers that store content (save_entry, chunk
        ingest) read naturally as plain `embed`. Anything on the SEARCH side
        must call `embed_query` instead — passing a query through here
        silently costs retrieval quality rather than failing.
        """
        return self.embed_passage(text)


def to_pgvector(vec: list[float]) -> str | None:
    """Format a vector as a pgvector literal ('[f1,f2,...]'), or None if empty."""
    if not vec:
        return None
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
