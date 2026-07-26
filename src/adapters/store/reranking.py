"""Local cross-encoder reranking — the calibration layer over hybrid recall.

Why this exists
---------------
Reciprocal Rank Fusion (db.recall_scored) is good at ORDERING and bad at
saying HOW RELEVANT anything is. Because RRF scores are 1/(K + rank) sums,
rank 1 and rank 5 differ by ~7% — after normalization a perfect hit scores
0.86 and an unrelated fact scores 0.79. Any "only inject memories above X"
threshold against that is guesswork.

A cross-encoder reads (query, passage) TOGETHER rather than comparing two
independently-computed vectors, so it can express real judgement. On the same
query the spread is 0.99 / 0.91 / 0.15 / 0.03 — a threshold means something
again. That is the entire job of this module: RRF picks the candidates,
the reranker decides which of them are actually worth the model's context
window.

Cost: ~9ms for 4 passages, ~35ms for 20, on an M4 CPU. Runs once per recall,
against an LLM call measured in seconds. The model is ~120MB — an order of
magnitude smaller than the embedding model, because it only ever sees a
handful of candidates.

Disable with `rerank.enabled: false`, in which case recall falls back to raw
RRF scores (and callers thresholding on them should expect the compressed
range described above).

Who chooses the settings
------------------------
Whoever constructs the Reranker — in practice the composition root, from
resolved config. These were module constants read from os.environ at import
time, which never worked: the composition root imports this package before it
loads the instance config, so every documented RERANK_* knob was silently
inert. Reading config at import time is reading it before it exists.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-12-v2"


@dataclass(frozen=True)
class RerankConfig:
    """Every reranking knob in one value, so `configure` is atomic — a
    partial update can't leave the calibration and the model disagreeing.
    """

    enabled: bool = True
    model: str = DEFAULT_MODEL

    # How many RRF candidates to rerank. Deeper costs linearly and buys less
    # each step: the point of fusion is that the right answer is almost always
    # inside the first ~20, and the reranker's job is to sort and score those,
    # not to rescue something RRF buried at rank 90.
    candidates: int = 20

    # Logit -> 0..1 calibration. A plain sigmoid is WRONG here and it took a
    # measurement to notice: this model's logits live in roughly [-11.3, +5],
    # so sigmoid() maps every negative one to ~1e-5 and they all print as
    # 0.0000 — the separation is still there, but squashed below float display
    # precision, and any threshold against it silently rejects everything.
    #
    # Measured on evals/recall_cases.yaml (max logit per query):
    #     off-topic queries   -11.23 .. -11.27   (a tight noise floor)
    #     on-topic queries    -10.47 .. +5.02
    #
    # So the decision boundary sits near -8, not 0. Recentering there and
    # widening the slope spreads the useful range across 0..1:
    #     -11.2  -> 0.17     (noise)
    #     -10.5  -> 0.23     (hard but real: "how should you talk to me")
    #      -4.4  -> 0.86
    #      +5.0  -> 1.00
    center: float = -8.0
    temperature: float = 2.0


class Reranker:
    """Scores (query, passage) pairs with one specific cross-encoder.

    Construct it from config and hand it to whatever recalls — one per
    process, like the Embedder, for the same reason (the model lives on the
    instance), though at a tenth the size.

    Unlike the embedder, nothing persisted is tagged with the reranker — it
    is a pure read-path enhancement, so switching it is free and failing to
    load it degrades to RRF rather than breaking recall.
    """

    def __init__(self, config: RerankConfig | None = None) -> None:
        self.config = config or RerankConfig()
        self._loaded = None
        self._unavailable = False
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        state = "on" if self.config.enabled else "off"
        return f"Reranker({self.config.model!r}, {state})"

    @property
    def candidates(self) -> int:
        """How deep a slice recall should fuse before reranking it."""
        return self.config.candidates

    def available(self) -> bool:
        return self.config.enabled and not self._unavailable

    def _model(self):
        """Lazily load the cross-encoder. Returns None (once, loudly) if it
        can't be loaded — reranking is an enhancement, not a dependency, and a
        missing model must degrade to RRF rather than break recall.
        """
        if self._unavailable:
            return None
        if self._loaded is None:
            with self._lock:
                if self._loaded is None and not self._unavailable:
                    try:
                        from fastembed.rerank.cross_encoder import TextCrossEncoder
                        log.info("loading reranker %s (first use)", self.config.model)
                        self._loaded = TextCrossEncoder(model_name=self.config.model)
                    except Exception:
                        self._unavailable = True
                        log.warning(
                            "reranker %s unavailable; falling back to RRF scores",
                            self.config.model, exc_info=True,
                        )
                        return None
        return self._loaded

    def rerank(self, query: str, passages: Sequence[str]) -> list[float] | None:
        """Score each passage against the query in 0..1, aligned to `passages`.

        Returns None when reranking is off or unavailable, so callers keep
        their existing scores rather than silently receiving a different scale.

        CPU-bound and synchronous — call via asyncio.to_thread.
        """
        if not self.config.enabled or not query or not passages:
            return None
        model = self._model()
        if model is None:
            return None
        try:
            raw = list(model.rerank(query, list(passages)))
        except Exception:
            log.warning("rerank failed; falling back to RRF scores", exc_info=True)
            return None
        if len(raw) != len(passages):
            log.warning(
                "reranker returned %d scores for %d passages; ignoring",
                len(raw), len(passages),
            )
            return None
        return [self._calibrate(float(s)) for s in raw]

    def _calibrate(self, logit: float) -> float:
        """Squash a cross-encoder logit to 0..1 around the measured decision
        boundary (see RerankConfig.center). Overflow-safe at the tails.
        """
        z = (logit - self.config.center) / (self.config.temperature or 1.0)
        if z < -60:
            return 0.0
        if z > 60:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))
