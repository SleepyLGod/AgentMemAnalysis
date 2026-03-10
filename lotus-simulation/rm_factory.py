from __future__ import annotations

import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lotus.models.rm import RM
from sim_config import RMConfig


class HashingRM(RM):
    """
    Lightweight local RM for environments where torch/sentence-transformers
    cannot be imported reliably. This is deterministic and dependency-light.
    """

    def __init__(self, dim: int = 1024) -> None:
        super().__init__()
        self.dim = max(64, int(dim))
        self._token_pattern = re.compile(r"\b\w+\b", re.UNICODE)

    def _embed(self, docs: list[str]) -> NDArray[np.float64]:
        out = np.zeros((len(docs), self.dim), dtype=np.float64)
        for row_idx, text in enumerate(docs):
            for token in self._token_pattern.findall((text or "").lower()):
                col = hash(token) % self.dim
                out[row_idx, col] += 1.0
            norm = np.linalg.norm(out[row_idx])
            if norm > 0:
                out[row_idx] /= norm
        return out


def build_rm(cfg: RMConfig) -> RM:
    backend = (cfg.backend or "hashing").strip().lower()
    kwargs: dict[str, Any] = dict(cfg.kwargs or {})

    if backend in {"sentence_transformers", "sentence-transformers", "st"}:
        from lotus.models.sentence_transformers_rm import SentenceTransformersRM

        return SentenceTransformersRM(model=cfg.model, **kwargs)

    if backend in {"litellm", "lite_lm", "lite-llm"}:
        from lotus.models.litellm_rm import LiteLLMRM

        return LiteLLMRM(model=cfg.model, **kwargs)

    if backend in {"hashing", "hash"}:
        return HashingRM(dim=int(kwargs.get("dim", 1024)))

    raise ValueError(
        f"Unsupported rm backend: {cfg.backend!r}. "
        "Use one of: sentence_transformers, litellm, hashing."
    )
