"""Local, free text embeddings (BAAI/bge-small-en-v1.5 via fastembed/ONNX).

fastembed runs the model with ONNX Runtime, without PyTorch, which keeps the
deployment image small. Vectors are L2-normalized, so cosine similarity is
a dot product.
"""

import copy
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
# bge models are trained with this instruction on the query side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

Vector = npt.NDArray[np.float32]


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> Vector: ...

    def embed_query(self, text: str) -> Vector: ...

    def count_tokens(self, text: str) -> int: ...


class FastEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, cache_dir: Path | None = None) -> None:
        import truststore

        truststore.inject_into_ssl()  # model download behind TLS-inspecting proxies
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.dim = DIM
        self._model = TextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None
        )
        # A non-truncating copy of the model's tokenizer, for sizing chunks.
        # (The model's own tokenizer truncates at 512, which would hide overflow.)
        self._tokenizer = copy.deepcopy(self._model.model.tokenizer)  # type: ignore[attr-defined]
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(list(self._model.embed(list(texts), batch_size=64)), dtype=np.float32)

    def embed_query(self, text: str) -> Vector:
        vec = next(iter(self._model.embed([QUERY_PREFIX + text])))
        return np.asarray(vec, dtype=np.float32)


@lru_cache
def get_embedder(cache_dir: Path | None = None) -> FastEmbedder:
    return FastEmbedder(cache_dir=cache_dir)
