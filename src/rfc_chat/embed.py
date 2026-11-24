import hashlib
import re
from typing import Protocol

import numpy as np

from rfc_chat.text import norm


class Embedder(Protocol):
    # Embedding interface returning L2-normalized float32 vectors.
    def embed(self, texts: list[str]) -> np.ndarray:
        ...

class FastEmbedder:
    # Lazy-loading FastEmbedder using BAAI/bge-small-en-v1.5.
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        # Instantiate FastEmbed TextEmbedding only on first access.
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        # Generates L2-normalized float32 embeddings.
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        embeddings = list(self.model.embed(texts))
        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

class HashingEmbedder:
    # Deterministic bag-of-words hashing embedder for testing.
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        # Computes deterministic L2-normalized bag-of-words hash vectors.
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9_@+-]+", norm(text))
            for token in tokens:
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                bucket = h % self.dim
                sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
                matrix[i, bucket] += sign
            norm_val = float(np.linalg.norm(matrix[i]))
            if norm_val > 0:
                matrix[i] /= norm_val
        return matrix
