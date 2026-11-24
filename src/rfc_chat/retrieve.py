import re
from collections import Counter

import numpy as np

from rfc_chat.embed import Embedder
from rfc_chat.text import Chunk, norm


def tokenize(text: str) -> list[str]:
    # Extracts lowercase alphanumeric and punctuation tokens after normalization.
    return re.findall(r"[a-z0-9_@+-]+", norm(text))

class Index:
    # Hybrid search index supporting dense, BM25 and reciprocal rank fusion.
    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        embedder: Embedder | None = None,
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self._init_bm25()

    def _init_bm25(self) -> None:
        # Precomputes BM25 parameters, document frequencies and IDF values.
        self.k1 = 1.5
        self.b = 0.75
        self.corpus_tokens = [tokenize(c.text) for c in self.chunks]
        self.doc_lens = [len(tokens) for tokens in self.corpus_tokens]
        self.n_docs = len(self.chunks)
        self.avgdl = sum(self.doc_lens) / self.n_docs if self.n_docs > 0 else 0.0

        df: Counter[str] = Counter()
        for tokens in self.corpus_tokens:
            df.update(set(tokens))
        self.df = df

        self.idf: dict[str, float] = {}
        for token, freq in df.items():
            self.idf[token] = float(np.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5)))

    def _bm25_score_doc(self, q_tokens: list[str], doc_idx: int) -> float:
        # Calculates BM25 score for a single document against query tokens.
        score = 0.0
        doc_tokens = self.corpus_tokens[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        if doc_len == 0:
            return 0.0
        tf_map = Counter(doc_tokens)
        denom_len = (
            self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
            if self.avgdl > 0
            else self.k1
        )
        for token in q_tokens:
            if token in tf_map and token in self.idf:
                tf = tf_map[token]
                term_score = self.idf[token] * (tf * (self.k1 + 1.0)) / (tf + denom_len)
                score += term_score
        return score

    def search(
        self,
        query: str,
        k: int = 3,
        method: str = "hybrid",
    ) -> list[tuple[Chunk, float]]:
        # Searches index using dense, bm25, or hybrid reciprocal rank fusion.
        if not self.chunks or k <= 0:
            return []

        if method == "dense":
            if self.embedder is None:
                raise ValueError("Embedder is required for dense retrieval")
            q_vec = self.embedder.embed([query])[0]
            scores = self.embeddings @ q_vec
            top_indices = np.argsort(-scores)[:k]
            return [(self.chunks[i], float(scores[i])) for i in top_indices]

        if method == "bm25":
            q_tokens = tokenize(query)
            scores = np.array([self._bm25_score_doc(q_tokens, i) for i in range(self.n_docs)])
            top_indices = np.argsort(-scores)[:k]
            return [(self.chunks[i], float(scores[i])) for i in top_indices]

        if method == "hybrid":
            if self.embedder is None:
                raise ValueError("Embedder is required for hybrid retrieval")
            q_vec = self.embedder.embed([query])[0]
            dense_scores = self.embeddings @ q_vec
            dense_order = np.argsort(-dense_scores)

            q_tokens = tokenize(query)
            bm25_scores = np.array([self._bm25_score_doc(q_tokens, i) for i in range(self.n_docs)])
            bm25_order = np.argsort(-bm25_scores)

            rrf_scores = np.zeros(self.n_docs, dtype=np.float64)
            for rank, idx in enumerate(dense_order, start=1):
                rrf_scores[idx] += 1.0 / (60.0 + rank)
            for rank, idx in enumerate(bm25_order, start=1):
                if bm25_scores[idx] > 0:
                    rrf_scores[idx] += 1.0 / (60.0 + rank)

            top_indices = np.argsort(-rrf_scores)[:k]
            return [(self.chunks[i], float(rrf_scores[i])) for i in top_indices]

        raise ValueError(f"Unknown retrieval method: {method}")
