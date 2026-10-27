"""Deterministic sparse retrieval and optional local-first dense embeddings."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..models import Document, SearchHit, nonempty


def tokenize(text: str) -> list[str]:
    """Unicode words including Devanagari combining marks; no stemming/translation."""
    chars = (
        c if unicodedata.category(c)[0] in "LNM" else " "
        for c in unicodedata.normalize("NFC", text).casefold()
    )
    return "".join(chars).split()


def validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def corpus(documents: Sequence[Document]) -> list[Document]:
    if not documents or len({d.id for d in documents}) != len(documents):
        raise ValueError("documents must be nonempty with unique ids")
    return sorted(documents, key=lambda d: d.id)


class Retriever(Protocol):
    def index(self, documents: Sequence[Document]) -> None: ...
    def search(self, query: str, k: int = 5) -> list[SearchHit]: ...
    def search_batch(self, queries: Sequence[str], k: int = 5) -> list[list[SearchHit]]: ...
    def configuration(self) -> dict[str, Any]: ...


class BaseRetriever:
    documents: list[Document]

    def _check(self, query: str, k: int) -> None:
        validate_k(k)
        nonempty(query, "query")
        if not getattr(self, "documents", None):
            raise ValueError("index documents before searching")

    def _rank(self, scores: Any, k: int) -> list[SearchHit]:
        values = np.asarray(scores).ravel()
        if len(values) != len(self.documents) or not np.isfinite(values).all():
            raise ValueError("retriever produced invalid scores")
        order = np.argsort(-values, kind="stable")[:k]
        return [SearchHit(self.documents[i].id, float(values[i])) for i in order]

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        raise NotImplementedError

    def search_batch(self, queries: Sequence[str], k: int = 5) -> list[list[SearchHit]]:
        validate_k(k)
        return [self.search(query, k) for query in queries]


class TfidfRetriever(BaseRetriever):
    def index(self, documents: Sequence[Document]) -> None:
        self.documents = corpus(documents)
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
        try:
            self.matrix = self.vectorizer.fit_transform([d.text for d in self.documents])
        except ValueError as exc:
            raise ValueError("TF-IDF corpus has no indexable words") from exc

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        self._check(query, k)
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray()
        return self._rank(scores, k)

    def configuration(self) -> dict[str, Any]:
        return {"name": "tfidf", "tokenizer": "unicode_NFC_casefold_v1", "norm": "l2"}


class BM25Retriever(BaseRetriever):
    """Internal Okapi BM25 with positive log(1 + (N-df+.5)/(df+.5)) IDF."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b

    def index(self, documents: Sequence[Document]) -> None:
        self.documents = corpus(documents)
        self.counts = [Counter(tokenize(d.text)) for d in self.documents]
        self.lengths = np.array([sum(c.values()) for c in self.counts])
        self.average = float(self.lengths.mean()) or 1.0
        df: Counter[str] = Counter()
        for count in self.counts:
            df.update(count.keys())
        n = len(self.documents)
        self.idf = {term: math.log1p((n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        self._check(query, k)
        scores = np.zeros(len(self.documents))
        for term in sorted(set(tokenize(query))):
            freq = np.array([count[term] for count in self.counts])
            denom = freq + self.k1 * (1 - self.b + self.b * self.lengths / self.average)
            scores += self.idf.get(term, 0) * freq * (self.k1 + 1) / denom
        return self._rank(scores, k)

    def configuration(self) -> dict[str, Any]:
        return {"name": "bm25", "k1": self.k1, "b": self.b, "tokenizer": "unicode_NFC_casefold_v1"}


class DenseRetriever(BaseRetriever):
    """Cosine similarity; all documents re-ranked to guarantee deterministic ties.

    Downloads are disabled by default. Inject an encoder implementing encode() for
    offline integrations. FAISS uses exact inner products, not approximate search.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        *,
        allow_download: bool = False,
        batch_size: int = 32,
        device: str = "cpu",
        use_faiss: bool = False,
        encoder: Any = None,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        nonempty(model_name, "model_name")
        validate_k(batch_size)
        self.model_name, self.revision = model_name, revision
        self.batch_size, self.device, self.use_faiss = batch_size, device, use_faiss
        self.query_prefix, self.document_prefix = query_prefix, document_prefix
        self.allow_download = allow_download
        self.injected = encoder is not None
        self.encoder = encoder

    def _encode(self, texts: Sequence[str]) -> Any:
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("Install hinglish-rag-probe[dense] for dense retrieval") from exc
            try:
                self.encoder = SentenceTransformer(
                    self.model_name,
                    revision=self.revision,
                    device=self.device,
                    local_files_only=not self.allow_download,
                    trust_remote_code=False,
                )
            except OSError as exc:
                raise ValueError(
                    "Model unavailable locally; explicitly set allow_download=True"
                ) from exc
        vectors = np.asarray(
            self.encoder.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
            raise ValueError("encoder must return a finite 2D embedding array")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("encoder returned zero-length embeddings")
        return np.ascontiguousarray(vectors / norms)

    def index(self, documents: Sequence[Document]) -> None:
        self.documents = corpus(documents)
        self.vectors = self._encode([self.document_prefix + d.text for d in self.documents])
        if self.use_faiss:
            try:
                import faiss
            except ImportError as exc:
                raise ImportError("Install hinglish-rag-probe[faiss] for FAISS indexing") from exc
            self.faiss_index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.faiss_index.add(self.vectors)

    def search_batch(self, queries: Sequence[str], k: int = 5) -> list[list[SearchHit]]:
        validate_k(k)
        if not queries:
            return []
        for query in queries:
            self._check(query, k)
        vectors = self._encode([self.query_prefix + q for q in queries])
        if vectors.shape[1] != self.vectors.shape[1]:
            raise ValueError("query and document embedding dimensions differ")
        if self.use_faiss:
            distances, indices = self.faiss_index.search(vectors, len(self.documents))
            scores = np.empty_like(distances)
            np.put_along_axis(scores, indices, distances, axis=1)
        else:
            scores = vectors @ self.vectors.T
        return [self._rank(row, k) for row in scores]

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        return self.search_batch([query], k)[0]

    def configuration(self) -> dict[str, Any]:
        return {
            "name": "dense",
            "model_name": self.model_name,
            "revision": self.revision,
            "batch_size": self.batch_size,
            "device": self.device,
            "use_faiss": self.use_faiss,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "allow_download": self.allow_download,
            "injected_encoder": self.injected,
        }


def make_retriever(name: str, **kwargs: Any) -> Retriever:
    constructors = {"tfidf": TfidfRetriever, "bm25": BM25Retriever, "dense": DenseRetriever}
    if name not in constructors:
        raise ValueError(f"Unknown retriever {name!r}; choose tfidf, bm25 or dense")
    return constructors[name](**kwargs)  # type: ignore[abstract]
