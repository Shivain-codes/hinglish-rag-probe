import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hinglish_rag_probe import Document
from hinglish_rag_probe.retrieval import (
    BM25Retriever,
    DenseRetriever,
    TfidfRetriever,
    make_retriever,
    tokenize,
)

DOCS = [
    Document("b", "water water roots"),
    Document("a", "sunlight plants"),
    Document("c", "पानी पौधे"),
]


@pytest.mark.parametrize("factory", [TfidfRetriever, BM25Retriever])
def test_sparse_rankings_ties_batch_and_reindex(factory):
    engine = factory()
    with pytest.raises(ValueError):
        engine.search("water")
    engine.index(DOCS)
    assert engine.search("water", 1)[0].id == "b"
    assert engine.search("पानी", 1)[0].id == "c"
    assert [h.id for h in engine.search("unknown", 5)] == ["a", "b", "c"]
    before = engine.search_batch(["water", "sunlight"], 2)
    engine.index(DOCS[::-1])
    assert before == engine.search_batch(["water", "sunlight"], 2)
    with pytest.raises(ValueError):
        engine.search("  ")
    with pytest.raises(ValueError):
        engine.index([DOCS[0], DOCS[0]])
    with pytest.raises(ValueError):
        engine.index([])
    assert engine.configuration()["name"] in ("tfidf", "bm25")


def test_bm25_score_and_tfidf_cosine():
    docs = [Document("a", "cat cat"), Document("b", "dog dog")]
    bm = BM25Retriever()
    bm.index(docs)
    assert bm.search("cat")[0].score == pytest.approx(math.log(2) * 2 * 2.5 / 3.5)
    tf = TfidfRetriever()
    tf.index(docs)
    assert tf.search("cat")[0].score == pytest.approx(1)
    assert tokenize("पानी, पौधे! CAT") == ["पानी", "पौधे", "cat"]
    with pytest.raises(ValueError):
        tf.index([Document("x", "!!!")])
    for kwargs in ({"k1": 0}, {"b": 2}, {"b": float("nan")}):
        with pytest.raises(ValueError):
            BM25Retriever(**kwargs)
    with pytest.raises(ValueError):
        make_retriever("bad")


class Encoder:
    def encode(self, texts, **kwargs):
        return np.array([[1, 0] if "water" in t else [0, 1] for t in texts], dtype=float)


def test_dense_injected_encoder_and_faiss():
    dense = DenseRetriever("fixture", encoder=Encoder())
    dense.index(DOCS)
    assert dense.search("water")[0].id == "b"
    assert [h.id for h in dense.search("sunlight")] == ["a", "c", "b"]
    assert len(dense.search_batch(["water", "sunlight"])) == 2
    assert dense.search_batch([]) == []
    assert dense.configuration()["injected_encoder"]
    pytest.importorskip("faiss")
    faiss = DenseRetriever("fixture", encoder=Encoder(), use_faiss=True)
    faiss.index(DOCS)
    assert faiss.search_batch(["water", "sunlight"]) == dense.search_batch(["water", "sunlight"])


def test_dense_lazy_loader_and_errors(monkeypatch):
    seen = {}

    def factory(*args, **kwargs):
        seen.update(kwargs)
        return Encoder()

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=factory)
    )
    dense = DenseRetriever("fixture", revision="abc")
    dense.index(DOCS)
    assert seen["local_files_only"] is True and seen["revision"] == "abc"
    assert seen["trust_remote_code"] is False
    assert dense.search("water")[0].id == "b"

    def broken(*args, **kwargs):
        raise OSError("missing")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=broken)
    )
    with pytest.raises(ValueError, match="unavailable locally"):
        DenseRetriever("missing").index(DOCS)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="dense"):
        DenseRetriever("missing").index(DOCS)
    monkeypatch.setitem(sys.modules, "faiss", None)
    with pytest.raises(ImportError, match="faiss"):
        DenseRetriever("fixture", encoder=Encoder(), use_faiss=True).index(DOCS)


@pytest.mark.parametrize("vectors", [[[0, 0]], [[float("nan"), 1]], [1, 2]])
def test_invalid_embeddings(vectors):
    encoder = SimpleNamespace(encode=lambda *args, **kwargs: vectors)
    with pytest.raises(ValueError):
        DenseRetriever("fixture", encoder=encoder).index([DOCS[0]])


@pytest.mark.model_download
@pytest.mark.parametrize("use_faiss", [False, True])
def test_real_sentence_transformer(use_faiss):
    model = os.environ.get("HRAG_TEST_MODEL")
    if not model:
        pytest.skip("Set HRAG_TEST_MODEL to explicitly select a local or downloadable model")
    pytest.importorskip("sentence_transformers")
    if use_faiss:
        pytest.importorskip("faiss")
    engine = DenseRetriever(
        model,
        revision=os.environ.get("HRAG_TEST_REVISION"),
        allow_download=os.environ.get("HRAG_ALLOW_DOWNLOAD") == "1",
        use_faiss=use_faiss,
    )
    docs = [
        Document("water", "Water is essential for plants."),
        Document("library", "A library lends books to readers."),
    ]
    engine.index(docs)
    assert engine.search("A library lends books to readers.", 1)[0].id == "library"
    assert len(engine.search_batch(["पानी", "paani"], 2)) == 2
    assert engine.configuration()["injected_encoder"] is False
