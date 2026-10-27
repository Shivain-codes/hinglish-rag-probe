import json
import tracemalloc
from dataclasses import asdict

import pytest
import yaml

from hinglish_rag_probe import Document, EvaluationReport, Query, evaluate
from hinglish_rag_probe.cli import main, run_config
from hinglish_rag_probe.datasets import (
    import_dataset,
    load_documents,
    load_queries,
    read_records,
    save_queries,
    write_records,
)
from hinglish_rag_probe.evaluation import compare_reports
from hinglish_rag_probe.reporting import render_report

DOCS = [Document("a", "sunlight plants"), Document("b", "water rain")]
QUERIES = [
    Query("q", english="water", hinglish="water chahiye", relevant_ids={"b"}),
    Query("empty", english="sunlight", relevant_ids=frozenset()),
    Query("missing", english="plants"),
]


def test_evaluation_missing_empty_serialization_and_pairs():
    report = evaluate(DOCS, QUERIES, k=1, resamples=20, permutations=20)
    assert report.summary["english"]["recall"] == 0.5
    assert report.summary["english"]["n_missing"] == 1
    assert report.summary["english"]["n_empty"] == 1
    assert report.summary["hinglish"]["recall"] == 1
    # English aggregate includes an extra judged-empty query; pairing must not.
    assert all(c["difference"] == 0 and c["n_pairs"] == 1 for c in report.comparisons)
    assert all(c["relative_degradation"] == 0 for c in report.comparisons)
    assert EvaluationReport.from_json(str(report)).to_dict() == report.to_dict()
    other = evaluate(
        DOCS[::-1], QUERIES[::-1], k=1, retriever="tfidf", resamples=20, permutations=20
    )
    assert other.metadata["dataset_sha256"] == report.metadata["dataset_sha256"]
    comparisons = compare_reports({"bm25": report, "tfidf": other}, resamples=20, permutations=20)
    assert len(comparisons) == 12
    assert all(c["difference"] == 0 for c in comparisons)
    with pytest.raises(ValueError):
        compare_reports({"one": report})
    other.metadata["k"] = 4
    with pytest.raises(ValueError):
        compare_reports({"one": report, "two": other})


def test_no_judgments_and_existing_tracing():
    tracemalloc.start()
    try:
        report = evaluate(
            DOCS, [Query("q", english="hi", hindi="पानी")], resamples=10, permutations=10
        )
        assert tracemalloc.is_tracing()
        assert report.metadata["peak_python_allocated_bytes"] is None
        assert report.summary["english"]["recall"] is None
        assert not report.comparisons
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize(
    "docs,queries",
    [
        ([], QUERIES),
        (DOCS, []),
        (DOCS, [QUERIES[0]] * 2),
        (DOCS, [Query("q", english="x", relevant_ids={"missing"})]),
    ],
)
def test_invalid_eval_inputs(docs, queries):
    with pytest.raises(ValueError):
        evaluate(docs, queries)


@pytest.mark.parametrize("suffix", ["csv", "json", "jsonl"])
def test_dataset_roundtrip(tmp_path, suffix):
    docs_path, query_path = tmp_path / f"docs.{suffix}", tmp_path / f"queries.{suffix}"
    write_records(docs_path, [asdict(d) for d in DOCS])
    save_queries(query_path, QUERIES)
    assert load_documents(docs_path) == DOCS
    assert load_queries(query_path) == QUERIES
    manifest = {
        "name": "test",
        "source": "local fixture",
        "license": "MIT",
        "version": "1",
        "relevance_definition": "binary",
    }
    imported = import_dataset(docs_path, query_path, manifest, tmp_path / "import")
    assert len(imported["source_checksums"]["documents"]) == 64
    assert load_queries(tmp_path / "import/queries.jsonl") == QUERIES
    for bad in ({}, {**manifest, "license": "unknown"}):
        with pytest.raises(ValueError):
            import_dataset(docs_path, query_path, bad, tmp_path / "bad")
    save_queries(query_path, [Query("q1", english="same"), Query("q2", english="same")])
    with pytest.raises(ValueError, match="duplicate"):
        import_dataset(docs_path, query_path, manifest, tmp_path / "duplicate")


def test_invalid_io(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("{}")
    with pytest.raises(ValueError):
        read_records(path)
    with pytest.raises(ValueError):
        write_records(path, [])
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        read_records(path)


def test_cli_full_pipeline_and_report(tmp_path, capsys):
    d, q = tmp_path / "docs.csv", tmp_path / "queries.csv"
    write_records(d, [asdict(doc) for doc in DOCS])
    save_queries(q, QUERIES)
    out = tmp_path / "results.json"
    assert (
        main(["evaluate", "--docs", str(d), "--queries", str(q), "--k", "1", "--output", str(out)])
        == 0
    )
    assert json.loads(out.read_text())["summary"]["english"]["recall"] == 0.5
    perturbed = tmp_path / "perturbed.csv"
    assert main(["perturb", "--input", str(q), "--output", str(perturbed)]) == 0
    assert load_queries(perturbed)[0].variants[0].settings["seed"] == 0
    config = {
        "documents": "docs.csv",
        "queries": "queries.csv",
        "k": 1,
        "retrievers": [{"name": "tfidf"}, {"name": "bm25"}],
        "resamples": 20,
        "permutations": 20,
        "output": "compare.json",
        "perturbations": [{"transformation": "insert", "intensity": 0.1}],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["compare", "--config", str(config_path)]) == 0
    assert (
        main(
            [
                "report",
                "--input",
                str(tmp_path / "compare.json"),
                "--output",
                str(tmp_path / "report"),
            ]
        )
        == 0
    )
    html = (tmp_path / "report/index.html").read_text()
    assert "data:image/png;base64," in html
    assert len(list((tmp_path / "report").glob("*.svg"))) == 6
    assert len(list((tmp_path / "report").glob("*.pdf"))) == 6
    assert len(list((tmp_path / "report").glob("*.png"))) == 6
    assert (
        main(
            [
                "evaluate",
                "--docs",
                str(d),
                "--queries",
                str(q),
                "--retriever",
                "dense",
                "--output",
                str(out),
            ]
        )
        == 2
    )
    assert "--model is required" in capsys.readouterr().err
    assert (
        main(["evaluate", "--docs", str(d), "--queries", str(q), "--k", "0", "--output", str(out)])
        == 2
    )
    config_path.write_text("[]")
    with pytest.raises(ValueError):
        run_config(config_path)
    config["retrievers"] = [{"name": "bm25"}, {"name": "bm25"}]
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="unique"):
        run_config(config_path)


def test_single_report_escapes_metadata_and_handles_no_pairs(tmp_path):
    report = evaluate(
        DOCS, [Query("q", english="water")], dataset_metadata={"name": "<script>alert(1)</script>"}
    )
    path = render_report(report.to_dict(), tmp_path)
    assert "<script>" not in path.read_text()
    assert "&lt;script&gt;" in path.read_text()


@pytest.mark.parametrize("kwargs", [{"resamples": 0}, {"permutations": 0}, {"confidence": 1}])
def test_invalid_statistics_even_without_pairs(kwargs):
    with pytest.raises(ValueError):
        evaluate(DOCS, [Query("q", english="water")], **kwargs)


def test_invalid_custom_retrieval_results():
    from hinglish_rag_probe.models import SearchHit
    from hinglish_rag_probe.retrieval import BM25Retriever

    class BrokenRetriever(BM25Retriever):
        def __init__(self, hits):
            super().__init__()
            self.hits = hits

        def search(self, query, k=5):
            return self.hits

    for hits in (
        [SearchHit("unknown", 1)],
        [SearchHit("a", float("nan"))],
        [SearchHit("a", 1), SearchHit("b", 0)],
    ):
        with pytest.raises(ValueError):
            evaluate(DOCS, QUERIES, k=1, retriever=BrokenRetriever(hits))
