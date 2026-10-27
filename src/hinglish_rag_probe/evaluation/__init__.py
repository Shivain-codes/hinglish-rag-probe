"""Evaluation orchestration with explicit missing-label and paired-cohort policies."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..metrics import METRICS, relative_degradation, score_ranking
from ..models import Document, EvaluationReport, Query
from ..retrieval import Retriever, make_retriever, validate_k
from ..statistics import paired_comparison


def validate_inputs(documents: Sequence[Document], queries: Sequence[Query]) -> None:
    ids = {d.id for d in documents}
    if not documents or len(ids) != len(documents):
        raise ValueError("documents must be nonempty with unique ids")
    if not queries or len({q.id for q in queries}) != len(queries):
        raise ValueError("queries must be nonempty with unique ids")
    for query in queries:
        missing = set(query.judgments() or {}) - ids
        if missing:
            raise ValueError(f"query {query.id} references unknown documents: {sorted(missing)}")


def environment() -> dict[str, Any]:
    versions = {}
    for name in (
        "hinglish-rag-probe",
        "numpy",
        "scikit-learn",
        "PyYAML",
        "sentence-transformers",
        "torch",
        "faiss-cpu",
        "matplotlib",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "python": sys.version,
        "dependencies": versions,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def evaluate(
    documents: Sequence[Document],
    queries: Sequence[Query],
    k: int = 5,
    retriever: str | Retriever = "bm25",
    *,
    seed: int = 0,
    dataset_metadata: dict[str, Any] | None = None,
    resamples: int = 2000,
    permutations: int = 2000,
    confidence: float = 0.95,
) -> EvaluationReport:
    validate_k(k)
    validate_inputs(documents, queries)
    engine = make_retriever(retriever) if isinstance(retriever, str) else retriever
    if not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not isinstance(permutations, int) or permutations < 1 or not 0 < confidence < 1:
        raise ValueError("permutations must be positive and confidence in (0, 1)")
    document_ids = {d.id for d in documents}
    records: list[dict[str, Any]] = []
    owned_trace = not tracemalloc.is_tracing()
    if owned_trace:
        tracemalloc.start()
    try:
        start = time.perf_counter()
        engine.index(documents)
        index_seconds = time.perf_counter() - start
        for query in queries:
            for variant in query.all_variants():
                start = time.perf_counter()
                hits = engine.search(variant.text, k)
                elapsed = time.perf_counter() - start
                ranking = [hit.id for hit in hits]
                if len(hits) > k or any(not math.isfinite(hit.score) for hit in hits):
                    raise ValueError("retriever returned invalid scores or more than k hits")
                if any(key not in document_ids for key in ranking):
                    raise ValueError("retriever returned an unknown document")
                records.append(
                    {
                        "query_id": query.id,
                        "condition": variant.condition,
                        "text": variant.text,
                        "variant": asdict(variant),
                        "ranking": [asdict(hit) for hit in hits],
                        "metrics": score_ranking(ranking, query.judgments(), k),
                        "latency_seconds": elapsed,
                        "judgment_status": "missing"
                        if query.judgments() is None
                        else "positive"
                        if any(v > 0 for v in (query.judgments() or {}).values())
                        else "empty",
                    }
                )
        peak = tracemalloc.get_traced_memory()[1] if owned_trace else None
    finally:
        if owned_trace:
            tracemalloc.stop()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(record)
    summary: dict[str, dict[str, Any]] = {}
    for condition, rows in grouped.items():
        judged = [r for r in rows if r["metrics"] is not None]
        elapsed = sum(r["latency_seconds"] for r in rows)
        summary[condition] = {
            **{
                m: float(np.mean([r["metrics"][m] for r in judged])) if judged else None
                for m in METRICS
            },
            "n_queries": len(rows),
            "n_judged": len(judged),
            "n_missing": len(rows) - len(judged),
            "n_empty": sum(r["judgment_status"] == "empty" for r in rows),
            "mean_latency_seconds": elapsed / len(rows),
            "throughput_queries_per_second": len(rows) / elapsed if elapsed > 0 else None,
        }
    comparisons = []
    clean = {r["query_id"]: r for r in grouped.get("english", []) if r["metrics"] is not None}
    for condition, rows in grouped.items():
        if condition == "english":
            continue
        paired = [r for r in rows if r["query_id"] in clean and r["metrics"] is not None]
        if not paired:
            continue
        for metric in METRICS:
            a = [clean[r["query_id"]]["metrics"][metric] for r in paired]
            b = [r["metrics"][metric] for r in paired]
            comparisons.append(
                {
                    "kind": "condition",
                    "baseline": "english",
                    "candidate": condition,
                    "metric": metric,
                    "relative_degradation": relative_degradation(
                        float(np.mean(a)), float(np.mean(b))
                    ),
                    **paired_comparison(
                        a,
                        b,
                        seed=seed,
                        resamples=resamples,
                        permutations=permutations,
                        confidence=confidence,
                    ),
                }
            )
    canonical = {
        "documents": [asdict(d) for d in sorted(documents, key=lambda d: d.id)],
        "queries": [asdict(q) for q in sorted(queries, key=lambda q: q.id)],
    }
    encoded = json.dumps(
        canonical, sort_keys=True, ensure_ascii=False, default=lambda value: sorted(value)
    ).encode()
    metadata = {
        **environment(),
        "schema_version": 1,
        "k": k,
        "seed": seed,
        "retriever": engine.configuration(),
        "dataset": dataset_metadata
        or {"name": "unspecified", "source": "user supplied", "license": "unknown"},
        "dataset_sha256": hashlib.sha256(encoded).hexdigest(),
        "n_documents": len(documents),
        "n_queries": len(queries),
        "index_seconds": index_seconds,
        "peak_python_allocated_bytes": peak,
        "memory_scope": (
            "tracemalloc Python allocations during index and search; "
            "excludes most native/GPU memory; null if tracing already active"
        ),
        "timing_scope": (
            "sequential queries, cold first query included, indexing excluded, tracemalloc enabled"
        ),
        "judgment_policy": "missing excluded, judged-empty zero, unjudged documents nonrelevant",
        "statistics": {
            "unit": "original query",
            "confidence": confidence,
            "resamples": resamples,
            "permutations": permutations,
            "multiplicity": "unadjusted exploratory",
        },
    }
    return EvaluationReport(metadata, records, summary, comparisons)


def compare_reports(
    reports: dict[str, EvaluationReport],
    *,
    seed: int = 0,
    resamples: int = 2000,
    permutations: int = 2000,
    confidence: float = 0.95,
) -> list[dict[str, Any]]:
    """Compare each method with the first method, paired within condition."""
    if len(reports) < 2:
        raise ValueError("comparison requires at least two retrievers")
    if len({(r.metadata["dataset_sha256"], r.metadata["k"]) for r in reports.values()}) != 1:
        raise ValueError("comparisons require identical datasets and k")
    baseline = next(iter(reports))
    indexed = {
        name: {
            (r["query_id"], r["condition"]): r["metrics"]
            for r in report.results
            if r["metrics"] is not None
        }
        for name, report in reports.items()
    }
    result = []
    for name in list(reports)[1:]:
        common = sorted(indexed[baseline].keys() & indexed[name].keys())
        for condition in sorted({key[1] for key in common}):
            keys = [key for key in common if key[1] == condition]
            for metric in METRICS:
                a = [indexed[baseline][key][metric] for key in keys]
                b = [indexed[name][key][metric] for key in keys]
                result.append(
                    {
                        "kind": "retriever",
                        "baseline": baseline,
                        "candidate": name,
                        "condition": condition,
                        "metric": metric,
                        **paired_comparison(
                            a,
                            b,
                            seed=seed,
                            resamples=resamples,
                            permutations=permutations,
                            confidence=confidence,
                        ),
                    }
                )
    return result
