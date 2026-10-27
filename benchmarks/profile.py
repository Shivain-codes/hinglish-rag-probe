"""Warm sequential latency and process high-water RSS; invoke once per method."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

from hinglish_rag_probe.datasets import load_documents, load_queries
from hinglish_rag_probe.evaluation import environment
from hinglish_rag_probe.retrieval import make_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", choices=["tfidf", "bm25", "dense"], default="bm25")
    parser.add_argument("--docs", default="benchmarks/data/documents.csv")
    parser.add_argument("--queries", default="benchmarks/data/queries.csv")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    kwargs = (
        {"model_name": args.model, "revision": args.revision} if args.retriever == "dense" else {}
    )
    engine = make_retriever(args.retriever, **kwargs)
    docs = load_documents(args.docs)
    texts = [v.text for q in load_queries(args.queries) for v in q.all_variants()]
    begin = time.perf_counter()
    engine.index(docs)
    indexing = time.perf_counter() - begin
    engine.search_batch(texts)  # explicit warmup
    times = []
    for _ in range(args.repeats):
        begin = time.perf_counter()
        engine.search_batch(texts)
        times.append(time.perf_counter() - begin)
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
    except ImportError:
        rss_bytes = None
    print(
        json.dumps(
            {
                **environment(),
                "retriever": engine.configuration(),
                "index_seconds": indexing,
                "batch_seconds": times,
                "n_queries": len(texts),
                "median_queries_per_second": len(texts) / statistics.median(times),
                "process_lifetime_peak_rss_bytes": rss_bytes,
                "scope": (
                    "fresh-process lifetime RSS including imports; one batch warmup; no GPU memory"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
