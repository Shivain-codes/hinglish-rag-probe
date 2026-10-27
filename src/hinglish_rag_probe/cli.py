"""hragprobe command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .datasets import load_documents, load_queries, save_queries
from .evaluation import compare_reports, evaluate
from .perturbations import TRANSFORMS, perturb_query
from .reporting import render_report
from .retrieval import make_retriever


def run_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    base = path.parent
    docs = load_documents(base / config["documents"])
    queries = load_queries(base / config["queries"])
    for settings in config.get("perturbations", []):
        queries = [perturb_query(q, seed=config.get("seed", 0), **settings) for q in queries]
    reports = {}
    stats = {
        key: config[key]
        for key in ("seed", "resamples", "permutations", "confidence")
        if key in config
    }
    dataset = config.get("dataset", {})
    if isinstance(dataset, str):
        dataset = json.loads((base / dataset).read_text(encoding="utf-8"))
    for spec in config["retrievers"]:
        spec = dict(spec)
        name = spec.pop("name")
        label = spec.pop("label", name)
        if label in reports:
            raise ValueError("retriever labels must be unique")
        print(f"Evaluating {label}…", file=sys.stderr)
        reports[label] = evaluate(
            docs,
            queries,
            k=config.get("k", 5),
            retriever=make_retriever(name, **spec),
            dataset_metadata=dataset,
            **stats,
        )
    comparisons = compare_reports(reports, **stats)
    payload = {
        "reports": {name: r.to_dict() for name, r in reports.items()},
        "comparisons": comparisons,
    }
    output = base / config.get("output", "results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    if config.get("report"):
        render_report(payload, base / config["report"])
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hragprobe", description="Evaluate multilingual retrieval robustness"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="Evaluate CSV/JSON/JSONL documents and queries")
    ev.add_argument("--docs", required=True)
    ev.add_argument("--queries", required=True)
    ev.add_argument("--retriever", choices=("bm25", "tfidf", "dense"), default="bm25")
    ev.add_argument("--k", type=int, default=5)
    ev.add_argument("--output", required=True)
    ev.add_argument("--model", help="Dense model name or local directory")
    ev.add_argument("--revision")
    ev.add_argument(
        "--allow-download", action="store_true", help="Explicitly permit dense model downloads"
    )
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--manifest", help="Dataset provenance JSON")
    cp = sub.add_parser(
        "compare", help="Run reproducible YAML experiment; paths relative to config"
    )
    cp.add_argument("--config", required=True)
    pe = sub.add_parser("perturb", help="Append seeded, unvalidated query variants")
    pe.add_argument("--input", required=True)
    pe.add_argument("--output", required=True)
    pe.add_argument("--transformation", choices=TRANSFORMS, default="typo")
    pe.add_argument("--intensity", type=float, default=0.1)
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--source-condition", default="english")
    rp = sub.add_parser("report", help="Create self-contained HTML and SVG/PDF/PNG figures")
    rp.add_argument("--input", required=True)
    rp.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            kwargs = {}
            if args.retriever == "dense":
                if not args.model:
                    raise ValueError("--model is required for dense retrieval")
                kwargs = {
                    "model_name": args.model,
                    "revision": args.revision,
                    "allow_download": args.allow_download,
                }
            metadata = json.loads(Path(args.manifest).read_text()) if args.manifest else None
            print(f"Evaluating {args.retriever}…", file=sys.stderr)
            report = evaluate(
                load_documents(args.docs),
                load_queries(args.queries),
                args.k,
                make_retriever(args.retriever, **kwargs),
                seed=args.seed,
                dataset_metadata=metadata,
            )
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(report.to_json(), encoding="utf-8")
        elif args.command == "compare":
            run_config(args.config)
        elif args.command == "perturb":
            queries = [
                perturb_query(
                    q,
                    transformation=args.transformation,
                    intensity=args.intensity,
                    seed=args.seed,
                    source_condition=args.source_condition,
                )
                for q in load_queries(args.input)
            ]
            save_queries(args.output, queries)
        else:
            render_report(json.loads(Path(args.input).read_text(encoding="utf-8")), args.output)
    except (ValueError, TypeError, KeyError, OSError, ImportError, yaml.YAMLError) as exc:
        print(f"hragprobe: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
