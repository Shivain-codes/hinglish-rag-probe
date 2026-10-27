"""Portable UTF-8 IO and manifest-gated local dataset imports."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..models import Document, Query, QueryVariant


def read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if path.suffix == ".csv":
            rows = list(csv.DictReader(handle))
        elif path.suffix == ".jsonl":
            rows = [json.loads(line) for line in handle if line.strip()]
        elif path.suffix == ".json":
            rows = json.load(handle)
        else:
            raise ValueError("supported dataset formats: .csv, .json, .jsonl")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("dataset must contain an array of records")
    return rows


def write_records(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix not in (".csv", ".json", ".jsonl"):
        raise ValueError("supported dataset formats: .csv, .json, .jsonl")
    with path.open("w", encoding="utf-8", newline="") as handle:
        if path.suffix == ".csv":
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict, bool))
                    else value
                    for key, value in row.items()
                }
                for row in rows
            )
        elif path.suffix == ".jsonl":
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            json.dump(rows, handle, ensure_ascii=False, indent=2, allow_nan=False)


def load_documents(path: str | Path) -> list[Document]:
    return [Document(id=row["id"], text=row["text"]) for row in read_records(path)]


def load_queries(path: str | Path) -> list[Query]:
    result = []
    for raw in read_records(path):
        row = {key: value for key, value in raw.items() if value is not None and value != ""}
        for key in ("relevant_ids", "relevance", "variants", "human_validated"):
            if key in row and isinstance(row[key], str):
                row[key] = json.loads(row[key])
        if row.get("relevant_ids") is not None:
            row["relevant_ids"] = frozenset(row["relevant_ids"])
        row["variants"] = tuple(QueryVariant(**v) for v in row.get("variants", []))
        result.append(Query(**row))
    return result


def save_queries(path: str | Path, queries: list[Query]) -> None:
    rows = []
    for query in queries:
        row = asdict(query)
        row["relevant_ids"] = sorted(query.relevant_ids) if query.relevant_ids is not None else None
        row["variants"] = list(row["variants"])
        rows.append(row)
    write_records(path, rows)


def checksum(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_dataset(
    documents_path: str | Path,
    queries_path: str | Path,
    manifest: dict[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    """Normalize licensed local data. Does not independently establish legal rights."""
    required = ("name", "source", "license", "version", "relevance_definition")
    if any(not isinstance(manifest.get(key), str) or not manifest[key].strip() for key in required):
        raise ValueError(f"manifest requires nonempty fields: {', '.join(required)}")
    if manifest["license"].casefold() in ("unknown", "tbd", "unlicensed"):
        raise ValueError("a known dataset license is required")
    docs, queries = load_documents(documents_path), load_queries(queries_path)
    from ..evaluation import validate_inputs

    validate_inputs(docs, queries)
    seen: set[str] = set()
    for query in queries:
        text = " ".join(query.all_variants()[0].text.casefold().split())
        if text in seen:
            raise ValueError("duplicate original query texts; resolve before importing")
        seen.add(text)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        **manifest,
        "source_checksums": {
            "documents": checksum(documents_path),
            "queries": checksum(queries_path),
        },
        "n_documents": len(docs),
        "n_queries": len(queries),
    }
    write_records(output / "documents.jsonl", [asdict(d) for d in docs])
    save_queries(output / "queries.jsonl", queries)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
