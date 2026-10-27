"""Explicit binary and graded top-k metric conventions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ..retrieval import validate_k

METRICS = ("precision", "recall", "hit_rate", "mrr", "map", "ndcg")


def score_ranking(
    ranking: Sequence[str],
    relevance: Mapping[str, float] | None,
    k: int,
) -> dict[str, float] | None:
    """AP denominator is min(number of positive judgments, k); P denominator is k.

    Missing judgments return None (excluded). Judged-empty queries score zero.
    Unjudged documents count as nonrelevant. NDCG uses 2**grade-1 gain.
    """
    validate_k(k)
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking contains duplicate document ids")
    if relevance is None:
        return None
    if any(not math.isfinite(v) or not 0 <= v <= 100 for v in relevance.values()):
        raise ValueError("relevance grades must be finite and in [0, 100]")
    relevant = {key for key, value in relevance.items() if value > 0}
    hits = [int(key in relevant) for key in ranking[:k]]
    count = sum(hits)
    ap = sum(sum(hits[:i]) / i for i, hit in enumerate(hits, 1) if hit)
    dcg = sum(
        (2 ** relevance.get(key, 0) - 1) / math.log2(i + 1) for i, key in enumerate(ranking[:k], 1)
    )
    ideal = sum(
        (2**grade - 1) / math.log2(i + 1)
        for i, grade in enumerate(sorted(relevance.values(), reverse=True)[:k], 1)
    )
    return {
        "precision": count / k,
        "recall": count / len(relevant) if relevant else 0.0,
        "hit_rate": float(count > 0),
        "mrr": next((1 / i for i, hit in enumerate(hits, 1) if hit), 0.0),
        "map": ap / min(len(relevant), k) if relevant else 0.0,
        "ndcg": dcg / ideal if ideal else 0.0,
    }


def relative_degradation(baseline: float, value: float) -> float | None:
    """(baseline - value) / baseline; undefined for a zero baseline."""
    if not math.isfinite(baseline) or not math.isfinite(value):
        raise ValueError("metrics must be finite")
    return (baseline - value) / baseline if baseline != 0 else None
