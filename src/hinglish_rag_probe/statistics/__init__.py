"""Paired query-level inference, with bounded-memory resampling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def paired_comparison(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    seed: int = 0,
    resamples: int = 2000,
    confidence: float = 0.95,
    permutations: int = 2000,
) -> dict[str, Any]:
    """Percentile bootstrap CI and two-sided paired sign-flip permutation test.

    Difference is candidate minus baseline. Monte Carlo p uses the +1 correction.
    The caller must align independent original query IDs before invoking this.
    """
    a, b = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    if a.ndim != 1 or b.shape != a.shape or a.size == 0:
        raise ValueError("paired samples must be nonempty equal-length vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired samples must be finite")
    if not 0 < confidence < 1 or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("confidence must be in (0,1) and resamples positive")
    if not isinstance(permutations, int) or permutations < 1:
        raise ValueError("permutations must be positive")
    delta = b - a
    observed = float(delta.mean())
    rng = np.random.default_rng(seed)
    boot = np.array(
        [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(resamples)]
    )
    alpha = (1 - confidence) / 2
    low, high = np.quantile(boot, [alpha, 1 - alpha])
    extreme = 0
    for _ in range(permutations):
        simulated = float((delta * rng.choice([-1, 1], size=len(delta))).mean())
        extreme += abs(simulated) >= abs(observed) - 1e-15
    return {
        "difference": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "p_value": (extreme + 1) / (permutations + 1),
        "n_pairs": len(delta),
        "seed": seed,
        "resamples": resamples,
        "permutations": permutations,
        "test": "two-sided paired sign-flip (Monte Carlo)",
        "multiplicity_adjustment": "none; exploratory",
    }
