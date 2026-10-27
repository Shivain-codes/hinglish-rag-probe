import math

import pytest

from hinglish_rag_probe.metrics import relative_degradation, score_ranking
from hinglish_rag_probe.statistics import paired_comparison


def test_hand_calculated_metrics():
    scores = score_ranking(["a", "x", "b"], {"a": 2, "b": 1, "c": 1}, 3)
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["hit_rate"] == 1
    assert scores["mrr"] == 1
    assert scores["map"] == pytest.approx((1 + 2 / 3) / 3)
    assert scores["ndcg"] == pytest.approx(3.5 / (3 + 1 / math.log2(3) + 0.5))
    later = score_ranking(["x", "a"], {"a": 1}, 2)
    assert later["mrr"] == 0.5
    assert later["map"] == 0.5
    assert later["ndcg"] == pytest.approx(1 / math.log2(3))


def test_empty_missing_and_truncation():
    assert score_ranking([], None, 3) is None
    assert set(score_ranking(["a"], {}, 3).values()) == {0}
    assert score_ranking(["a"], {"a": 1}, 5)["precision"] == 0.2
    assert score_ranking(["a", "b"], {"a": 1, "b": 1, "c": 1}, 1)["map"] == 1
    assert relative_degradation(0, 1) is None
    assert relative_degradation(0.8, 0.6) == pytest.approx(0.25)
    assert relative_degradation(0.5, 1) == -1


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        score_ranking([], {}, k)


def test_bad_ranking_or_labels():
    with pytest.raises(ValueError):
        score_ranking(["a", "a"], {}, 2)
    for grade in (-1, float("nan"), float("inf"), 101):
        with pytest.raises(ValueError):
            score_ranking(["a"], {"a": grade}, 1)
    with pytest.raises(ValueError):
        relative_degradation(float("nan"), 0)


def test_paired_inference_numerics_and_reproducibility():
    result = paired_comparison([0] * 10, [1] * 10, seed=12, resamples=300, permutations=2000)
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 1
    assert result["p_value"] < 0.01
    assert result == paired_comparison(
        [0] * 10, [1] * 10, seed=12, resamples=300, permutations=2000
    )
    same = paired_comparison([0.2, 0.5], [0.2, 0.5], resamples=20, permutations=20)
    assert same["difference"] == same["ci_low"] == same["ci_high"] == 0
    assert same["p_value"] == 1
    opposite = paired_comparison([0, 1], [1, 0], resamples=20, permutations=20)
    assert opposite["difference"] == 0 and opposite["p_value"] == 1


@pytest.mark.parametrize(
    "a,b,kwargs",
    [
        ([], [], {}),
        ([1], [1, 2], {}),
        ([float("nan")], [0], {}),
        ([1], [1], {"confidence": 1}),
        ([1], [1], {"resamples": 0}),
        ([1], [1], {"permutations": 0}),
    ],
)
def test_invalid_statistics(a, b, kwargs):
    with pytest.raises(ValueError):
        paired_comparison(a, b, **kwargs)
