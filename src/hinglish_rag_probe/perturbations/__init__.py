"""Seeded character perturbations; no unverified automatic translation claims."""

from __future__ import annotations

import hashlib
import random
import unicodedata
from dataclasses import replace
from typing import Any

from ..models import Query, QueryVariant

TRANSFORMS = ("typo", "delete", "swap", "insert", "whitespace", "normalize")


def perturb_query(
    query: Query,
    *,
    transformation: str = "typo",
    intensity: float = 0.1,
    seed: int = 0,
    source_condition: str = "english",
    alphabet: str = "abcdefghijklmnopqrstuvwxyz",
) -> Query:
    """Append one automatically generated variant; RNG is stable per query ID.

    Character edits operate on Unicode code points, including combining marks.
    NFC normalization is supported; transliterations/paraphrases are user supplied.
    """
    if transformation not in TRANSFORMS or not 0 <= intensity <= 1 or not alphabet:
        raise ValueError("invalid transformation, intensity or alphabet")
    sources = {v.condition: v for v in query.all_variants()}
    if source_condition not in sources:
        raise ValueError(f"query {query.id} has no {source_condition} variant")
    source = sources[source_condition]
    settings: dict[str, Any] = {
        "seed": seed,
        "intensity": intensity,
        "source_condition": source_condition,
        "alphabet": alphabet,
        "original_query_id": query.id,
        "original_text": source.text,
        "algorithm": "codepoint_v1",
    }
    key = f"{seed}:{query.id}:{source_condition}:{transformation}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))
    text = list(source.text)
    count = round(len(text) * intensity)
    for _ in range(count):
        operation = (
            rng.choice(("delete", "swap", "insert")) if transformation == "typo" else transformation
        )
        index = rng.randrange(len(text))
        if operation == "delete" and len(text) > 1:
            del text[index]
        elif operation == "swap" and len(text) > 1:
            other = min(index + 1, len(text) - 1)
            text[index], text[other] = text[other], text[index]
        elif operation == "insert":
            text.insert(index, rng.choice(alphabet))
        elif operation == "whitespace":
            text.insert(index, rng.choice((" ", "\t", "  ")))
    transformed = (
        unicodedata.normalize("NFC", source.text)
        if transformation == "normalize"
        else "".join(text)
    )
    if not transformed.strip():
        transformed = source.text
        settings["fallback"] = "empty result reverted"
    variant = QueryVariant(
        f"{source_condition}_{transformation}", transformed, False, transformation, settings
    )
    return replace(query, variants=(*query.variants, variant))


def add_variant(
    query: Query,
    *,
    condition: str,
    text: str,
    human_validated: bool = False,
    kind: str = "paraphrase",
) -> Query:
    if kind not in ("paraphrase", "transliteration", "translation"):
        raise ValueError("kind must be paraphrase, transliteration or translation")
    return replace(
        query,
        variants=(
            *query.variants,
            QueryVariant(
                condition,
                text,
                human_validated,
                f"user_{kind}",
                {"original_query_id": query.id},
            ),
        ),
    )
