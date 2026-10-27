"""Public data contracts. Missing judgments (None) differ from judged-empty sets."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


def nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class Document:
    id: str
    text: str

    def __post_init__(self) -> None:
        nonempty(self.id, "document id")
        nonempty(self.text, "document text")


@dataclass(frozen=True)
class QueryVariant:
    condition: str
    text: str
    human_validated: bool = False
    transformation: str = "user_supplied"
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nonempty(self.condition, "condition")
        nonempty(self.text, "variant text")
        if not isinstance(self.human_validated, bool):
            raise ValueError("human_validated must be a boolean")


@dataclass(frozen=True)
class Query:
    id: str
    english: str | None = None
    hinglish: str | None = None
    relevant_ids: frozenset[str] | None = None
    hindi: str | None = None
    romanized_hindi: str | None = None
    variants: tuple[QueryVariant, ...] = ()
    relevance: dict[str, float] | None = None
    human_validated: bool = False

    def __post_init__(self) -> None:
        nonempty(self.id, "query id")
        if not isinstance(self.human_validated, bool):
            raise ValueError("human_validated must be a boolean")
        if self.relevant_ids is not None:
            if isinstance(self.relevant_ids, str):
                raise ValueError("relevant_ids must be a collection, not a string")
            object.__setattr__(self, "relevant_ids", frozenset(self.relevant_ids))
            for key in self.relevant_ids:
                nonempty(key, "relevant document id")
        if self.relevance is not None:
            for key, grade in self.relevance.items():
                nonempty(key, "relevance document id")
                if (
                    not isinstance(grade, (int, float))
                    or not math.isfinite(grade)
                    or grade < 0
                    or grade > 100
                ):
                    raise ValueError("relevance grades must be finite numbers in [0, 100]")
            if self.relevant_ids is not None and self.relevant_ids != frozenset(
                key for key, grade in self.relevance.items() if grade > 0
            ):
                raise ValueError("relevant_ids and graded relevance disagree")
        variants = self.all_variants()
        if not variants:
            raise ValueError("query must contain at least one text variant")
        if len({v.condition for v in variants}) != len(variants):
            raise ValueError("conditions must be unique within a query")

    def all_variants(self) -> tuple[QueryVariant, ...]:
        supplied = tuple(
            QueryVariant(name, text, self.human_validated)
            for name in ("english", "hindi", "romanized_hindi", "hinglish")
            if (text := getattr(self, name)) is not None
        )
        return supplied + tuple(self.variants)

    def judgments(self) -> dict[str, float] | None:
        if self.relevance is not None:
            return dict(self.relevance)
        if self.relevant_ids is None:
            return None
        return {key: 1.0 for key in self.relevant_ids}


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float


@dataclass
class EvaluationReport:
    metadata: dict[str, Any]
    results: list[dict[str, Any]]
    summary: dict[str, dict[str, Any]]
    comparisons: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> EvaluationReport:
        return cls(**json.loads(text))

    def __str__(self) -> str:
        return self.to_json()
