"""Retrieval robustness evaluation for Hindi, English and Hinglish."""

from .evaluation import evaluate
from .models import Document, EvaluationReport, Query, QueryVariant

__version__ = "0.1.0"
__all__ = ["Document", "Query", "QueryVariant", "EvaluationReport", "evaluate"]
