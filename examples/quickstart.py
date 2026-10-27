"""Run with python examples/quickstart.py after installation."""

from hinglish_rag_probe import Document, Query, evaluate

report = evaluate(
    [
        Document("doc1", "Machine learning enables computers to learn from data."),
        Document("doc2", "Photosynthesis converts sunlight into chemical energy."),
    ],
    [
        Query(
            "q1",
            english="How does photosynthesis work?",
            hinglish="Photosynthesis kaise kaam karta hai?",
            relevant_ids=frozenset({"doc2"}),
        )
    ],
    k=2,
)
print(report.summary)
