# HinglishRAGProbe

A Python package for measuring retrieval robustness across English, Hindi,
romanized Hindi and Hinglish query variants. It evaluates retrieval, not answer
generation. Sparse retrieval works offline; no paid API is required.

**Status:** 0.1.0 release candidate. Synthetic demonstrations are software checks,
not evidence of multilingual effectiveness. See `VERIFICATION.md` for executed
checks and unresolved release gates.

## Install

Python 3.10+ (CI covers 3.10–3.13). From this repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[reports,dev]'
# Optional only:
python -m pip install -e '.[dense,faiss]'
```

For the locked development environment, use `uv sync --frozen --extra dev --extra reports`.

The project has not been published. Do not assume `pip install hinglish-rag-probe`
works until a release is approved. Dependencies are downloaded at installation;
sparse indexing and evaluation require no network afterwards.

## Quick start

```python
from hinglish_rag_probe import Document, Query, evaluate

documents = [
    Document(id="doc1", text="Machine learning enables computers to learn from data."),
    Document(id="doc2", text="Photosynthesis converts sunlight into chemical energy."),
]
queries = [
    Query(
        id="q1",
        english="How does photosynthesis work?",
        hinglish="Photosynthesis kaise kaam karta hai?",
        relevant_ids=frozenset({"doc2"}),
    )
]
report = evaluate(documents=documents, queries=queries, k=2)
print(report.summary)
print(report.to_json())
```

## CLI

Run from the repository root:

```bash
hragprobe --help
hragprobe evaluate --docs benchmarks/data/documents.csv --queries benchmarks/data/queries.csv --retriever bm25 --k 3 --manifest benchmarks/data/manifest.json --output results/single.json
hragprobe perturb --input benchmarks/data/queries.csv --output results/perturbed.csv --seed 42 --intensity 0.15
hragprobe compare --config benchmarks/sparse.yaml
hragprobe report --input results/comparison.json --output reports/demo
```

CSV, JSON arrays and JSONL records are supported. CSV `relevant_ids`, `relevance`
and `variants` cells contain JSON; booleans use `true`/`false`. Empty relevance
means **missing**, whereas `[]` means **judged and no relevant documents**.
`relevance` supports graded judgments, e.g. `{"doc1": 2, "doc2": 1}`.
See [data contracts](docs/API.md) and [methodology](research/METHODOLOGY.md).

## Retrieval methods

- `TfidfRetriever`: scikit-learn L2-normalized word TF-IDF cosine similarity.
- `BM25Retriever(k1=1.5, b=0.75)`: documented internal Okapi BM25, positive IDF;
  query term repetitions are ignored. Unicode tokenization retains Hindi marks.
- `DenseRetriever(model_name=..., revision=...)`: configurable Sentence Transformers,
  normalized cosine embeddings, batch encoding, CPU default; optional exact FAISS.

```python
from hinglish_rag_probe.retrieval import DenseRetriever

engine = DenseRetriever(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    revision=None,  # Set an immutable model commit for a publishable experiment.
    allow_download=False,  # True explicitly authorizes model download.
)
# report = evaluate(documents, queries, retriever=engine)
```

Dense construction is lazy. Model weights must be cached/local unless downloads
are explicitly enabled. Prefix-sensitive models can use `query_prefix` and
`document_prefix`. No universal dense model suitability is assumed. Equal scores
sort by document ID; FAISS requests all exact scores to preserve tie-breaking.
This is deliberately a correctness baseline, not a million-document index.

## Perturbations and provenance

Seeded deletion, adjacent swaps, insertion, mixed typos, whitespace and NFC
normalization are supported. `add_variant` accepts user translations,
transliterations and paraphrases. No automatic Hindi translation/transliteration
is claimed. Every generated variant stores source text, source condition, query
ID, seed, intensity and algorithm. All generated variants are unvalidated.
`human_validated=True` is an assertion by the dataset provider, never inferred.
Perturbations act on code points; Hindi combining marks can be disrupted.

## Metrics and inference

Precision, recall, hit rate, MRR, MAP and NDCG at K; mean query latency, sequential
throughput and Python allocation peak. P@K divides by K even when fewer documents
exist. AP@K divides by min(K, number of positive judgments). NDCG uses exponential
gain. Missing labels are excluded, judged-empty queries score zero. Zero-baseline
relative degradation is JSON null. Positive degradation means loss; positive
paired difference means candidate improvement.

Paired percentile bootstrap intervals and two-sided Monte Carlo sign-flip tests
operate on original query IDs. Comparisons are exploratory, without multiplicity
adjustment. Original queries must be independent; do not count perturbation
replicates as independent samples. No p-value is a claim of scientific validity.

## Benchmarking and reports

`benchmarks/sparse.yaml` runs the small explicitly synthetic demonstration.
`benchmarks/multilingual.yaml` adds a configurable multilingual dense model and
requires installing the dense extra and explicitly permitting downloads in the
config (or populating the local cache). Model revisions must be pinned for papers.
Outputs record dataset checksums, provenance, model settings, versions, hardware,
seed, timestamp and measurement scope. The report exports six figure families in
SVG, PDF and PNG plus self-contained HTML. Values come only from completed runs.

`datasets.import_dataset` normalizes externally sourced local data only when
license/source/version/relevance metadata are supplied. It validates references
and duplicate original query texts. It does not certify license rights or download
data. Preserve source attribution and independently review redistribution rights.

## Limitations

The demo has a tiny English-only document collection and unreviewed synthetic
query variants. It cannot establish language fairness, model rankings or novelty.
BM25 and TF-IDF do not align languages. Latency is sequential, includes the cold
first query and instrumentation; allocation peaks exclude most native/GPU memory.
Model computations can vary across hardware even with deterministic ranking ties.
On the tested Apple Silicon environment, loading FAISS before PyTorch caused a
native crash with unconstrained threads. `OMP_NUM_THREADS=1` passed the combined
suite; set this before process startup when combining these optional libraries.
The package does not silently override global thread settings. See VERIFICATION.md.
Real benchmark collection, human review, contamination auditing and repeated
hardware-controlled runs remain required. See [protocol](research/METHODOLOGY.md).

## Development and citation

```bash
pytest --cov=hinglish_rag_probe --cov-report=term-missing
ruff check .
ruff format --check .
mypy src/hinglish_rag_probe
python -m build
python -m twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [release instructions](docs/RELEASE.md),
[CITATION.cff](CITATION.cff) and [paper draft](research/paper/manuscript.md).
Source repository: [https://github.com/Shivain-codes/hinglish-rag-probe](https://github.com/Shivain-codes/hinglish-rag-probe).
Author identities for formal citation have not yet been confirmed. Set verified citation
metadata before publication; no author names or affiliations are invented here.
