# API and data contracts

`Document(id, text)` requires nonempty strings. IDs must be unique per corpus.
`Query(id, english=None, hinglish=None, relevant_ids=None, hindi=None,
romanized_hindi=None, variants=(), relevance=None, human_validated=False)`
requires at least one text. Query IDs and condition names are unique. If both
binary and graded judgments are present their positive sets must agree.
Grades are finite numbers in [0,100]. A positive grade means relevant; zero means
nonrelevant. Unknown document references are errors. Frozen dataclasses discourage
mutation; embedded mappings should also be treated as immutable by callers.

`QueryVariant(condition, text, human_validated=False, transformation="user_supplied",
settings={})` represents a condition, never an independent query. Custom conditions
can hold human translations, transliterations or paraphrases. A variant inherits
the original query's labels; validate semantic equivalence before research use.

`evaluate(documents, queries, k=5, retriever="bm25", *, seed=0,
dataset_metadata=None, resamples=2000, permutations=2000, confidence=.95)`
returns `EvaluationReport(metadata, results, summary, comparisons)`.
`to_dict()`, `to_json()` and `from_json()` support JSON serialization. Query rows
record rankings/scores, latency, label status and complete variant metadata.
Reports use schema version 1; latency/timestamps vary between repeated runs.
The SHA-256 fingerprint covers normalized corpus/query content and provenance.

Custom retrievers implement `index(documents)`, `search(query, k)`,
`search_batch(queries, k)`, and `configuration()`. Return `SearchHit` instances.
Do not return duplicate IDs, nonfinite scores, unknown IDs or more than K hits.
Built-in sparse batch search loops sequentially; dense batch search batches encoding.
Evaluation measures individual calls so each query has a latency observation.

`score_ranking(ids, relevance, k)` returns six named scores or None for missing
labels. Rankings shorter than K retain P@K's denominator K. AP@K normalizes by
min(K, total positives). NDCG normalizes against the ideal sorted graded judgments,
including relevant documents absent from the retrieved list. This convention may
differ from software using total positives for truncated AP; report it explicitly.

`paired_comparison(baseline, candidate, seed=0, resamples=2000,
confidence=.95, permutations=2000)` requires aligned equal-length finite vectors.
It returns candidate-minus-baseline mean difference, a percentile bootstrap CI,
and a two-sided paired sign-flip Monte Carlo p-value with +1 correction.
`compare_reports` aligns query IDs within each condition and refuses different
fingerprints/K. It compares every method against the first dictionary entry.

`perturb_query` appends one condition, preserving the source and setting validation
false. Intensity is an approximate edit budget `round(codepoints * intensity)`;
some swaps can be no-ops. Seeds are SHA-256-derived per query/condition/transform,
so query order does not alter results. NFC normalization ignores edit intensity.
Explicit user variants can be added with `add_variant`.

Dataset CSV fields mirror constructors. Nested values use JSON cells. JSON and
JSONL use JSON objects directly. `save_queries` preserves judgments and metadata.
`import_dataset(docs, queries, manifest, output)` requires nonempty name, source,
license, version and relevance_definition. Unknown/unlicensed placeholders are
rejected. Review actual rights separately. The normalized bundle records original
file checksums and preserves source metadata; source files are not modified.

Invalid inputs raise ValueError; optional dependency absence raises ImportError;
filesystem failures raise OSError. CLI catches these and emits an error with exit
status 2. Unexpected programmer errors retain their traceback.

## Native-library threading on macOS

For the tested FAISS/PyTorch combination, run with `OMP_NUM_THREADS=1` before
Python starts when both libraries share a process. An unconstrained suite with
FAISS loaded first segfaulted inside PyTorch layer normalization. Standalone dense
and torch-first FAISS checks passed, and the complete combined suite is verified
with one OpenMP thread. This is a native dependency interaction, not a caught
Python exception. No KMP_DUPLICATE_LIB_OK bypass is used. Revalidate other versions,
thread counts and hardware; the framework records explicit thread environment.
