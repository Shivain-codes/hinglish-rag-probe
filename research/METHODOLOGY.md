# Human-reviewed multilingual retrieval robustness protocol

## Scope and preregistration

Evaluate retrieval quality under Hindi-English language and script variation,
not end-to-end generation or factuality. Freeze research questions, primary
metric (NDCG@10), recall secondary metric, conditions, model revisions, domain
strata, exclusions and analysis before observing test results. Determine sample
size from a separate pilot and a smallest practically meaningful paired effect;
do not choose a sample size solely to obtain significance. No power calculation
or sufficient sample size is claimed by the bundled demonstration.

RQ1: What paired retrieval change occurs between English, Hindi, romanized Hindi
and natural Hinglish? RQ2: How do sparse and multilingual dense systems differ
under these conditions? RQ3: How does robustness change with corruption intensity?
RQ4: What quality/latency/memory tradeoffs persist across repeated controlled runs?

## Collection and licensing

Select multiple documented domains with a fixed retrieval corpus. Obtain sources
with verified redistribution/research rights; retain source URL, license text,
version, retrieval date, document hashes and exclusions. The local importer checks
manifest completeness, not legal sufficiency. Avoid private personal data and
remove identifying text. Keep a source-to-normalized-ID mapping outside evaluation.
Record both query language and document language; an English-only corpus creates
cross-lingual asymmetry and cannot isolate all effects of code mixing.

Create queries reflecting realistic needs rather than copying answer passages.
Split by intent/topic and source document, not by translated query variant. Detect
exact and normalized duplicates, near duplicates and paraphrase leakage across
train/dev/test. Preserve all variants of an intent in a single split and cluster.
The importer detects normalized original-text duplicates; near-duplicate and
cross-split checks require additional human review.

## Human review and judgments

Have at least two proficient Hindi-English speakers independently produce/review
variants for meaning, naturalness, dialect, script, spelling and answerability.
Adjudicate disagreement with a third reviewer; retain original decisions and
reasons. Do not mark machine-generated text human validated without actual review.
Separate natural code-mixing from forced word substitution. Record transliteration
style, mixing fraction, language-identification uncertainty and variant provenance.
Measure inter-annotator agreement (including raw agreement and a suitable chance-
corrected coefficient) on a predeclared sample, with uncertainty.

Pool retrieved documents from diverse systems and random negatives, blind annotators
to system identities, and grade relevance using a written rubric (0 irrelevant,
1 partially useful, 2 directly answers). Audit label completeness and negatives.
Missing query judgments are excluded with counts; judged-empty queries score zero.
Unjudged documents count as nonrelevant under this framework, which may bias pooled
evaluation. Report judgment coverage and consider expanded pools. Re-adjudicate
labels for variants that change intent; do not silently inherit invalid labels.

## Systems and controls

Compare TF-IDF, positive-IDF Okapi BM25, and at least one configurable multilingual
Sentence Transformer. Pin package lock, model commit, tokenizer, prefixes, precision,
device, batch size and K. Use identical documents/judgments for every method. Record
indexing time separately. Sparse tokenization uses NFC, casefold and Unicode
letter/number/mark tokens, without stemming or cross-language alignment. Exact
cosine/FAISS ranking uses ID tie-breaking. Do not claim deterministic floating
point across operating systems or CPU/GPU implementations.

Apply independently seeded perturbations over an intensity grid chosen beforehand.
A seed and stable original-query key identify each variant. Retain human validation
status. Code-point corruption can damage Hindi graphemes; treat this as a distinct
noise model, not a validated model of human typing. User-supplied translations and
transliterations are preferable when linguistic equivalence matters.

Audit model training documentation and benchmark overlap. Record release dates,
source corpus exposure, known contamination and unknowable exposure; a post-cutoff
claim alone is insufficient. Do not fine-tune on test queries or choose the best
model after inspecting test outcomes.

## Statistics

Compute per-query P/R/Hit/MRR/AP/NDCG@K using the documented API conventions.
Macro-average over judged queries; publish both judged-empty-inclusive and a
predeclared positive-only sensitivity analysis if needed. Match original IDs for
comparisons; never subtract aggregates from different cohorts. Difference is
candidate minus baseline, relative degradation is (baseline-candidate)/baseline.
Zero baseline yields undefined relative loss (null), not zero or infinity.

Use paired bootstrap resampling of original independent intents for percentile
confidence intervals. For related intents or repeated perturbation seeds, first
aggregate within intent or implement a hierarchical cluster bootstrap; the current
API does not automatically infer clusters. Use two-sided paired sign-flip tests
when exchangeability under the null is justified. Predeclare resample counts and
seeds; increase counts for final analysis. The software's default p-values are
unadjusted and exploratory. For a confirmatory family, apply a preregistered Holm
adjustment, report effect sizes and adjusted/unadjusted p-values, and avoid treating
lack of significance as equivalence. Small synthetic intervals have no inferential
standing. No significance claim is made in the draft.

## Performance and reproducibility

The evaluator measures sequential search latency including the first query, with
tracemalloc enabled; throughput is number of queries / summed search time. This
is an instrumented diagnostic, not production throughput. Use the profiling
script in fresh processes to separate warm latency and process lifetime RSS;
report warmup, repeats, thread count, thermal conditions and background workload.
Process high-water RSS includes interpreter/imports/index and is platform-specific.
Tracemalloc excludes most native and GPU allocations. No GPU peak is claimed.
Model download time must be excluded from scientific indexing measurements by
pre-populating the cache. Run order should be randomized across repeated processes.

Archive immutable inputs, checksums, configs, dependency lock, raw rankings,
per-query scores, metadata and generated figures. Release an executable reproduction
command, license/consent records, annotation rubric and split manifest. Confirm
all paper numbers can be regenerated from raw outputs.

## Current evidence boundary

Bundled data: 12 English documents and 8 hand-constructed synthetic intents,
with automatically authored/unreviewed language variants. This tests software
behavior only. It is not a representative benchmark, has no independent human
validation, and cannot support claims about superiority, fairness or novelty.
