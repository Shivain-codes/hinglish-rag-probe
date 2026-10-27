# HinglishRAGProbe: A Reproducible Framework for Evaluating Multilingual Retrieval Robustness Under Hindi-English Code-Mixing

**Manuscript draft — not submitted. Authors and affiliations: pending user-supplied information.**

## Abstract

Retrieval robustness under Hindi-English code-mixing requires comparisons that
preserve query intent and distinguish human-reviewed variants from automatic noise.
We present HinglishRAGProbe, an executable Python framework for sparse and dense
retrieval evaluation, provenance-preserving query perturbations, paired statistical
analysis and reproducible reporting. The implementation records input fingerprints,
model configurations and measurement scope. A synthetic demonstration exercises the
workflow; it does not establish retrieval effectiveness. A human-reviewed benchmark
and confirmatory experiments remain pending. No empirical superiority is claimed.

## 1. Introduction

A query can express a similar need in English, Devanagari Hindi, romanized Hindi
or mixed language. Lexical overlap, transliteration and embedding alignment can
change retrieved documents. Comparing these inputs requires explicit judgments
and equivalence review, rather than assuming every transformed string preserves
intent. The contribution of this draft is an implemented measurement framework
and a proposed review protocol. Whether it provides an original publishable
research contribution requires a fuller novelty assessment and real experiments.

## 2. Related Work

Sentence-BERT derives sentence embeddings suitable for similarity comparison [1].
Multilingual knowledge distillation extends sentence representations using parallel
sentences [2]. These motivate configurable dense baselines rather than a single
hardcoded language model. Hindi-BEIR evaluates Hindi retrieval over multiple tasks
and domains [3]; Hindi-only and Hindi-English code-mixed evaluation are different
settings, and this work must be compared carefully with that benchmark.

BEIR provides a heterogeneous multi-domain retrieval evaluation setting [4].
The probabilistic relevance framework motivates the BM25 family [5]; our exact
positive-IDF variant and tokenization are documented separately.

These references were verified against ACL Anthology, NeurIPS proceedings and
the publisher record on 2026-10-26. The review is preliminary, not exhaustive. Dedicated Hinglish retrieval
projects also exist; claims of being the first framework or benchmark are withheld.
A final review must cover code-mixed IR, robustness evaluation, BM25, statistical
IR evaluation, multilingual benchmarks and concurrent software before submission.

## 3. Research Questions

1. How does paired quality vary across English, Hindi, romanized Hindi and Hinglish?
2. Which retrieval methods preserve quality under natural mixing and artificial noise?
3. How do perturbation intensity and script interact with method performance?
4. Which quality/latency/memory tradeoffs remain under controlled repeated runs?

## 4. Methodology

Use a fixed licensed corpus, original intent IDs, multiple human-reviewed variants,
and adjudicated relevance grades. Keep variants grouped in splits and analysis.
Compare word TF-IDF, positive-IDF Okapi BM25 and configurable multilingual sentence
embeddings. Rank by score, resolving exact ties by document ID. All automatic
variants retain source metadata and remain explicitly unvalidated.

Measure P/R/Hit/MRR/MAP/NDCG@K. AP@K divides by min(K, positive judgments), and
NDCG uses exponential gain. Exclude queries without judgments, include judged-empty
queries as zero, and publish counts. Pair differences by intent within a condition;
use percentile bootstrap intervals and two-sided sign-flip tests. Confirmatory
families require a preregistered multiplicity correction. See ../METHODOLOGY.md.

## 5. Experimental Setup

The software demonstration contains 12 synthetic English documents and 8 intents
with English/Hindi/romanized Hindi/Hinglish variants and seeded noise conditions.
No variants have received independent human review. The bundled configurations
record K, seed, resample counts and method settings; each completed run records
versions, machine information, timestamps, checksums and raw results. The dense
configuration pins a multilingual MiniLM model revision. Local execution evidence
and any download limitations are documented in ../../VERIFICATION.md.

For the real study, dataset domains, corpus sizes, sample size justification,
annotation agreement, hardware-controlled repetitions and final model choices
remain **pending**. These fields must be filled from actual work, not estimates.

## 6. Results

**Research results pending.** No empirical table here should be read as evidence
about multilingual retrieval. Synthetic execution outputs are stored in
../../results/ and the self-contained demonstration reports in ../../reports/.
The verification ledger records executed tests. A software demonstration cannot
support population-level confidence intervals or comparative scientific claims.

Final tables must include per-condition judged counts, primary metric and paired
confidence intervals, adjusted hypothesis tests where appropriate, repeated timing
and measured memory scope. Include null/negative findings and predeclared exclusions.

## 7. Discussion

Explicit variant provenance and paired cohort alignment reduce avoidable analysis
errors. They do not guarantee semantic equivalence, complete judgments or realistic
noise. Sparse systems cannot bridge languages without shared tokens, while dense
models must be examined for domain and script sensitivity. Exact FAISS scoring is
a correctness baseline; scalability requires separate index experiments.

## 8. Threats to Validity

Synthetic selection, English-only documents, incomplete relevance pools, reviewer
bias, unnatural mixing, code-point noise, correlated intents, model contamination,
post-hoc model selection and instrumented latency can affect conclusions. Hardware
variation and unpinned dependencies limit reproducibility. The protocol specifies
controls, but those controls have not yet been completed for a research dataset.

## 9. Conclusion

HinglishRAGProbe supplies executable evaluation and reporting components and a
human-review protocol. Establishing a scientific contribution requires a licensed,
reviewed benchmark and completed reproducible experiments. This manuscript remains
a methods draft until those steps and an originality review are complete.

## References

[1] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using
Siamese BERT-Networks. EMNLP-IJCNLP. https://aclanthology.org/D19-1410/

[2] Nils Reimers and Iryna Gurevych. 2020. Making Monolingual Sentence Embeddings
Multilingual using Knowledge Distillation. EMNLP. https://aclanthology.org/2020.emnlp-main.365/

[3] Arkadeep Acharya, Rudra Murthy, Vishwajeet Kumar, and Jaydeep Sen. 2025.
Benchmarking and Building Zero-Shot Hindi Retrieval Model with Hindi-BEIR and
NLLB-E5. NAACL. https://aclanthology.org/2025.naacl-long.220/

[4] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna
Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of
Information Retrieval Models. NeurIPS Datasets and Benchmarks.
https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html

[5] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance
Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval.
https://www.nowpublishers.com/article/DownloadEBook/INR-019
