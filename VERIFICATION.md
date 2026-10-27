# Verification ledger — 2026-10-26

GitHub update (2026-10-27): repository created at [https://github.com/Shivain-codes/hinglish-rag-probe](https://github.com/Shivain-codes/hinglish-rag-probe).
The local validation record below predates the GitHub upload; PyPI publication
and manuscript submission remain pending.

## Delivered software

New local Git repository at `hinglish-rag-probe`. No remote, push,
tag, public release, package upload or manuscript submission was performed.
Version 0.1.0 is an unpublished release candidate.

Implemented: validated Document/Query/QueryVariant API; TF-IDF, internal BM25,
configurable Sentence Transformers and exact FAISS; seeded provenance-preserving
perturbations; binary/graded metrics; paired bootstrap and permutation statistics;
CSV/JSON/JSONL IO and manifest-gated imports; complete evaluate/compare/perturb/report
CLI; six plot families exported in SVG/PDF/PNG and embedded HTML; reproducibility
metadata and profiling script; lockfile, CI and Trusted Publishing workflows;
research protocol, five verified references and manuscript draft.

## Executed checks

| Check | Actual outcome |
|---|---|
| Python 3.11.15 full suite, real multilingual model + FAISS, OMP_NUM_THREADS=1 | 56 passed, 0 skipped; 97.99% statement coverage |
| Python 3.10.20 sparse/default environment | 53 passed, 3 optional tests skipped; 97.56% coverage |
| Python 3.12.13 sparse/default environment | 53 passed, 3 optional tests skipped; 97.56% coverage |
| Python 3.13.12 sparse/default environment | 53 passed, 3 optional tests skipped; 97.56% coverage |
| Ruff lint and formatting | Passed |
| Mypy package plus release preflight script | Passed, 12 source files |
| Wheel and source-distribution build | Passed |
| Twine metadata/readme checks | Both distributions passed |
| Fresh minimal wheel and sdist installation | Passed; API and CLI smoke checks; no torch/FAISS dependency |
| Dependency audit | 106 environment entries; no known vulnerabilities; editable project excluded |
| Sparse synthetic pipeline | TF-IDF and BM25 completed, 48 query-condition rows each |
| Multilingual synthetic pipeline | TF-IDF, BM25, pinned multilingual model completed, 48 rows each |
| Figures and reports | Six SVG/PDF/PNG figures and self-contained HTML per pipeline; figures visually inspected |
| PyPI name check | Official JSON endpoint returned 404; not a reservation or guaranteed registration |
| Publication preflight | Intentionally fails for missing author/repository/copyright metadata |

The three default-environment skips are the FAISS integration and two explicitly
opt-in real-model cases (with/without FAISS). No failed assertions were suppressed.
Cross-version logs are under `results/tests-python*.txt`; full 3.11 evidence is
`results/test-results.txt`. The installed 3.11 dependency snapshot is
`results/environment-python311.txt`. GitHub Actions have not run on a remote host.

The real model was explicitly downloaded once, then final tests/benchmarks ran
with `HF_HUB_OFFLINE=1`. Model:
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, immutable revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Final pipeline metadata records
`OMP_NUM_THREADS=1`. No synthetic encoder output is substituted for that model.

## Native dependency limitation discovered and reproduced

An unconstrained combined test run segfaulted inside PyTorch layer normalization
after FAISS had already loaded. Standalone dense testing and a fresh torch-first
real-model-plus-FAISS process passed. Setting `OMP_NUM_THREADS=1` before startup
passed all 56 tests together, including real model and FAISS integration. The
unconstrained failure is retained locally in `results/native-crash-diagnostic.txt` (excluded from Git).
This establishes a verified workaround on this Apple Silicon Mac, not a universal
fix for native OpenMP/library interactions. The package does not mutate global
thread settings or use unsafe duplicate-runtime bypasses. Other devices/thread
counts, GPU memory and large-scale approximate indexes remain unverified.

## Reproduce locally

```bash
cd hinglish-rag-probe
uv sync --frozen --extra dev --extra reports
uv run hragprobe compare --config benchmarks/sparse.yaml
uv run pytest --cov=hinglish_rag_probe --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy src/hinglish_rag_probe scripts
uv run python -m build
uv run python -m twine check dist/*
```

Full optional integration, after selecting/download-authorizing the pinned model:

```bash
uv sync --frozen --extra dev --extra reports --extra dense --extra faiss
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
HRAG_TEST_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
HRAG_TEST_REVISION=e8f8c211226b894fcb81acc59f3b34ba3efd5f42 \
uv run pytest --cov=hinglish_rag_probe
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 uv run hragprobe compare --config benchmarks/multilingual.yaml
```

On a new machine, remove HF_HUB_OFFLINE and explicitly set HRAG_ALLOW_DOWNLOAD=1
for the test download, or set allow_download:true in an experiment config. The
config defaults to false. Prefer a separate model-fetch step before measured runs.

## Evidence boundary and remaining gates

The 12-document/8-intent corpus and language variants are explicitly synthetic
and have not been independently human reviewed. Statistical outputs merely test
computation. No generalizable ranking, significance or scientific novelty is
claimed. Actual benchmark collection/licensing, bilingual review, label-quality,
duplicate and contamination checks, preregistration, power analysis, multiplicity
handling and controlled repeated performance experiments remain research work.

Publication requires verified authors/affiliations where applicable, repository
URL, copyright holder and confirmation of the proposed MIT license. CITATION.cff
is an intentionally incomplete template rather than invented identity metadata.
The release workflow enforces `scripts/release_preflight.py`, which currently
fails on these missing fields. Remote CI and protected Trusted Publishers must
be configured, name availability rechecked, TestPyPI validated and explicit
publication approval obtained. See docs/RELEASE.md for separate package and paper
checklists. Twine validation alone does not complete these gates.
