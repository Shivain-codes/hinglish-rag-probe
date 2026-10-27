# Contributing

Use Python 3.10+ and install `.[dev,reports]` in an isolated environment. Run pytest
with coverage, Ruff check/format, mypy and builds before proposing changes. Add
hand-calculated metric fixtures for mathematical changes, preserve public APIs or
document migrations, and retain provenance on every variant. Do not call generated
text human-validated. Keep model-download tests opt-in and offline tests default.
Do not suppress failures or fill missing research results with invented numbers.

Include the problem, behavior change and validation in pull requests. Report
security issues privately to the maintainer once a verified contact is supplied;
no maintainer contact has yet been assigned. Never commit credentials or private
corpora. Review licenses before adding dependencies or data.
