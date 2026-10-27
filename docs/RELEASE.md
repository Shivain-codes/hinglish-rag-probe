# Release procedure (approval required)

PyPI JSON lookup for `hinglish-rag-probe` returned HTTP 404 on 2026-10-26. This
indicates no project record at that time, not guaranteed availability or reservation.
Repeat immediately before publishing; normalized name collisions/reservations can
still prevent registration. Nothing has been uploaded or pushed.

Author and copyright holder Shivain Gupta and the MIT license were confirmed
on 2026-10-27. The repository URL is recorded in CITATION.cff and pyproject.toml.
The publishing workflow enforces `python scripts/release_preflight.py`.

1. Supply verified author/maintainer details, repository URL and copyright holder.
   Confirm MIT license choice and complete LICENSE's copyright notice. Complete
   CITATION.cff authors and validate with `cffconvert --validate` before release.
2. Run all CI checks on the real GitHub repository and review dependency audit.
   Resolve findings; do not bypass failing gates. Review supported Python matrix.
3. Pin model commits and use `uv.lock` for reproduction. Keep lightweight install
   independent of torch/FAISS; optional model-download tests require explicit opt-in.
4. Set project version in pyproject.toml and __init__.py consistently, update changelog,
   build wheel/sdist, run `twine check`, and install each in fresh environments.
5. On TestPyPI configure a pending Trusted Publisher with the actual owner/repo,
   workflow `publish.yml`, and environment `testpypi`. On GitHub require human
   approval on that environment. No API tokens should be committed.
6. After explicit approval dispatch publish.yml with target=testpypi. Test install
   using a separate environment and TestPyPI index with PyPI dependencies as needed.
7. Configure the equivalent `pypi` environment/publisher and require approval.
   Recheck name, artifacts and metadata, then request explicit publication approval.
   Only then dispatch target=pypi. GitHub releases/tags/pushes also require approval.

Suggested local validation:

```bash
python -m build
python -m twine check dist/*
python -m venv /tmp/hrag-wheel-test
/tmp/hrag-wheel-test/bin/python -m pip install dist/*.whl
/tmp/hrag-wheel-test/bin/hragprobe --help
```

## Package publication checklist

- [ ] Verified owner, repository, contact, copyright and citation authors
- [ ] CI matrix passes on remote; dependency findings resolved
- [ ] Name rechecked; README/package contents reviewed
- [ ] Fresh wheel and sdist installation checks pass
- [ ] TestPyPI upload and install approved and verified
- [ ] Protected Trusted Publishing environments configured
- [ ] Explicit PyPI and GitHub release approval received

## Paper submission checklist (separate from package release)

- [ ] Full related-work and novelty review
- [ ] Licensed representative corpus and documented splits
- [ ] Independent bilingual review, adjudication and agreement analysis
- [ ] Relevance completeness, duplicates and contamination audits
- [ ] Preregistered hypotheses, sample size and multiplicity handling
- [ ] Real multilingual experiments and repeated hardware-controlled profiling
- [ ] Every claim/table traceable to archived raw results
- [ ] Verified authors, affiliations, contributions and venue requirements
- [ ] Internal review and explicit submission approval

Package availability does not imply scientific validity or peer-reviewed acceptance.
