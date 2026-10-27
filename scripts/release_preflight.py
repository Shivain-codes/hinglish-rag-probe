"""Fail publication until identity and citation metadata have been supplied."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    citation = yaml.safe_load((root / "CITATION.cff").read_text())
    authors = citation.get("authors")
    errors = []
    if not isinstance(authors, list) or not authors:
        errors.append("Supply verified authors in CITATION.cff and validate its schema.")
    if not citation.get("repository-code"):
        errors.append("Supply the actual repository-code URL in CITATION.cff.")
    if not re.search(r"Copyright\s+\(c\)\s+\d{4}\s+\S+", (root / "LICENSE").read_text()):
        errors.append("Supply the verified copyright holder in LICENSE.")
    project = (root / "pyproject.toml").read_text()
    package = (root / "src/hinglish_rag_probe/__init__.py").read_text()
    project_version = re.search(r'^version = "([^"]+)"', project, re.M)
    package_version = re.search(r'^__version__ = "([^"]+)"', package, re.M)
    if not project_version or not package_version or project_version[1] != package_version[1]:
        errors.append("Package and distribution versions must match.")
    elif str(citation.get("version")) != project_version[1]:
        errors.append("Citation version must match distribution version.")
    if errors:
        print("\n".join(errors))
        return 1
    print("Identity/version preflight passed; complete the release checklist before publishing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
