"""
scripts/verify_phase1_hardening.py

Verify selected hardening properties against repository files.

The checks cover WAL mode, bounded cached search, PubMed API-key handling,
paired evaluation statistics, CI configuration, and the dependency lockfile.
Each check reports whether its expected pattern is present or absent.

Run from your repo root:
    python scripts/verify_phase1_hardening.py

Exit code is 0 when all checks pass and 1 when any check fails.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()

CHECKS: list[tuple[str, str, str]] = [
    # (description, file relative to repo root, regex that must be found)
    ("WAL mode in storage.py", "src/storage.py", r"journal_mode\s*=\s*WAL|PRAGMA\s+journal_mode"),
    ("WAL mode in embedder.py", "src/embedder.py", r"journal_mode\s*=\s*WAL|PRAGMA\s+journal_mode"),
    ("WAL mode in knowledge_base.py", "src/knowledge_base.py", r"journal_mode\s*=\s*WAL|PRAGMA\s+journal_mode"),
    ("search_cached() has a LIMIT clause", "src/storage.py", r"search_cached[\s\S]{0,600}LIMIT"),
    ("PubMed rate-limit no longer mutates os.environ directly for api_key",
     "src/fetcher_pubmed.py", r"os\.environ\[.NCBI_API_KEY.\]\s*=" ),  # NOTE: this one should NOT match
    ("Paired statistical test present in evaluator.py", "src/evaluator.py",
     r"wilcoxon|sign_test|bootstrap|paired|confidence_interval|scipy\.stats"),
    ("CI workflow file exists", ".github/workflows/tests.yml", r"pytest"),
    ("requirements-lock.txt exists and is non-trivial", "requirements-lock.txt", r"[a-zA-Z0-9_-]+==\d"),
]

# Checks where a MATCH means FAILURE (i.e. we're checking something is
# ABSENT, not present) — index into CHECKS by description substring.
INVERTED_CHECKS = {"PubMed rate-limit no longer mutates os.environ directly for api_key"}


def _read(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def main() -> int:
    print(f"Verifying Phase 1 hardening claims against: {REPO_ROOT}\n")
    print(f"{'Check':<62}{'Result'}")
    print("-" * 76)

    all_ok = True
    for description, rel_path, pattern in CHECKS:
        path = REPO_ROOT / rel_path
        content = _read(path)

        if content is None:
            result = "FILE NOT FOUND"
            ok = False
        else:
            found = re.search(pattern, content, flags=re.IGNORECASE) is not None
            if description in INVERTED_CHECKS:
                ok = not found  # good if NOT found
                result = "PASS (absent, as expected)" if ok else "FAIL (still present)"
            else:
                ok = found
                result = "PASS" if ok else "FAIL (not found)"

        all_ok = all_ok and ok
        print(f"{description:<62}{result}")

    print("-" * 76)
    if all_ok:
        print("\nAll Phase 1 hardening claims verified against real files.")
    else:
        print("\nOne or more claims could NOT be independently verified.")
        print("This does not necessarily mean the fix is missing — regex checks")
        print("are approximate. Open the flagged file(s) and check by eye before")
        print("concluding anything is actually wrong.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
