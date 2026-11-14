"""scripts/make_manifest.py - write benchmark/MANIFEST.json (SHA-256 + size of every file) for Zenodo/HF release.

    python scripts/make_manifest.py [--dir benchmark]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_manifest(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json" and "__pycache__" not in p.parts:
            out[p.relative_to(root).as_posix()] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="benchmark")
    a = ap.parse_args()
    root = Path(a.dir)
    m = build_manifest(root)
    (root / "MANIFEST.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {root / 'MANIFEST.json'} ({len(m)} files)")


if __name__ == "__main__":
    main()
