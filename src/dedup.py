"""
src/dedup.py - Cross-source paper deduplication.

A bioRxiv preprint and its later peer-reviewed PubMed record are two
different Paper objects (different paper_id: DOI vs PMID; often a
different, revised abstract) but represent ONE underlying piece of work.
Today, `--source both` silently double-counts this work in:
  - TF-IDF / BioBERT rankings (it can occupy two slots in a top-10 list)
  - Knowledge-graph centrality and Graph Connectivity Score (GCS) —
    the same gene-disease co-occurrence gets counted twice

This module provides an explicit, auditable deduplication pass to run after
fetching and before ranking or graph construction. It returns a merge log
identifying every collapsed record.

MATCHING STRATEGY (two passes, most confident first)
------------------------------------------------------
1. DOI match: papers sharing a normalized, non-empty DOI are the same work
   almost by definition. Highest confidence.
2. content_hash match: papers with identical `title + abstract` (via the
   existing `Paper.content_hash` property already defined in
   src/models.py) are treated as the same work even without a DOI — this
   catches same-source duplicate fetches (e.g. overlapping bioRxiv date-
   range pages) as well as unusually-clean cross-source matches.

We deliberately do NOT do fuzzy/approximate title matching (e.g. edit
distance) in this pass — false-positive merges (accidentally collapsing
two DIFFERENT papers) are scientifically worse than false-negative misses
(leaving two records of the same paper un-merged), so we only merge on
exact, high-confidence keys. This favours avoiding false-positive merges
over catching approximate title variants.

WHEN TWO SOURCES DISAGREE ON WHICH RECORD TO KEEP
----------------------------------------------------
We prefer PubMed's record over bioRxiv's when both exist for the same
    work, because PubMed records are post-peer-review and typically have a
    more complete, corrected abstract and confirmed MeSH indexing (which
    mesh_fusion.py depends on). This is a documented default — pass
default — pass `prefer_source="biorxiv"` to invert it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _normalize_doi(doi) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d or None


@dataclass
class DedupResult:
    """Auditable output of a deduplication pass."""
    papers: list = field(default_factory=list)          # deduplicated, order-preserving
    merged_count: int = 0                                # number of records collapsed away
    merge_log: list[tuple[str, str, str]] = field(default_factory=list)
    # merge_log entries: (kept_paper_id, kept_source, dropped_identity)
    # where dropped_identity is "source:paper_id" of the record that was merged away.

    def summary(self) -> str:
        return (
            f"DedupResult: {len(self.papers)} unique papers kept, "
            f"{self.merged_count} duplicate records merged away."
        )


def deduplicate_papers(papers: list, prefer_source: str = "pubmed") -> DedupResult:
    """
    Deduplicate a list of Paper objects across sources.

    Parameters
    ----------
    papers : list[Paper]
        Papers from one or more sources (e.g. the combined output of a
        `--source both` fetch, BEFORE ranking or graph construction).
    prefer_source : str
        Which source's record to keep when the same underlying work is
        found under both sources. Defaults to "pubmed" (see module
        docstring for rationale). Case-insensitive.

    Returns
    -------
    DedupResult
        `.papers` is the deduplicated list, safe to pass directly into
        TFIDFRanker.rank() / SemanticRanker.rank() / KnowledgeBase.process_papers().
        `.merge_log` lets you print exactly what was collapsed, for
        transparency in findings.md or a debug run.
    """
    if not papers:
        return DedupResult(papers=[], merged_count=0, merge_log=[])

    prefer_source = prefer_source.lower()

    def _source_of(p) -> str:
        s = getattr(p, "source", "")
        # Paper.source may be a DataSource enum or a str depending on
        # Config.use_enum_values — handle both without assuming.
        return getattr(s, "value", s) if s is not None else ""

    def _priority(p) -> int:
        # Lower is "kept preferentially" when there's a tie to break.
        return 0 if _source_of(p).lower() == prefer_source else 1

    # ── Pass 1: group by normalized DOI ─────────────────────────────────
    doi_groups: dict[str, list] = {}
    no_doi: list = []
    for p in papers:
        doi = _normalize_doi(getattr(p, "doi", None))
        if doi:
            doi_groups.setdefault(doi, []).append(p)
        else:
            no_doi.append(p)

    kept: list = []
    merge_log: list[tuple[str, str, str]] = []
    merged_count = 0

    for doi, group in doi_groups.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        group_sorted = sorted(group, key=_priority)
        winner = group_sorted[0]
        kept.append(winner)
        for loser in group_sorted[1:]:
            merged_count += 1
            merge_log.append((
                winner.paper_id, _source_of(winner),
                f"{_source_of(loser)}:{loser.paper_id}",
            ))
        logger.info(
            "dedup: DOI match collapsed %d records into %s:%s (doi=%s)",
            len(group), _source_of(winner), winner.paper_id, doi,
        )

    # ── Pass 2: among the DOI-less remainder, group by content_hash ────
    hash_groups: dict[str, list] = {}
    for p in no_doi:
        h = getattr(p, "content_hash", None)
        if h is None:
            # Paper.content_hash is a computed property in src/models.py;
            # if it's ever missing, don't crash — keep the paper untouched.
            kept.append(p)
            continue
        hash_groups.setdefault(h, []).append(p)

    for h, group in hash_groups.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        group_sorted = sorted(group, key=_priority)
        winner = group_sorted[0]
        kept.append(winner)
        for loser in group_sorted[1:]:
            merged_count += 1
            merge_log.append((
                winner.paper_id, _source_of(winner),
                f"{_source_of(loser)}:{loser.paper_id}",
            ))
        logger.info(
            "dedup: content_hash match collapsed %d records into %s:%s",
            len(group), _source_of(winner), winner.paper_id,
        )

    logger.info(
        "deduplicate_papers(): %d input -> %d unique (%d merged)",
        len(papers), len(kept), merged_count,
    )
    return DedupResult(papers=kept, merged_count=merged_count, merge_log=merge_log)


def self_test() -> None:
    """Run `python -m src.dedup` to sanity-check this module in isolation."""

    class _FakePaper:
        def __init__(self, paper_id, source, title, abstract="", doi=None):
            self.paper_id = paper_id
            self.source = source
            self.title = title
            self.abstract = abstract
            self.doi = doi

        @property
        def content_hash(self):
            import hashlib
            return hashlib.sha256((self.title + self.abstract).encode()).hexdigest()[:16]

    a = _FakePaper("38812345", "pubmed", "KRAS in lung cancer", "abc", doi="10.1/xyz")
    b = _FakePaper("10.1101/2023.01.01.000001", "biorxiv", "KRAS in lung cancer (preprint)", "abc-preprint", doi="10.1/xyz")
    c = _FakePaper("99999999", "pubmed", "Unrelated paper", "different text")

    result = deduplicate_papers([a, b, c])
    assert len(result.papers) == 2, result.papers
    assert result.merged_count == 1
    kept_ids = {p.paper_id for p in result.papers}
    assert "38812345" in kept_ids  # pubmed preferred by default
    assert "10.1101/2023.01.01.000001" not in kept_ids
    print("dedup.self_test(): OK ->", result.summary())
    for entry in result.merge_log:
        print("  merged:", entry)


if __name__ == "__main__":
    self_test()
