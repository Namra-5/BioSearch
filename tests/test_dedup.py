# tests/test_dedup.py

import hashlib

import pytest

from src.dedup import deduplicate_papers, DedupResult, _normalize_doi


class _FakePaper:
    def __init__(self, paper_id, source, title, abstract="", doi=None):
        self.paper_id = paper_id
        self.source = source
        self.title = title
        self.abstract = abstract
        self.doi = doi

    @property
    def content_hash(self):
        raw = (self.title + self.abstract).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]


# -- _normalize_doi --

def test_normalize_doi_strips_url_prefix():
    assert _normalize_doi("https://doi.org/10.1/xyz") == "10.1/xyz"
    assert _normalize_doi("http://doi.org/10.1/xyz") == "10.1/xyz"


def test_normalize_doi_strips_doi_colon_prefix():
    assert _normalize_doi("doi:10.1/xyz") == "10.1/xyz"


def test_normalize_doi_lowercases_and_strips_whitespace():
    assert _normalize_doi("  10.1/XYZ  ") == "10.1/xyz"


def test_normalize_doi_handles_none_and_empty():
    assert _normalize_doi(None) is None
    assert _normalize_doi("") is None


# -- deduplicate_papers: empty / trivial cases --

def test_empty_input_returns_empty_result():
    result = deduplicate_papers([])
    assert result.papers == []
    assert result.merged_count == 0


def test_single_paper_is_unchanged():
    p = _FakePaper("1", "pubmed", "Title", "Abstract")
    result = deduplicate_papers([p])
    assert result.papers == [p]
    assert result.merged_count == 0


def test_no_duplicates_keeps_all_papers():
    a = _FakePaper("1", "pubmed", "Title A", "Abstract A", doi="10.1/a")
    b = _FakePaper("2", "pubmed", "Title B", "Abstract B", doi="10.1/b")
    result = deduplicate_papers([a, b])
    assert len(result.papers) == 2
    assert result.merged_count == 0


# -- DOI-based matching --

def test_same_doi_across_sources_is_merged():
    a = _FakePaper("38812345", "pubmed", "KRAS study", doi="10.1/xyz")
    b = _FakePaper("10.1101/preprint", "biorxiv", "KRAS study preprint", doi="10.1/xyz")
    result = deduplicate_papers([a, b])
    assert len(result.papers) == 1
    assert result.merged_count == 1


def test_prefer_source_pubmed_default_wins_doi_tie():
    a = _FakePaper("38812345", "pubmed", "KRAS study", doi="10.1/xyz")
    b = _FakePaper("10.1101/preprint", "biorxiv", "KRAS study preprint", doi="10.1/xyz")
    result = deduplicate_papers([a, b])
    assert result.papers[0].source == "pubmed"


def test_prefer_source_can_be_inverted():
    a = _FakePaper("38812345", "pubmed", "KRAS study", doi="10.1/xyz")
    b = _FakePaper("10.1101/preprint", "biorxiv", "KRAS study preprint", doi="10.1/xyz")
    result = deduplicate_papers([a, b], prefer_source="biorxiv")
    assert result.papers[0].source == "biorxiv"


def test_doi_case_and_url_variants_still_match():
    a = _FakePaper("1", "pubmed", "Study", doi="10.1/XYZ")
    b = _FakePaper("2", "biorxiv", "Study preprint", doi="https://doi.org/10.1/xyz")
    result = deduplicate_papers([a, b])
    assert len(result.papers) == 1


def test_merge_log_records_dropped_identity():
    a = _FakePaper("38812345", "pubmed", "KRAS study", doi="10.1/xyz")
    b = _FakePaper("10.1101/preprint", "biorxiv", "KRAS study preprint", doi="10.1/xyz")
    result = deduplicate_papers([a, b])
    assert result.merge_log == [("38812345", "pubmed", "biorxiv:10.1101/preprint")]


# -- content_hash-based matching (no DOI present) --

def test_same_content_hash_no_doi_is_merged():
    a = _FakePaper("1", "pubmed", "Same title", "Same abstract")
    b = _FakePaper("2", "biorxiv", "Same title", "Same abstract")
    result = deduplicate_papers([a, b])
    assert len(result.papers) == 1
    assert result.merged_count == 1


def test_different_content_no_doi_is_not_merged():
    a = _FakePaper("1", "pubmed", "Title A", "Abstract A")
    b = _FakePaper("2", "pubmed", "Title B", "Abstract B")
    result = deduplicate_papers([a, b])
    assert len(result.papers) == 2
    assert result.merged_count == 0


def test_missing_content_hash_attribute_does_not_crash():
    class _NoHash:
        paper_id = "1"
        source = "pubmed"
        doi = None
    result = deduplicate_papers([_NoHash()])
    assert len(result.papers) == 1
    assert result.merged_count == 0


# -- Mixed scenario --

def test_mixed_doi_and_hash_matches_in_one_pass():
    a = _FakePaper("1", "pubmed", "DOI match", "text1", doi="10.1/a")
    b = _FakePaper("2", "biorxiv", "DOI match preprint", "text1-pre", doi="10.1/a")
    c = _FakePaper("3", "pubmed", "Hash match", "same text")
    d = _FakePaper("4", "biorxiv", "Hash match", "same text")
    e = _FakePaper("5", "pubmed", "Unique paper", "unique text")

    result = deduplicate_papers([a, b, c, d, e])
    assert len(result.papers) == 3  # (a or b) + (c or d) + e
    assert result.merged_count == 2


def test_summary_string_is_human_readable():
    a = _FakePaper("1", "pubmed", "T", "A", doi="10.1/x")
    b = _FakePaper("2", "biorxiv", "T2", "A2", doi="10.1/x")
    result = deduplicate_papers([a, b])
    assert "1 unique papers kept" in result.summary()
    assert "1 duplicate records merged" in result.summary()
