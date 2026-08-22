"""
tests/test_canonicalization_edge_cases.py 

Targeted stress tests for src/ner_extractor.py's `_canonical()` and
`_is_gene_like()` — the two functions responsible for making sure that
"BRCA1", "brca1", "BRCA-1", and "  BRCA1  " all collapse onto the SAME
knowledge-graph node instead of silently fragmenting centrality/GCS
metrics across near-duplicate nodes.
"""
import pytest

from src.ner_extractor import _canonical, _is_gene_like

# -- _canonical(): case, whitespace, punctuation --

@pytest.mark.parametrize("raw,expected", [
    ("BRCA1", "brca1"),
    ("brca1", "brca1"),
    ("Brca1", "brca1"),
    ("  BRCA1  ", "brca1"),
    ("BRCA1\n", "brca1"),
    ("BRCA1\t", "brca1"),
])
def test_canonical_collapses_case_and_whitespace_variants(raw, expected):
    assert _canonical(raw) == expected

def test_canonical_hyphen_variant_is_distinct_from_no_hyphen():
    # Known limitation: _canonical() only lowercases + strips.
    # BRCA-1 and BRCA1 do not canonicalize to the same node.
    assert _canonical("BRCA-1") != _canonical("BRCA1")

def test_canonical_multiple_internal_spaces_not_collapsed():
    # _canonical() does not collapse internal whitespace runs. "breast  cancer" 
    # (two spaces) stays distinct from "breast cancer" (one space). 
    assert _canonical("breast  cancer") == "breast  cancer"

def test_canonical_empty_and_whitespace_only():
    assert _canonical("") == ""
    assert _canonical("   ") == ""

# -- _is_gene_like(): the heuristic gate for statistical-model hits --

@pytest.mark.parametrize("text", [
    "TP53", "BRCA1", "BRCA2", "EGFR", "KRAS", "APOE", "PTEN", "MYC",
])
def test_is_gene_like_true_for_canonical_gene_symbols(text):
    assert _is_gene_like(text) is True

@pytest.mark.parametrize("text", [
    "tp53",      # lowercase - regex requires leading uppercase per spec
    "Tp53",      # mixed case
    "brca1",     # fully lowercase
])
def test_is_gene_like_false_for_lowercase_variants(text):
    # Regex requires ALL-CAPS symbols; lowercase mentions are rejected here.
    assert _is_gene_like(text) is False

@pytest.mark.parametrize("text", [
    "CI", "IQR", "HR", "OR", "RR", "SD", "SE", "AUC", "ROC",
])
def test_is_gene_like_false_for_known_stopword_abbreviations(text):
    assert _is_gene_like(text) is False

def test_is_gene_like_false_for_too_short_symbol():
    # Regex requires >= 3 total characters ({2,7} after the first char).
    assert _is_gene_like("P5") is False

def test_is_gene_like_false_for_too_long_symbol():
    # Regex caps at 8 total characters ({2,7} after the first char).
    assert _is_gene_like("ABCDEFGHIJ") is False

def test_is_gene_like_false_for_symbol_with_internal_space():
    assert _is_gene_like("BR CA1") is False

def test_is_gene_like_false_for_pure_digit_string():
    assert _is_gene_like("12345") is False

def test_is_gene_like_true_for_hyphenated_gene_symbol():
    # HER-2-style hyphenated symbols must pass.
    assert _is_gene_like("HER-2") is True
