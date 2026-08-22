# tests/test_mesh_fusion.py 

import pytest

from src.mesh_fusion import (
    classify_mesh_heading,
    extract_mesh_diseases,
    extract_mesh_diseases_batch,
    fuse_mesh_into_entities,
    MESH_NOISE_TERMS,
    MESH_SYNONYM_MAP,
)


class _FakePaper:
    def __init__(self, paper_id="P1", keywords=None):
        self.paper_id = paper_id
        self.keywords = keywords or []


# -- classify_mesh_heading --

def test_classify_rejects_known_noise_term():
    assert classify_mesh_heading("Humans") is False
    assert classify_mesh_heading("Female") is False
    assert classify_mesh_heading("Case Reports") is False


def test_classify_is_case_insensitive():
    assert classify_mesh_heading("humans") is False
    assert classify_mesh_heading("HUMANS") is False


def test_classify_accepts_plausible_disease_term():
    assert classify_mesh_heading("Breast Neoplasms") is True
    assert classify_mesh_heading("Diabetes Mellitus, Type 2") is True


def test_classify_rejects_empty_or_whitespace():
    assert classify_mesh_heading("") is False
    assert classify_mesh_heading("   ") is False


# -- extract_mesh_diseases --

def test_extract_filters_noise_and_canonicalizes():
    paper = _FakePaper(keywords=["Humans", "Female", "Breast Neoplasms"])
    result = extract_mesh_diseases(paper)
    assert result == ["breast cancer"]  # synonym-mapped + noise filtered


def test_extract_applies_synonym_map():
    paper = _FakePaper(keywords=["Carcinoma, Non-Small-Cell Lung"])
    result = extract_mesh_diseases(paper)
    assert result == ["non-small cell lung cancer"]


def test_extract_deduplicates_preserving_order():
    paper = _FakePaper(keywords=["Neoplasms", "Carcinoma", "Neoplasms"])
    result = extract_mesh_diseases(paper)
    # "Neoplasms" and "Carcinoma" both map to "cancer" via MESH_SYNONYM_MAP
    assert result == ["cancer"]


def test_extract_empty_keywords_returns_empty_list():
    paper = _FakePaper(keywords=[])
    assert extract_mesh_diseases(paper) == []


def test_extract_missing_keywords_attribute_does_not_crash():
    class _NoKeywords:
        paper_id = "P2"
    assert extract_mesh_diseases(_NoKeywords()) == []


def test_extract_all_noise_returns_empty_list():
    paper = _FakePaper(keywords=["Humans", "Female", "Case Reports"])
    assert extract_mesh_diseases(paper) == []


# -- extract_mesh_diseases_batch --

def test_batch_skips_papers_with_no_disease_mesh_terms():
    papers = [
        _FakePaper("P1", ["Humans", "Female"]),
        _FakePaper("P2", ["Breast Neoplasms"]),
    ]
    result = extract_mesh_diseases_batch(papers)
    assert "P1" not in result
    assert result["P2"] == ["breast cancer"]


def test_batch_skips_papers_with_no_paper_id():
    class _NoId:
        keywords = ["Breast Neoplasms"]
    result = extract_mesh_diseases_batch([_NoId()])
    assert result == {}


# -- fuse_mesh_into_entities --

class _FakePaperEntities:
    """Mimics the documented shape of src/ner_extractor.py's PaperEntities."""
    def __init__(self, paper_id, genes=None, diseases=None, raw_entities=None):
        self.paper_id = paper_id
        self.genes = genes or []
        self.diseases = diseases or []
        self.raw_entities = raw_entities or []


def test_fuse_merges_new_diseases():
    pe = _FakePaperEntities("P1", diseases=["diabetes"])
    fused = fuse_mesh_into_entities(pe, ["breast cancer"])
    assert set(fused.diseases) == {"diabetes", "breast cancer"}


def test_fuse_does_not_duplicate_existing_disease():
    pe = _FakePaperEntities("P1", diseases=["breast cancer"])
    fused = fuse_mesh_into_entities(pe, ["breast cancer"])
    assert fused.diseases.count("breast cancer") == 1


def test_fuse_with_no_mesh_diseases_is_a_no_op():
    pe = _FakePaperEntities("P1", diseases=["diabetes"])
    fused = fuse_mesh_into_entities(pe, [])
    assert fused.diseases == ["diabetes"]


def test_fuse_raises_clear_error_on_shape_mismatch():
    class _WrongShape:
        pass
    with pytest.raises(AssertionError, match="diseases"):
        fuse_mesh_into_entities(_WrongShape(), ["breast cancer"])


def test_fuse_preserves_other_fields():
    pe = _FakePaperEntities("P1", genes=["tp53"], diseases=["diabetes"])
    fused = fuse_mesh_into_entities(pe, ["breast cancer"])
    assert fused.genes == ["tp53"]
    assert fused.paper_id == "P1"


# -- Curated list sanity (guards against accidental edits shrinking it) --

def test_noise_list_has_reasonable_minimum_size():
    # Guards against accidental shrinkage of the noise-term list.
    assert len(MESH_NOISE_TERMS) >= 80


def test_synonym_map_covers_standard_query_terms():
    # Losing these mappings silently weakens MeSH/NER cross-validation across 
    # the project's five STANDARD_QUERIES in evaluator.py.
    for expected_key in ("breast neoplasms", "lung neoplasms", "alzheimer disease"):
        assert expected_key in MESH_SYNONYM_MAP
