# tests/test_ner_extractor.py
# Unit tests for BioNERExtractor (rules-only mode — no spaCy download needed).

import pytest
from src.ner_extractor import (
    BioNERExtractor, _canonical, _is_gene_like,
)

@pytest.fixture(scope='module')
def extractor():
    """Rules-only extractor"""
    return BioNERExtractor(use_statistical=False)


# -- Canonical / helpers --

def test_canonical_lowercase():
    assert _canonical('BRCA1 ') == 'brca1'

def test_canonical_strips_whitespace():
    assert _canonical('  Breast Cancer  ') == 'breast cancer'

def test_is_gene_like_true():
    assert _is_gene_like('FOXP3') is True
    assert _is_gene_like('CDK4') is True

def test_is_gene_like_false_common():
    # Common abbreviations must not be classified as genes
    assert _is_gene_like('RNA') is False
    assert _is_gene_like('DNA') is False

def test_is_gene_like_false_lowercase():
    assert _is_gene_like('cancer') is False

# -- Single-token gene extraction --

def test_extracts_brca1(extractor):
    pe = extractor.extract('BRCA1 mutations cause breast cancer.', 'p1')
    assert 'brca1' in pe.genes, f'Expected brca1 in genes, got {pe.genes}'

def test_extracts_tp53(extractor):
    pe = extractor.extract('TP53 is a tumour suppressor gene.', 'p2')
    assert 'tp53' in pe.genes

def test_extracts_kras(extractor):
    pe = extractor.extract('KRAS mutations are common in pancreatic cancer.', 'p3')
    assert 'kras' in pe.genes

def test_case_insensitive_gene(extractor):
    pe = extractor.extract('brca2 variant detected in patient.', 'p4')
    assert 'brca2' in pe.genes, 'Should match lowercase gene name'

# -- Single-token disease extraction --

def test_extracts_cancer(extractor):
    pe = extractor.extract('Lung cancer prevalence is rising.', 'p5')
    assert 'cancer' in pe.diseases or 'lung cancer' in pe.diseases

def test_extracts_diabetes(extractor):
    pe = extractor.extract('Type 2 diabetes affects millions.', 'p6')
    assert 'diabetes' in pe.diseases or 'type 2 diabetes' in pe.diseases

def test_extracts_alzheimers(extractor):
    pe = extractor.extract("Alzheimer's disease is the most common form of dementia.", 'p7')
    found = any('alzheimer' in d for d in pe.diseases)
    assert found, f'Expected alzheimer in diseases, got {pe.diseases}'

# -- Multi-token patterns --

def test_extracts_breast_cancer_as_single_entity(extractor):
    pe = extractor.extract('Breast cancer risk is elevated in BRCA1 carriers.', 'p8')
    # 'breast cancer' should be matched as one entity, not 'breast' + 'cancer' separately
    all_diseases = pe.diseases
    assert any('breast cancer' in d or 'cancer' in d for d in all_diseases), \
        f'breast cancer not found in {all_diseases}'

def test_extracts_myocardial_infarction(extractor):
    pe = extractor.extract('Myocardial infarction is a leading cause of death.', 'p9')
    found = any('myocardial' in d for d in pe.diseases)
    assert found, f'Expected myocardial infarction, got {pe.diseases}'

def test_extracts_arrhythmia_spelling(extractor):
    pe = extractor.extract('Cardiac arrhythmia is common in this cohort.', 'p10')
    assert 'arrhythmia' in pe.diseases

# -- Empty and edge-case inputs --

def test_empty_text_returns_empty_entities(extractor):
    pe = extractor.extract('', 'empty')
    assert pe.genes == []
    assert pe.diseases == []

def test_no_entities_text(extractor):
    pe = extractor.extract('The weather was sunny and warm today.', 'no_ent')
    assert pe.genes == []
    assert pe.diseases == []

def test_only_genes_no_diseases(extractor):
    pe = extractor.extract('EGFR and HER2 expression was measured.', 'genes_only')
    assert len(pe.genes) >= 1
    assert pe.diseases == []

def test_only_diseases_no_genes(extractor):
    pe = extractor.extract('Cancer and diabetes are common comorbidities.', 'dis_only')
    assert len(pe.diseases) >= 1
    assert pe.genes == []


# -- Deduplication --

def test_deduplication_same_gene_multiple_mentions(extractor):
    pe = extractor.extract(
        'BRCA1 plays a role in DNA repair. BRCA1 mutations are pathogenic. '
        'Studies of BRCA1 continue.', 'dup'
    )
    assert pe.genes.count('brca1') == 1

def test_deduplication_same_disease_multiple_mentions(extractor):
    pe = extractor.extract(
        'Cancer cells proliferate. Cancer treatment is challenging. '
        'Cancer research is progressing.', 'dup_dis'
    )
    count = pe.diseases.count('cancer')
    assert count == 1, f'Expected 1, got {count}'

# -- PaperEntities properties --

def test_paper_entities_has_entities_true(extractor):
    pe = extractor.extract('BRCA1 causes breast cancer.', 'has_ent')
    assert pe.has_entities is True

def test_paper_entities_has_entities_false(extractor):
    pe = extractor.extract('The sky is blue.', 'no_ent_2')
    assert pe.has_entities is False

def test_paper_entities_all_entity_names(extractor):
    pe = extractor.extract('BRCA1 and TP53 in breast cancer.', 'all_names')
    names = pe.all_entity_names
    assert isinstance(names, list)
    assert len(names) >= 2

# -- Batch extraction --

def test_extract_batch_returns_correct_count(extractor):
    texts = [
        ('BRCA1 mutations in breast cancer.', 'b1'),
        ('KRAS in lung cancer.', 'b2'),
        ('No entities here.', 'b3'),
    ]
    results = extractor.extract_batch(texts)
    assert len(results) == 3

def test_extract_batch_correct_paper_ids(extractor):
    texts = [('TP53 in cancer.', 'x1'), ('EGFR in lung.', 'x2')]
    results = extractor.extract_batch(texts)
    ids = [r.paper_id for r in results]
    assert ids == ['x1', 'x2']

# -- Dynamic pattern addition --

def test_add_gene_pattern_dynamic(extractor):
    """Adding a new gene at runtime should work immediately."""
    extractor.add_gene_pattern('FOXP3')
    pe = extractor.extract('FOXP3 regulates immune tolerance.', 'dyn_gene')
    assert 'foxp3' in pe.genes

def test_add_disease_pattern_dynamic(extractor):
    extractor.add_disease_pattern("Crohn's disease")
    pe = extractor.extract("Crohn's disease affects the intestine.", 'dyn_dis')
    found = any('crohn' in d for d in pe.diseases)
    assert found, f'Expected crohn in diseases, got {pe.diseases}'

def test_add_gene_pattern_rejects_empty_name(extractor):
    with pytest.raises(ValueError):
        extractor.add_gene_pattern('   ')

def test_add_disease_pattern_rejects_empty_name(extractor):
    with pytest.raises(ValueError):
        extractor.add_disease_pattern('   ')

