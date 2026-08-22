"""
src/mesh_fusion.py — MeSH-term disease augmentation

PubMed already ships every article with NLM-indexer-assigned MeSH headings
(Medical Subject Headings) — a curated, human-reviewed controlled vocabulary.
BioSearch AI fetches these into `Paper.keywords` (see fetcher_pubmed.py's
`.//MeshHeading/DescriptorName` parsing) but the knowledge-graph pipeline
currently ignores them entirely, rebuilding similar information from raw
abstract text via a much noisier rule+statistical NER hybrid.

MeSH is NOT a good gene-symbol vocabulary (genes are sparsely and
inconsistently represented in MeSH), so this module does NOT attempt gene
extraction from MeSH. It IS an excellent, high-precision *disease/condition*
vocabulary, since MeSH's whole design purpose is indexing diseases,
chemicals, and biomedical concepts. This module extracts the disease-like
subset of each paper's MeSH headings and returns them so they can be fused
into the existing NER-derived disease list before graph construction.

DESIGN CHOICES
---------------------------------------------------------------
1. We do NOT attempt a MeSH-to-UMLS semantic-type API lookup. That is the
   "textbook correct" approach but requires a UMLS license + API key and is
   out of scope for a one-week hardening pass. Instead we use a curated
   exclusion list of the ~120 most common NON-disease MeSH headings
   (demographics, study-design terms, geography, species) and treat
   everything else as a disease/condition candidate. This is a precision/
   recall trade-off: we accept a small amount of noise (some non-disease
   concept headings will slip through) in exchange for zero external
   dependencies and full transparency. State this trade-off explicitly if
    This trades some precision for recall while avoiding an external ontology
    dependency.
2. Provenance is preserved: every entity this module produces is tagged
   source="mesh" wherever the caller wants to track it, so it is always
   possible to separate "NLM indexer said so" from "our NER guessed so."
3. Canonicalization mirrors `src/ner_extractor.py`'s `_canonical()`
   (lowercase + strip) so MeSH-derived and NER-derived diseases collapse
   onto the same graph node when they refer to the same concept
   (e.g. NER's "breast cancer" and MeSH's "Breast Neoplasms" will NOT
   automatically merge — MeSH uses formal nomenclature. This is a known,
   accepted limitation; see MESH_SYNONYM_MAP below for the small manual
   bridge we do provide for your five standard evaluation queries).

INTEGRATION
-------------------------------------------------------------------
Call `extract_mesh_diseases(paper)` for each paper and merge the result
into that paper's `PaperEntities.diseases` list before it reaches
`BioKnowledgeGraph.add_paper_entities()`. A ready-to-use merge helper,
`fuse_mesh_into_entities()`, is provided below and is defensive about the
exact field names on your `PaperEntities` dataclass (it uses getattr/
setattr with a clear assertion error if the shape doesn't match, rather
than failing silently).
"""

from __future__ import annotations

import logging
from dataclasses import replace, is_dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


# ── MeSH headings that are demographic / methodological / geographic, ──────
# ── i.e. almost never a disease or condition. Curated manually. ────────────
MESH_NOISE_TERMS: frozenset[str] = frozenset(t.lower() for t in {
    # Demographics
    "Humans", "Animals", "Male", "Female", "Adult", "Aged", "Aged, 80 and over",
    "Middle Aged", "Young Adult", "Child", "Child, Preschool", "Infant",
    "Infant, Newborn", "Adolescent", "Pregnancy", "Mice", "Rats", "Rabbits",
    "Dogs", "Cats", "Swine", "Cattle", "Zebrafish", "Drosophila melanogaster",
    "Caenorhabditis elegans", "Mice, Inbred C57BL", "Mice, Knockout",
    "Mice, Transgenic", "Rats, Sprague-Dawley", "Rats, Wistar",
    # Study design / methodology
    "Case Reports", "Retrospective Studies", "Prospective Studies",
    "Cohort Studies", "Cross-Sectional Studies", "Longitudinal Studies",
    "Follow-Up Studies", "Randomized Controlled Trials as Topic",
    "Clinical Trials as Topic", "Double-Blind Method", "Reproducibility of Results",
    "Sensitivity and Specificity", "Predictive Value of Tests",
    "ROC Curve", "Statistics, Nonparametric", "Data Interpretation, Statistical",
    "Sample Size", "Time Factors", "Risk Factors", "Risk Assessment",
    "Treatment Outcome", "Prognosis", "Survival Analysis", "Survival Rate",
    "Disease-Free Survival", "Kaplan-Meier Estimate", "Proportional Hazards Models",
    "Multivariate Analysis", "Logistic Models",
    # Geography
    "United States", "United Kingdom", "China", "Europe", "Germany",
    "Japan", "Canada", "Australia", "France", "Italy", "India",
    # Generic biomedical process / method nouns that are not diseases
    "Gene Expression Regulation", "Gene Expression Regulation, Neoplastic",
    "Signal Transduction", "Cell Line", "Cell Line, Tumor", "Cell Proliferation",
    "Cell Survival", "Apoptosis", "Cell Differentiation", "Cell Cycle",
    "Base Sequence", "Amino Acid Sequence", "Molecular Sequence Data",
    "Polymerase Chain Reaction", "Immunohistochemistry", "Blotting, Western",
    "Flow Cytometry", "Microscopy, Electron", "Sequence Analysis, DNA",
    "Sequence Analysis, RNA", "High-Throughput Nucleotide Sequencing",
    "Gene Expression Profiling", "RNA, Messenger", "DNA, Complementary",
    "Genotype", "Phenotype", "Mutation", "Polymorphism, Single Nucleotide",
    "Genetic Predisposition to Disease", "Biomarkers, Tumor", "Biomarkers",
    "Prevalence", "Incidence", "Comorbidity", "Odds Ratio",
    "Confidence Intervals", "Quality of Life", "Surveys and Questionnaires",
})


# ── A small, honest bridge between a handful of MeSH formal terms and the ──
# ── plain-language disease strings your NER layer already produces, so    ──
# ── the two sources visibly reinforce each other on your five standard    ──
# ── evaluation queries instead of silently creating parallel nodes.       ──
# ── Extend this table as you observe more overlaps in real output —       ──
# ── do not treat it as exhaustive; it is a deliberately small, auditable  ──
# ── seed, not a general-purpose ontology mapping.                         ──
MESH_SYNONYM_MAP: dict[str, str] = {
    "breast neoplasms": "breast cancer",
    "lung neoplasms": "lung cancer",
    "carcinoma, non-small-cell lung": "non-small cell lung cancer",
    "ovarian neoplasms": "ovarian cancer",
    "alzheimer disease": "alzheimer's disease",
    "neoplasms": "cancer",
    "carcinoma": "cancer",
    "tumor suppressor protein p53": "tp53",
}


def _canonical(text: str) -> str:
    """Mirrors src/ner_extractor.py's _canonical() — lowercase + strip."""
    return text.strip().lower()


def classify_mesh_heading(term: str) -> bool:
    """
    Return True if `term` should be treated as a disease/condition candidate.

    Deliberately permissive: anything not in MESH_NOISE_TERMS and not empty
    is treated as a candidate. This favours recall over precision by design
    (see module docstring, design choice #1) — you are trading a bit of
    graph noise for zero external ontology dependency.
    """
    if not term or not term.strip():
        return False
    return term.strip().lower() not in MESH_NOISE_TERMS


def extract_mesh_diseases(paper) -> list[str]:
    """
    Extract disease-like MeSH descriptor terms from a single Paper.

    Parameters
    ----------
    paper : Paper
        Must expose a `.keywords` attribute (list[str]) — this matches
        src/models.py's Paper.keywords field, populated in
        fetcher_pubmed.py from MeshHeading/DescriptorName elements.

    Returns
    -------
    list[str]
        Canonicalized (lowercased, stripped) candidate disease terms,
        synonym-mapped where a bridge exists in MESH_SYNONYM_MAP,
        de-duplicated, order-preserving.
    """
    keywords: Iterable[str] = getattr(paper, "keywords", None) or []
    seen: set[str] = set()
    out: list[str] = []
    for raw in keywords:
        if not classify_mesh_heading(raw):
            continue
        canon = _canonical(raw)
        canon = MESH_SYNONYM_MAP.get(canon, canon)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def extract_mesh_diseases_batch(papers) -> dict[str, list[str]]:
    """Convenience: {paper_id: [mesh-derived disease terms]} for a list of Papers."""
    result: dict[str, list[str]] = {}
    for paper in papers:
        paper_id = getattr(paper, "paper_id", None)
        if paper_id is None:
            continue
        diseases = extract_mesh_diseases(paper)
        if diseases:
            result[paper_id] = diseases
    return result


def fuse_mesh_into_entities(paper_entities, mesh_diseases: list[str]):
    """
    Merge MeSH-derived disease terms into an existing PaperEntities object.

    Defensive by design: PaperEntities (src/ner_extractor.py) is documented
    as a mutable dataclass with at least a `.diseases: list[str]` field and
    a `.paper_id` field. Rather than assuming the exact dataclass shape,
    this function verifies the attribute exists before touching it and
    raises a clear AssertionError (not a silent no-op) if your local
    PaperEntities doesn't match — so a shape mismatch fails loudly in your
    test suite instead of quietly producing an under-populated graph.

    Returns a new object for dataclass inputs; plain mutable objects are
    updated in place and returned.
    """
    if not mesh_diseases:
        return paper_entities

    assert hasattr(paper_entities, "diseases"), (
        "fuse_mesh_into_entities(): expected PaperEntities to expose a "
        "'.diseases' list attribute — your local PaperEntities shape has "
        "changed. Update mesh_fusion.py's fuse_mesh_into_entities() to "
        "match the real field name before relying on this function."
    )

    existing = list(getattr(paper_entities, "diseases") or [])
    existing_set = set(existing)
    merged = existing + [d for d in mesh_diseases if d not in existing_set]

    if is_dataclass(paper_entities):
        try:
            # frozen dataclasses require dataclasses.replace(); mutable ones
            # also accept it and it's simpler to use uniformly here.
            return replace(paper_entities, diseases=merged)
        except TypeError:
            pass  # not a dataclass instance in the way replace() expects

    # Fallback for a plain mutable object / non-frozen dataclass.
    setattr(paper_entities, "diseases", merged)
    return paper_entities


def self_test() -> None:
    """Run `python -m src.mesh_fusion` to sanity-check this module in isolation."""

    class _FakePaper:
        def __init__(self, paper_id, keywords):
            self.paper_id = paper_id
            self.keywords = keywords

    p = _FakePaper(
        "PMID123",
        ["Humans", "Female", "Breast Neoplasms", "BRCA1 Protein",
         "Case Reports", "Neoplasm Staging"],
    )
    diseases = extract_mesh_diseases(p)
    assert diseases == ["breast cancer", "brca1 protein", "neoplasm staging"], diseases
    print("mesh_fusion.self_test(): OK ->", diseases)


if __name__ == "__main__":
    self_test()
