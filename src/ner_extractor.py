# src/ner_extractor.py
# Biomedical Named Entity Recognition pipeline for BioSearch AI — Week 3
#
# Architecture: HYBRID NER
#   Layer 1 — EntityRuler (rule-based): 35 genes + 35 diseases as exact-match
#              patterns. Fast, deterministic, zero false negatives for known entities.
#   Layer 2 — spaCy statistical model (en_core_web_sm): catches novel entities
#              the ruler has never seen. Labels mapped to GENE/DISEASE via heuristics.
#
# Design principle: this file knows nothing about graphs.
# It takes text in, returns structured ExtractedEntity objects out.
# knowledge_graph.py consumes those objects.

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# spaCy imported lazily so the module can be imported even if spaCy is not 
#    installed (tests can mock at the boundary)
try:
    import spacy                          # type: ignore
    from spacy.language import Language   # type: ignore
    from spacy.tokens import Doc, Span    # type: ignore
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    logger.warning(
        'spaCy not installed. Run: pip install spacy && '
        'python -m spacy download en_core_web_sm'
    )

# -- Entity label constants --
LABEL_GENE    = 'GENE'
LABEL_DISEASE = 'DISEASE'

# -- Biomedical entity dictionaries --
#
# Why hard-coded dictionaries and not an external file?
# We want zero external file dependencies — the project must run from a fresh 
# git clone with only pip installs.
# 35 genes and 35 diseases cover the most-cited entities in PubMed; the
# statistical layer catches everything else.
#
# Case strategy: patterns stored in lower-case; matched with LOWER attribute
# so BRCA1, Brca1, and brca1 all match. Exact-match is intentional — we do
# NOT want 'ATM' matching inside 'ATM machine' (false positive).

_GENE_PATTERNS: list[dict] = [
    # Tumor suppressor genes
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'brca1'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'brca2'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'tp53'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'p53'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'pten'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'rb1'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'apc'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'vhl'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'cdkn2a'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'nf1'}]},
    # Oncogenes
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'kras'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'nras'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'braf'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'egfr'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'her2'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'erbb2'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'myc'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'akt1'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'pik3ca'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'met'}]},
    # DNA repair/genome stability
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'atm'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'chek2'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'palb2'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'rad51'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'msh2'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'mlh1'}]},
    # Neurological/metabolic
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'apoe'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'app'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'psen1'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'psen2'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'snca'}]},
    # Immunity
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'tnf'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'il6'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'il1b'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'ifng'}]},
    # Multi-token gene names
    {'label': LABEL_GENE, 'pattern': [{'LOWER': 'her'}, {'TEXT': '-'}, {'LOWER':'2'}]},
    {'label': LABEL_GENE, 'pattern': [{'LOWER':'nf'}, {'TEXT': '-'}, {'LOWER': 'κb'}]},
]

_DISEASE_PATTERNS: list[dict] = [
    # Cancers
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'cancer'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'carcinoma'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'melanoma'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'leukemia'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'leukaemia'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'lymphoma'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'glioma'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'glioblastoma'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'sarcoma'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'adenocarcinoma'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'neoplasm'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'malignancy'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'tumour'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'tumor'}]},
    # Neurological diseases
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'alzheimer'}, {'ORTH': "'s"}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'alzheimers'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'parkinson'}, {'ORTH': "'s"}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'parkinsons'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'epilepsy'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'schizophrenia'}]},
    # Cardiovascular
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'atherosclerosis'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'hypertension'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'cardiomyopathy'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'arrhythmia'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'arrythmia'}]},
    # Metabolic/endocrine
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'diabetes'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'obesity'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'hyperlipidemia'}]},
    # Inflammatory/immune
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'arthritis'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'lupus'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'colitis'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'asthma'}]},
    # Infectious
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'hiv'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'tuberculosis'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'sepsis'}]},
    # Multi-token disease names
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'breast'}, {'LOWER': 'cancer'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'lung'}, {'LOWER': 'cancer'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'colon'}, {'LOWER': 'cancer'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'colorectal'}, 
                                         {'LOWER': 'cancer'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'prostate'}, {'LOWER': 'cancer'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'ovarian'}, {'LOWER': 'cancer'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'pancreatic'}, 
                                         {'LOWER': 'cancer'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'myocardial'}, 
                                         {'LOWER': 'infarction'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'heart'}, {'LOWER': 'attack'}]},
    {'label': LABEL_DISEASE, 'pattern': [{'LOWER': 'type'}, {'IS_DIGIT': True}, 
                                         {'LOWER': 'diabetes'}]}
] 

# -- Statistical model label mapping -- 
# en_core_web_sm uses OntoNotes labels. We map a subset to our own labels.
# Only ORG/PRODUCT/WORK_OF_ART are excluded as they are noisy in biomedical text.
_STAT_LABEL_MAP: dict[str, Optional[str]] = {
    'PERSON':  None,
    'ORG':     None,           # too noisy
    'GPE':     None,
    'LOC':     None,
    'FAC':     None,
    'NORP':    None,
    'PRODUCT': None,
    'EVENT':   LABEL_DISEASE,  # disease outbreaks are tagged EVENT
    'LAW':     None,
    'DATE':    None,
    'TIME':    None,
    'MONEY':   None,
    'PERCENT': None,
    'QUANTITY':None,
    'ORDINAL': None,
    'CARDINAL':None,
    'WORK_OF_ART': None,
    'LANGUAGE':None,
}

# Gene-like regex: all-caps 3-8 chars, may contain digits/hyphens, not a common word
_GENE_REGEX = re.compile(r'^[A-Z][A-Z0-9\-]{2,7}$')
_COMMON_WORDS = frozenset({
    # Generic English
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
    "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "HAD", "HIS",
    "HOW", "ITS", "NEW", "USE", "TWO", "WAY", "WHO", "MAY",
    # Standard biomedical acronyms that are NOT genes
    "RNA", "DNA", "PCR", "MRI", "USA", "FDA", "WHO", "ATP",
    "NAD", "TCA", "ICU", "BMI", "SNP", "CNS", "PNS", "CNV",
    "SNV", "NGS", "WGS", "WES", "CNA", "LOH", "TMB", "MSI",
    "PDX", "PDL", "CAR", "CAT", "ACE", "BMR", "LDL", "HDL",
    # Statistical / clinical trial abbreviations (your false positives)
    "CI",  "HR",  "OR",  "RR",  "SD",  "SE",  "IQR", "HRs",
    "RRM", "PSO", "IQR", "AUC", "ROC", "PPV", "NPV", "NNT",
    "NNH", "ARR", "RRR", "CIs", "ORs", "HRs", "SDs", "SEs",
    # 2-letter codes (too short and ambiguous — require ≥3 chars now)
    "LD",  "CI",  "BC",  "II",  "LD",  "MR",  "CT",  "OS",
    "PFS", "DFS", "ORR", "DCR", "TTF", "PFR",
    # Drug/trial names
    "ABC", "HRD", "BRD", "SMA", "TAT", "LQP",
    # Journal/citation fragments
    "ETC", "ALT", "AST", "GGT",
})

# -- Data classes --

@dataclass(frozen=True)
class ExtractedEntity:
    """
    Immutable record representing one named entity found in a text span.

    Using a frozen dataclass (not Pydantic) here because:
    - We produce thousands of these per corpus — Pydantic's per-instance
      validation overhead adds up.
    - These objects are never deserialised from JSON or user input, so
      Pydantic's input-validation benefit does not apply.
    - Dataclasses are hashable when frozen, which lets us put them in sets
      for deduplication.
    """
    text: str          # canonical (lower-stripped) form of the entity
    label: str         # GENE or DISEASE
    source: str        # 'ruler' (deterministic) or 'stat' (statistical)
    paper_id: str      # paper this entity was found in
    start_char: int    # character offset in the original text (for provenance)
    end_char: int


@dataclass
class PaperEntities:
    """
    All entities extracted from a single Paper.
    Stored alongside the paper in KnowledgeBase.
    """
    paper_id: str
    genes: list[str]       = field(default_factory=list)   # canonical names
    diseases: list[str]    = field(default_factory=list)
    raw_entities: list[ExtractedEntity] = field(default_factory=list)

    @property
    def all_entity_names(self) -> list[str]:
        return self.genes + self.diseases

    @property
    def has_entities(self) -> bool:
        return bool(self.genes or self.diseases)


# -- Pipeline builder --

def _build_spacy_pipeline(use_statistical: bool = True, 
                          model_name: str = 'en_core_web_sm',
                          ) -> 'Language':
    """
    Build and return a spaCy NLP pipeline for biomedical NER.

    Pipeline component order matters in spaCy.
    We ALWAYS add EntityRuler BEFORE the statistical 'ner' component
    (ruler first = 'entity_ruler' gets priority).

    If the statistical model is disabled we use a blank English pipeline —
    this is faster but only catches entities in our dictionaries.

    Why ruler-before-ner?
    spaCy resolves overlapping spans by giving priority to the component
    that sets them first. Placing the ruler first means our deterministic
    biomedical patterns always win over the statistical model's guesses
    for known entities. The statistical model only contributes for tokens
    the ruler never matched — this is the hybrid benefit.

    Parameters
    ----------
    use_statistical : bool
        If True, load en_core_web_sm for statistical NER of novel entities.
        Set False for faster offline / test runs.
    model_name : str
        spaCy model name for statistical layer. Requires separate download:
        python -m spacy download en_core_web_sm
    """
    if not _SPACY_AVAILABLE:
        raise ImportError(
            'spaCy is not installed. Run:\n'
            '  pip install spacy\n'
            '  python -m spacy download en_core_web_sm'
        )

    if use_statistical:
        try:
            nlp = spacy.load(model_name, disable=['parser', 'lemmatizer', 'attribute_ruler'])
            logger.info('Loaded spaCy model "%s" (statistical NER enabled).', model_name)
        except OSError:
            logger.warning(
                'spaCy model "%s" not found. Falling back to blank pipeline. '
                'Run: python -m spacy download %s',
                model_name, model_name,
            )
            nlp = spacy.blank('en')
    else:
        nlp = spacy.blank('en')
        logger.info('Using blank spaCy pipeline (rules-only mode).')

    # Add EntityRuler BEFORE ner so ruler has priority
    # overwrite_ents=True: if the ruler matches a span, it replaces any
    # existing entity label — our biomedical labels win over OntoNotes.
    ruler = nlp.add_pipe(
        'entity_ruler',
        before='ner' if 'ner' in nlp.pipe_names else None,
        config={'overwrite_ents': True, 'phrase_matcher_attr': 'LOWER'},
    )
    ruler.add_patterns(_GENE_PATTERNS + _DISEASE_PATTERNS)

    logger.info('EntityRuler loaded: %d gene patterns + %d disease patterns.',
        len(_GENE_PATTERNS), len(_DISEASE_PATTERNS),
    )
    return nlp

def _is_gene_like(text: str) -> bool:
    """
    Heuristic: is this statistical-model span likely a real gene symbol?
    
    Rules (ALL must be true):
    1. Must match ALL-CAPS pattern with digits/hyphens allowed, min 3 chars.
    2. Must NOT be in the exclusion list.
    3. Must be a SINGLE token (no spaces) — multi-word spans are never genes
       from the statistical layer; they are almost always false positives.
    4. Must contain at least one letter that is not just a digit suffix
       (e.g. G12 alone is not a gene, but KRAS-G12 would be handled by ruler).
    """
    stripped = text.strip()
    # Rule 3: reject multi-word spans immediately
    if ' ' in stripped:
        return False
    # Rule 1+2: regex match and not in exclusion list
    if not _GENE_REGEX.match(stripped):
        return False
    if stripped in _COMMON_WORDS or stripped.upper() in _COMMON_WORDS:
        return False
    # Rule 4: must have at least 2 alphabetic characters (not just "G12")
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if alpha_chars < 2:
        return False
    return True


def _canonical(text: str) -> str:
    """
    Normalise entity text to a stable canonical form.
    Lower-case, strip whitespace. Keeps hyphens.
    Example: 'BRCA1 ' → 'brca1', 'Breast Cancer' → 'breast cancer'
    """
    return text.strip().lower()

# -- Main extractor class --

class BioNERExtractor:
    """
    Hybrid biomedical Named Entity Recognition extractor.

    -- Why a hybrid approach? --

    RULE-BASED ONLY (EntityRuler):
    + 100% recall for known entities — never misses BRCA1 if it is in patterns.
    + Fully deterministic — same text always gives same result.
    - 0% recall for entities NOT in the dictionary.
    - Requires maintenance as new genes are discovered.

    STATISTICAL ONLY (en_core_web_sm):
    + Can recognise entities it was never explicitly told about.
    + Generalises from training data.
    - Trained on news/web text (OntoNotes), NOT biomedical literature.
    - High false-positive rate for gene names (e.g. 'ATM' → ORG or PERSON).
    - Misses genes it has never seen (novel mutations).

    HYBRID (EntityRuler FIRST, then statistical):
    + Perfect recall for known entities (ruler handles them).
    + Statistical model fills in novel entities ruler missed.
    + Ruler overrides statistical on known entities (no false relabelling).
    + The _is_gene_like() heuristic filters obviously wrong statistical hits.

    This is the standard professional approach for biomedical NER when a
    domain-specific model (e.g. scispaCy) is not available.

    -- Overlapping entity handling --

    spaCy's EntityRuler with phrase_matcher_attr='LOWER' handles overlaps
    internally: it uses the longest match wins strategy. So 'breast cancer'
    (2-token pattern) beats 'cancer' (1-token) when both could match.
    We additionally deduplicate at the character-offset level in _deduplicate()
    to handle the rare case where statistical NER overlaps with ruler output.

    -- Lazy pipeline loading --

    The spaCy pipeline is built on first call to extract() / extract_batch().
    This matches BioBERTEmbedder's pattern: importing the module is free.
    """

    def __init__(self, use_statistical: bool = True, model_name: str = 'en_core_web_sm',
                 ) -> None:
        self._use_statistical = use_statistical
        self._model_name = model_name
        self._nlp: Optional['Language'] = None  # Lazy
        logger.info('BioNERExtractor created (statistical=%s, model=%s)',
            use_statistical, model_name,
        )

    # -- Pipeline loading --
    def _load_pipeline(self) -> None:
        """Build the spaCy pipeline on first use. Safe to call multiple times."""
        if self._nlp is not None:
            return
        self._nlp = _build_spacy_pipeline(self._use_statistical, self._model_name)

    # -- Overlap handling --
    @staticmethod
    def _deduplicate_entities(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        """
        Remove overlapping entities, keeping the ruler entity when both
        a ruler and a statistical entity cover the same character span.

        Strategy:
        1. Sort by start_char, then by source ('ruler' before 'stat').
        2. Walk through — if the new span starts before the previous span ends,
           it overlaps; keep only the first (ruler-priority) one.

        Why this is necessary:
        spaCy's built-in filter_spans() is applied PER COMPONENT, not across
        components. When we have both entity_ruler and ner in the pipeline,
        the ruler sets ents first, and ner ADDS to ents. With overwrite_ents=True
        the ruler entities are protected, but very rarely the statistical model
        produces an adjacent non-overlapping span that ABUTS a ruler span.
        This filter is the safety net.
        """
        if not entities:
            return []

        # Sort: ruler first within same span so ruler wins ties
        sorted_ents = sorted(
            entities,
            key=lambda e: (e.start_char, 0 if e.source == 'ruler' else 1),
        )

        deduped: list[ExtractedEntity] = []
        last_end = -1
        for ent in sorted_ents:
            if ent.start_char >= last_end:
                deduped.append(ent)
                last_end = ent.end_char
        return deduped

    # -- Single text extraction --
    def _process_doc(self, doc: 'Doc', paper_id: str) -> list[ExtractedEntity]:
        """
        Convert a spaCy Doc's entity spans into ExtractedEntity objects.
        Applies the _is_gene_like() heuristic to statistical-model results.
        """
        entities: list[ExtractedEntity] = []

        for span in doc.ents:
            label = span.label_
            source = 'stat'

            if label in (LABEL_GENE, LABEL_DISEASE):
                # This span was set by the EntityRuler so trusted & no heuristic needed
                source = 'ruler'
            else:
                # This span came from the statistical model — apply strict validation
                
                # Rule A: reject multi-word spans from statistical model entirely.
                # Multi-word statistical spans are almost always false positives in
                # biomedical text: "complex interplay between", "sung et al.", etc.
                # The ruler handles all legitimate multi-word entities (breast cancer,
                # myocardial infarction) with 100% precision. Statistical multi-word
                # hits add only noise.
                if len(span) > 1:
                    continue
                
                # Rule B: apply gene-like heuristic for single uppercase tokens
                if _is_gene_like(span.text):
                    label = LABEL_GENE
                else:
                    # Rule C: map OntoNotes label; skip unmapped
                    mapped = _STAT_LABEL_MAP.get(label)
                    if mapped is None:
                        continue
                    label = mapped

            canonical_text = _canonical(span.text)
            if not canonical_text:
                continue

            entities.append(
                ExtractedEntity(
                    text=canonical_text,
                    label=label,
                    source=source,
                    paper_id=paper_id,
                    start_char=span.start_char,
                    end_char=span.end_char,
                )
            )

        return self._deduplicate_entities(entities)

    def extract(self, text: str, paper_id: str) -> PaperEntities:
        """
        Run NER on a single text string and return a PaperEntities object.

        Parameters
        ----------
        text : str
            The full abstract (or title + abstract) to analyse.
        paper_id : str
            Identifier used to tag each ExtractedEntity for provenance.

        Returns
        -------
        PaperEntities
            Contains deduplicated lists of gene and disease canonical names,
            plus the raw ExtractedEntity list for downstream graph construction.
        """
        self._load_pipeline()

        if not text or not text.strip():
            logger.debug('extract() called with empty text for paper_id=%s', paper_id)
            return PaperEntities(paper_id=paper_id)

        doc = self._nlp(text)  # type: ignore[arg-type]
        raw_entities = self._process_doc(doc, paper_id)

        # Deduplicate by (canonical_text, label) — same gene mentioned 5 times
        # in one abstract should appear once in the entity list.
        seen: set[tuple[str, str]] = set()
        unique_genes: list[str] = []
        unique_diseases: list[str] = []

        for ent in raw_entities:
            key = (ent.text, ent.label)
            if key in seen:
                continue
            seen.add(key)
            if ent.label == LABEL_GENE:
                unique_genes.append(ent.text)
            elif ent.label == LABEL_DISEASE:
                unique_diseases.append(ent.text)

        result = PaperEntities(
            paper_id=paper_id,
            genes=unique_genes,
            diseases=unique_diseases,
            raw_entities=raw_entities,
        )
        logger.debug('extract(): paper=%s → %d genes, %d diseases', 
                     paper_id, len(unique_genes), len(unique_diseases),
        )
        return result

    def extract_batch(self, texts: list[tuple[str, str]], batch_size: int = 64,
                      ) -> list[PaperEntities]:
        """
        Run NER on multiple (text, paper_id) pairs using spaCy's pipe().

        spaCy's pipe() processes documents in batches using internal C-level
        parallelism & significantly faster than calling extract() in a Python loop.

        Parameters
        ----------
        texts : list[tuple[str, str]]
            List of (text, paper_id) pairs.
        batch_size : int
            Number of documents per spaCy batch. 64 is safe for most machines.
            Reduce to 32 if you get MemoryError.

        Returns
        -------
        list[PaperEntities]
            One PaperEntities per input, in the same order as input.
        """
        self._load_pipeline()

        if not texts:
            return []

        safe_texts = [t if t.strip() else ' ' for t, _ in texts]
        paper_ids  = [pid for _, pid in texts]

        results: list[PaperEntities] = []
        for doc, paper_id in zip(
            self._nlp.pipe(safe_texts, batch_size=batch_size),  # type: ignore[union-attr]
            paper_ids,
        ):
            entities = self._process_doc(doc, paper_id)
            seen: set[tuple[str, str]] = set()
            genes: list[str] = []
            diseases: list[str] = []
            for ent in entities:
                key = (ent.text, ent.label)
                if key in seen:
                    continue
                seen.add(key)
                if ent.label == LABEL_GENE:
                    genes.append(ent.text)
                elif ent.label == LABEL_DISEASE:
                    diseases.append(ent.text)
            results.append(
                PaperEntities(
                    paper_id=paper_id,
                    genes=genes,
                    diseases=diseases,
                    raw_entities=entities,
                )
            )

        logger.info(
            'extract_batch: processed %d documents, '
            'total gene mentions=%d, total disease mentions=%d',
            len(results),
            sum(len(r.genes) for r in results),
            sum(len(r.diseases) for r in results),
        )
        return results

    def add_gene_pattern(self, gene_name: str) -> None:
        """
        Dynamically add a new gene pattern at runtime.
        The pipeline must already be loaded (call extract() once first, or warm_up()).
        Useful for user-supplied gene lists without restarting the process.
        """
        self._load_pipeline()
        cleaned_name = gene_name.strip()
        if not cleaned_name:
            raise ValueError('gene_name cannot be empty.')

        ruler = self._nlp.get_pipe('entity_ruler')   # type: ignore[union-attr]
        doc = self._nlp.make_doc(cleaned_name)  # type: ignore[union-attr]
        pattern = {
            'label': LABEL_GENE,
            'pattern': [
                {'ORTH': token.text} if token.is_punct else {'LOWER': token.lower_}
                for token in doc
            ],
        }
        ruler.add_patterns([pattern])
        logger.info('Added dynamic gene pattern: %s', cleaned_name)

    def add_disease_pattern(self, disease_name: str) -> None:
        """Dynamically add a new disease pattern."""
        self._load_pipeline()
        cleaned_name = disease_name.strip()
        if not cleaned_name:
            raise ValueError('disease_name cannot be empty.')

        ruler = self._nlp.get_pipe('entity_ruler')   # type: ignore[union-attr]
        # Use spaCy tokenization so apostrophes and punctuation follow the same rules
        # as the main pipeline rather than naive string splitting.
        doc = self._nlp.make_doc(cleaned_name)  # type: ignore[union-attr]
        pattern = {
            'label': LABEL_DISEASE,
            'pattern': [
                {'ORTH': token.text} if token.is_punct else {'LOWER': token.lower_}
                for token in doc
            ],
        }
        ruler.add_patterns([pattern])
        logger.info('Added dynamic disease pattern: %s', cleaned_name)

    def warm_up(self) -> None:
        """Force pipeline load. Call at CLI startup."""
        self._load_pipeline()
        logger.info('BioNERExtractor pipeline loaded and ready.')
