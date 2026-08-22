# src/ranker_semantic.py
# Semantic ranker for BioSearch AI.

# Ranks Paper objects against a query using BioBERT dense embeddings
# and cosine similarity. Designed to be a drop-in replacement for
# TFIDFRanker — both expose the same rank() interface so main.py
# can swap between them with a single flag.

# Key architectural rule: this file consumes BioBERTEmbedder.
# It does NOT call sentence-transformers or torch directly.
# All model/device complexity lives in embedder.py.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.embedder import BioBERTEmbedder, _EMBEDDING_DIM, _DEFAULT_CACHE_PATH
from src.models import Paper, SearchResult

logger = logging.getLogger(__name__)


def _paper_identity_key(paper: Paper) -> str:
    """Build a stable key for cross-source ranking lookups."""
    source_value = paper.source if isinstance(paper.source, str) else paper.source.value
    return f'{source_value}:{paper.paper_id}'


class SemanticRanker:
    """
    Ranks papers against a query using BioBERT dense vector cosine similarity.

    -- Why cosine similarity works differently here than in TF-IDF ---

    In TF-IDF space, vectors are SPARSE (most dimensions = 0) and dimensions
    represent vocabulary slots. Cosine similarity measures vocabulary overlap
    weighted by IDF scores.

    In BioBERT embedding space, vectors are dense and dimensions represent
    learned semantic features rather than individual words.
    Cosine similarity measures how similarly two texts are "understood"
    by the model, not how many words they share.

    Mathematical detail:
        cosine(A, B)  = (A · B) / (||A|| × ||B||)

    Because BioBERTEmbedder returns L2-NORMALISED vectors (||A|| = ||B|| = 1),
    this simplifies to a plain dot product:
        cosine(A, B)  = A · B        (when both are unit vectors)

    np.dot(query_vec, paper_vecs.T) computes this for all papers at once —
    a single matrix multiplication rather than a Python loop.

    -- Normalisation: why it matters --

    Without normalisation, longer texts produce vectors with larger L2 norms.
    A 300-word abstract would score higher than an identical 50-word abstract
    purely because it has more tokens contributing to the embedding sum.
    L2 normalisation removes this length bias — only DIRECTION matters.

    -- Interface contract with TFIDFRanker --

    Both rankers expose:
        rank(papers: list[Paper], query: str, top_n: Optional[int]) -> list[SearchResult]

    main.py can use either with:
        ranker = SemanticRanker()   # or TFIDFRanker()
        results = ranker.rank(papers, query, top_n=10)

    The method='biobert' label on each SearchResult distinguishes the results.
    """

    def __init__(
        self, 
        model_id: str = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb',
        device: Optional[str] = None,
        cache_path: Path = _DEFAULT_CACHE_PATH,
        batch_size: int = 16) -> None:
        """
        Parameters
        ----------
        model_id : str
            HuggingFace model identifier. Passed through to BioBERTEmbedder.
        device : str | None
            'cuda', 'cpu', or None for auto-detection.
        cache_path : Path
            Path to the embedding cache SQLite database.
        batch_size : int
            Number of texts per encoding batch. Reduce to 8 on low-RAM machines.
        """
        self._embedder = BioBERTEmbedder(model_id=model_id, device=device,
                                         cache_path=cache_path, batch_size=batch_size)
        logger.info('SemanticRanker initialised (model_id=%s, device=%s, cache_path=%s)',
                    model_id, device or 'auto', cache_path)
        
    # Core ranking method
    def rank(self, papers: list[Paper], query: str, top_n: Optional[int] = None
            )-> list[SearchResult]:
        """
        Rank papers by semantic similarity to the query using BioBERT.

        Algorithm:
            1. Embed all papers (cache hits are free; misses hit the model).
            2. Embed the query (always re-computed — fast for a short string).
            3. Compute dot product of query vector against all paper vectors
               (equivalent to cosine similarity since vectors are L2-normalised).
            4. Sort descending, assign ranks, return SearchResult list.

        Parameters
        ----------
        papers : list[Paper]
            The corpus to rank. Can include papers from both PubMed and bioRxiv.
        query : str
            The natural language biomedical query.
        top_n : int | None
            Return only the top N results. None returns all papers ranked.

        Returns
        -------
        list[SearchResult]
            Sorted by score descending. method='biobert' on each result.
            Scores are cosine similarities in [0.0, 1.0] (negative cosines
            are clamped to 0.0 — they are near-zero in practice for biomedical
            text and indicate complete semantic unrelation).
        """
        if not papers:
            logger.warning('SemanticRanker.rank() called with empty paper list.')
            return []
        if not query.strip():
            logger.warning('SemanticRanker.rank() called with empty query.')
            return []
        
        # Step 1: Embed all papers
        # Returns dict {source:paper_id -> np.ndarray shape (768,)}
        paper_vectors: dict[str, np.ndarray] = self._embedder.embed_papers(papers)

        # Step 2: Embed the query
        # Return shape (768,), L2-normalised
        try:
            query_vector: np.ndarray = self._embedder.embed_query(query)
        except ValueError as exc:
            logger.error('Query embedding failed: %s', exc)
            return []
        
        # Step 3: Build aligned paper vector matrix
        # We need to maintain paper order so scores map back to the right Paper.
        # Some papers may have failed embedding (empty title + abstract edge case).
        ordered_papers: list[Paper] = []
        vector_rows: list[np.ndarray] = []
        
        for paper in papers:
            vec = paper_vectors.get(_paper_identity_key(paper))
            if vec is None:
                # Paper had no embeddable text so assign score 0 at the end
                logger.warning('No embedding found for paper_id=%s. Assigning score 0.',
                                paper.paper_id)
                ordered_papers.append(paper)
                vector_rows.append(np.zeros(_EMBEDDING_DIM, dtype=np.float32))
            else:
                ordered_papers.append(paper)
                vector_rows.append(vec)
        
        if not vector_rows:
            logger.error('No paper vectors available for ranking.')
            return []
        
        # Stack into matrix: shape (n_papers, 768)
        paper_matrix: np.ndarray = np.vstack(vector_rows)

        # Step 4: Dot product = cosine similarity for unit vectors
        # query_vector shape: (768,)
        # paper_matrix.T shape: (768, n_papers)
        # result shape: (n_papers,)
        raw_scores: np.ndarray = paper_matrix @ query_vector

        # Step 5: Clamp to [0,1]
        # Cosine similarity for normalised BERT vectors is technically in [-1, 1].
        # In practice, biomedical text almost never produces negative cosines —
        # they indicate semantic opposition which doesn't occur in literature.
        # We clamp to 0 to satisfy Pydantic's ge=0.0 constraint on SearchResult.score.
        scores: np.ndarray = np.clip(raw_scores, 0.0, 1.0)

        # Step 6: Build SearchResult objects
        results: list[SearchResult] = []
        for i, (paper,score) in enumerate(zip(ordered_papers, scores)):
            results.append(SearchResult(
                paper=paper,
                score=float(score),  
                rank=i+1,            # Temporary rank; will be re-assigned after sort
                method='biobert',
            ))

        # Step 7: Sort by score descending and re-assign final ranks
        results.sort(key=lambda r: r.score, reverse=True)
        for final_rank, result in enumerate(results, start=1):
            result.rank = final_rank

        # Step 8: Apply top_n cap
        if top_n is not None:
            results = results[:top_n]

        logger.info(
            'SemanticRanker complete. Top score: %.4f | Bottom: %.4f | Papers ranked: %d',
            results[0].score if results else 0.0,
            results[-1].score if results else 0.0,
            len(results))
        
        return results
    
    # Utility
    def warm_up(self) -> None:
        """
        Pre-load the BioBERT model into memory.
        Call once at CLI startup so users see the load message immediately.
        """
        self._embedder.warm_up()

    def cache_stats(self) -> dict:
        """Return embedding cache statistics (delegates to embedder)."""
        return self._embedder.cache_stats()
    
    def compare_texts(self, text_1:str, text_2:str) -> float:
        """
        Return the cosine similarity between two arbitrary text strings.

        Useful for debugging and comparing semantic similarity:
            ranker.compare_texts("malignancy", "cancer")
            ranker.compare_texts("malignancy", "photosynthesis")

        Returns a float in [0.0, 1.0].

        """
        vecs = self._embedder.embed_texts([text_1, text_2])
        # Both vectors are L2-normalised and dot product = cosine similarity
        sim = float(np.dot(vecs[0], vecs[1]))
        return float(np.clip(sim, 0.0, 1.0))