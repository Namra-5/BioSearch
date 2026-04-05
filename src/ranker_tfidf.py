# src/ranker_tfidf.py
# TF-IDF baseline ranker using scikit-learn.
# This is the RESEARCH CONTROL — Week 4 will compare its scores against BioBERT.
# Structured as a class to match the interface of the Week 2 semantic ranker.

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models import Paper, SearchResult

logger = logging.getLogger(__name__)

class TFIDFRanker:
    """
    Ranks papers against a query using TF-IDF vectorisation and cosine similarity.

    - How TF-IDF works (the maths behind it)
    Given N documents and a vocabulary V:

    TF(term, doc)  = (count of term in doc) / (total terms in doc)
        → Normalised frequency. A paper of 200 words with 'cancer' 10 times
          has TF = 0.05.

    IDF(term)      = log( N / (1 + df(term)) ) + 1   [scikit-learn smooth IDF]
        → df(term) = number of docs containing the term.
        → If 'cancer' appears in every doc, df = N, IDF ≈ 1 (useless).
        → If 'BRCA1' appears in 2 of 100 docs, IDF = log(100/3)+1 ≈ 4.2.
        → Rare domain-specific terms get high IDF.

    TF-IDF(term, doc) = TF * IDF
        → High when: the term is frequent in THIS doc AND rare globally.

    Cosine Similarity between query vector Q and paper vector P:
        sim(Q, P) = (Q · P) / (|Q| × |P|)
        → Dot product of vectors divided by product of their magnitudes.
        → Range: [0, 1] for TF-IDF (all non-negative values).
        → 1.0 = identical direction = maximally similar.
        → 0.0 = no shared vocabulary = unrelated.

    - Why this is the BASELINE (not the final system) 
    TF-IDF treats every word as independent. It cannot recognise:
    - 'Malignancy' and 'Cancer' as synonymous.
    - 'BRCA1' and 'breast cancer susceptibility gene 1' as the same concept.
    - Context: 'cold' in 'cold virus' vs 'cold temperature'.

    BioBERT (Week 2) solves all three by embedding words in a 768-dimensional
    semantic space learned from 29 billion words of biomedical text.

    - Design decision: fit once, reuse many times 
    The vectoriser is fitted on the CORPUS (all paper texts combined with the
    query). We store it as self._vectorizer so that if you call rank() on the
    same papers with a different query, you can re-use the vocabulary without
    re-fitting from scratch (though a fresh fit is cleaner for new corpora).
    """
    def __init__(self, max_features: int = 10_000, ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1, sublinear_tf: bool = True,) -> None:
        """
        Parameters
        ----------
        max_features : int
            Vocabulary size cap. 10,000 covers most biomedical corpora.
        ngram_range : (1, 2)
            Include unigrams AND bigrams. Bigrams capture 'gene expression',
            'breast cancer', 'CRISPR cas9' as single features — crucial for
            biomedical text where compound terms dominate.
        min_df : int
            Ignore terms appearing in fewer than min_df documents.
            With small corpora (20 papers) keep min_df=1.
        sublinear_tf : bool
            Use log(1 + tf) instead of raw tf. Prevents a paper that says
            'cancer' 50 times from vastly outscoring one that says it 10 times.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self._vectorizer: Optional[TfidfVectorizer] = None
        logger.info('TFIDFRanker initialised (max_features=%d, ngram_range=%s, sublinear_tf=%s)',
                    max_features, ngram_range, sublinear_tf)
        
    def _build_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            stop_words='english',
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b'
            # Custom pattern keeps hyphenated biomedical terms: 'HER-2', 'BRCA-1'
        )
    def rank(self, papers: list[Paper], query:str, top_n: Optional[int] = None,
                ) -> list[SearchResult]:
        """
        Rank papers by TF-IDF cosine similarity to the query.

        Parameters
        ----------
        papers : list[Paper]
            The corpus to rank.
        query : str
            The natural language query.
        top_n : int | None
            Return only top N results. None returns all.

        Returns
        -------
        list[SearchResult]
            Sorted descending by score. Each item has rank, score, paper, method.
        """
        if not papers:
            logger.warning('rank() called with empty paper list.')
            return []

        if not query.strip(): 
            logger.warning('rank() called with empty query.')
            return []

        # Step 1: Build corpus texts 
        # Query appended as the last document so it shares the same vocabulary.
        paper_texts: list[str] = [p.combined_text for p in papers]
        all_texts: list[str] = paper_texts + [query]
        logger.info('Fitting TF-IDF on %d texts (corpus=%d, +1 query).', 
                    len(all_texts), len(papers))
        
        # Step 2: Fit and transform
        self._vectorizer = self._build_vectorizer()
        try:
            tfidf_matrix = self._vectorizer.fit_transform(all_texts)
        except ValueError as exc:
            # Raised when the vocabulary is empty (e.g., all words are stop words)
            logger.warning('TF-IDF vectorisation produced empty vocabulary: %s', exc)
            fallback: list[SearchResult] = []
            for idx, paper in enumerate(papers, start=1):
                fallback.append(
                    SearchResult(
                        paper=paper,
                        score=0.0,
                        rank=idx,
                        method='tfidf',
                    )
                )
            if top_n is not None:
                return fallback[:top_n]
            return fallback
        
        # Step 3: Seperate query and paper vectors
        # tfidf matrix shape: (len(papers)+1, vocabulary_size)
        # Query is the last row: index -1
        query_vector = tfidf_matrix[-1]    # shape: (1, vocab)
        paper_vectors = tfidf_matrix[:-1]  # shape: (n_papers, vocab)

        # Step 4: Compute cosine similarities
        # cosine_similarity returns shape (1, n_papers) - we unpack with [0] to get a 1D array of scores.
        similarities: np.ndarray = cosine_similarity(query_vector, paper_vectors)[0]

        # Step 5: Build SearchResult objects
        results: list[SearchResult] = []
        for i, (paper,score) in enumerate(zip(papers, similarities)):
            # Clamp to [0,1] to satisfy pydantic validation
            clamped = float(np.clip(score, 0.0, 1.0))
            results.append(SearchResult(
                paper=paper, 
                score=clamped,
                rank=i+1,    # temporary rank - will be reassigned after sort
                method='tfidf'))
            
        # Step 6: Sort and re-rank
        results.sort(key=lambda r: r.score, reverse=True)
        for new_rank,result in enumerate(results, start=1):
            result.rank = new_rank    
        
        # Step 7: Apply top_n cap
        if top_n is not None:
            return results[:top_n]
        
        logger.info('TF-IDF ranking complete. Top score: %.4f | Bottom score: %.4f',
        results[0].score if results else 0.0, results[-1].score if results else 0.0)

        return results
    
    def get_top_terms(self, query: str, n: int = 10) -> list[str]:
        """
        Return the top N terms from the fitted vocabulary closest to the query.
        Useful for debugging: shows which terms are driving the ranking.
        """

        if self._vectorizer is None: 
            logger.warning('Vectorizer not fitted yet. Call rank() first')
            return []
        try:
            q_vec = self._vectorizer.transform([query])
            feature_names = self._vectorizer.get_feature_names_out()
            top_indices = np.argsort(q_vec.toarray()[0])[::-1][:n]
            return [feature_names[i] for i in top_indices if q_vec.toarray()[0][i] > 0]
        except Exception as exc:
            logger.warning('get_top_terms failed: %s', exc)
            return []




                
 









    

    




