# src/embedder.py
# BioBERT semantic embedder for BioSearch AI.
#
# Responsibilities:
#   1. Load the BioBERT sentence-transformer model exactly once (lazy singleton).
#   2. Truncate long abstracts to the model's 512-token window safely.
#   3. Embed any list of texts into 768-dimensional L2-normalised dense vectors.
#   4. Cache every embedding in a dedicated SQLite table to avoid repeated
#      embedding work across sessions.
#
# Design principle: this file knows NOTHING about ranking or storage.py.
# It is a pure "text-in / vector-out" service. The ranker consumes it.

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

import numpy as np

# sentence-transformers is imported lazily inside BioBERTEmbedder.__init__
# so that importing this module never crashes when the library is absent.
# The user gets a clean ImportError with install instructions.

from src.models import Paper

logger = logging.getLogger(__name__)


def _paper_identity_key(paper: Paper) -> str:
    """Build a stable per-paper key across sources to avoid ID collisions."""
    source_value = paper.source if isinstance(paper.source, str) else paper.source.value
    return f'{source_value}:{paper.paper_id}'

# ── Constants ──────────────────────────────────────────────────────────────────

# Official HuggingFace identifier for the BioBERT STS model.
# This checkpoint was fine-tuned on medical NLI and STS tasks.
_MODEL_ID = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'

# BioBERT (and all BERT-family models) have a hard limit of 512 WordPiece tokens.
# Long structured abstracts can exceed 512 tokens and will be truncated by the
# tokenizer if they are not shortened first. The word cap is a coarse bound;
# sentence-transformers applies the model's token limit afterward.
_MAX_WORDS = 400

# Dimension of the BioBERT output embedding. Fixed by the architecture.
_EMBEDDING_DIM = 768

# Default path for the embedding vector cache database.
_DEFAULT_CACHE_PATH = Path('data/embedding_cache.db')

# DDL for the embedding cache table.
# PRIMARY KEY is the content_hash (16-char SHA-256 prefix from Paper.content_hash).
# Storing numpy arrays as BLOBs is the most compact and fastest option for SQLite.
_CREATE_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS embeddings (
    content_hash    TEXT PRIMARY KEY,
    paper_id        TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    vector_blob     BLOB NOT NULL,
    embedded_at     TEXT NOT NULL
);
"""
_CREATE_EMBEDDING_INDEX = """
CREATE INDEX IF NOT EXISTS idx_embed_paper ON embeddings(paper_id);
"""


# ── Cache helper ───────────────────────────────────────────────────────────────

class _EmbeddingCache:
    """
    Thin SQLite wrapper that stores and retrieves numpy embedding vectors.

    Kept separate from PaperStorage (storage.py) intentionally:
    - Embedding vectors can be very large (768 × 4 bytes = 3 KB each).
    - Mixing BLOBs into the papers table would make all paper queries slower.
    - A separate DB file can be deleted without losing paper metadata.
    - This also makes it easy to invalidate the cache if you switch models.

    BLOB storage keeps vectors compact and avoids JSON or pickle overhead.
    numpy arrays serialised with np.tobytes() / np.frombuffer() are:
    - Faster to read/write than JSON or pickle.
    - Compact: 768 float32 values = 3,072 bytes per vector.
    - Reconstructed as float32 arrays without serialization-format overhead.
    """

    def __init__(self, db_path: Path = _DEFAULT_CACHE_PATH, model_id:str = _MODEL_ID) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self._init_db()
        logger.info('EmbeddingCache initialised at %s (model=%s)', self.db_path, model_id)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        try:
            yield conn 
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.executescript(_CREATE_EMBEDDINGS_TABLE + _CREATE_EMBEDDING_INDEX)
            logger.debug('EmbeddingCache schema ready.')

    # public API

    def get(self, content_hash:str) -> Optional[np.ndarray]:
        """
        Return the cached vector for content_hash, or None if not found.
        Only returns a hit if the stored model_id matches the current one —
        changing models automatically invalidates old embeddings.
        """
        with self._conn() as conn:
            row = conn.execute("SELECT vector_blob FROM embeddings WHERE content_hash = ? AND model_id = ?", 
                               (content_hash, self.model_id)).fetchone()
            if row is None:
                return None
            # Reconstruct numpy array from raw bytes, dtype=float32 matches what
            # sentence-transformers uses internally for BERT embeddings
            return np.frombuffer(row['vector_blob'], dtype=np.float32).copy() 
        
    def put(self, content_hash: str, paper_id: str, vector: np.ndarray) -> None:
        """Store a single embedding vector. Silently replaces any existing entry"""
        blob = vector.astype(np.float32).tobytes()
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute("""INSERT OR REPLACE INTO embeddings
                         (content_hash, paper_id, model_id, vector_blob, embedded_at) VALUES (?, ?, ?, ?, ?)""",
                         (content_hash, paper_id, self.model_id, blob, now))
            
    def put_batch(self, items: list[tuple[str, str, np.ndarray]]) -> int:
        """
        Bulk insert (content_hash, paper_id, vector) tuples.
        Returns number of rows written.
        Used for the batch embedding path to avoid per-row transaction overhead.
        """
        now = datetime.now(timezone.utc).isoformat()
        rows = [(ch, pid, self.model_id, vec.astype(np.float32).tobytes(),now)
                for ch, pid, vec in items]
        with self._conn() as conn:
            conn.executemany("""INSERT OR REPLACE INTO embeddings
                             (content_hash, paper_id, model_id, vector_blob, embedded_at)
                             Values (?, ?, ?, ?, ?)""", rows)
            return len(rows)
        
    def count(self) -> int:
        """Total number of cached vectors (for diagnostics)."""
        with self._conn() as conn:
            return conn.execute("SELECT Count(*) FROM embeddings").fetchone()[0]
        
    def clear(self) -> int:
        """Delete all cached vectors. Returns number of deleted rows."""
        with self._conn() as conn:
            n = conn.execute("DELETE FROM embeddings").rowcount
            logger.warning('EmbeddingCache cleared (%d rows deleted).', n)
            return n
    
    def stats(self) -> dict:
        """Return diagnostics dict for CLI / logging."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            models = conn.execute(
                "SELECT model_id, COUNT(*) FROM embeddings GROUP BY model_id").fetchall()
            return{
                'total_embeddings': total,
                'by_model': {r[0]: r[1] for r in models},
                'db_path': str(self.db_path) 
            }
    
            
# Text preprocessing

def _truncate_to_word_limit(text: str, max_words: int = _MAX_WORDS) -> str:
    """
    Truncate text to at most max_words words.

    The tokenizer runs inside sentence-transformers and is opaque to this
    module. The word limit is a coarse bound; the model tokenizer applies the
    final 512-token limit.

    We preserve the BEGINNING of the text because PubMed abstracts are
    structured: the first sentences contain Background and Objective, which
    carry the highest semantic signal for relevance scoring.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    logger.debug('Truncated text from %d to %d words.', len(words), max_words)
    return truncated

def _prepare_text_for_embedding(paper: Paper) -> str:
    """
    Build the canonical text representation of a paper for embedding.

    We prepend the title twice: once as the natural title, once as a
    "sentence" prefix. This is a well-established trick in biomedical IR:
    titles are the highest-signal text and increasing their representation
    in the input shifts the embedding toward the paper's core topic.

    Then we append the abstract (truncated). This gives a combined input
    that is rich enough for semantic understanding but stays within
    BioBERT's 512-token limit.
    """
    title = paper.title.strip()
    abstract = paper.abstract.strip()

    if abstract:
        # Pattern: "TITLE. TITLE. ABSTRACT_WORDS..."
        # Double title weighting is subtle but measurable in retrieval benchmarks.
        combined = f'{title}. {title}. {abstract}'
    else:
        combined = title

    return _truncate_to_word_limit(combined, _MAX_WORDS)

# Main embedder class

class BioBERTEmbedder:
    """
    Wraps the pritamdeka/BioBERT STS model from sentence-transformers.

    -- Model characteristics --
    General BERT (bert-base-uncased) was pre-trained on Wikipedia + BookCorpus.
    It has never seen medical literature in any meaningful volume, so it
    represents biomedical synonyms as unrelated vectors:

    BioBERT was pre-trained on 29 billion words of PubMed abstracts + PMC
    full-text articles before fine-tuning on NLI and STS tasks. This gives it:

    1. Biomedical synonym awareness: BRCA1 ≈ "breast cancer susceptibility gene 1"
    2. Abbreviation expansion: NF-κB ≈ "nuclear factor kappa-light-chain-enhancer"
    3. Contextual disambiguation: "cold" in "common cold" vs "cold shock protein"

    -- Dense vs Sparse vectors --
    TF-IDF produces SPARSE vectors: most dimensions are 0 (vocabulary slots
    for words not in this document). A corpus of 10,000-word vocab has
    10,000-dimensional vectors where 9,800+ entries are zero.

    BioBERT produces dense vectors whose dimensions encode learned semantic
    features.
    Every dimension encodes a learned semantic feature. Two papers that share
    no vocabulary but describe the same concept will still have high cosine
    similarity because they activate the same learned features.

    -- The 512-token limit --
    BERT's self-attention mechanism computes pairwise interactions between
    all tokens. Memory grows as O(n²) with sequence length. At pre-training
    time, sequences were capped at 512 tokens to fit in GPU memory. This limit
    is baked into the positional embeddings — you cannot simply extend it.
    We handle it with _truncate_to_word_limit() above.

    -- Device management (CUDA vs CPU) --
    sentence-transformers auto-detects CUDA. On your laptop (CPU), encoding
    The device parameter lets you select CPU or CUDA for reproducibility:
        embedder = BioBERTEmbedder(device='cpu')

    -- Lazy loading --
    The model is loaded in __init__ only when first accessed. This means
    importing this module costs zero time - the 15-second download/load only
    happens when you actually create a BioBERTEmbedder instance.
    """

    def __init__(self, model_id: str = _MODEL_ID, device: Optional[str] = None,
                 cache_path: Path = _DEFAULT_CACHE_PATH, batch_size: int = 16) -> None:
        """
        Parameters
        ----------
        model_id : str
            HuggingFace model identifier. Defaults to the BioBERT STS checkpoint.
        device : str | None
            'cuda', 'cpu', or None (auto-detect). On first run, sentence-transformers
            downloads the model (~440 MB) to ~/.cache/huggingface/hub/.
        cache_path : Path
            SQLite file for embedding vectors. Separate from the paper cache.
        batch_size : int
            How many texts to embed in one forward pass. 16 is safe for CPU.
            Increase to 32–64 on GPU. Reduce to 8 if you get OOM errors.
        """
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._cache = _EmbeddingCache(db_path=cache_path, model_id=model_id)
        self._model = None # Loaded lazily in _load_model()

    # Model loading
    def _load_model(self) -> None:
        """
        Load the sentence-transformer model into memory.
        Called automatically on first use. Safe to call multiple times.
        """
        if self._model is not None:
            return # already loaded
        
        try:
            from sentence_transformers import SentenceTransformer # type: ignore
        except ImportError as exc:
            raise ImportError(
                'sentence-transformers is not installed.'
                'Run: pip install sentence-transformers torch\n'
                'For CPU-only (no CUDA): pip install sentence-transformers'
                'torch --index-url https://download.pytorch.org/whl/cpu'
            ) from exc
        
        logger.info('Loading BioBERT model: %s (device=%s) ...', 
                    self.model_id, self.device or 'auto')
        t0 = time.perf_counter()

        try: 
            self._model = SentenceTransformer(self.model_id, device=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{self.model_id}'."
                'Check your internet connection on first run (model download ~440 MB). '
                f"Original error: {exc}"
            ) from exc
        
        elapsed = time.perf_counter() - t0
        logger.info('BioBERT model loaded in %.1fs.', elapsed)

        # Verify the model produces the expected embedding dimension.
        probe = self._model.encode(['test'], convert_to_numpy=True)
        actual_dim = probe.shape[1]
        if actual_dim != _EMBEDDING_DIM:
            logger.warning('Model dimension mismatch: expected %d, got %d.'
                           "Proceeding, but ranker_semantic.py may need updating.",
                           _EMBEDDING_DIM, actual_dim)
    
    # Core embedding logic
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of raw text strings into a 2D numpy array of shape
        (len(texts), 768). Vectors are L2-normalised (unit length).

        L2-normalisation means cosine_similarity(A, B) == np.dot(A, B) —
        the dot product of normalised vectors IS the cosine similarity.
        This simplifies the ranker and is standard practice for retrieval.

        Parameters
        ----------
        texts : list[str]
            Pre-processed strings (already truncated). Empty strings are
            replaced with a single space to prevent tokenizer crashes.

        Returns
        -------
        np.ndarray
            Shape (N, 768), dtype float32.
        """
        self._load_model()

        # Guard against empty strings - the tokenizer raises if input is ''
        safe_texts = [t if t.strip() else ' ' for t in texts]
        logger.info('Embedding %d texts in batches of %d ...', len(safe_texts), self.batch_size)
        t0 = time.perf_counter()

        # normalize_embeddings=True applies L2 normalisation after encoding.
        # show_progress_bar=False keeps logs clean in production.
        vectors : np.ndarray = self._model.encode(
            safe_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        elapsed = time.perf_counter() - t0  
        logger.info('Encoded %d texts in %.2fs (%.0f ms/text).', 
                    len(safe_texts), elapsed, 
                    (elapsed/len(safe_texts)) * 1000 if safe_texts else 0)
        return vectors # shape (N, 768),  float32, L2-normalised

    # Paper embedding with cache 

    def embed_papers(self, papers: list[Paper]) -> dict[str, np.ndarray]:
        """
        Embed a list of Paper objects, using the cache for any already-embedded papers.

        Cache key: paper.content_hash (SHA-256[:16] of title+abstract defined in models.py).
        If the paper's text changes (e.g. abstract corrected), the hash changes and
        we automatically re-embed — no manual cache invalidation needed.

        Returns
        -------
        dict[str, np.ndarray]
            Maps source:paper_id -> 768-dim L2-normalised vector.
        """
        if not papers:
            return {}

        result: dict[str, np.ndarray] = {}
        to_embed: list[tuple[int, Paper, str]] = []  # (original_index, paper, prepared_text)

        # Pass 1: check cache 
        cache_hits = 0
        for paper in papers:
            cached_vec = self._cache.get(paper.content_hash)
            if cached_vec is not None:
                result[_paper_identity_key(paper)] = cached_vec
                cache_hits += 1
            else:
                text = _prepare_text_for_embedding(paper)
                to_embed.append((len(to_embed), paper, text))

        logger.info(
            'embed_papers: %d cache hits, %d need embedding.', cache_hits, len(to_embed))

        if not to_embed:
            return result  # everything was cached
        
        self._load_model()
        # Pass 2: embed the misses 
        texts_to_embed = [text for _, _, text in to_embed]
        new_vectors = self.embed_texts(texts_to_embed)

        # Pass 3: store new vectors in cache and result
        cache_batch: list[tuple[str, str, np.ndarray]] = []
        for (_, paper, _), vector in zip(to_embed, new_vectors):
            result[_paper_identity_key(paper)] = vector
            cache_batch.append((paper.content_hash, paper.paper_id, vector))

        written = self._cache.put_batch(cache_batch)
        logger.info('Stored %d new embedding vectors in cache.', written)

        return result

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string. Returns shape (768,).

        Queries are NOT cached — they are short, fast to embed, and change
        every search. Caching them would add complexity with no real benefit.
        """
        self._load_model()

        if not query.strip():
            raise ValueError('Query string cannot be empty.')

        query_truncated = _truncate_to_word_limit(query, max_words=64)
        
        # encode() returns shape (1, 768) for a single string input
        vector: np.ndarray = self._model.encode(
            [query_truncated],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]  # unpack to shape (768,)

        logger.debug('Query embedded. Vector norm: %.4f', float(np.linalg.norm(vector)))
        return vector

    # Diagnostics

    def cache_stats(self) -> dict:
        """Return embedding cache statistics."""
        return self._cache.stats()

    def warm_up(self) -> None:
        """
        Force model loading now (instead of lazily on first embed call).
        Call this at CLI startup so the user sees the loading message immediately
        rather than during what appears to be the ranking step.
        """
        self._load_model()
        logger.info('BioBERTEmbedder warmed up. Model is in memory.')



