import pytest
import numpy as np

from src.embedder import BioBERTEmbedder, _EmbeddingCache
from src.models import Paper

pytestmark = [pytest.mark.slow, pytest.mark.integration]


# Reusable embedder instance
@pytest.fixture(scope="module")
def embedder():
    """
    Module-scoped fixture:
    - Loads BioBERT only once (very important for speed)
    - Reused across all tests
    """
    return BioBERTEmbedder()

# 1. Semantic Validation Tests

def test_semantic_ranking(embedder):
    """
    Test that synonyms are more similar than unrelated terms.
    """
    synonym_vecs = embedder.embed_texts(["myocardial infarction", "heart attack"])
    unrelated_vecs = embedder.embed_texts(["myocardial infarction", "volleyball"])
    
    synonym_sim = float(np.dot(synonym_vecs[0], synonym_vecs[1]))
    unrelated_sim = float(np.dot(unrelated_vecs[0], unrelated_vecs[1]))
    
    # Assert that synonyms are ranked higher than unrelated terms
    assert synonym_sim > unrelated_sim, "Model failed to rank synonyms higher than unrelated terms"

def test_semantic_related_concepts(embedder):
    """
    Related biomedical concepts should have moderate similarity.
    """
    vecs = embedder.embed_texts([
        "cancer therapy",
        "tumor treatment"
    ])

    sim = float(np.dot(vecs[0], vecs[1]))

    assert sim > 0.55, f"Expected moderate similarity, got {sim:.4f}"


def test_semantic_unrelated(embedder):
    """
    Unrelated concepts should have low similarity.
    """
    vecs = embedder.embed_texts([
        "brain neuron",
        "volleyball game"
    ])

    sim = float(np.dot(vecs[0], vecs[1]))

    assert sim < 0.40, f"Expected low similarity, got {sim:.4f}"


# 2. CACHE Integration Test

def test_cache_behavior(tmp_path):
    """
    - First run inserts embeddings
    - Second run uses cache (no growth)
    """

    cache_db = tmp_path / "test_cache.db"

    embedder = BioBERTEmbedder(cache_path=cache_db)

    papers = [
        Paper(paper_id="P1", source="pubmed", title="Cancer study", abstract="Tumor cells"),
        Paper(paper_id="P2", source="pubmed", title="Lung research", abstract="Respiratory system"),
    ]

    # First run = embeddings created
    embedder.embed_papers(papers)
    stats_1 = embedder.cache_stats()

    # Second run = should hit cache
    embedder.embed_papers(papers)
    stats_2 = embedder.cache_stats()

    assert stats_1["total_embeddings"] == stats_2["total_embeddings"], \
        "Cache size changed — expected cache hit, not recomputation"


# 3. CACHE Unit Test (LOW-LEVEL)

def test_embedding_cache_roundtrip(tmp_path):
    """
    Ensure vector stored == vector retrieved (bit-level correctness)
    """
    db_path = tmp_path / "cache_unit.db"

    cache = _EmbeddingCache(db_path, model_id="test-model")

    np.random.seed(42)
    vec = np.random.rand(768).astype(np.float32)

    chash = "test_hash"

    cache.put(chash, "paper_1", vec)
    retrieved = cache.get(chash)

    assert retrieved is not None, "Retrieved vector is None"
    assert retrieved.shape == (768,), "Shape mismatch"
    assert np.allclose(vec, retrieved), "Vector mismatch after round-trip"


# 4. Robustness Test 

def test_empty_query_raises(embedder):
    """
    Empty query should raise ValueError.
    """
    with pytest.raises(ValueError):
        embedder.embed_query("")


def test_empty_texts_safe(embedder):
    """
    embed_texts should NOT crash on empty strings.
    """
    vecs = embedder.embed_texts(["", ""])

    assert vecs.shape == (2, 768), "Unexpected output shape"


# 5. Shape and Consistency Test
def test_embedding_shape(embedder):
    """
    Ensure correct embedding dimensions.
    """
    vecs = embedder.embed_texts(["test text"])

    assert vecs.shape == (1, 768), "Embedding dimension incorrect"


def test_l2_normalization(embedder):
    """
    Ensure vectors are unit-length (critical for cosine similarity).
    """
    vecs = embedder.embed_texts(["biology"])

    norm = np.linalg.norm(vecs[0])

    assert abs(norm - 1.0) < 1e-5, f"Vector not normalized: norm={norm}"