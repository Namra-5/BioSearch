import pytest
from src.embedder import BioBERTEmbedder
from src.models import Paper

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _paper_result_key(paper: Paper) -> str:
    return f"{paper.source}:{paper.paper_id}"


@pytest.fixture(scope="module")
def embedder():
    """
    Single embedder instance reused across tests.
    Prevents repeated model loading (critical for performance).
    """
    return BioBERTEmbedder()


# 1. Empty Abstract Paper (Must not crash)
def test_empty_abstract_handling(embedder):
    """
    Case:
        Paper has empty abstract.
    Expected:
        System should NOT crash and should still return a vector.
    """

    paper = Paper(
        paper_id="EMPTY_ABSTRACT",
        source="pubmed",
        title="Gene expression analysis",
        abstract=""   # edge case
    )

    try:
        result = embedder.embed_papers([paper])
        key = _paper_result_key(paper)

        assert key in result, "Paper missing from results"
        assert result[key].shape == (768,), "Invalid embedding shape"

    except Exception as e:
        pytest.fail(f"System crashed on empty abstract: {e}")


# 2. Long Abstract (800 words - must truncate safely)
def test_long_abstract_truncation(embedder):
    """
    Case:
        Abstract exceeds model limit (~512 tokens equivalent).
    Expected:
        No crash + embedding produced normally.
        (We cannot directly see truncation, but we ensure stability.)
    """

    long_text = "gene therapy " * 800  # synthetic 800-word abstract

    paper = Paper(
        paper_id="LONG_ABSTRACT",
        source="pubmed",
        title="Large scale gene therapy study",
        abstract=long_text
    )

    result = embedder.embed_papers([paper])
    key = _paper_result_key(paper)

    assert key in result, "Long abstract paper missing"
    assert result[key].shape == (768,), "Embedding failed for long text"


# 3. Empty Query (Must raise ValueError)
def test_empty_query_raises_error(embedder):
    """
    Case:
        Empty query string
    Expected:
        ValueError with meaningful message
    """

    with pytest.raises(ValueError) as exc_info:
        embedder.embed_query("")

    assert "Query string cannot be empty" in str(exc_info.value)


# 4. Whitespace Query Edge Case
def test_whitespace_query_raises_error(embedder):
    """
    Case:
        Query is only spaces
    Expected:
        Treated as invalid input
    """

    with pytest.raises(ValueError):
        embedder.embed_query("     ")