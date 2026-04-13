import pytest

from src.models import Paper
from src.ranker_semantic import SemanticRanker

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope='module')
def papers() -> list[Paper]:
    return [
        Paper(
            paper_id='1',
            title='BRCA1 and Breast Cancer',
            abstract='Mutations in the BRCA1 gene are strongly associated with breast cancer development.',
            source='biorxiv',
        ),
        Paper(
            paper_id='2',
            title='Photosynthesis in Plants',
            abstract='This study explores light absorption and energy conversion in plant chloroplasts.',
            source='biorxiv',
        ),
        Paper(
            paper_id='3',
            title='Gene Mutation Mechanisms',
            abstract='Different types of gene mutations and their biological consequences are analyzed.',
            source='biorxiv',
        ),
        Paper(
            paper_id='4',
            title='Cancer Immunotherapy Advances',
            abstract='Recent advances in immunotherapy for treating various types of cancer.',
            source='biorxiv',
        ),
        Paper(
            paper_id='5',
            title='Protein Folding Dynamics',
            abstract='Investigation of protein folding and misfolding in cellular environments.',
            source='biorxiv',
        ),
    ]


@pytest.fixture(scope='module')
def ranker() -> SemanticRanker:
    return SemanticRanker()


def test_semantic_ranker_returns_top_n(ranker: SemanticRanker, papers: list[Paper]) -> None:
    results = ranker.rank(papers, 'cancer gene mutation', top_n=3)
    assert len(results) == 3, 'Expected exactly 3 results'


def test_semantic_ranker_scores_are_bounded_and_sorted(
    ranker: SemanticRanker,
    papers: list[Paper],
) -> None:
    results = ranker.rank(papers, 'cancer gene mutation', top_n=3)
    scores = [r.score for r in results]
    assert all(0.0 <= s <= 1.0 for s in scores), 'Scores not in [0,1]'
    assert scores == sorted(scores, reverse=True), 'Scores not sorted descending'