from __future__ import annotations

import unittest

from src.models import DataSource, Paper
from src.ranker_tfidf import TFIDFRanker


class TestTFIDFRanker(unittest.TestCase):
    def test_rank_returns_zero_scores_when_vocabulary_is_empty(self) -> None:
        papers = [
            Paper(
                paper_id='p1',
                title='the and of',
                abstract='',
                authors=['Doe J'],
                source=DataSource.PUBMED,
            )
        ]

        ranker = TFIDFRanker()
        results = ranker.rank(papers, query='the and of')

        self.assertEqual(1, len(results))
        self.assertEqual(0.0, results[0].score)
        self.assertEqual(1, results[0].rank)


if __name__ == '__main__':
    unittest.main()
