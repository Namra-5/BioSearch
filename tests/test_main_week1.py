from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

import main_week1


class TestMainWeek1CacheBehavior(unittest.TestCase):
    def _fake_cached_items(self, n: int) -> list[object]:
        return [object() for _ in range(n)]

    @patch('main_week1.setup_logging')
    @patch('main_week1.TFIDFRanker')
    @patch('main_week1.PubMedFetcher')
    @patch('main_week1.PaperStorage')
    @patch('main_week1.NCBI_EMAIL', 'test@example.com')
    @patch('main_week1.NCBI_API_KEY', 'key')
    def test_refetches_when_cache_has_too_few_items(
        self,
        mock_storage_cls: MagicMock,
        mock_pubmed_cls: MagicMock,
        mock_ranker_cls: MagicMock,
        _mock_setup_logging: MagicMock,
    ) -> None:
        mock_storage = mock_storage_cls.return_value
        mock_storage.was_recently_fetched.return_value = True
        mock_storage.search_cached.return_value = self._fake_cached_items(2)

        fetched = self._fake_cached_items(10)
        mock_pubmed = mock_pubmed_cls.return_value
        mock_pubmed.fetch.return_value = fetched

        mock_ranker = mock_ranker_cls.return_value
        mock_ranker.rank.return_value = []
        mock_ranker.get_top_terms.return_value = []

        argv = ['main_week1.py', '--query', 'BRCA1 breast cancer', '--max', '10', '--source', 'pubmed']
        with patch.object(sys, 'argv', argv):
            main_week1.main()

        mock_pubmed.fetch.assert_called_once_with(query='BRCA1 breast cancer', max_results=10)
        mock_storage.cache_query_results.assert_called_once_with(
            query='BRCA1 breast cancer',
            source='pubmed',
            papers=fetched,
        )

    @patch('main_week1.setup_logging')
    @patch('main_week1.TFIDFRanker')
    @patch('main_week1.PubMedFetcher')
    @patch('main_week1.PaperStorage')
    @patch('main_week1.NCBI_EMAIL', 'test@example.com')
    @patch('main_week1.NCBI_API_KEY', 'key')
    def test_uses_cache_when_items_are_sufficient(
        self,
        mock_storage_cls: MagicMock,
        mock_pubmed_cls: MagicMock,
        mock_ranker_cls: MagicMock,
        _mock_setup_logging: MagicMock,
    ) -> None:
        mock_storage = mock_storage_cls.return_value
        mock_storage.was_recently_fetched.return_value = True
        mock_storage.search_cached.return_value = self._fake_cached_items(12)

        mock_ranker = mock_ranker_cls.return_value
        mock_ranker.rank.return_value = []
        mock_ranker.get_top_terms.return_value = []

        argv = ['main_week1.py', '--query', 'BRCA1 breast cancer', '--max', '10', '--source', 'pubmed']
        with patch.object(sys, 'argv', argv):
            main_week1.main()

        mock_pubmed_cls.return_value.fetch.assert_not_called()
        mock_storage.cache_query_results.assert_not_called()

    @patch('main_week1.setup_logging')
    @patch('main_week1.PaperStorage')
    def test_stats_does_not_require_query(
        self,
        mock_storage_cls: MagicMock,
        _mock_setup_logging: MagicMock,
    ) -> None:
        mock_storage = mock_storage_cls.return_value
        mock_storage.stats.return_value = {
            'total_papers': 1,
            'by_source': {'pubmed': 1},
            'recent_queries': [],
        }

        argv = ['main_week1.py', '--stats']
        with patch.object(sys, 'argv', argv):
            main_week1.main()

        mock_storage.stats.assert_called_once()


if __name__ == '__main__':
    unittest.main()
