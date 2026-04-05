from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.models import DataSource, Paper
from src.storage import PaperStorage


class TestPaperStorage(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / 'cache.db'
        self.storage = PaperStorage(db_path=self.db_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _make_paper(self, paper_id: str, title: str) -> Paper:
        return Paper(
            paper_id=paper_id,
            title=title,
            abstract='BRCA1 related abstract',
            authors=['Doe J'],
            source=DataSource.PUBMED,
        )

    def test_query_and_source_are_normalized_for_cache_lookup(self) -> None:
        papers = [self._make_paper('1', 'BRCA1 study')]
        self.storage.cache_query_results(
            query='  BRCA1 Breast Cancer  ',
            source=' PubMed ',
            papers=papers,
        )

        self.assertTrue(
            self.storage.was_recently_fetched('brca1 breast cancer', source='PUBMED')
        )
        cached = self.storage.search_cached(' BRCA1 BREAST CANCER ', source='pUbMeD')
        self.assertEqual(1, len(cached))
        self.assertEqual('1', cached[0].paper_id)

    def test_clear_removes_query_metadata(self) -> None:
        papers = [self._make_paper('2', 'Breast cancer risk')]
        self.storage.cache_query_results(
            query='BRCA1 breast cancer',
            source='pubmed',
            papers=papers,
        )

        self.storage.clear(source='pubmed')

        self.assertFalse(
            self.storage.was_recently_fetched('BRCA1 breast cancer', source='pubmed')
        )
        self.assertEqual([], self.storage.search_cached('BRCA1 breast cancer', source='pubmed'))


if __name__ == '__main__':
    unittest.main()
