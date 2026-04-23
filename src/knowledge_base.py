# src/knowledge_base.py
# KnowledgeBase: integrates Paper objects, NER extraction, and the graph.
# This is the Week 3 'glue' layer — it wires ner_extractor + knowledge_graph
# together and provides a clean single interface for main_week3.py.
#
# It also persists extracted entities alongside papers so we never re-run
# NER on papers we have already processed.

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

from src.models import Paper
from src.ner_extractor import BioNERExtractor, PaperEntities
from src.knowledge_graph import BioKnowledgeGraph

logger = logging.getLogger(__name__)

_DEFAULT_KB_PATH  = Path('data/knowledge_base.db')
_DEFAULT_GRAPH_PATH = Path('data/knowledge_graph.json')

_CREATE_ENTITIES_TABLE = """
CREATE TABLE IF NOT EXISTS paper_entities (
    paper_id        TEXT PRIMARY KEY,
    genes_json      TEXT NOT NULL DEFAULT '[]',
    diseases_json   TEXT NOT NULL DEFAULT '[]',
    extracted_at    TEXT NOT NULL
);
"""
_CREATE_ENTITIES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_entities_paper ON paper_entities(paper_id);
"""


class KnowledgeBase:
    """
    Integrates Paper retrieval, NER extraction, and graph construction.

    Responsibilities:
    1. Cache NER results in SQLite so papers are never re-processed.
    2. Expose process_papers(list[Paper]) → BioKnowledgeGraph.
    3. Provide query methods that combine entity data with paper metadata.

    Design decision: keep this class thin.
    It delegates to BioNERExtractor for extraction
    and to BioKnowledgeGraph for graph logic.
    """

    def __init__(
        self,
        db_path: Path = _DEFAULT_KB_PATH,
        use_statistical_ner: bool = True,
        spacy_model: str = 'en_core_web_sm',
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._extractor = BioNERExtractor(
            use_statistical=use_statistical_ner,
            model_name=spacy_model,
        )
        self._init_db()
        logger.info('KnowledgeBase initialised at %s', self.db_path)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
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
            conn.executescript(_CREATE_ENTITIES_TABLE + _CREATE_ENTITIES_INDEX)

    # ── NER caching ───────────────────────────────────────────────────────────

    def _is_processed(self, paper_id: str) -> bool:
        with self._conn() as conn:
            return conn.execute(
                'SELECT 1 FROM paper_entities WHERE paper_id = ? LIMIT 1',
                (paper_id,),
            ).fetchone() is not None

    def _save_entities(self, pe: PaperEntities) -> None:
        with self._conn() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO paper_entities '
                '(paper_id, genes_json, diseases_json, extracted_at) VALUES (?, ?, ?, ?)',
                (
                    pe.paper_id,
                    json.dumps(pe.genes),
                    json.dumps(pe.diseases),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def _load_entities(self, paper_id: str) -> Optional[PaperEntities]:
        with self._conn() as conn:
            row = conn.execute(
                'SELECT * FROM paper_entities WHERE paper_id = ?', (paper_id,)
            ).fetchone()
        if row is None:
            return None
        return PaperEntities(
            paper_id=row['paper_id'],
            genes=json.loads(row['genes_json']),
            diseases=json.loads(row['diseases_json']),
        )

    def _load_entities_bulk(self, paper_ids: list[str]) -> dict[str, PaperEntities]:
        """Load cached entities for many paper IDs in batched SQL queries."""
        if not paper_ids:
            return {}

        result: dict[str, PaperEntities] = {}
        chunk_size = 500  # Keep comfortably below SQLite parameter limit.

        with self._conn() as conn:
            for i in range(0, len(paper_ids), chunk_size):
                chunk = paper_ids[i:i + chunk_size]
                placeholders = ', '.join('?' for _ in chunk)
                query = (
                    'SELECT paper_id, genes_json, diseases_json '
                    f'FROM paper_entities WHERE paper_id IN ({placeholders})'
                )
                rows = conn.execute(query, chunk).fetchall()
                for row in rows:
                    result[row['paper_id']] = PaperEntities(
                        paper_id=row['paper_id'],
                        genes=json.loads(row['genes_json']),
                        diseases=json.loads(row['diseases_json']),
                    )

        return result

    def _save_entities_batch(self, entities: list[PaperEntities]) -> None:
        """Persist extracted entities in a single transaction."""
        if not entities:
            return

        rows = [
            (
                pe.paper_id,
                json.dumps(pe.genes),
                json.dumps(pe.diseases),
                datetime.now(timezone.utc).isoformat(),
            )
            for pe in entities
        ]

        with self._conn() as conn:
            conn.executemany(
                'INSERT OR REPLACE INTO paper_entities '
                '(paper_id, genes_json, diseases_json, extracted_at) VALUES (?, ?, ?, ?)',
                rows,
            )

    # ── Core processing pipeline ──────────────────────────────────────────────

    def process_papers(self, papers: list[Paper]) -> BioKnowledgeGraph:
        """
        Run NER on all papers, cache results, build and return a knowledge graph.

        Papers already processed (in SQLite) are loaded from cache — NER is not
        re-run. This means re-processing the same corpus is near-instant.

        Parameters
        ----------
        papers : list[Paper]
            Papers to process. May be from any source (PubMed, bioRxiv).

        Returns
        -------
        BioKnowledgeGraph
            Graph populated with all co-occurrences found in the corpus.
        """
        if not papers:
            logger.warning('process_papers called with empty list.')
            return BioKnowledgeGraph()

        # Split into cached and uncached
        cached_map = self._load_entities_bulk([p.paper_id for p in papers])
        cached_entities: list[PaperEntities] = []
        to_process: list[tuple[str, str]] = []  # (text, paper_id)

        for paper in papers:
            pe = cached_map.get(paper.paper_id)
            if pe is not None:
                cached_entities.append(pe)
            else:
                to_process.append((paper.combined_text, paper.paper_id))

        logger.info(
            'process_papers: %d cached, %d to extract.',
            len(cached_entities), len(to_process),
        )

        # Batch-extract the uncached ones
        new_entities: list[PaperEntities] = []
        if to_process:
            # Warm up only when we actually need model inference.
            self._extractor.warm_up()
            new_entities = self._extractor.extract_batch(to_process, batch_size=64)
            self._save_entities_batch(new_entities)

        # Build the graph from all entities
        graph = BioKnowledgeGraph()
        all_entities = cached_entities + new_entities
        graph.add_batch(all_entities)

        return graph

    def stats(self) -> dict:
        """Return KB statistics."""
        with self._conn() as conn:
            total = conn.execute(
                'SELECT COUNT(*) FROM paper_entities'
            ).fetchone()[0]
        return {'papers_with_entities': total, 'db_path': str(self.db_path)}

