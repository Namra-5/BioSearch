# src/storage.py
# Local persistence layer using SQLite.
# Prevents redundant API calls by caching Paper objects.
# Uses SQLite instead of plain JSON for query performance and concurrency safety.

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Optional

from src.models import Paper

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path('data/biosearch_cache.db')

# DDL — executed once on first connection
_CREATE_PAPERS_TABLE = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id        TEXT NOT NULL,
    source          TEXT NOT NULL,
    title           TEXT NOT NULL,
    abstract        TEXT DEFAULT '',
    authors_json    TEXT DEFAULT '[]',
    published_date  TEXT,
    doi             TEXT,
    journal         TEXT,
    keywords_json   TEXT DEFAULT '[]',
    url             TEXT,
    fetched_at      TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    PRIMARY KEY (paper_id, source)
);
"""

_CREATE_QUERIES_TABLE = """
CREATE TABLE IF NOT EXISTS query_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT NOT NULL,
    source      TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    ran_at      TEXT NOT NULL
);
"""

_CREATE_QUERY_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS query_results (
    query       TEXT NOT NULL,
    source      TEXT NOT NULL,
    paper_id    TEXT NOT NULL,
    paper_rank  INTEGER NOT NULL,
    PRIMARY KEY (query, source, paper_id)
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
CREATE INDEX IF NOT EXISTS idx_papers_fetched ON papers(fetched_at);
CREATE INDEX IF NOT EXISTS idx_papers_hash ON papers(content_hash);
CREATE INDEX IF NOT EXISTS idx_query_results_lookup ON query_results(query, source, paper_rank);
CREATE INDEX IF NOT EXISTS idx_query_log_lookup ON query_log(query, source, ran_at);
"""


class PaperStorage:
    """
    SQLite-backed cache for Paper objects.

    Why SQLite and not a plain JSON file?
    - JSON files require loading the entire file into memory to check for one paper.
    - SQLite gives O(log n) lookups by primary key, even with 100k records.
    - SQLite is ACID-compliant: if script crashes mid-write, we do not
      end up with a corrupted file.
    - SQLite ships with Python; no extra dependency.
    - The PRIMARY KEY (paper_id, source) prevents duplicates automatically.

    The contextmanager pattern (_get_connection) ensures connections are always
    closed even if an exception is raised inside a with block.
    """

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise_db()
        logger.info('PaperStorage initialised at %s', self.db_path)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager that opens, yields, commits, and closes a connection.
        Using isolation_level=None would give us autocommit but we want
        explicit transaction control for batch inserts.
        """
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row   # rows behave like dicts
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialise_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                _CREATE_PAPERS_TABLE
                + _CREATE_QUERIES_TABLE
                + _CREATE_QUERY_RESULTS_TABLE
                + _CREATE_INDEXES)
            # Keep historical rows index-friendly by normalizing persisted keys once on startup.
            conn.execute('UPDATE query_log SET query = LOWER(TRIM(query)), source = LOWER(TRIM(source))')
            conn.execute('UPDATE query_results SET query = LOWER(TRIM(query)), source = LOWER(TRIM(source))')
            self._migrate_legacy_timestamps(conn)
        logger.debug('Database schema initialised.')

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize query keys before writing/reading to keep lookups index-friendly."""
        return query.strip().lower()

    @staticmethod
    def _normalize_source(source: str) -> str:
        """Normalize source keys for consistent storage and retrieval."""
        return source.strip().lower()

    @staticmethod
    def _normalize_iso_timestamp(value: str) -> str:
        """Normalize timestamp strings to UTC ISO-8601 for consistent comparisons."""
        raw = value.strip()
        if raw.endswith('Z'):
            raw = raw[:-1] + '+00:00'
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()

    def _migrate_legacy_timestamps(self, conn: sqlite3.Connection) -> None:
        """Normalize older timestamp rows so lexical ordering and filtering remain reliable."""
        query_log_updates: list[tuple[str, int]] = []
        for row in conn.execute('SELECT id, ran_at FROM query_log').fetchall():
            ran_at = row['ran_at']
            if not ran_at:
                continue
            try:
                normalized = self._normalize_iso_timestamp(ran_at)
            except ValueError:
                continue
            if normalized != ran_at:
                query_log_updates.append((normalized, row['id']))

        paper_updates: list[tuple[str, Optional[str], str, str]] = []
        for row in conn.execute('SELECT paper_id, source, fetched_at, published_date FROM papers').fetchall():
            fetched_at = row['fetched_at']
            if not fetched_at:
                continue
            try:
                normalized_fetched = self._normalize_iso_timestamp(fetched_at)
            except ValueError:
                normalized_fetched = fetched_at

            published_date = row['published_date']
            normalized_published = published_date
            if published_date:
                try:
                    normalized_published = self._normalize_iso_timestamp(published_date)
                except ValueError:
                    normalized_published = published_date

            if normalized_fetched != fetched_at or normalized_published != published_date:
                paper_updates.append(
                    (normalized_fetched, normalized_published, row['paper_id'], row['source'])
                )

        if query_log_updates:
            conn.executemany('UPDATE query_log SET ran_at = ? WHERE id = ?', query_log_updates)
        if paper_updates:
            conn.executemany(
                'UPDATE papers SET fetched_at = ?, published_date = ? WHERE paper_id = ? AND source = ?',
                paper_updates,
            )

    # serialisation helpers 

    @staticmethod
    def _paper_to_row(paper: Paper) -> dict:
        return {
            'paper_id': paper.paper_id,
            'source': paper.source if isinstance(paper.source, str) else paper.source.value,
            'title': paper.title,
            'abstract': paper.abstract,
            'authors_json': json.dumps(paper.authors),
            'published_date': paper.published_date.isoformat() if paper.published_date else None,
            'doi': paper.doi,
            'journal': paper.journal,
            'keywords_json': json.dumps(paper.keywords),
            'url': paper.url,
            'fetched_at': paper.fetched_at.isoformat(),
            'content_hash': paper.content_hash,
        }

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> Optional[Paper]:
        try:
            return Paper(
                paper_id=row['paper_id'],
                source=row['source'],
                title=row['title'],
                abstract=row['abstract'] or '',
                authors=json.loads(row['authors_json'] or '[]'),
                published_date=datetime.fromisoformat(row['published_date'])
                               if row['published_date'] else None,
                doi=row['doi'],
                journal=row['journal'],
                keywords=json.loads(row['keywords_json'] or '[]'),
                url=row['url'],
                fetched_at=datetime.fromisoformat(row['fetched_at']),
            )
        except Exception as exc:
            logger.warning('Failed to deserialise row paper_id=%s: %s', row['paper_id'], exc)
            return None

    # public API 

    def save_papers(self, papers: list[Paper]) -> int:
        """
        Insert or replace a list of Paper objects.
        Returns the number of rows actually written.
        INSERT OR REPLACE silently overwrites existing records with the same
        (paper_id, source) primary key — this handles re-fetched data gracefully.
        """
        if not papers:
            return 0

        rows = [self._paper_to_row(p) for p in papers]
        sql = """
            INSERT OR REPLACE INTO papers
            (paper_id, source, title, abstract, authors_json, published_date,
             doi, journal, keywords_json, url, fetched_at, content_hash)
            VALUES
            (:paper_id, :source, :title, :abstract, :authors_json, :published_date,
             :doi, :journal, :keywords_json, :url, :fetched_at, :content_hash)
        """
        with self._get_connection() as conn:
            conn.executemany(sql, rows)

        logger.info('Saved %d papers to cache.', len(papers))
        return len(papers)

    def get_paper(self, paper_id: str, source: str) -> Optional[Paper]:
        """Retrieve a single paper by ID and source. Returns None if not found."""
        with self._get_connection() as conn:
            row = conn.execute(
                'SELECT * FROM papers WHERE paper_id = ? AND source = ?',
                (paper_id, source)
            ).fetchone()
        return self._row_to_paper(row) if row else None

    def paper_exists(self, paper_id: str, source: str) -> bool:
        """Fast existence check - avoids deserialising the full row."""
        with self._get_connection() as conn:
            result = conn.execute(
                'SELECT 1 FROM papers WHERE paper_id = ? AND source = ? LIMIT 1',
                (paper_id, source)
            ).fetchone()
        return result is not None

    def get_all_papers(self, source: Optional[str] = None, limit: int = 1000,
        offset: int = 0,
    ) -> list[Paper]:
        """
        Retrieve papers, optionally filtered by source.
        Ordered by fetched_at DESC so the most recent results come first.
        """
        with self._get_connection() as conn:
            if source:
                rows = conn.execute(
                    'SELECT * FROM papers WHERE source = ? ORDER BY fetched_at DESC LIMIT ? OFFSET ?',
                    (source, limit, offset)
                ).fetchall()
            else:
                rows = conn.execute(
                    'SELECT * FROM papers ORDER BY fetched_at DESC LIMIT ? OFFSET ?',
                    (limit, offset)
                ).fetchall()

        papers = [self._row_to_paper(r) for r in rows]
        return [p for p in papers if p is not None]

    def search_cached(self, query: str, source: Optional[str] = None) -> list[Paper]:
        """
        Prefer exact cached query results when available.

        For legacy caches without query-to-paper links, fall back to the most
        recent fetch for the same query/source pair, then to a simple text
        match on title/abstract.
        """
        normalized_query = self._normalize_query(query)
        normalized_source = self._normalize_source(source) if source else None
        with self._get_connection() as conn:
            if normalized_source:
                rows = conn.execute(
                    """
                    SELECT p.*
                    FROM query_results qr
                    JOIN papers p
                      ON p.paper_id = qr.paper_id
                     AND p.source = qr.source
                    WHERE qr.query = ?
                      AND qr.source = ?
                    ORDER BY qr.paper_rank ASC
                    """,
                    (normalized_query, normalized_source),
                ).fetchall()
                if not rows:
                    recent_log = conn.execute(
                        """
                        SELECT result_count
                        FROM query_log
                        WHERE query = ? AND source = ?
                        ORDER BY ran_at DESC
                        LIMIT 1
                        """,
                        (normalized_query, normalized_source),
                    ).fetchone()
                    if recent_log:
                        rows = conn.execute(
                            """
                            SELECT *
                            FROM papers
                            WHERE source = ?
                            ORDER BY fetched_at DESC
                            LIMIT ?
                            """,
                            (normalized_source, recent_log[0]),
                        ).fetchall()
                if not rows:
                    pattern = f'%{normalized_query.lower()}%'
                    rows = conn.execute(
                        'SELECT * FROM papers WHERE source = ? AND (LOWER(title) LIKE ? OR LOWER(abstract) LIKE ?)',
                        (normalized_source, pattern, pattern)
                    ).fetchall()
            else:
                pattern = f'%{normalized_query.lower()}%'
                rows = conn.execute(
                    'SELECT * FROM papers WHERE LOWER(title) LIKE ? OR LOWER(abstract) LIKE ?',
                    (pattern, pattern)
                ).fetchall()

        return [p for r in rows if (p := self._row_to_paper(r)) is not None]

    def cache_query_results(self, query: str, source: str, papers: list[Paper]) -> int:
        """Save papers, record the query log, and link the query to the returned papers."""
        normalized_query = self._normalize_query(query)
        normalized_source = self._normalize_source(source)
        if not papers:
            self.log_query(query=normalized_query, source=normalized_source, result_count=0)
            return 0

        self.save_papers(papers)
        with self._get_connection() as conn:
            conn.execute(
                'DELETE FROM query_results WHERE query = ? AND source = ?',
                (normalized_query, normalized_source),
            )
            conn.executemany(
                'INSERT INTO query_results (query, source, paper_id, paper_rank) VALUES (?, ?, ?, ?)',
                [
                    (normalized_query, normalized_source, paper.paper_id, index)
                    for index, paper in enumerate(papers, start=1)
                ],
            )
            conn.execute(
                'INSERT INTO query_log (query, source, result_count, ran_at) VALUES (?, ?, ?, ?)',
                (normalized_query, normalized_source, len(papers), datetime.now(timezone.utc).isoformat())
            )

        logger.info('Cached %d papers for query %r from %s.', len(papers), normalized_query, normalized_source)
        return len(papers)

    def log_query(self, query: str, source: str, result_count: int) -> None:
        """Record that a query was run. Useful for analysis and de-duplication."""
        normalized_query = self._normalize_query(query)
        normalized_source = self._normalize_source(source)
        with self._get_connection() as conn:
            conn.execute(
                'INSERT INTO query_log (query, source, result_count, ran_at) VALUES (?, ?, ?, ?)',
                (normalized_query, normalized_source, result_count, datetime.now(timezone.utc).isoformat())
            )

    def was_recently_fetched(self, query: str, source: str, within_hours: int = 24) -> bool:
        """
        Check if this exact query was already run recently.
        Prevents hammering the API with the same query multiple times per day.
        """
        from datetime import timedelta
        normalized_query = self._normalize_query(query)
        normalized_source = self._normalize_source(source)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=within_hours)).isoformat()
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT 1 FROM query_log WHERE query = ? AND source = ? AND ran_at > ? LIMIT 1",
                (normalized_query, normalized_source, cutoff)
            ).fetchone()
        return result is not None

    def count(self, source: Optional[str] = None) -> int:
        """Return total number of cached papers."""
        with self._get_connection() as conn:
            if source:
                return conn.execute(
                    "SELECT COUNT(*) FROM papers WHERE source = ?", (source,)
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def clear(self, source: Optional[str] = None) -> int:
        """Delete cached papers. Returns number of deleted rows."""
        normalized_source = self._normalize_source(source) if source else None
        with self._get_connection() as conn:
            if normalized_source:
                n = conn.execute("DELETE FROM papers WHERE source = ?", (normalized_source,)).rowcount
                conn.execute("DELETE FROM query_results WHERE source = ?", (normalized_source,))
                conn.execute("DELETE FROM query_log WHERE source = ?", (normalized_source,))
            else:
                n = conn.execute("DELETE FROM papers").rowcount
                conn.execute("DELETE FROM query_results")
                conn.execute("DELETE FROM query_log")
        logger.warning("Cleared %d papers from cache (source=%s).", n, normalized_source or "all")
        return n

    def prune_stale(self, older_than_days: int = 180, source: Optional[str] = None) -> dict[str, int]:
        """Remove stale cache rows and orphaned query mappings to keep DB size bounded."""
        if older_than_days <= 0:
            raise ValueError('older_than_days must be > 0')

        normalized_source = self._normalize_source(source) if source else None
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()

        with self._get_connection() as conn:
            if normalized_source:
                papers_deleted = conn.execute(
                    'DELETE FROM papers WHERE source = ? AND fetched_at < ?',
                    (normalized_source, cutoff),
                ).rowcount
                logs_deleted = conn.execute(
                    'DELETE FROM query_log WHERE source = ? AND ran_at < ?',
                    (normalized_source, cutoff),
                ).rowcount
                query_results_deleted = conn.execute(
                    '''
                    DELETE FROM query_results
                    WHERE source = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM papers p
                          WHERE p.paper_id = query_results.paper_id
                            AND p.source = query_results.source
                      )
                    ''',
                    (normalized_source,),
                ).rowcount
            else:
                papers_deleted = conn.execute(
                    'DELETE FROM papers WHERE fetched_at < ?',
                    (cutoff,),
                ).rowcount
                logs_deleted = conn.execute(
                    'DELETE FROM query_log WHERE ran_at < ?',
                    (cutoff,),
                ).rowcount
                query_results_deleted = conn.execute(
                    '''
                    DELETE FROM query_results
                    WHERE NOT EXISTS (
                        SELECT 1 FROM papers p
                        WHERE p.paper_id = query_results.paper_id
                          AND p.source = query_results.source
                    )
                    '''
                ).rowcount

        summary = {
            'papers_deleted': papers_deleted,
            'query_log_deleted': logs_deleted,
            'query_results_deleted': query_results_deleted,
        }
        logger.info(
            'Pruned stale cache rows (source=%s, older_than_days=%d): %s',
            normalized_source or 'all',
            older_than_days,
            summary,
        )
        return summary

    def stats(self) -> dict:
        """Return a summary dict for CLI display."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            by_source = conn.execute(
                "SELECT source, COUNT(*) FROM papers GROUP BY source"
            ).fetchall()
            recent = conn.execute(
                "SELECT query, ran_at FROM query_log ORDER BY ran_at DESC LIMIT 5"
            ).fetchall()

        return {
            "total_papers": total,
            "by_source": {row[0]: row[1] for row in by_source},
            "recent_queries": [{"query": r[0], "ran_at": r[1]} for r in recent],
        }
