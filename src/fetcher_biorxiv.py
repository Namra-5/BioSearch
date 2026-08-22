# src/fetcher_biorxiv.py
# bioRxiv / medRxiv REST API fetcher.
# Uses the official API: https://api.biorxiv.org/details/{server}/{interval}/{cursor}
# No authentication required. Rate limiting applied defensively.

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.models import DataSource, Paper

logger = logging.getLogger(__name__)

_BIORXIV_BASE = 'https://api.biorxiv.org/details'
_EUROPE_PMC_SEARCH = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
_REQUEST_TIMEOUT = 60   # seconds
_SLEEP_BETWEEN = 0.5    # seconds between page requests
_MAX_RETRIES = 3


def _tokenize_text(text: str) -> set[str]:
    """Tokenize text to lowercase alphanumeric terms for precise keyword matching."""
    return set(re.findall(r'[a-z0-9]+', text.lower()))

def _build_session() -> requests.Session:
    """
    Build a requests.Session with automatic retry on network errors.

    Why a Session with Retry?
    A plain requests.get() raises an exception on any connection error.
    urllib3's Retry adapter automatically retries on connection resets,
    DNS failures, and 5xx responses — the kinds of transient failures
    common when hitting public preprint APIs.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=_MAX_RETRIES,
        backoff_factor=1.5,          # waits 1.5s, 3s, 4.5s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET'],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def _parse_biorxiv_paper(item: dict, server: str) -> Optional[Paper]:
    """
    Convert one item from the bioRxiv JSON response into a Paper object.
    Returns None on failure so the caller can skip bad records gracefully.
    """
    try:
        doi = item.get('doi', '').strip()
        title = item.get('title','').strip()
        if not doi or not title:
            return None
        
        abstract = item.get('abstract','').strip()
        authors_raw = item.get('authors','')
        # bioRxiv returns authors as 'Last F; Last F; ...' semicolon-separated
        if ';' in authors_raw:
            authors = [a.strip() for a in authors_raw.split(';') if a.strip()]
        else:
            authors = [a.strip() for a in authors_raw.split(',') if a.strip()]

        # Date is 'YYYY-MM-DD'
        date_str = item.get('date', '')
        pub_date : Optional[datetime] = None
        if date_str:
            try: 
                pub_date = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                pass

        category = item.get('category','')
        journal = f'{server.capitalize()} - {category}' if category else server.capitalize()

        return Paper(
            paper_id=doi,
            title=title,
            abstract=abstract,
            authors=authors,
            published_date=pub_date,
            source=DataSource.BIORXIV,
            doi=doi,
            journal=journal,
            keywords=[category] if category else [],
        )
    except Exception as exc:
        logger.warning('Failed to parse bioRxiv item: %s | item: %s', exc, str(item)[:120])
        return None


def _parse_europe_pmc_preprint(item: dict, server: str) -> Optional[Paper]:
    """Convert an Europe PMC preprint search result into a Paper object."""
    try:
        doi = (item.get('doi') or '').strip()
        title = re.sub(r'<[^>]+>', '', item.get('title') or '').strip()
        if not doi or not title:
            return None

        abstract = re.sub(r'<[^>]+>', '', item.get('abstractText') or '').strip()
        authors = [
            author.get('fullName', '').strip()
            for author in item.get('authorList', {}).get('author', [])
            if author.get('fullName', '').strip()
        ]
        pub_date: Optional[datetime] = None
        pub_year = str(item.get('pubYear') or '').strip()
        if pub_year.isdigit():
            pub_date = datetime(int(pub_year), 1, 1)

        return Paper(
            paper_id=doi,
            title=title,
            abstract=abstract,
            authors=authors,
            published_date=pub_date,
            source=DataSource.BIORXIV,
            doi=doi,
            journal=f'{server.capitalize()} (Europe PMC)',
            url=f'https://doi.org/{doi}',
        )
    except Exception as exc:
        logger.warning('Failed to parse Europe PMC preprint: %s', exc)
        return None
    
class BioRxivFetcher:
    """
    Fetches preprints from bioRxiv or medRxiv using their public REST API.

    The API supports two modes:
      - Date interval:  /details/{server}/{start_date}/{end_date}/{cursor}
      - Single DOI:     /details/{server}/{doi}

    We use the date-interval mode and filter title and abstract locally
    because the API does not provide full-text search.

    Parameters
    ----------
    server : str
        'biorxiv' or 'medrxiv'
    days_back : int
        How many days of preprints to retrieve.
    """

    def __init__(self, server : str = 'biorxiv', days_back : int = 180) -> None:
        if server.lower() not in ['biorxiv', 'medrxiv']:
            raise ValueError("server must be 'biorxiv' or 'medrxiv'")
        self.server = server.lower()
        self.days_back = days_back
        self._session = _build_session()
        logger.info('BioRxivFetcher initialised (server=%s, days_back=%d)', server, days_back)
    
    def _date_range(self) -> tuple[str,str]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.days_back)
        return start .strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
    
    def _fetch_page(self, start_date: str, end_date: str, cursor: int) -> dict:
        """
        Fetch one page of results from the bioRxiv API.
        cursor is the 0-based offset; the API returns 100 results per page.
        """
        url = f'{_BIORXIV_BASE}/{self.server}/{start_date}/{end_date}/{cursor}'
        logger.debug('GET %s', url)
        try:
            response = self._session.get(url, timeout = _REQUEST_TIMEOUT)
            if response.status_code != 200:
                logger.error('bioRxiv API returned HTTP %d for %s', response.status_code, url)
                return {}
            return response.json()
        except requests.RequestException as exc:
            logger.error('Request to bioRxiv failed: %s', exc)
            return {}

    def _fetch_europe_pmc_fallback(self, query: str, max_results: int) -> list[Paper]:
        """Fetch bioRxiv-indexed preprints through Europe PMC when needed."""
        params = {
            'query': f'SRC:PPR AND ({query})' if query else 'SRC:PPR',
            'format': 'json',
            'resultType': 'core',
            'pageSize': min(max(max_results * 3, 25), 100),
        }
        try:
            response = self._session.get(_EUROPE_PMC_SEARCH, params=params, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            results = response.json().get('resultList', {}).get('result', [])
            papers = [
                paper for item in results
                if (paper := _parse_europe_pmc_preprint(item, self.server)) is not None
            ]
            logger.info('Europe PMC fallback returned %d preprints for query %r', len(papers), query)
            return papers
        except (requests.RequestException, ValueError) as exc:
            logger.error('Europe PMC preprint fallback failed: %s', exc)
            return []
        
    def _local_keyword_filter(self, papers: list[Paper], query: str) -> list[Paper]:
        """
        The bioRxiv API has NO search endpoint — it only supports date ranges.
        We fetch a window of recent preprints and filter locally.

        This is intentional: it gives us a real corpus to rank with TF-IDF
        and later BioBERT. Papers with zero query-term overlap will score
        near 0 and sink to the bottom of ranked results.
        """
        if not query:
            return papers

        query_terms = _tokenize_text(query)
        if not query_terms:
            return papers

        def _filter_with_minimum(minimum_matches: int) -> list[Paper]:
            selected: list[Paper] = []
            for paper in papers:
                paper_terms = _tokenize_text(paper.title + ' ' + paper.abstract)
                overlap_count = len(query_terms & paper_terms)
                if overlap_count >= minimum_matches:
                    selected.append(paper)
            return selected

        strict_minimum = 1 if len(query_terms) == 1 else 2
        matched = _filter_with_minimum(strict_minimum)
        if matched:
            logger.info(
                'Local filter: %d/%d papers matched query "%s" (min_terms=%d)',
                len(matched),
                len(papers),
                query,
                strict_minimum,
            )
            return matched

        if strict_minimum > 1:
            relaxed = _filter_with_minimum(1)
            logger.info(
                'Local filter strict pass yielded 0; relaxed pass matched %d/%d for query "%s"',
                len(relaxed),
                len(papers),
                query,
            )
            return relaxed

        logger.info('Local filter: 0/%d papers matched query "%s"', len(papers), query)
        return []
    
    def fetch(self, query: str = "", max_results: int = 50) -> list[Paper]:
        """
        Fetch preprints from bioRxiv, optionally filtered by keyword.

        Steps:
        1. Build date range (today minus days_back).
          2. Page through API results (100 per page), up to the fetch window
              used to compensate for local keyword filtering.
        3. Filter locally by query keyword(s).
        4. Return up to max_results papers.
        """
        start_date, end_date = self._date_range()
        logger.info('Fetching %s preprints from %s to %s (max_results=%d)',
            self.server, start_date, end_date, max_results)
        all_papers: list[Paper] = []
        cursor = 0
        page_size = 100 # Api fixed page size

        while len(all_papers) < max_results * 3:
            # We fetch 3x max_results to have enough for filtering
            data = self._fetch_page(start_date, end_date, cursor)
            if not data:
                logger.warning('Empty response from bioRxiv at cursor=%d', cursor)
                if not all_papers and query:
                    all_papers = self._fetch_europe_pmc_fallback(query, max_results)
                break
            collection = data.get('collection', [])
            if not collection:
                logger.info('No more preprints returned at cursor=%d. Stopping.', cursor)
                break

            messages = data.get('messages', [])
            total = messages[0].get('total',0) if messages else 0
            logger.info('Page cursor=%d: got %d items (total available: %s)',
                cursor, len(collection), total)
            
            for item in collection:
                paper = _parse_biorxiv_paper(item, self.server)
                if paper:
                    all_papers.append(paper)

             # Check if we have fetched everything available
            if len(collection) < page_size or (total and cursor + page_size >= int(total)):
                logger.info('Reached end of available preprints.')
                break

            cursor += page_size
            time.sleep(_SLEEP_BETWEEN)

        # Apply local keyword filter
        if query:
            all_papers = self._local_keyword_filter(all_papers, query)

        # Deduplicate by DOI
        seen_dois: set[str] = set()
        unique_papers: list[Paper] = []
        for p in all_papers:
            if p.doi not in seen_dois:
                seen_dois.add(p.doi)
                unique_papers.append(p)

        result = unique_papers[:max_results]
        logger.info('Returning %d unique bioRxiv papers', len(result))
        return result






 




 

