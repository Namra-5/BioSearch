# src/fetcher_pubmed.py
# PubMed fetcher using Biopython Entrez.
# Two-step process: ESearch (get IDs) -> EFetch (get full records).
# Includes exponential backoff, rate limiting, and API key support.

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from typing import Optional
from xml.etree import ElementTree as ET

import requests
from Bio import Entrez

from src.models import DataSource, Paper

logger = logging.getLogger(__name__)

# NCBI allows 3 requests/sec without key, 10/sec with key.
# We stay conservative to avoid getting blocked.
_RATE_LIMIT_NO_KEY: float = 0.34     # ~3 requests/sec
_RATE_LIMIT_WITH_KEY: float = 0.11   # ~9 requests/sec - Allowed 10/sec but stay under to be safe
_MAX_RETRIES: int = 4
_BACKOFF_BASE: float = 2.0    # seconds; doubles with each retry

_MONTH_LOOKUP = {
    'jan': 1,
    'january': 1,
    'feb': 2,
    'february': 2,
    'mar': 3,
    'march': 3,
    'apr': 4,
    'april': 4,
    'may': 5,
    'jun': 6,
    'june': 6,
    'jul': 7,
    'july': 7,
    'aug': 8,
    'august': 8,
    'sep': 9,
    'sept': 9,
    'september': 9,
    'oct': 10,
    'october': 10,
    'nov': 11,
    'november': 11,
    'dec': 12,
    'december': 12,
}


def _parse_pubmed_date(date_elem: ET.Element | None) -> Optional[datetime]:
    """Parse PubDate values robustly across Year/Month/Day and MedlineDate variants."""
    if date_elem is None:
        return None

    year_text = (date_elem.findtext('Year') or '').strip()
    month_text = (date_elem.findtext('Month') or '').strip()
    day_text = (date_elem.findtext('Day') or '').strip()

    year = int(year_text) if year_text.isdigit() else None

    month = 1
    if month_text:
        if month_text.isdigit():
            month = min(max(int(month_text), 1), 12)
        else:
            month = _MONTH_LOOKUP.get(month_text.lower(), 1)

    day = 1
    if day_text.isdigit():
        day = min(max(int(day_text), 1), 31)

    if year is not None:
        try:
            return datetime(year, month, day)
        except ValueError:
            return datetime(year, 1, 1)

    medline = (date_elem.findtext('MedlineDate') or '').strip()
    if medline:
        match = re.search(r'(19|20)\d{2}', medline)
        if match:
            return datetime(int(match.group(0)), 1, 1)

    return None

def _get_sleep_interval() -> float:
    """Determines sleep interval based on presence of API key."""
    return _RATE_LIMIT_WITH_KEY if os.environ.get("NCBI_API_KEY") else _RATE_LIMIT_NO_KEY

def _configure_entrez(email: str, api_key: Optional[str] = None) -> None:
    """
    Configure Biopython Entrez globals.
    NCBI requires an email so they can contact you if your script misbehaves.
    The API key raises your rate limit from 3 to 10 requests/second.
    """
    if not email or '@' not in email:
        raise ValueError(
            'A valid email address is required by NCBI Entrez policy'
            'See: https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.Usage_Guidelines_and_Requirements')
    
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key
        os.environ['NCBI_API_KEY'] = api_key  # Ensure it's in env for rate limit checks
        logger.info('Entrez configured with API key (rate limit: 10 req/sec)')
    else:
        logger.warning(
            'No NCBI API key provided. Rate limit is 3 req/sec.'
            'Get a free key at: https://www.ncbi.nlm.nih.gov/account/'
        )

def _entrez_with_backoff(func_name:str, **kwargs) -> ET.Element:
    """
    Call any Biopython Entrez function with exponential backoff.

    Why exponential backoff?
    If NCBI returns a 429 (Too Many Requests) or a transient 5xx error,
    immediately retrying makes things worse. Waiting 2s, then 4s, then 8s
    gives the server time to recover and avoids a permanent IP ban.
    """
    entrez_func = getattr(Entrez, func_name)
    last_exception: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            handle = entrez_func(**kwargs)
            record = Entrez.read(handle)
            handle.close()
            time.sleep(_get_sleep_interval())
            return record
        except Exception as exc:
            last_exception = exc
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                'Entrez.%s attempt %d/%d failed: %s. Retrying in %.1s...',
                func_name, attempt, _MAX_RETRIES, exc, wait
            )
            time.sleep(wait)
    raise RuntimeError(
        f'Entrez.{func_name} failed after {_MAX_RETRIES} attempts.' 
        f'Last error: {last_exception}'
    )

def _parse_pubmed_xml(xml_text: str) -> list[Paper]:
    """
    Parse PubMed XML returned by efetch into Paper objects.

    Why manual XML parsing instead of Entrez.read()?
    Entrez.read() with rettype='xml' produces deeply nested dicts that
    differ between PubMed record types (journal article vs book section vs
    clinical trial). Parsing the XML directly gives us full control and
    consistent output.
    """
    papers: list[Paper] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParserError as exc:
        logger.error('failed to parse PubMed XML: %s', exc)
        return papers
    
    for article in root.findall('.//PubmedArticle'):
        try:
            # PMID
            pmid_elem = article.find('.//PMID')
            if pmid_elem is None or not pmid_elem.text:
                continue
            pmid = pmid_elem.text.strip()

            # Title
            title_elem = article.find('.//ArticleTitle')
            title = "".join(title_elem.itertext()).strip() if title_elem is not None else 'No title'

            # Abstract
            # AbstractText can be multiple elements (structured abstracts)
            abstract_parts = article.findall('.//AbstractText')
            abstract = " ".join("".join(p.itertext()) for p in abstract_parts).strip()

            # Authors
            authors: list[str] = []
            for author in article.findall('.//Author'):
                last = author.findtext("LastName", default = '')
                initials = author.findtext("Initials", default = '')
                if last:
                    authors.append(f' {last} {initials}'.strip())
            
            # Date
            pub_date = _parse_pubmed_date(article.find('.//PubDate'))

            # Journal
            journal = article.findtext(".//Journal/Title") or article.findtext(".//MedlineTA")

            # MeSH Keywords
            keywords = [mh.findtext('DescriptorName', default = '') 
                        for mh in article.findall(".//MeshHeading") 
                        if mh.findtext("DescriptorName")]
            
            paper = Paper(
                paper_id = pmid,
                title = title,
                abstract = abstract,
                authors = authors,
                published_date = pub_date,
                source = DataSource.PUBMED,                
                journal = journal,
                keywords = keywords
            )
            papers.append(paper)
            logger.debug("Parsed PubMed paper: %s", pmid)

        except Exception as exc:
            logger.warning("Skipped a PubMed record due to parsing error: %s", exc)
            continue
    return papers


class PubMedFetcher:
    """
    Fetches papers from PubMed using a two-step Entrez pipeline:
      1. ESearch  -> returns a list of PubMed IDs (PMIDs) matching a query
      2. EFetch   -> fetches full XML records for those PMIDs

    Object-oriented design rationale:
    Encapsulating state (email, api_key) in __init__ avoids passing them to
    every function call and makes the fetcher injectable/mockable in tests.
    """
    def __init__(self, email: str, api_key: Optional[str] = None) -> None:
        self.email = email
        self.api_key = api_key
        _configure_entrez(email,api_key)
        logger.info(' PubMedFetcher initialised (email=%s)', email)

    def search_ids(self, query: str, max_results: int = 20) -> list[str]:
        """
        Step 1: ESearch.
        Sends query to PubMed and returns a list of matching PMIDs.
        Returns empty list on failure (never raises — callers handle empty gracefully).
        """
        logger.info("ESearch: query=%r max_results=%d", query, max_results)
        try:
            record = _entrez_with_backoff(
                'esearch',
                db='pubmed',
                term=query,
                retmax=max_results,
                usehistory='y', # stores results server-side for EFetch
                sort='relevance'
            )
            ids: list[str] = record.get('IdList', [])
            logger.info('ESearch returned %d PMIDs', len(ids))
            return ids
        except RuntimeError as exc:
                logger.error('ESearch permanently failed: %s', exc)
                return []
    
    def fetch_by_ids(self, pmids: list[str]) -> list[Paper]:
        """
        Step 2: EFetch.
        Fetches full XML records for a list of PMIDs in batches of 100.
        Batching avoids URL-length limits and keeps each request fast.
        """
        if not pmids:
            logger.warning('fetch_by_ids called with empty PMID list')
            return []
        all_papers : list[Paper] = []
        batch_size = 100 # NCBI recommend <=200, we use 100 for safety

        for start in range(0, len(pmids), batch_size):
            batch = pmids[start : start + batch_size]
            logger.info(
                "EFetch batch %d-%d of %d",
                start + 1, start + len(batch), len(pmids)
            )
            try:
                # We use the requests library here instead of Biopython's
                # efetch because it gives us the raw XML string, which our
                # custom parser _parse_pubmed_xml() expects.
                params = {
                    'db' : 'pubmed',
                    'id' : ','.join(batch),
                    'rettype': 'xml',
                    'rettmode': 'xml',
                    'email' : self.email
                }
                if self.api_key:
                    params['api_key'] = self.api_key

                for attempt in range(1, _MAX_RETRIES + 1):
                    try:
                        response = requests.get(
                            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                            params=params,
                            timeout=30,
                        )
                        response.raise_for_status()
                        time.sleep(_get_sleep_interval())
                        break
                    except requests.RequestException as exc:
                        wait = _BACKOFF_BASE ** attempt
                        logger.warning('' \
                        'EFetch attempt %d/%d failed: %s. Waiting %.1s',
                        attempt, _MAX_RETRIES, exc, wait)
                        time.sleep(wait)
                else:
                    logger.error('EFetch permanently failed for batch starting at %d', start)
                    continue

                batch_papers = _parse_pubmed_xml(response.text)
                all_papers.extend(batch_papers)
                logger.info("Parsed %d papers from batch", len(batch_papers))
            except Exception as exc:
                logger.error("Unexpected error in EFetch batch: %s", exc)
                continue
        return all_papers
    
    def fetch(self, query:str, max_results:int = 20) -> list[Paper]:
        """
        Convenience method combining search_ids + fetch_by_ids.
        This is the single method most callers should use.
        """
        pmids = self.search_ids(query, max_results)
        if not pmids:
            logger.warning("No PMIDs found for query: %r", query)
            return []
        return self.fetch_by_ids(pmids)






                



            