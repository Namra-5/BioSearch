# src/models.py
"""
Data models for BioSearch AI
Defines the core Pydantic schemas for paper representation, search queries, and results. 
All incoming data is validated here before touching any other system.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import textwrap

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Use __name__ so the log identifies this specific file
logger = logging.getLogger(__name__)

class DataSource(str, Enum):
    PUBMED = 'pubmed'
    BIORXIV = 'biorxiv'

class Paper(BaseModel):
    """
    Canonical representation of a  single biomedical paper. 
    Validates and normalizes data from disparate API sources into a unified format.    
    """
    # Allow mutation for easier data cleaning and enrichment during model validation.
    model_config = ConfigDict(
        frozen=False,
        use_enum_values=True
    )
    
    paper_id: str = Field(..., description='PMID for PubMed, DOI for bioRXIV')
    title: str = Field(..., min_length=1)
    abstract: str = Field(default='')
    authors: list[str] = Field(default_factory=list)
    published_date:  Optional[datetime] = Field(default=None)
    source: DataSource = Field(...)
    doi: Optional[str] = Field(default=None)
    journal: Optional[str] = Field(default=None)
    keywords: list[str] = Field(default_factory=list)
    url: Optional[str] = Field(default=None)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def content_hash(self) -> str:
        """Generate a hash based on the title and abstract to detect duplicates."""
        raw = (self.title + self.abstract).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()[:16]

    @property
    def combined_text(self) -> str:
        """Concatenates title and abstract for TF-IDF or Semantic ranking."""
        return f"{self.title}. {self.abstract}".strip()
    
    @property
    def author_string(self) ->str:
        """Formats author list for display"""
        if not self.authors:
            return 'Unknown authors'
        if len(self.authors) <=3:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:3]) + " et al."
    
    @field_validator("paper_id")
    @classmethod
    def strip_id(cls, v:str) -> str:
        return v.strip()
    
    @field_validator("authors", mode ='before')
    @classmethod
    def coerce_Authors(cls,v:Any) -> list[str]:
        """
        Coerces input author data into a list of strings.
        Supports comma-separated strings (common in bioRxiv API) or already parsed lists (common in PubMed API).
        """
        if isinstance(v, str):
            return [a.strip() for a in v.split(",") if a.strip()]
        return v
    
    @field_validator("abstract", mode='before')
    @classmethod
    def clean_abstract(cls, v:Any) -> str:
        """Removes common NCBI prefixes and normalizes whitespace."""
        if v is None:
            return ""
        text = str(v).strip()
        if text.upper().startswith('ABSTRACT'):
            text = text[8:].strip()
            if text.startswith(':'):
                text = text[1:].strip()
        return text
    
    @model_validator(mode="after")
    def build_url_if_missing(self) -> "Paper":
        """Constructs a direct web link if none was provided by the source API."""
        if self.url:
            return self
        if self.source == DataSource.PUBMED:
            self.url = f"https://pubmed.ncbi.nlm.nih.gov/{self.paper_id}/"
        elif self.source == DataSource.BIORXIV and self.doi:
            self.url = f"https://doi.org/{self.doi}"
        return self

    def __str__(self) -> str:
        date_str = self.published_date.strftime("%Y-%m-%d") if self.published_date else "Unknown date"
        return (
            f"[{self.source.upper()}]  {self.paper_id}\n"
            f"Title:    {self.title}\n"
            f"Authors:  {self.author_string}\n"
            f"Date:     {date_str}\n"
            f"URL:      {self.url or 'N/A'}\n"
            f"Abstract: {' '.join(self.abstract.split()[:30])}..."
        )

    def __repr__(self) -> str:
        # Use getattr to safely check for .value, otherwise use the object itself
        source_val = getattr(self.source, 'value', self.source)
        return f"Paper(id={self.paper_id!r}, source={source_val!r})"

class SearchResult(BaseModel):
    """
    Represents a single search result with its associated paper, relevance score, and ranking.
    """
    paper: Paper
    score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1)
    method: str = Field(...)

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(float(v), 6)

    def __str__(self) -> str:
        display_title = textwrap.fill(self.paper.title, width=80, initial_indent="", subsequent_indent="      ")
        return (
            f"#{self.rank:02d}  [{self.method.upper()}]  Score: {self.score:.4f}\n"
            f"      {display_title}\n"
            f"      {self.paper.author_string}  |  {self.paper.url or 'no URL'}"
        )


class SearchQuery(BaseModel):
    """Represents a search query with validation rules for the query string, maximum results, and data sources."""
    query: str = Field(..., min_length=2, max_length=500)
    max_results: int = Field(default=20, ge=1, le=200)
    sources: list[DataSource] = Field(default_factory=lambda: [DataSource.PUBMED, DataSource.BIORXIV])

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty.")
        return cleaned

    