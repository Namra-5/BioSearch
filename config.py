# config.py
# Central configuration loader.
# Uses python-dotenv to load .env file automatically.

from __future__ import annotations
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ

# NCBI credentials
NCBI_EMAIL:str = os.environ.get("NCBI_EMAIL", "")
NCBI_API_KEY:str | None = os.environ.get("NCBI_API_KEY") or None

# Paths
DATA_DIR = Path('data')
DB_PATH = DATA_DIR / 'biosearch_cache.db'

# Logging Setup - call setup_logging() at the start of main script
def setup_logging(level:str = 'INFO'):
    """Configures logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt = '%H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(DATA_DIR / 'biosearch.log', mode='a')
        ]
    )
    DATA_DIR.mkdir(exist_ok=True)  # Ensure data directory exists