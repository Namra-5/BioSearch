# BioSearch AI

BioSearch AI is a modular biomedical search system evolving from lexical retrieval to semantic embedding-based intelligence and knowledge graph construction.

---

## System Overview

- **Week 1 (Lexical):** Focused on robust data ingestion from PubMed/bioRxiv, SQLite-based query caching, and TF-IDF baselines.  
- **Week 2 (Semantic):** Advanced retrieval using BioBERT (768-d vectors) with persistent vector caching and comparative ranking logic.  
- **Week 3 (Knowledge Graph + NER):** Introduces hybrid biomedical Named Entity Recognition (NER) and constructs a gene–disease knowledge graph from literature.

---

## What it does

- Fetches biomedical papers from PubMed and bioRxiv.
- Caches query results in SQLite to avoid redundant API calls.
- Stores embeddings as 768-dimensional BioBERT vectors.
- Supports both lexical (TF-IDF) and semantic (BioBERT) ranking.
- Extracts biomedical entities (genes, diseases) using hybrid NER.
- Builds a co-occurrence-based knowledge graph from extracted entities.
- Enables graph-based exploration of gene–disease relationships.

---

## Current Status

- `src/models.py`: validated data models.  
- `src/fetcher_pubmed.py`: PubMed Entrez client with retries and rate limiting.  
- `src/fetcher_biorxiv.py`: bioRxiv REST client with adaptive local filtering.  
- `src/storage.py`: SQLite cache with query normalization and pruning.  
- `src/ranker_tfidf.py`: baseline ranking and safe fallback behavior.  
- `src/ranker_semantic.py`: BioBERT-based semantic ranking with embedding cache.  
- `src/ner_extractor.py`: hybrid biomedical NER (rules + spaCy statistical model).  
- `src/knowledge_base.py`: entity processing and persistence layer.  
- `src/knowledge_graph.py`: graph construction, analytics, and export utilities.  
- `main_week3.py`: extended CLI with NER and knowledge graph support.

---

## Technical Architecture

- **Fetch Layer:** PubMed + bioRxiv API clients.  
- **Storage Layer:** SQLite caching for papers and embeddings.  
- **Embedding Layer:** BioBERT (768-d dense vector representations).  
- **NER Layer:** Hybrid entity extraction (rules + statistical NLP).  
- **Graph Layer:** Co-occurrence-based biomedical knowledge graph.  
- **Ranking Layer:** Hybrid approach (Sparse TF-IDF vs Dense Semantic BioBERT).

---

## Week 3 Features

- **Hybrid NER**
  - Combines rule-based patterns (EntityRuler) with spaCy statistical model.
  - Supports `--rules-only` mode for high-precision extraction.

- **Knowledge Graph Construction**
  - Nodes: genes and diseases  
  - Edges: co-occurrence relationships across papers  
  - Weighted edges based on frequency  

- **Graph Analytics**
  - Degree centrality ranking  
  - Connected components  
  - Graph density  

- **Entity Exploration**
  - Subgraph extraction using `--entity`
  - Neighbor analysis (top co-occurring entities)

- **Export Support**
  - JSON graph export (`--save-graph`)
  - TSV edgelist export (`--save-edgelist`) for Gephi/Cytoscape

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
````

Install NLP backend:

```bash
pip install sentence-transformers torch spacy
python -m spacy download en_core_web_sm
```

Optional GPU acceleration:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Configuration

Create a `.env` file:

```env
NCBI_EMAIL=your_email@example.com
NCBI_API_KEY=your_api_key
```

---

## Example Usage

```bash
# Semantic ranking (Week 2)
python main_week2.py --query "BRCA1 breast cancer" --method biobert

# Compare TF-IDF vs BioBERT
python main_week2.py --query "CRISPR genomics" --method both --max 10

# NER extraction (Week 3)
python main_week3.py --query "BRCA1 breast cancer" --ner

# Knowledge graph generation
python main_week3.py --query "KRAS lung cancer" --graph

# Graph + entity exploration
python main_week3.py --query "KRAS lung cancer" --graph --entity kras

# Export graph
python main_week3.py --query "cancer genomics" --graph \
  --save-graph data/cancer_graph.json \
  --save-edgelist data/cancer_edges.tsv

# Rules-only NER (high precision)
python main_week3.py --query "EGFR lung cancer" --ner --rules-only
```

---

## Testing

All modules are validated using `pytest`:

* Unit tests for embedding cache integrity
* Integration tests for embedding pipeline
* NER extraction validation
* Knowledge graph construction tests
* Edge case handling (empty corpora, no entities found)

Run tests:

```bash
pytest
```

---

## Notes

* Week 1 focuses on lexical retrieval and caching.
* Week 2 introduces semantic understanding via BioBERT embeddings.
* Week 3 extends the system into **biomedical knowledge extraction and graph-based analysis**.
* Hybrid NER improves recall but may introduce noise from statistical predictions.
* Knowledge graph is based on co-occurrence and does not imply causation.
* System is designed for extension into advanced biomedical reasoning and graph learning in future phases.

