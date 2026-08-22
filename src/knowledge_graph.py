# src/knowledge_graph.py
# Biomedical knowledge graph for BioSearch AI.
#
# Constructs a NetworkX graph from co-occurrence of Genes and Diseases
# within paper abstracts. Each node is a canonical entity name.
# Each edge carries a weight = number of papers in which both endpoints
# co-occurred. Degree-centrality and hub detection answer the research
# question: 'Which genes / diseases are most interconnected in this corpus?'
#
# Design principle: this file consumes PaperEntities objects from ner_extractor.py.
# It knows nothing about spaCy, BERT, or the API fetchers.

from __future__ import annotations

import itertools
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx

from src.ner_extractor import LABEL_DISEASE, LABEL_GENE, PaperEntities

logger = logging.getLogger(__name__)

# -- Graph node attribute keys --
_ATTR_LABEL       = 'entity_type'   # GENE or DISEASE
_ATTR_PAPER_COUNT = 'paper_count'   # how many papers mention this entity
_ATTR_WEIGHT      = 'weight'        # edge attribute: co-occurrence count


# -- Data classes --

@dataclass
class GraphStats:
    """
    Summary statistics for the knowledge graph.
    Returned by BioKnowledgeGraph.summary_stats().
    """
    total_nodes: int
    total_edges: int
    gene_nodes: int
    disease_nodes: int
    density: float
    top_genes_by_degree: list[tuple[str, float]]       # (name, centrality)
    top_diseases_by_degree: list[tuple[str, float]]
    top_edges_by_weight: list[tuple[str, str, int]]    # (node_a, node_b, weight)
    connected_components: int
    largest_component_size: int

    def to_dict(self) -> dict:
        return {
            'total_nodes': self.total_nodes,
            'total_edges': self.total_edges,
            'gene_nodes': self.gene_nodes,
            'disease_nodes': self.disease_nodes,
            'density': round(self.density, 6),
            'top_genes_by_degree': [
                {'entity': n, 'centrality': round(c, 4)} for n, c in self.top_genes_by_degree
            ],
            'top_diseases_by_degree': [
                {'entity': n, 'centrality': round(c, 4)} for n, c in self.top_diseases_by_degree
            ],
            'top_edges_by_weight': [
                {'node_a': a, 'node_b': b, 'co_occurrences': w}
                for a, b, w in self.top_edges_by_weight
            ],
            'connected_components': self.connected_components,
            'largest_component_size': self.largest_component_size,
        }


# -- Main graph class --

class BioKnowledgeGraph:
    """
    Builds and queries a weighted undirected graph of biomedical entities.

    -- Graph Theory Concepts Used --

    NODE:   
        An entity (gene or disease canonical name). Attribute: entity_type.

    EDGE:   
        A co-occurrence relationship between two entities found in the
        same abstract. Attribute: weight = number of papers they co-occurred in.

    DEGREE CENTRALITY:
        centrality(v) = degree(v) / (N - 1)
        where N = total nodes, degree(v) = number of edges connected to v.

        A gene with high degree centrality (e.g. TP53 often near cancer, diabetes,
        lung cancer, BRCA1, etc.) is a 'hub' — it connects many different biological
        concepts. Hub genes/diseases are often the most biologically significant
        because they participate in many pathways. This mirrors the 'scale-free'
        property observed in real biological networks (Barabási & Albert, 1999).

    EDGE WEIGHT:
        weight(u, v) = number of abstracts containing BOTH entity u and entity v.
        High weight means a strong, replicated association — e.g. BRCA1 + breast cancer
        appearing together in 45 papers is stronger evidence than TP53 + melanoma
        appearing in 2 papers.

    DENSITY:
        density = 2E / (N × (N-1))     (for undirected graphs)
        Range [0, 1]. A sparse graph (density ≈ 0.01) means most entities
        are not directly associated — the corpus covers many independent topics.
        A dense graph means most entities co-occur — corpus is very focused.

    CONNECTED COMPONENTS:
        A component is a set of nodes where every node can reach every other.
        Multiple components suggest the corpus contains research clusters that
        never interact (e.g. cardiovascular + neurology papers with no shared genes).

    -- Memory management --

    NetworkX stores nodes and edges as Python dicts. For a corpus of 1,000 papers
    with ~20 entities each, we get at most ~20,000 nodes and ~200,000 edges —
    both fit in RAM comfortably (~50 MB).

    For very large corpora (>100,000 papers), the graph would need either:
    a) Pruning low-weight edges: prune_low_weight_edges(min_weight=5)
    b) Subgraph extraction: subgraph_for_entity('brca1', depth=2)
    c) A graph database (Neo4j) — not implemented here.

    Both (a) and (b) are implemented below.
    """

    def __init__(self) -> None:
        # Undirected weighted graph — direction of co-occurrence is meaningless
        self._graph: nx.Graph = nx.Graph()
        # Track which papers contributed to each entity for provenance
        self._entity_papers: dict[str, set[str]] = defaultdict(set)
        logger.info('BioKnowledgeGraph initialised (empty).')

    # -- Graph population --

    def add_paper_entities(self, paper_entities: PaperEntities) -> int:
        """
        Add all entity co-occurrences from one paper to the graph.

        For a paper with genes=[g1, g2] and diseases=[d1, d2], we add edges:
            g1–d1, g1–d2, g2–d1, g2–d2  (gene-disease pairs)
            g1–g2                       (gene-gene pair)
            d1–d2                       (disease-disease pair)

        Gene-gene and disease-disease edges are included because the graph
        represents all entity co-occurrences, regardless of entity type.

        Returns
        -------
        int
            Number of new or updated edges added.
        """
        # Deduplicate mentions within the same paper so one paper contributes
        # at most +1 node paper_count and +1 edge weight per entity pair.
        unique_genes = list(dict.fromkeys(paper_entities.genes))
        unique_diseases = list(dict.fromkeys(paper_entities.diseases))
        paper_id = paper_entities.paper_id

        # Update node attributes even when a paper has only a single entity.
        # This preserves mention counts and prevents singleton papers from
        # disappearing from node-level statistics.
        for gene in unique_genes:
            if not self._graph.has_node(gene):
                self._graph.add_node(gene, **{_ATTR_LABEL: LABEL_GENE, _ATTR_PAPER_COUNT: 0})
            if paper_id not in self._entity_papers[gene]:
                self._entity_papers[gene].add(paper_id)
                self._graph.nodes[gene][_ATTR_PAPER_COUNT] += 1

        for disease in unique_diseases:
            if not self._graph.has_node(disease):
                self._graph.add_node(disease, **{_ATTR_LABEL: LABEL_DISEASE, _ATTR_PAPER_COUNT: 0})
            if paper_id not in self._entity_papers[disease]:
                self._entity_papers[disease].add(paper_id)
                self._graph.nodes[disease][_ATTR_PAPER_COUNT] += 1

        all_entities = unique_genes + unique_diseases
        if len(all_entities) < 2:
            # Can't form a co-occurrence with fewer than 2 entities
            return 0

        # Add edges for all unique pairs
        edges_modified = 0
        for entity_a, entity_b in itertools.combinations(all_entities, 2):
            if entity_a == entity_b:
                continue
            if self._graph.has_edge(entity_a, entity_b):
                self._graph[entity_a][entity_b][_ATTR_WEIGHT] += 1
            else:
                self._graph.add_edge(entity_a, entity_b, **{_ATTR_WEIGHT: 1})
            edges_modified += 1

        return edges_modified

    def add_batch(self, paper_entities_list: list[PaperEntities]) -> dict[str, int]:
        """
        Add entities from multiple papers at once.
        Returns summary dict: {papers_processed, edges_added, total_nodes, total_edges}.
        """
        total_edges_added = 0
        for pe in paper_entities_list:
            total_edges_added += self.add_paper_entities(pe)

        summary = {
            'papers_processed': len(paper_entities_list),
            'edges_added_or_updated': total_edges_added,
            'total_nodes': self._graph.number_of_nodes(),
            'total_edges': self._graph.number_of_edges(),
        }
        logger.info('add_batch complete: %s', summary)
        return summary

    # -- Queries --

    def degree_centrality(self) -> dict[str, float]:
        """
        Compute degree centrality for all nodes.
        Returns dict {entity_name: centrality_score}.
        Centrality = degree / (N-1). Range [0, 1].
        """
        if self._graph.number_of_nodes() == 0:
            return {}
        return nx.degree_centrality(self._graph)

    def top_entities_by_centrality(self, label: Optional[str] = None, top_n: int = 10,
                                   ) -> list[tuple[str, float]]:
        """
        Return the top_n most central entities, optionally filtered by label.

        Parameters
        ----------
        label : str | None
            LABEL_GENE, LABEL_DISEASE, or None (all entities).
        top_n : int
            Number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            Sorted (entity, centrality) pairs, descending.
        """
        centrality = self.degree_centrality()
        if label:
            centrality = {
                node: score
                for node, score in centrality.items()
                if self._graph.nodes[node].get(_ATTR_LABEL) == label
            }
        sorted_items = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_n]

    def top_edges_by_weight(self, top_n: int = 20) -> list[tuple[str, str, int]]:
        """
        Return the top_n strongest co-occurrence edges.
        Returns list of (node_a, node_b, weight) sorted by weight descending.
        """
        edges = [
            (u, v, data.get(_ATTR_WEIGHT, 1))
            for u, v, data in self._graph.edges(data=True)
        ]
        return sorted(edges, key=lambda x: x[2], reverse=True)[:top_n]

    def neighbours(self, entity: str) -> list[tuple[str, int]]:
        """
        Return all entities that co-occur with entity, sorted by co-occurrence count.
        Returns list of (neighbour_name, weight).
        """
        if not self._graph.has_node(entity):
            logger.warning("Entity '%s' not found in graph.", entity)
            return []
        return sorted(
            [
                (nbr, data.get(_ATTR_WEIGHT, 1))
                for nbr, data in self._graph[entity].items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )

    def subgraph_for_entity(self, entity: str, depth: int = 1) -> nx.Graph:
        """
        Extract a subgraph containing entity and all nodes within `depth` hops.

        A subgraph contains the requested entity and all nodes within the
        specified number of hops.

        Memory note: nx.ego_graph() returns a VIEW into the original graph
        for depth=1, which is O(degree) memory. For depth>1 it materialises
        a copy — safe for depth <= 3 on typical corpora.

        Parameters
        ----------
        entity : str
            Canonical entity name (lower-case).
        depth : int
            Number of hops from entity. 1 = immediate neighbours, 2 = neighbours of neighbours.
        """
        entity_lower = entity.lower()
        if not self._graph.has_node(entity_lower):
            logger.warning("Entity '%s' not in graph. Returning empty subgraph.", entity_lower)
            return nx.Graph()
        return nx.ego_graph(self._graph, entity_lower, radius=depth)

    def gene_disease_edges(self) -> list[tuple[str, str, int]]:
        """
        Return only the gene-disease edges (no gene-gene or disease-disease).
        Returns list of (gene, disease, weight).
        """
        result: list[tuple[str, str, int]] = []
        for u, v, data in self._graph.edges(data=True):
            u_type = self._graph.nodes[u].get(_ATTR_LABEL)
            v_type = self._graph.nodes[v].get(_ATTR_LABEL)
            weight = data.get(_ATTR_WEIGHT, 1)
            if u_type == LABEL_GENE and v_type == LABEL_DISEASE:
                result.append((u, v, weight))
            elif u_type == LABEL_DISEASE and v_type == LABEL_GENE:
                result.append((v, u, weight))  # normalise: gene always first
        return sorted(result, key=lambda x: x[2], reverse=True)

    # -- Graph manipulation --

    def prune_low_weight_edges(self, min_weight: int = 2) -> int:
        """
        Remove edges whose co-occurrence count is below min_weight.
        This keeps only well-evidenced relationships and reduces memory.
        Returns number of edges removed.
        """
        to_remove = [
            (u, v)
            for u, v, data in self._graph.edges(data=True)
            if data.get(_ATTR_WEIGHT, 1) < min_weight
        ]
        self._graph.remove_edges_from(to_remove)
        # Also remove now-isolated nodes (degree 0 after pruning)
        isolates = list(nx.isolates(self._graph))
        self._graph.remove_nodes_from(isolates)

        logger.info(
            'prune_low_weight_edges(min=%d): removed %d edges, %d isolated nodes.',
            min_weight, len(to_remove), len(isolates),
        )
        return len(to_remove)

    def merge(self, other: 'BioKnowledgeGraph') -> None:
        """
        Merge another BioKnowledgeGraph into this one.
        Edge weights are SUMMED (additive merge — co-occurrences accumulate).
        Node paper_counts are summed.
        Useful for merging PubMed and bioRxiv sub-graphs.
        """
        for node, data in other._graph.nodes(data=True):
            if not self._graph.has_node(node):
                self._graph.add_node(node, **data)
            else:
                self._graph.nodes[node][_ATTR_PAPER_COUNT] += data.get(_ATTR_PAPER_COUNT, 0)

            # Merge provenance when available so paper_count can be de-duplicated
            # across graphs that share paper IDs.
            if node in other._entity_papers:
                before = len(self._entity_papers[node])
                self._entity_papers[node].update(other._entity_papers[node])
                if before > 0:
                    self._graph.nodes[node][_ATTR_PAPER_COUNT] = len(self._entity_papers[node])

        for u, v, data in other._graph.edges(data=True):
            w = data.get(_ATTR_WEIGHT, 1)
            if self._graph.has_edge(u, v):
                self._graph[u][v][_ATTR_WEIGHT] += w
            else:
                self._graph.add_edge(u, v, **{_ATTR_WEIGHT: w})

        logger.info(
            'Merged graph: now %d nodes, %d edges.',
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    # -- Statistics --

    def summary_stats(self, top_n: int = 10) -> GraphStats:
        """
        Compute and return a GraphStats object with key metrics.
        Safe to call on empty graphs (returns zeroed stats).
        """
        n_nodes = self._graph.number_of_nodes()
        n_edges = self._graph.number_of_edges()

        gene_nodes = [
            n for n, d in self._graph.nodes(data=True)
            if d.get(_ATTR_LABEL) == LABEL_GENE
        ]
        disease_nodes = [
            n for n, d in self._graph.nodes(data=True)
            if d.get(_ATTR_LABEL) == LABEL_DISEASE
        ]
        density = nx.density(self._graph) if n_nodes > 1 else 0.0

        top_genes    = self.top_entities_by_centrality(LABEL_GENE, top_n)
        top_diseases = self.top_entities_by_centrality(LABEL_DISEASE, top_n)
        top_edges    = self.top_edges_by_weight(top_n)

        # Connected components (treat as undirected)
        components = list(nx.connected_components(self._graph))
        n_components = len(components)
        largest_size = max((len(c) for c in components), default=0)

        return GraphStats(
            total_nodes=n_nodes,
            total_edges=n_edges,
            gene_nodes=len(gene_nodes),
            disease_nodes=len(disease_nodes),
            density=density,
            top_genes_by_degree=top_genes,
            top_diseases_by_degree=top_diseases,
            top_edges_by_weight=top_edges,
            connected_components=n_components,
            largest_component_size=largest_size,
        )

    # -- Persistence --

    def save_json(self, path: Path) -> None:
        """
        Save the graph as node-link JSON (standard NetworkX format).
        Loadable with BioKnowledgeGraph.load_json().
        The file can also be visualised directly in D3.js or Gephi.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._graph)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2)
        logger.info('Graph saved to %s (%d nodes, %d edges).', 
                    path, self._graph.number_of_nodes(), self._graph.number_of_edges())

    @classmethod
    def load_json(cls, path: Path) -> 'BioKnowledgeGraph':
        """Load a previously saved graph. Returns a new BioKnowledgeGraph."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Graph file not found: {path}')
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        instance = cls()
        instance._graph = nx.node_link_graph(data)
        logger.info('Graph loaded from %s.', path)
        return instance

    def save_edgelist(self, path: Path) -> None:
        """
        Save edges as a tab-separated edgelist: gene\tdisease\tweight
        Useful for import into Gephi or Cytoscape for visualisation.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('entity_a\tentity_b\ttype_a\ttype_b\tweight\n')
            for u, v, data in sorted(
                self._graph.edges(data=True),
                key=lambda x: x[2].get(_ATTR_WEIGHT, 1),
                reverse=True,
            ):
                type_a = self._graph.nodes[u].get(_ATTR_LABEL, 'UNKNOWN')
                type_b = self._graph.nodes[v].get(_ATTR_LABEL, 'UNKNOWN')
                w = data.get(_ATTR_WEIGHT, 1)
                fh.write(f'{u}\t{v}\t{type_a}\t{type_b}\t{w}\n')
        logger.info('Edgelist saved to %s.', path)

    # -- Properties --

    @property
    def graph(self) -> nx.Graph:
        """Direct access to the underlying NetworkX graph. Handle with care."""
        return self._graph

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def __repr__(self) -> str:
        return (
            f'BioKnowledgeGraph('
            f'nodes={self.node_count}, edges={self.edge_count})'
        )
