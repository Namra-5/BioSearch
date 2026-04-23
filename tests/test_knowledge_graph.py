# tests/test_knowledge_graph.py
# Pure unit tests for BioKnowledgeGraph — no NLP model required.

import pytest
from pathlib import Path

from src.knowledge_graph import BioKnowledgeGraph
from src.ner_extractor import PaperEntities, LABEL_GENE, LABEL_DISEASE

# -- Fixtures --

@pytest.fixture
def sample_entities():
    return [
        PaperEntities(paper_id='p1', genes=['brca1', 'tp53'], diseases=['breast cancer', 'cancer']),
        PaperEntities(paper_id='p2', genes=['kras', 'tp53'], diseases=['lung cancer', 'cancer']),
        PaperEntities(paper_id='p3', genes=['egfr'],         diseases=['lung cancer']),
        PaperEntities(paper_id='p4', genes=['brca1'],        diseases=['ovarian cancer']),
        PaperEntities(paper_id='p5', genes=[],               diseases=[]),  # no entities
    ]

@pytest.fixture
def populated_graph(sample_entities):
    g = BioKnowledgeGraph()
    g.add_batch(sample_entities)
    return g

# -- Basic structure --

def test_graph_has_nodes_after_adding(populated_graph):
    assert populated_graph.node_count > 0

def test_graph_has_edges_after_adding(populated_graph):
    assert populated_graph.edge_count > 0

def test_gene_nodes_labelled_correctly(populated_graph):
    g = populated_graph.graph
    assert g.nodes['brca1']['entity_type'] == LABEL_GENE
    assert g.nodes['tp53']['entity_type'] == LABEL_GENE

def test_disease_nodes_labelled_correctly(populated_graph):
    g = populated_graph.graph
    assert g.nodes['breast cancer']['entity_type'] == LABEL_DISEASE
    assert g.nodes['lung cancer']['entity_type'] == LABEL_DISEASE

def test_empty_paper_entities_does_not_add_nodes():
    g = BioKnowledgeGraph()
    pe = PaperEntities(paper_id='empty', genes=[], diseases=[])
    edges_added = g.add_paper_entities(pe)
    assert edges_added == 0
    assert g.node_count == 0

def test_singleton_entity_still_adds_node():
    g = BioKnowledgeGraph()
    edges_added = g.add_paper_entities(PaperEntities(paper_id='single_1', genes=['brca1'], diseases=[]))
    assert edges_added == 0
    assert g.graph.has_node('brca1')
    assert g.graph.nodes['brca1']['paper_count'] == 1


# -- Edge weights --

def test_tp53_cancer_edge_has_weight_2(populated_graph):
    """tp53 and cancer both appear in p1 and p2 — weight should be 2."""
    g = populated_graph.graph
    assert g.has_edge('tp53', 'cancer')
    assert g.edges['tp53', 'cancer']['weight'] == 2

def test_brca1_lung_cancer_no_direct_edge(populated_graph):
    """brca1 and lung cancer never appear in the same paper — no edge."""
    g = populated_graph.graph
    assert not g.has_edge('brca1', 'lung cancer')


# -- Degree centrality --

def test_degree_centrality_returns_all_nodes(populated_graph):
    centrality = populated_graph.degree_centrality()
    assert len(centrality) == populated_graph.node_count

def test_tp53_has_higher_centrality_than_egfr(populated_graph):
    """tp53 appears in 2 papers with many co-entities; egfr in only 1."""
    centrality = populated_graph.degree_centrality()
    assert centrality['tp53'] >= centrality['egfr']

def test_top_entities_by_centrality_length(populated_graph):
    top = populated_graph.top_entities_by_centrality(LABEL_GENE, top_n=3)
    assert len(top) <= 3

def test_top_genes_are_all_genes(populated_graph):
    top = populated_graph.top_entities_by_centrality(LABEL_GENE, top_n=5)
    g = populated_graph.graph
    for name, _ in top:
        assert g.nodes[name]['entity_type'] == LABEL_GENE

def test_top_diseases_are_all_diseases(populated_graph):
    top = populated_graph.top_entities_by_centrality(LABEL_DISEASE, top_n=5)
    g = populated_graph.graph
    for name, _ in top:
        assert g.nodes[name]['entity_type'] == LABEL_DISEASE


# -- Top edges --

def test_top_edges_sorted_descending(populated_graph):
    edges = populated_graph.top_edges_by_weight(top_n=10)
    weights = [w for _, _, w in edges]
    assert weights == sorted(weights, reverse=True)

def test_tp53_cancer_in_top_edges(populated_graph):
    edges = populated_graph.top_edges_by_weight(top_n=20)
    pairs = {(a, b) for a, b, _ in edges} | {(b, a) for a, b, _ in edges}
    assert ('tp53', 'cancer') in pairs, 'tp53–cancer edge not in top edges'


# -- Neighbours --

def test_neighbours_of_brca1(populated_graph):
    nbrs = populated_graph.neighbours('brca1')
    assert len(nbrs) > 0
    nbr_names = [n for n, _ in nbrs]
    assert 'breast cancer' in nbr_names or 'cancer' in nbr_names

def test_neighbours_unknown_entity(populated_graph):
    nbrs = populated_graph.neighbours('nonexistent_gene_xyz')
    assert nbrs == []


# -- Gene-disease edges --

def test_gene_disease_edges_gene_first(populated_graph):
    """All returned tuples should have gene at position 0, disease at position 1."""
    g = populated_graph.graph
    for gene, disease, _ in populated_graph.gene_disease_edges():
        assert g.nodes[gene]['entity_type'] == LABEL_GENE, f'{gene} is not a gene'
        assert g.nodes[disease]['entity_type'] == LABEL_DISEASE, f'{disease} is not a disease'

def test_gene_disease_edges_sorted_descending(populated_graph):
    edges = populated_graph.gene_disease_edges()
    weights = [w for _, _, w in edges]
    assert weights == sorted(weights, reverse=True)


# -- Pruning --

def test_prune_removes_low_weight_edges(populated_graph):
    before_edges = populated_graph.edge_count
    removed = populated_graph.prune_low_weight_edges(min_weight=2)
    # Should have fewer or equal edges
    assert populated_graph.edge_count <= before_edges

def test_after_prune_no_low_weight_edges(sample_entities):
    g = BioKnowledgeGraph()
    g.add_batch(sample_entities)
    g.prune_low_weight_edges(min_weight=3)
    for u, v, data in g.graph.edges(data=True):
        assert data.get('weight', 1) >= 3, f'Edge {u}–{v} has weight below threshold'


# -- Subgraph --

def test_subgraph_contains_entity(populated_graph):
    sub = populated_graph.subgraph_for_entity('brca1', depth=1)
    assert 'brca1' in sub.nodes

def test_subgraph_unknown_entity_returns_empty(populated_graph):
    sub = populated_graph.subgraph_for_entity('xyz_not_exist', depth=1)
    assert sub.number_of_nodes() == 0


# -- Summary stats --

def test_summary_stats_correct_counts(populated_graph):
    stats = populated_graph.summary_stats()
    assert stats.total_nodes == populated_graph.node_count
    assert stats.total_edges == populated_graph.edge_count
    assert stats.gene_nodes + stats.disease_nodes == stats.total_nodes

def test_summary_stats_empty_graph():
    g = BioKnowledgeGraph()
    stats = g.summary_stats()
    assert stats.total_nodes == 0
    assert stats.total_edges == 0
    assert stats.density == 0.0

def test_summary_stats_to_dict_keys(populated_graph):
    d = populated_graph.summary_stats().to_dict()
    required_keys = {
        'total_nodes', 'total_edges', 'gene_nodes', 'disease_nodes',
        'density', 'top_genes_by_degree', 'top_diseases_by_degree',
        'top_edges_by_weight', 'connected_components', 'largest_component_size',
    }
    assert required_keys.issubset(d.keys())


# -- Merge --

def test_merge_two_graphs():
    g1 = BioKnowledgeGraph()
    g1.add_paper_entities(PaperEntities('p1', genes=['brca1'], diseases=['breast cancer']))
    g2 = BioKnowledgeGraph()
    g2.add_paper_entities(PaperEntities('p2', genes=['tp53'],  diseases=['cancer']))

    g1.merge(g2)
    assert g1.graph.has_node('tp53')
    assert g1.graph.has_node('breast cancer')

def test_merge_accumulates_edge_weights():
    g1 = BioKnowledgeGraph()
    g1.add_paper_entities(PaperEntities('p1', genes=['brca1'], diseases=['cancer']))
    g2 = BioKnowledgeGraph()
    g2.add_paper_entities(PaperEntities('p2', genes=['brca1'], diseases=['cancer']))

    g1.merge(g2)
    # Both graphs had brca1–cancer edge with weight 1; merged should be 2
    assert g1.graph.edges['brca1', 'cancer']['weight'] == 2


# -- Persistence --

def test_save_and_load_json(populated_graph, tmp_path):
    path = tmp_path / 'test_graph.json'
    populated_graph.save_json(path)
    assert path.exists()

    loaded = BioKnowledgeGraph.load_json(path)
    assert loaded.node_count == populated_graph.node_count
    assert loaded.edge_count == populated_graph.edge_count

def test_save_edgelist_has_header(populated_graph, tmp_path):
    path = tmp_path / 'edges.tsv'
    populated_graph.save_edgelist(path)
    with open(path) as f:
        header = f.readline()
    assert 'entity_a' in header
    assert 'weight' in header

def test_load_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        BioKnowledgeGraph.load_json(Path('data/does_not_exist.json'))


# -- Repr --

def test_repr_contains_node_and_edge_count(populated_graph):
    r = repr(populated_graph)
    assert 'nodes=' in r
    assert 'edges=' in r

# -- Stress & Logic Tests --

def test_add_batch_returns_correct_summary(sample_entities):
    g = BioKnowledgeGraph()
    summary = g.add_batch(sample_entities)
    assert 'papers_processed' in summary
    assert 'total_nodes' in summary
    assert summary['papers_processed'] == 5

def test_duplicate_entities_in_same_paper_weight_is_one():
    g = BioKnowledgeGraph()
    # brca1 listed twice, but should only create 1 edge with weight 1
    pe = PaperEntities(paper_id='p_dup', genes=['brca1', 'brca1'], diseases=['cancer'])
    g.add_paper_entities(pe)
    assert g.graph.edges['brca1', 'cancer']['weight'] == 1

def test_same_paper_id_does_not_double_count_node_mentions():
    g = BioKnowledgeGraph()
    pe = PaperEntities(paper_id='p_dup_id', genes=['brca1'], diseases=['cancer'])
    g.add_paper_entities(pe)
    g.add_paper_entities(pe)
    assert g.graph.nodes['brca1']['paper_count'] == 1
    assert g.graph.nodes['cancer']['paper_count'] == 1

def test_summary_stats_detects_islands():
    g = BioKnowledgeGraph()
    g.add_paper_entities(PaperEntities('p1', genes=['brca1'], diseases=['cancer']))
    g.add_paper_entities(PaperEntities('p2', genes=['app'], diseases=['alzheimers']))
    stats = g.summary_stats()
    assert stats.connected_components == 2

def test_json_persistence_preserves_weights(populated_graph, tmp_path):
    path = tmp_path / 'weight_test.json'
    populated_graph.save_json(path)
    loaded = BioKnowledgeGraph.load_json(path)
    # Check weight of tp53-cancer (which was 2 in populated_graph)
    assert loaded.graph.edges['tp53', 'cancer']['weight'] == 2