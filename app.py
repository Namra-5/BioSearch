"""
app.py - BioSearch AI web dashboard.

A single-file Streamlit UI over the existing CLI system (main.py). Every
data operation below calls the same classes main.py calls - PubMedFetcher,
BioRxivFetcher, TFIDFRanker, SemanticRanker, KnowledgeBase, dedup, and the
BioSearchEvaluator - with no changes to any of them. This file is a
presentation layer only.

Four tabs:
  1. Search        - live retrieval against PubMed and/or bioRxiv, with the
                      same --source semantics as main.py (pubmed / biorxiv /
                      both, the latter triggering automatic deduplication).
    2. Live Evaluation - runs the GCS evaluator on demand and displays fresh
                      results every time, using the exact same
                      BioSearchEvaluator / save_csv / save_json_summary /
                      generate_findings_md pipeline as `python main.py
                      --evaluate`. This tab regenerates `findings.md` live -
                      it is a real run, not a cached screenshot.
  3. Precision and Recall - read-only display of PRECISION_RECALL_FINDINGS.md,
                      clearly and repeatedly marked as hand-labeled ground
                      truth that this app does not and must never regenerate.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

@st.cache_resource(show_spinner="Loading BioBERT model (first run only)...")
def get_semantic_ranker():
    from src.ranker_semantic import SemanticRanker
    return SemanticRanker()

@st.cache_resource(show_spinner="Loading NER/knowledge-graph pipeline (first run only)...")
def get_knowledge_base():
    from src.knowledge_base import KnowledgeBase
    return KnowledgeBase()

@st.cache_resource
def get_pubmed_fetcher(email: str, api_key: str | None):
    from src.fetcher_pubmed import PubMedFetcher
    return PubMedFetcher(email=email, api_key=api_key)

@st.cache_resource
def get_biorxiv_fetcher():
    from src.fetcher_biorxiv import BioRxivFetcher
    return BioRxivFetcher()

@st.cache_resource
def get_storage(db_path):
    from src.storage import PaperStorage
    return PaperStorage(db_path=db_path)

def _graph_to_dot(graph, max_edges: int = 150):
    """Render a compact Graphviz diagram, keeping the strongest edges."""
    all_edges = sorted(
        graph.graph.edges(data=True),
        key=lambda e: e[2].get("weight", 1),
        reverse=True,
    )

    truncated = len(all_edges) > max_edges
    edges = all_edges[:max_edges]

    used_nodes = set()
    for first, second, _ in edges:
        used_nodes.add(first)
        used_nodes.add(second)

    node_ids = {
        node: f"n{index}"
        for index, node in enumerate(used_nodes)
    }

    lines = [
        "graph BioSearch {",
        '  layout=sfdp;',
        '  graph [bgcolor="transparent", overlap=false, ranksep=1.2, nodesep=0.45];',
        '  node [shape=circle, style=filled, fontname="Arial", fontsize=10, color="#FFFFFF"];',
        '  edge [color="#8AA9AD", fontname="Arial", fontsize=9, penwidth=1.2];',
    ]

    for node in used_nodes:
        data = graph.graph.nodes[node]
        node_color = (
            "#C1541C"
            if data.get("entity_type") == "GENE"
            else "#176B70"
        )
        lines.append(
            f"  {node_ids[node]} "
            f"[label={json.dumps(str(node))}, "
            f"fillcolor={json.dumps(node_color)}];"
        )

    for first, second, data in edges:
        weight = data.get("weight", 1)
        lines.append(
            f"  {node_ids[first]} -- {node_ids[second]} "
            f"[label={json.dumps(str(weight))}];"
        )

    lines.append("}")

    return "\n".join(lines), truncated

st.set_page_config(page_title="BioSearch AI", layout="wide")

# -- Visual styling --
# Custom styles distinguish auto-generated and hand-labeled results while
# keeping metrics, tables, and containers as native Streamlit widgets.
st.markdown(
    """
    <style>
    :root {
        --bs-ink: #173042;
        --bs-muted: #617482;
        --bs-teal: #0F4C5C;
        --bs-teal-soft: #EAF5F5;
        --bs-coral: #C1541C;
        --bs-line: #D8E4E6;
    }
    .stApp {
        background: #F7FAF9;
        color: var(--bs-ink);
    }
    [data-testid="stAppViewContainer"] > .main {
        background:
            radial-gradient(circle at 92% 2%, rgba(206, 231, 226, 0.5), transparent 24rem),
            #F7FAF9;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, [data-testid="stMetricValue"] {
        font-family: Georgia, "Times New Roman", serif;
        color: var(--bs-ink);
    }
    h3 {
        letter-spacing: 0.01em;
    }
    .bs-hero {
        background: linear-gradient(135deg, #0F4C5C 0%, #176B70 58%, #2B7B70 100%);
        padding: 1.55rem 2rem 1.65rem;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 1.35rem;
        box-shadow: 0 14px 34px rgba(15, 76, 92, 0.16);
        position: relative;
        overflow: hidden;
    }
    .bs-hero::after {
        content: "";
        position: absolute;
        width: 15rem;
        height: 15rem;
        right: -4rem;
        top: -7rem;
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 50%;
        box-shadow: 0 0 0 2.5rem rgba(255, 255, 255, 0.06), 0 0 0 5rem rgba(255, 255, 255, 0.04);
    }
    .bs-hero h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .bs-hero-kicker {
        margin: 0 0 0.45rem;
        color: #BCE5DE;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }
    .bs-hero p {
        margin: 0.35rem 0 0;
        font-size: 0.98rem;
        opacity: 0.92;
        max-width: 62ch;
        line-height: 1.45;
    }
    .bs-hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 1rem;
    }
    .bs-hero-meta span {
        border: 1px solid rgba(255, 255, 255, 0.28);
        border-radius: 999px;
        color: #E8F7F3;
        font-size: 0.72rem;
        padding: 0.27rem 0.62rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--bs-line);
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--bs-muted);
        font-weight: 600;
        padding: 0.75rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: var(--bs-teal);
    }
    .stButton > button[kind="primary"] {
        background: var(--bs-coral);
        border: 0;
        box-shadow: 0 5px 12px rgba(193, 84, 28, 0.2);
    }
    .stButton > button[kind="primary"]:hover {
        background: #A94414;
        border: 0;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--bs-line);
        background: rgba(255, 255, 255, 0.72);
        box-shadow: 0 5px 16px rgba(23, 48, 66, 0.045);
    }
    [data-testid="stMetric"] {
        background: var(--bs-teal-soft);
        border: 1px solid #D4E7E5;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    .bs-section-note {
        color: var(--bs-muted);
        font-size: 0.92rem;
        margin: -0.5rem 0 1.2rem;
    }
    .bs-banner-auto {
        border-left: 5px solid #0F4C5C;
        background: #EEF6F7;
        padding: 0.8rem 1.1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-size: 0.94rem;
    }
    .bs-banner-human {
        border-left: 5px solid #C1541C;
        background: #FBF0E8;
        padding: 0.8rem 1.1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-size: 0.94rem;
    }
    .bs-banner-human strong {
        color: #8A3A12;
    }
    .bs-source-tag {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        padding: 0.12rem 0.5rem;
        border-radius: 4px;
        margin-left: 0.4rem;
    }
    .bs-tag-pubmed { background: #E3EEF0; color: #0F4C5C; }
    .bs-tag-biorxiv { background: #F0E8F5; color: #5B2C82; }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 0.8rem 2rem;
        }
        .bs-hero {
            padding: 1.3rem 1.15rem 1.4rem;
            border-radius: 10px;
        }
        .bs-hero h1 {
            font-size: 1.8rem;
        }
        .bs-hero p {
            font-size: 0.9rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.65rem 0.45rem;
            font-size: 0.82rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="bs-hero">
        <div class="bs-hero-kicker">Biomedical literature intelligence</div>
        <h1>BioSearch AI</h1>
        <p>Compare fast lexical search with semantic BioBERT retrieval across
        real biomedical literature, then follow the evidence into a
        gene-disease knowledge graph.</p>
        <div class="bs-hero-meta">
            <span>PubMed + bioRxiv</span>
            <span>TF-IDF + BioBERT</span>
            <span>Evidence-led evaluation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_search, tab_eval, tab_precision, tab_about = st.tabs(
    ["Search", "Live Evaluation (GCS)", "Precision and Recall", "About"]
)

# -- Tab 1: Search -- 
with tab_search:
    st.subheader("Search biomedical literature")
    st.markdown(
        '<p class="bs-section-note">Search recent biomedical records, compare ranking signals, and inspect the resulting entity network.</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        query = st.text_input(
            "Query",
            placeholder="e.g. BRCA1 BRCA2 breast cancer hereditary mutations",
            key="search_query",
        )
    with col2:
        source = st.selectbox("Source", ["pubmed", "biorxiv", "both"], key="search_source")
    with col3:
        method = st.selectbox("Method", ["tfidf", "biobert"], key="search_method")
    with col4:
        max_results = st.number_input("Max papers", min_value=5, max_value=50, value=15, key="search_max")

    show_graph = st.checkbox("Also build knowledge graph for results", value=False, key="search_show_graph")

    if st.button("Search", type="primary", disabled=not query):
        with st.spinner(
            "Loading BioBERT (first run only, roughly 20 seconds)..."
            if method == "biobert"
            else "Fetching and ranking..."
        ):
            try:
                from config import DB_PATH, NCBI_API_KEY, NCBI_EMAIL

                needs_pubmed = source in ("pubmed", "both")
                if needs_pubmed and not NCBI_EMAIL:
                    st.error(
                        "NCBI_EMAIL is not set in your .env file - required "
                        "for any PubMed access. Set it exactly as main.py "
                        "expects, or switch Source to 'biorxiv' only."
                    )
                    st.stop()

                from src.storage import PaperStorage

                storage = get_storage(DB_PATH)
                fetchers = []
                if needs_pubmed:
                    from src.fetcher_pubmed import PubMedFetcher
                    fetchers.append(("pubmed", get_pubmed_fetcher(NCBI_EMAIL, NCBI_API_KEY)))
                if source in ("biorxiv", "both"):
                    from src.fetcher_biorxiv import BioRxivFetcher
                    fetchers.append(("biorxiv", get_biorxiv_fetcher()))

                papers = []
                for source_name, fetcher in fetchers:
                    cached = storage.get_papers_for_query(query, source=source_name)
                    if cached and len(cached) >= max_results:
                        papers.extend(cached[:max_results])
                        continue
                    fetched = fetcher.fetch(query, max_results=max_results)
                    storage.cache_query_results(query, source_name, fetched)
                    papers.extend(fetched)

                if not papers:
                    st.warning("No papers found for this query.")
                    st.session_state["search_ranked"] = None
                    st.session_state["search_graph"] = None
                    st.stop()

                if source == "both" and len(papers) > 1:
                    from src.dedup import deduplicate_papers
                    dedup_result = deduplicate_papers(papers)
                    st.caption(dedup_result.summary())
                    papers = dedup_result.papers

                if method == "tfidf":
                    from src.ranker_tfidf import TFIDFRanker
                    ranked = TFIDFRanker().rank(papers, query, top_n=10)
                else:
                    from src.ranker_semantic import SemanticRanker
                    ranked = get_semantic_ranker().rank(papers, query, top_n=10)

                # Persist so results survive the rerun triggered by opening
                # the graph expander below - Streamlit reruns this whole
                # script on every interaction, and this block only executes
                # again on an actual Search click, not later ones.
                st.session_state["search_ranked"] = ranked

                graph = None
                if show_graph:
                    from src.knowledge_base import KnowledgeBase
                    top_papers = [r.paper for r in ranked]
                    graph = get_knowledge_base().process_papers(top_papers)

                    dot_source, truncated = _graph_to_dot(graph, max_edges=150)
                    st.session_state["search_graph_dot"] = (dot_source, truncated)
                st.session_state["search_graph"] = graph

            except Exception as exc:
                st.error(f"Search failed: {exc}")
                st.caption(
                    "This surfaces whatever error the underlying fetcher or "
                    "ranker raised - compare against a direct "
                    "`python main.py --query ...` run if the cause is unclear."
                )
                st.session_state["search_ranked"] = None
                st.session_state["search_graph"] = None

    # -- Persisted results: render on every rerun, not just on Search click --
    ranked = st.session_state.get("search_ranked")
    if ranked:
        st.success(f"Showing top {len(ranked)} results")
        for r in ranked:
            with st.container(border=True):
                paper_source = getattr(r.paper.source, "value", r.paper.source)
                tag_class = "bs-tag-biorxiv" if str(paper_source) == "biorxiv" else "bs-tag-pubmed"
                st.markdown(
                    f"**{r.rank}. [{r.score:.4f}] {r.paper.title}**"
                )

                st.markdown(
                    f'<span class="bs-source-tag {tag_class}">{paper_source}</span>',
                    unsafe_allow_html=True,
                )
                if r.paper.abstract:
                    st.caption(r.paper.abstract[:300] + "...")
                if r.paper.url:
                    st.markdown(f"[View source]({r.paper.url})")

    graph = st.session_state.get("search_graph")

    if graph is not None:
        node_count = getattr(graph, "node_count", None)
        edge_count = getattr(graph, "edge_count", None)

        if node_count is None and hasattr(graph, "number_of_nodes"):
            node_count = graph.number_of_nodes()
            edge_count = graph.number_of_edges()

        m1, m2 = st.columns(2)
        m1.metric("Graph nodes", node_count)
        m2.metric("Graph edges", edge_count)

        if not node_count:
            st.info(
                "This graph has 0 nodes for this result set - no genes or "
                "diseases were extracted from these particular top-10 papers."
            )
        else:
            if "graph_viz_visible" not in st.session_state:
                st.session_state["graph_viz_visible"] = False

            if st.session_state["graph_viz_visible"]:
                if st.button(
                    "Hide knowledge graph visualization",
                    key="toggle_graph_viz",
                ):
                    st.session_state["graph_viz_visible"] = False
                    st.rerun()
            else:
                if st.button(
                    "Show knowledge graph visualization",
                    key="toggle_graph_viz",
                ):
                    st.session_state["graph_viz_visible"] = True
                    st.rerun()

            if st.session_state["graph_viz_visible"]:
                st.caption(
                    "Orange nodes are genes. Teal nodes are diseases. "
                    "Edge labels show co-occurrence counts."
                )

                dot_source, truncated = st.session_state["search_graph_dot"]

                if truncated:
                    st.caption(
                        f"Showing the top 150 highest-weight edges "
                        f"of {edge_count} total."
                    )

                try:
                    st.graphviz_chart(dot_source, width="stretch")
                except Exception as exc:
                    st.error(f"Could not render the graph: {exc}")

    elif ranked and show_graph:
        st.info("No graph was built for this search. " 
                "Tick the checkbox and click Search again.")

# -- Tab 2: Live Evaluation (GCS) --
with tab_eval:
    st.markdown(
        """
        <div class="bs-banner-auto">
        <strong>Auto-generated, regenerated live.</strong> This tab runs the
        real evaluation pipeline (the same one <code>python main.py
        --evaluate</code> runs) and overwrites <code>findings.md</code> with
        a fresh result each time you click Run. Values here will drift from
        any number previously reported elsewhere - including in this app's
        own README - because the underlying PubMed corpus and knowledge
        graph can change between runs. Treat this tab, not any static
        document, as the current source of truth for the GCS metric.
        </div>
        """,
        unsafe_allow_html=True,
    )

    include_biobert = st.checkbox(
        "Include BioBERT (roughly 20-40 seconds total; unchecked runs TF-IDF only in a few seconds)",
        value=True,
    )

    if st.button("Run evaluation now", type="primary"):
        try:
            from config import DB_PATH, NCBI_API_KEY, NCBI_EMAIL

            if not NCBI_EMAIL:
                st.error("NCBI_EMAIL is not set in your .env file - required to run the evaluation.")
                st.stop()

            from src.storage import PaperStorage
            from src.fetcher_pubmed import PubMedFetcher
            from src.evaluator import (
                BioSearchEvaluator,
                save_csv,
                save_json_summary,
                generate_findings_md,
            )

            storage = get_storage(DB_PATH)
            fetcher = get_pubmed_fetcher(NCBI_EMAIL, NCBI_API_KEY)
            paper_cache: dict = {}

            def _cached_fetch(q: str, max_results: int):
                if q not in paper_cache:
                    cached = storage.get_papers_for_query(q, source="pubmed")
                    if cached:
                        paper_cache[q] = cached
                    else:
                        fetched = fetcher.fetch(q, max_results=max_results)
                        storage.cache_query_results(q, "pubmed", fetched)
                        paper_cache[q] = fetched
                return paper_cache[q]

            methods = ["tfidf", "biobert"] if include_biobert else ["tfidf"]

            progress = st.empty()
            t0 = time.perf_counter()
            with st.spinner("Running evaluation across the standard query set..."):
                progress.info("Fetching and ranking for each query...")
                evaluator = BioSearchEvaluator(fetcher_fn=_cached_fetch)
                run = evaluator.run(methods=methods)

                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)
                save_csv(run, data_dir / "comparison_results.csv")
                save_json_summary(run, data_dir / "findings_summary.json")
                generate_findings_md(run, Path("findings.md"))

            elapsed = time.perf_counter() - t0
            progress.empty()
            st.success(f"Evaluation complete in {elapsed:.1f}s. findings.md has been regenerated.")

            summary = run.summary()
            agg = summary.get("aggregate", {})
            tf = agg.get("tfidf", {})
            bio = agg.get("biobert", {})
            paired = agg.get("paired_gcs_stats", {})

            c1, c2, c3 = st.columns(3)
            c1.metric("TF-IDF mean GCS", f"{tf.get('mean_graph_connectivity_score', 0):.1f}")
            if bio:
                c2.metric("BioBERT mean GCS", f"{bio.get('mean_graph_connectivity_score', 0):.1f}")
            if paired:
                pvalue = paired.get("sign_test_pvalue")
                c3.metric(
                    "Sign-test p-value",
                    f"{pvalue:.3f}" if isinstance(pvalue, float) else "n/a",
                    help="A paired sign test across the standard query set. "
                         "p > 0.05 means no statistically significant "
                         "difference was detected at this sample size.",
                )

            with st.expander("Full findings.md (as just regenerated)", expanded=True):
                st.markdown(Path("findings.md").read_text(encoding="utf-8"))

        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    elif Path("findings.md").exists():
        st.caption("Showing the last saved findings.md. Click 'Run evaluation now' for a fresh result.")
        with st.expander("Last saved findings.md"):
            st.markdown(Path("findings.md").read_text(encoding="utf-8"))
    else:
        st.info("No findings.md found yet. Click 'Run evaluation now' to generate one.")

# -- Tab 3: Precision and Recall (hand-labeled, read-only) --
with tab_precision:
    st.markdown(
        """
        <div class="bs-banner-human">
        <strong>Hand-labeled ground truth. Not auto-generated, not
        regenerated by this app.</strong> Every relevance judgment behind
        these numbers was read and labeled by the project author under a
        documented strict-intent policy - see the file itself for the
        methodology and worked examples. Unlike the Live Evaluation tab,
        nothing in this tab can be re-run from here, by design: these
        numbers only change if a human relabels the underlying data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    precision_recall_path = Path("PRECISION_RECALL_FINDINGS.md")
    if precision_recall_path.exists():
        st.markdown(precision_recall_path.read_text(encoding="utf-8"))
    else:
        st.warning(
            "PRECISION_RECALL_FINDINGS.md not found. This file is "
            "hand-maintained - see the project README for how it is produced."
        )

# -- Tab 4: About --
with tab_about:
    st.subheader("About this dashboard")
    st.markdown(
        """
        This is a thin presentation layer over the BioSearch AI CLI
        (`main.py`) - every search, ranking, and evaluation call here uses
        the same underlying classes the command line does, unmodified.

        **What each tab actually does:**

        - **Search** calls `PubMedFetcher` and/or `BioRxivFetcher`,
          `TFIDFRanker` or `SemanticRanker`, and optionally `KnowledgeBase`,
          exactly as `python main.py --query ... --source ...` would.
        - **Live Evaluation** calls `BioSearchEvaluator.run()` and the same
          `save_csv` / `save_json_summary` / `generate_findings_md`
          functions as `python main.py --evaluate`, and genuinely
          overwrites `findings.md` on disk - this is a live run, not a
          cached screenshot.
        - **Precision and Recall** only reads a file from disk. It never
          calls a ranker, fetcher, or evaluator, and cannot regenerate or
          modify the hand-labeled ground truth it displays.

        For the full methodology behind either evaluation, see
        `findings.md`, `PRECISION_RECALL_FINDINGS.md`, and the project
        README.
        """
    )
