"""
DFM Knowledge Graph Agent -- Streamlit Interface

Visual interface for the DFM RAG agent. Retrieves knowledge frames from
Qdrant, traverses the provenance graph, and synthesizes structured DFM
reports via GPT with full interactive frame inspection.

Usage:
    streamlit run streamlit_app.py
"""

import os
import sys
import io
import json
import pickle
import hashlib
import datetime
from pathlib import Path

import streamlit as st
import networkx as nx
import plotly.graph_objects as go

# Import agent module first -- it handles the cp1252 encoding fix at load time.
# Do NOT duplicate the fix here (double TextIOWrapper wrapping causes I/O errors).
import dfm_agent as agent
from dfm_agent import (
    query_dfm,
    collect_cited_frame_ids,
    get_frame_details,
    gather_provenance_context,
    summarize_context,
    GRAPH_PICKLE,
    QDRANT_PATH,
    SEMANTIC_EDGE_TYPES,
    MODEL,
)
import dfm_agent as _agent_module
from openai import OpenAI
from dfm_report_schema import (
    DFMReportV2, BackfillOutput, QueryDecomposition,
    DECOMPOSE_PROMPT, SYNTHESIS_PROMPT_V2, BACKFILL_PROMPT,
)


# ─────────────────────────────────────────────
# ANALYSIS MODE CONFIGS
# ─────────────────────────────────────────────

ANALYSIS_MODES = {
    "Quick Search": {
        "model": "gpt-4.1",
        "reasoning_effort": None,
        "icon": "\u26a1",
        "time_range": "1-2 min",
        "token_range": "~11K tokens",
        "cost_estimate": "~$0.05",
        "description": "Fast, concise reports. Best for interactive use.",
    },
    "Efficient Thinking": {
        "model": "gpt-5",
        "reasoning_effort": "low",
        "icon": "\U0001f9e0",
        "time_range": "3-4 min",
        "token_range": "~15K tokens",
        "cost_estimate": "~$0.39",
        "description": "Deeper analysis with more design rules and disciplines.",
    },
    "Deep Reasoning": {
        "model": "gpt-5",
        "reasoning_effort": None,  # default = medium
        "icon": "\U0001f52c",
        "time_range": "7-10 min",
        "token_range": "~26K tokens",
        "cost_estimate": "~$0.92",
        "description": "Most thorough analysis. Richest rules and knowledge gaps.",
    },
}


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="DFM Knowledge Graph Agent",
    layout="wide",
    page_icon="\U0001f527",  # wrench
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# CUSTOM CSS + JS
# ─────────────────────────────────────────────

_CUSTOM_STYLES = """
<style>
/* ── Frame popover buttons: keep icon + type label on one line, hide arrow ── */

/* The trigger button (path: stPopover > div > button) */
[data-testid="stPopoverButton"] {
    white-space: nowrap !important;
    padding: 0.2rem 0.5rem !important;
    min-height: unset !important;
    font-size: 0.85rem !important;
    min-width: unset !important;
    gap: 0 !important;
    border: 1px solid rgba(250, 250, 250, 0.2) !important;
    border-radius: 6px !important;
    background-color: rgba(250, 250, 250, 0.06) !important;
}

/* Hide the expand_more dropdown arrow (last child of the button container) */
[data-testid="stPopoverButton"] > div > div:last-child:not(:first-child) {
    display: none !important;
}

/* Fallback: hide any material icon inside the popover button */
[data-testid="stPopoverButton"] [data-testid="stIconMaterial"] {
    display: none !important;
}

/* Label text div: center content, cancel the negative margin from hidden arrow */
[data-testid="stPopoverButton"] > div {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0 !important;
    margin: 0 !important;
}

/* Vertically center the popover column with the text column */
[data-testid="stPopover"] {
    display: flex;
    align-items: center;
    justify-content: flex-end;
}

/* Ensure the column holding the popover has a usable minimum width */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child:has([data-testid="stPopover"]) {
    min-width: 60px !important;
    flex-shrink: 0 !important;
}
</style>
"""
st.markdown(_CUSTOM_STYLES, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────

@st.cache_resource
def load_graph():
    """Load the provenance graph once per server lifetime."""
    if not GRAPH_PICKLE.exists():
        return None
    with open(GRAPH_PICKLE, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def get_openai_client():
    """Create OpenAI client once per server lifetime.

    Checks Streamlit secrets first (for Cloud deployment), then env var.
    """
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    # Also inject into os.environ so dfm_agent.py can find it
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI(api_key=api_key, timeout=300.0)


@st.cache_data
def get_graph_stats(_G):
    """Count frames and semantic edges for display."""
    n_frames = sum(1 for _, d in _G.nodes(data=True) if d.get("node_type") == "Frame")
    n_sem = sum(
        1 for _, _, d in _G.edges(data=True)
        if d.get("edge_type") in SEMANTIC_EDGE_TYPES
    )
    n_authors = sum(1 for _, d in _G.nodes(data=True) if d.get("node_type") == "Author")
    n_threads = sum(1 for _, d in _G.nodes(data=True) if d.get("node_type") == "Thread")
    return {
        "frames": n_frames,
        "semantic_edges": n_sem,
        "authors": n_authors,
        "threads": n_threads,
        "total_nodes": _G.number_of_nodes(),
        "total_edges": _G.number_of_edges(),
    }


_EXAMPLE_CACHE_FILE = Path(__file__).parent / "example_cache.json"


@st.cache_resource
def load_example_cache():
    """Load pre-computed example answers (once per server lifetime)."""
    if not _EXAMPLE_CACHE_FILE.exists():
        return {}
    try:
        with open(_EXAMPLE_CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

def init_session_state():
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "active_idx" not in st.session_state:
        st.session_state.active_idx = -1
    if "context_cache" not in st.session_state:
        st.session_state.context_cache = {}
    if "is_querying" not in st.session_state:
        st.session_state.is_querying = False

init_session_state()


# ─────────────────────────────────────────────
# PERSISTENT TOKEN TALLY
# ─────────────────────────────────────────────

_TOKEN_TALLY_FILE = Path(__file__).parent / "token_tally.json"


def _load_token_tally():
    """Load cumulative token tally from disk."""
    if _TOKEN_TALLY_FILE.exists():
        try:
            with open(_TOKEN_TALLY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    defaults = {
        "since": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "embedding_tokens": 0,
        "total_tokens": 0,
        "queries": 0,
        "unique_ips": [],
        "page_views": 0,
    }
    return defaults


def _update_token_tally(token_usage):
    """Add a query's token usage to the persistent tally."""
    tally = _load_token_tally()
    tally["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
    tally["completion_tokens"] += token_usage.get("completion_tokens", 0)
    tally["embedding_tokens"] += token_usage.get("embedding_tokens", 0)
    tally["total_tokens"] += token_usage.get("total_tokens", 0)
    tally["queries"] += 1
    tally.setdefault("unique_ips", [])
    tally.setdefault("page_views", 0)
    with open(_TOKEN_TALLY_FILE, "w") as f:
        json.dump(tally, f, indent=2)


def _reset_token_tally():
    """Reset the persistent token tally."""
    tally = {
        "since": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "embedding_tokens": 0,
        "total_tokens": 0,
        "queries": 0,
        "unique_ips": [],
        "page_views": 0,
    }
    with open(_TOKEN_TALLY_FILE, "w") as f:
        json.dump(tally, f, indent=2)
    return tally


def _get_visitor_ip_hash():
    """Get a privacy-safe hash of the visitor's IP address."""
    try:
        ip = st.context.headers.get("X-Forwarded-For", "")
        if ip:
            ip = ip.split(",")[0].strip()
        if not ip:
            ip = st.context.headers.get("Host", "")
        if not ip:
            return None
        return hashlib.sha256(ip.encode()).hexdigest()[:12]
    except Exception:
        return None


def _record_visitor():
    """Record a unique visitor (once per session)."""
    if st.session_state.get("_visitor_recorded"):
        return
    st.session_state["_visitor_recorded"] = True
    ip_hash = _get_visitor_ip_hash()
    tally = _load_token_tally()
    tally.setdefault("unique_ips", [])
    tally.setdefault("page_views", 0)
    tally["page_views"] += 1
    if ip_hash and ip_hash not in tally["unique_ips"]:
        tally["unique_ips"].append(ip_hash)
    with open(_TOKEN_TALLY_FILE, "w") as f:
        json.dump(tally, f, indent=2)


# ─────────────────────────────────────────────
# SCROLL-DISMISS JS
# ─────────────────────────────────────────────

def inject_scroll_dismiss(pin_popovers=False):
    """Inject JS that closes open popovers on scroll (unless pinned).

    Uses st.components.v1.html so the <script> actually executes, then
    reaches into window.parent.document (same-origin Streamlit frame).
    """
    import streamlit.components.v1 as components

    pinned_val = "true" if pin_popovers else "false"
    components.html(f"""
    <script>
    (function() {{
        var doc = window.parent.document;

        // Update pin flag every rerun
        var pin = doc.getElementById('dfm-pin-popovers');
        if (!pin) {{
            pin = doc.createElement('div');
            pin.id = 'dfm-pin-popovers';
            pin.style.display = 'none';
            doc.body.appendChild(pin);
        }}
        pin.dataset.pinned = '{pinned_val}';

        // Attach scroll listener only once
        if (window.parent._dfmScrollDismissAttached) return;
        window.parent._dfmScrollDismissAttached = true;

        var observer = new MutationObserver(function() {{
            var main = doc.querySelector('[data-testid="stMain"]');
            if (!main || main._dfmScrollListener) return;
            main._dfmScrollListener = true;
            var scrollTimer = null;
            main.addEventListener('scroll', function() {{
                var p = doc.getElementById('dfm-pin-popovers');
                if (p && p.dataset.pinned === 'true') return;
                if (scrollTimer) clearTimeout(scrollTimer);
                scrollTimer = setTimeout(function() {{
                    var open = doc.querySelectorAll('[data-testid="stPopoverBody"]');
                    if (open.length > 0) {{
                        // Close by clicking the main content area (acts as outside-click)
                        var mc = doc.querySelector('[data-testid="stMainBlockContainer"]');
                        if (mc) {{
                            mc.dispatchEvent(new MouseEvent('mousedown', {{bubbles:true}}));
                            mc.dispatchEvent(new MouseEvent('mouseup', {{bubbles:true}}));
                            mc.dispatchEvent(new MouseEvent('click', {{bubbles:true}}));
                        }}
                    }}
                }}, 150);
            }});
        }});
        observer.observe(doc.body, {{childList: true, subtree: true}});
    }})();
    </script>
    """, height=0)


# ─────────────────────────────────────────────
# RENDERING HELPERS
# ─────────────────────────────────────────────

# Map edge types to colors for graph viz
EDGE_COLORS = {
    "CAUSAL_LINK": "#e74c3c",    # red
    "MITIGATES": "#2ecc71",      # green
    "SPECIFIES": "#3498db",      # blue
    "CONTRADICTS": "#e67e22",    # orange
}

# Map frame types to colors for graph viz
FRAME_TYPE_COLORS = {
    "risk": "#e74c3c",
    "heuristic": "#3498db",
    "principle": "#9b59b6",
    "case": "#f39c12",
    "workaround": "#2ecc71",
    "observation": "#1abc9c",
    "comparison": "#e67e22",
    "other": "#95a5a6",
}

# Map chain roles to visual indicators
ROLE_INDICATORS = {
    "root_cause": ("\U0001f534", "Root Cause"),       # red circle
    "effect": ("\U0001f7e0", "Effect"),               # orange circle
    "mitigation": ("\U0001f7e2", "Mitigation"),       # green circle
    "condition": ("\U0001f535", "Condition"),          # blue circle
    "origin": ("\u26aa", "Origin"),                    # white circle
}


def render_frame_popover(frame_id, frame_lookup, short_label=None):
    """Render a frame_id as a clickable popover with full details."""
    detail = frame_lookup.get(frame_id)
    # Default label: frame_type from lookup, else frame_id
    if short_label is None:
        label = detail["frame_type"] if detail else frame_id
    else:
        label = short_label

    if not detail:
        st.caption(f"`{frame_id}`")
        return

    with st.popover(f"\U0001f4cb {label}"):
        # Header
        ft = detail["frame_type"]
        color = FRAME_TYPE_COLORS.get(ft, "#95a5a6")
        st.markdown(
            f"**{ft.upper()}** &nbsp; | &nbsp; "
            f"Scope: `{detail['scope']}` &nbsp; | &nbsp; "
            f"Applicability: `{detail['applicability']}`"
        )

        st.markdown(f"**Subject:** {detail['subject']}")
        st.markdown(f"**Main Point:** {detail['main_point']}")

        if detail.get("source_quote"):
            st.caption("Source Quote")
            st.markdown(f"> {detail['source_quote']}")

        st.markdown(f"**Epistemic stance:** `{detail['epistemic_stance']}` "
                    f"&nbsp; | &nbsp; Post role: `{detail['post_role']}`")

        if detail.get("example_value"):
            st.markdown(f"**Example value:** {detail['example_value']}")
        if detail.get("example_context"):
            st.markdown(f"**Example context:** {detail['example_context']}")

        st.divider()

        # Author & credibility
        c1, c2, c3 = st.columns(3)
        c1.metric("Author", detail.get("author", "?"))
        c2.metric("Credibility", detail.get("credibility", "?"))
        rep = detail.get("reputation", -1)
        c3.metric("Reputation", f"{rep:,}" if rep != -1 else "N/A")

        c4, c5 = st.columns(2)
        c4.metric("Post Score", detail.get("post_score", 0))
        if detail.get("is_accepted"):
            c5.success("\u2713 Accepted Answer")
        else:
            c5.caption("Not accepted")

        if detail.get("source_url"):
            st.markdown(f"[\U0001f517 View Original Thread]({detail['source_url']})")
        if detail.get("se_tags"):
            st.caption(f"Tags: {detail['se_tags']}")


def render_diagnostic(diagnostic):
    """Show retrieval diagnostic as a metrics row."""
    if not diagnostic:
        return

    relevant = diagnostic.get("relevant", 0)
    total = diagnostic.get("total", 0)
    avg_sim = diagnostic.get("avg_sim", 0)

    # Reinterpret grade based on avg similarity (more useful than the 0.7 threshold)
    if avg_sim >= 0.6:
        match_label = "High"
    elif avg_sim >= 0.45:
        match_label = "Moderate"
    elif avg_sim >= 0.3:
        match_label = "Indirect"
    else:
        match_label = "Low"

    c1, c2, c3 = st.columns(3)
    c1.metric("Retrieval Match", match_label,
              help="How directly the query matched knowledge graph content. "
                   "'Indirect' or 'Moderate' can still produce strong reports "
                   "because graph traversal pulls in related frames via semantic edges.")
    c2.metric("Seed Frames", f"{total}",
              help="Number of unique frames retrieved from vector search before graph traversal.")
    c3.metric("Avg Similarity", f"{avg_sim:.3f}",
              help="Mean cosine similarity of the top seed frames to the query embedding. "
                   "Values above 0.4 typically produce useful results.")


def render_design_rules(report, frame_lookup, concise=False):
    """Render actionable design rules with frame popovers."""
    rules = report.design_rules
    if not rules:
        st.caption("No design rules extracted.")
        return

    show_rules = rules[:5] if concise else rules
    for i, rule in enumerate(show_rules, 1):
        col_text, col_ref = st.columns([7, 1])
        with col_text:
            st.markdown(f"**{i}.** {rule.claim}")
        with col_ref:
            render_frame_popover(rule.frame_id, frame_lookup)

    if concise and len(rules) > 5:
        with st.expander(f"Show all {len(rules)} rules"):
            for i, rule in enumerate(rules[5:], 6):
                col_text, col_ref = st.columns([7, 1])
                with col_text:
                    st.markdown(f"**{i}.** {rule.claim}")
                with col_ref:
                    render_frame_popover(rule.frame_id, frame_lookup)


def render_evidence(evidence_list, frame_lookup):
    """Render a list of SourcedClaim evidence items."""
    if not evidence_list:
        st.caption("No evidence in this discipline.")
        return

    for ev in evidence_list:
        col_text, col_ref = st.columns([7, 1])
        with col_text:
            st.markdown(f"- {ev.claim}")
        with col_ref:
            render_frame_popover(ev.frame_id, frame_lookup)


def render_reasoning_chains(chains, frame_lookup):
    """Render reasoning chains as individual dropdowns."""
    if not chains:
        return

    for chain in chains:
        with st.expander(f"\U0001f517 {chain.chain_label}", expanded=False):
            for cf in chain.frames:
                indicator, role_label = ROLE_INDICATORS.get(
                    cf.role_in_chain, ("\u2b55", cf.role_in_chain)
                )
                edge_info = ""
                if cf.edge_type and cf.edge_from:
                    edge_info = f" &larr; `{cf.edge_type}` from `{cf.edge_from}`"

                col_text, col_ref = st.columns([7, 1])
                with col_text:
                    st.markdown(
                        f"{indicator} **{role_label}**: {cf.claim}{edge_info}"
                    )
                with col_ref:
                    render_frame_popover(cf.frame_id, frame_lookup)


def render_formulas(formulas):
    """Render formula entries."""
    if not formulas:
        return

    st.markdown("**Governing Equations**")
    for f in formulas:
        with st.expander(f"\U0001f4d0 {f.formula_name}"):
            st.code(f.expression, language=None)
            st.caption(f"**Variables:** {f.variables}")
            st.markdown(f"*{f.relevance}*")


def _split_rationale_to_bullets(text):
    """Split a long first-principles rationale into bullet points by sentence."""
    import re
    # Split on sentence boundaries (period/semicolon followed by space and uppercase or end)
    sentences = re.split(r'(?<=[.;])\s+(?=[A-Z])', text.strip())
    # Filter out very short fragments
    return [s.strip().rstrip(".") + "." for s in sentences if len(s.strip()) > 15]


def render_disciplines(report, frame_lookup):
    """Render discipline sections as tabs."""
    disciplines = report.disciplines
    if not disciplines:
        st.caption("No discipline sections.")
        return

    tab_names = [d.discipline.replace("_", " ").title() for d in disciplines]
    tabs = st.tabs(tab_names)

    for tab, disc in zip(tabs, disciplines):
        with tab:
            # First principles -- split into bullets for readability
            if disc.first_principles:
                bullets = _split_rationale_to_bullets(disc.first_principles)
                if len(bullets) > 1:
                    bullet_md = "\n".join(f"- {b}" for b in bullets)
                    st.info(f"**Engineering Rationale**\n\n{bullet_md}")
                else:
                    st.info(disc.first_principles)

            # Formulas
            render_formulas(disc.formulas)

            # Evidence
            if disc.evidence:
                st.markdown(f"**Evidence** ({len(disc.evidence)} claims)")
                render_evidence(disc.evidence, frame_lookup)

            # Reasoning chains
            if disc.reasoning_chains:
                st.markdown(f"**Reasoning Chains** ({len(disc.reasoning_chains)})")
                render_reasoning_chains(disc.reasoning_chains, frame_lookup)


def render_contradictions(report, frame_lookup):
    """Render contradictions as warning blocks with source details."""
    contras = report.contradictions
    if not contras:
        return

    st.markdown("### \u26a0\ufe0f Contradictions")
    for c in contras:
        with st.container(border=True):
            st.warning(f"**{c.summary}**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Position A** (`{c.frame_id_a}`):\n\n{c.position_a}")
                fa = frame_lookup.get(c.frame_id_a)
                if fa:
                    if fa.get("source_quote"):
                        st.caption(f'> "{fa["source_quote"][:200]}..."' if len(fa.get("source_quote", "")) > 200 else f'> "{fa["source_quote"]}"')
                    link = fa.get("thread_url") or fa.get("url")
                    author = fa.get("author", "")
                    if link:
                        st.caption(f"Source: [{author or c.frame_id_a}]({link})")
            with c2:
                st.markdown(f"**Position B** (`{c.frame_id_b}`):\n\n{c.position_b}")
                fb = frame_lookup.get(c.frame_id_b)
                if fb:
                    if fb.get("source_quote"):
                        st.caption(f'> "{fb["source_quote"][:200]}..."' if len(fb.get("source_quote", "")) > 200 else f'> "{fb["source_quote"]}"')
                    link = fb.get("thread_url") or fb.get("url")
                    author = fb.get("author", "")
                    if link:
                        st.caption(f"Source: [{author or c.frame_id_b}]({link})")
            if hasattr(c, "edge_reasoning") and c.edge_reasoning:
                st.caption(f"Edge reasoning: {c.edge_reasoning}")


def render_knowledge_gaps(report, backfill_result, concise=False):
    """Render knowledge gaps and optional backfill."""
    gaps = report.knowledge_gaps
    if not gaps:
        return

    n_backfills = len(backfill_result.backfills) if backfill_result and backfill_result.backfills else 0
    st.markdown(f"### \U0001f50d Knowledge Gaps & Textbook Backfill")

    with st.expander(f"Knowledge Gaps ({len(gaps)})", expanded=False):
        st.caption("Areas where the knowledge graph lacks coverage")
        for gap in gaps:
            st.markdown(f"- {gap}")

    # Backfill
    if backfill_result and backfill_result.backfills:
        with st.expander(
            f"\U0001f4da Textbook Backfill ({n_backfills} gaps filled)",
            expanded=False,
        ):
            st.caption(
                "General engineering knowledge -- NOT from the knowledge graph. "
                "Use as starting guidance only."
            )
            for bf in backfill_result.backfills:
                with st.container(border=True):
                    st.markdown(f"**Gap:** {bf.gap}")
                    st.markdown(bf.backfill_answer)
                    footer_parts = [f"*{bf.confidence_note}*"]
                    if bf.relevant_standards:
                        footer_parts.append(f"Standards: {bf.relevant_standards}")
                    st.caption(" | ".join(footer_parts))


def build_graph_figure(report, G, frame_lookup):
    """Build a Plotly network graph of cited frames and their semantic edges."""
    cited = collect_cited_frame_ids(report)
    if not cited or len(cited) < 2:
        return None

    # Build subgraph of cited frames
    cited_in_graph = [fid for fid in cited if G.has_node(fid)]
    if len(cited_in_graph) < 2:
        return None

    subG = G.subgraph(cited_in_graph).copy()

    # Keep only semantic edges
    edges_to_remove = [
        (u, v, k) for u, v, k, d in subG.edges(data=True, keys=True)
        if d.get("edge_type") not in SEMANTIC_EDGE_TYPES
    ]
    for u, v, k in edges_to_remove:
        subG.remove_edge(u, v, key=k)

    if subG.number_of_edges() == 0:
        return None

    # Layout -- higher k = more separation between nodes
    pos = nx.spring_layout(subG, seed=42, k=3.0, iterations=80)

    # Build edge traces (one per edge type for legend)
    edge_traces = {}
    for u, v, d in subG.edges(data=True):
        etype = d.get("edge_type", "UNKNOWN")
        if etype not in edge_traces:
            edge_traces[etype] = {"x": [], "y": []}
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces[etype]["x"].extend([x0, x1, None])
        edge_traces[etype]["y"].extend([y0, y1, None])

    fig = go.Figure()

    # Add edge traces
    for etype, coords in edge_traces.items():
        color = EDGE_COLORS.get(etype, "#bdc3c7")
        fig.add_trace(go.Scatter(
            x=coords["x"], y=coords["y"],
            mode="lines",
            line=dict(width=1.5, color=color),
            name=etype,
            hoverinfo="none",
        ))

    # Build node trace
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    node_hover = []

    for fid in cited_in_graph:
        if fid not in pos:
            continue
        x, y = pos[fid]
        node_x.append(x)
        node_y.append(y)

        detail = frame_lookup.get(fid, {})
        ft = detail.get("frame_type", "other")
        node_colors.append(FRAME_TYPE_COLORS.get(ft, "#95a5a6"))

        # Short label: last part of frame_id
        parts = fid.split("-")
        short = parts[-1] if len(parts) > 1 else fid[:8]
        node_text.append(short)

        # Hover text
        subject = detail.get("subject", "?")
        author = detail.get("author", "?")
        node_hover.append(
            f"<b>{fid}</b><br>"
            f"Type: {ft}<br>"
            f"Subject: {subject}<br>"
            f"Author: {author}"
        )

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=16, color=node_colors, line=dict(width=1.5, color="white")),
        text=node_text,
        textposition="top center",
        textfont=dict(size=10),
        hovertext=node_hover,
        hoverinfo="text",
        name="Frames",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text="Cited Frames & Semantic Edges", y=0.98, x=0.01,
                   xanchor="left", yanchor="top"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        height=600,
        margin=dict(l=20, r=20, t=80, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ─────────────────────────────────────────────
# SUMMARY FORMATTING
# ─────────────────────────────────────────────

def _render_summary(text):
    """Render summary text with improved readability.

    Splits long text into paragraphs on sentence boundaries and
    bolds the first sentence of each paragraph as a topic sentence.
    """
    import re
    text = text.strip()
    if not text:
        return

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 3:
        # Short summary: just bold the first sentence
        sentences[0] = f"**{sentences[0]}**"
        st.markdown(" ".join(sentences))
        return

    # Group sentences into paragraphs of 2-3 sentences
    paragraphs = []
    i = 0
    while i < len(sentences):
        chunk_size = 3 if len(sentences) - i > 4 else 2
        chunk = sentences[i:i + chunk_size]
        # Bold the first sentence of each paragraph (topic sentence)
        chunk[0] = f"**{chunk[0]}**"
        paragraphs.append(" ".join(chunk))
        i += chunk_size

    st.markdown("\n\n".join(paragraphs))


# ─────────────────────────────────────────────
# RENDER FULL REPORT
# ─────────────────────────────────────────────

def render_report(result, G, concise=False):
    """Render a complete DFM report from a result dict."""
    report = result["report"]
    enrichment = result["enrichment"]
    diagnostic = result["diagnostic"]
    sub_aspects = result["sub_aspects"]
    backfill_result = result["backfill_result"]
    frame_lookup = {d["frame_id"]: d for d in result["frame_details"]}
    query = result["query"]

    # Diagnostic
    render_diagnostic(diagnostic)

    st.divider()

    # Sub-aspects
    if sub_aspects:
        with st.expander("Query Decomposition", expanded=False):
            for asp in sub_aspects:
                st.markdown(f"- {asp}")

    # Summary -- improve readability by splitting into paragraphs
    st.markdown("### Summary")
    _render_summary(report.summary)

    st.divider()

    # Design Rules
    n_rules = len(report.design_rules)
    st.markdown(f"### Design Rules ({n_rules})")
    render_design_rules(report, frame_lookup, concise=concise)

    st.divider()

    # Disciplines
    st.markdown("### Analysis by Discipline")
    render_disciplines(report, frame_lookup)

    # Contradictions
    render_contradictions(report, frame_lookup)

    st.divider()

    # Knowledge Gaps + Backfill
    render_knowledge_gaps(report, backfill_result, concise=concise)

    st.divider()

    # Context Summary & Graph
    st.markdown("### \U0001f50e Source Context & Knowledge Graph")
    cited_ids = sorted(collect_cited_frame_ids(report))
    cache_key = tuple(cited_ids) + (query,)

    if cache_key in st.session_state.context_cache:
        with st.expander("Source Context & Credibility Summary", expanded=False):
            st.markdown(st.session_state.context_cache[cache_key])
    else:
        if st.button("\U0001f50e Get Source Context & Credibility Summary", key=f"ctx_{query[:20]}"):
            client = get_openai_client()
            if not client:
                st.error("OPENAI_API_KEY not set. Cannot generate context summary.")
            else:
                with st.spinner("Walking provenance graph and generating summary..."):
                    provenance = gather_provenance_context(G, cited_ids)
                    summary = summarize_context(client, provenance, query)
                    st.session_state.context_cache[cache_key] = summary
                    st.rerun()

    # Graph visualization
    with st.expander("\U0001f578\ufe0f Knowledge Graph View", expanded=False):
        fig = build_graph_figure(report, G, frame_lookup)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Not enough connected frames to visualize.")

    # Stats footer
    n_cited = len(collect_cited_frame_ids(report))
    n_rules = len(report.design_rules)
    n_chains = sum(len(d.reasoning_chains) for d in report.disciplines)
    n_gaps = len(report.knowledge_gaps)

    st.divider()
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Cited Frames", n_cited)
    fc2.metric("Design Rules", n_rules)
    fc3.metric("Reasoning Chains", n_chains)
    fc4.metric("Knowledge Gaps", n_gaps)

    # ── Token usage & timing for this query ──
    token_usage = result.get("token_usage", {})
    elapsed_s = result.get("elapsed_s")
    if token_usage or elapsed_s:
        with st.expander("\u23f1\ufe0f Pipeline stats", expanded=False):
            sc1, sc2, sc3 = st.columns(3)
            if elapsed_s:
                mins, secs = divmod(int(elapsed_s), 60)
                sc1.metric("Elapsed", f"{mins}m {secs}s" if mins else f"{secs}s")
            if token_usage:
                sc2.metric("API calls", token_usage.get("api_calls", 0))
                sc3.metric("Total tokens", f"{token_usage.get('total_tokens', 0):,}")

            if token_usage:
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("Prompt tokens", f"{token_usage.get('prompt_tokens', 0):,}",
                           help="Tokens sent to the model (system + user messages)")
                tc2.metric("Completion tokens", f"{token_usage.get('completion_tokens', 0):,}",
                           help="Tokens generated by the model")
                tc3.metric("Embedding tokens", f"{token_usage.get('embedding_tokens', 0):,}",
                           help="Tokens used for query/sub-aspect embedding")


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    # Load resources
    G = load_graph()
    client = get_openai_client()
    example_cache = load_example_cache()

    # Record visitor (once per session)
    _record_visitor()

    # ── SIDEBAR ──
    with st.sidebar:
        st.title("\U0001f527 DFM KG Agent")

        if st.session_state.get("active_idx", -1) >= 0:
            if st.button("\u2190 Home", use_container_width=True):
                st.session_state.active_idx = -1
                st.rerun()

        if G is None:
            st.error("Graph not found. Run `build_graph.py` first.")
            st.stop()

        stats = get_graph_stats(G)
        st.caption(
            f"**{stats['frames']:,}** frames &nbsp;|&nbsp; "
            f"**{stats['semantic_edges']:,}** semantic edges\n\n"
            f"**{stats['authors']:,}** authors &nbsp;|&nbsp; "
            f"**{stats['threads']:,}** threads"
        )

        st.divider()
        st.markdown("**Controls**")

        # Analysis mode selector
        mode_names = list(ANALYSIS_MODES.keys())
        mode_labels = [
            f"{ANALYSIS_MODES[m]['icon']} {m}" for m in mode_names
        ]
        selected_mode = st.radio(
            "Analysis mode",
            mode_names,
            format_func=lambda m: f"{ANALYSIS_MODES[m]['icon']} {m}",
            index=0,
            help="Controls the LLM used for synthesis. Affects speed, depth, and cost.",
        )
        mode_cfg = ANALYSIS_MODES[selected_mode]
        st.caption(
            f"{mode_cfg['description']}  \n"
            f"**{mode_cfg['time_range']}** &nbsp;|&nbsp; "
            f"**{mode_cfg['token_range']}** &nbsp;|&nbsp; "
            f"**{mode_cfg['cost_estimate']}**/query"
        )

        top_k = st.slider("Top K (frames per query)", min_value=3, max_value=25, value=10, step=1,
                          help="Number of frames to retrieve from the vector index per sub-aspect")
        max_hops = st.slider("Max Hops (graph traversal)", min_value=1, max_value=4, value=2, step=1,
                             help="How many edge hops to follow in the semantic graph")
        max_frames = st.slider("Max Frames (for synthesis)", min_value=10, max_value=40, value=20, step=5,
                               help="Maximum frames passed to the LLM for report synthesis")
        decompose = st.checkbox("Decompose query", value=True,
                                help="Break question into engineering sub-aspects for broader retrieval")
        backfill = st.checkbox("Backfill knowledge gaps", value=True,
                               help="Generate textbook answers for areas not covered by the knowledge graph")
        concise = st.checkbox("Concise output", value=False,
                              help="Shorter report: max 5 design rules, collapsed evidence, compact knowledge gaps")
        pin_popovers = st.checkbox("Pin popovers", value=False,
                                   help="Keep frame detail popovers open while scrolling (default: auto-dismiss on scroll)")

        # Query history
        if st.session_state.query_history:
            st.divider()
            st.markdown("**Query History**")
            for i, res in enumerate(st.session_state.query_history):
                q_short = res["query"][:40] + ("..." if len(res["query"]) > 40 else "")
                ts = res.get("timestamp", "")
                is_active = (i == st.session_state.active_idx)
                btn_type = "primary" if is_active else "secondary"
                btn_label = f"{ts}  {q_short}" if ts else q_short
                if st.button(
                    btn_label,
                    key=f"hist_{i}",
                    type=btn_type,
                    use_container_width=True,
                ):
                    st.session_state.active_idx = i
                    st.session_state._scroll_to_report = True
                    st.rerun()

        # ── Token tally ──
        st.divider()
        tally = _load_token_tally()
        st.markdown("**Token Usage**")
        st.caption(f"Since {tally['since']}")
        tc1, tc2 = st.columns(2)
        tc1.metric("Queries", tally["queries"])
        tc2.metric("Total tokens", f"{tally['total_tokens']:,}")
        with st.expander("Breakdown", expanded=False):
            st.caption(f"Prompt: **{tally['prompt_tokens']:,}**")
            st.caption(f"Completion: **{tally['completion_tokens']:,}**")
            st.caption(f"Embedding: **{tally['embedding_tokens']:,}**")
        if st.button("\U0001f504 Reset tally", key="reset_tally",
                     help="Reset the token counter and start a new tracking period"):
            _reset_token_tally()
            st.rerun()

        # ── About the Developer ──
        st.divider()
        st.markdown("**About the Developer**")
        st.markdown(
            "**Douglas McGowan**  \n"
            "Ph.D. Researcher, Mechanical Engineering  \n"
            "[Co-Design Lab](https://codesign.berkeley.edu/), UC Berkeley  \n"
            "Advisor: Dr. Kosa Goucher-Lambert"
        )
        st.caption(
            "Research focus: computational design, AI/ML in product "
            "development, and design for manufacturability."
        )
        st.markdown(
            "\U0001f4e7 [dpm5970@berkeley.edu](mailto:dpm5970@berkeley.edu)  \n"
            "\U0001f517 [LinkedIn](https://www.linkedin.com/in/douglas-mcgowan/) "
            " | [Co-Design Lab](https://codesign.berkeley.edu/team/douglas-mcgowan/)"
        )

    # ── MAIN AREA ──

    # API key check
    if not client:
        st.warning(
            "\u26a0\ufe0f **OPENAI_API_KEY not set.** "
            "Add it to `.streamlit/secrets.toml` or set it as an environment variable.\n\n"
            "```toml\n"
            "# .streamlit/secrets.toml\n"
            'OPENAI_API_KEY = "sk-..."\n'
            "```"
        )
        st.stop()

    title_col, info_col = st.columns([6, 1])
    with title_col:
        st.title("DFM Knowledge Graph Agent")
    with info_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        with st.popover("\u2753 What is this?"):
            st.markdown(
                "### Design-for-Manufacturability Knowledge Agent\n\n"
                "This tool helps **mechanical and product designers** get "
                "manufacturing feedback *before* sending parts to the shop.\n\n"
                "**The problem:** Experienced machinists carry decades of practical "
                "knowledge about what's easy or hard to machine -- wall thickness "
                "limits, fixturing tricks, tolerance traps -- but most of that "
                "knowledge lives in forum posts and shop-floor conversations, not "
                "in engineering textbooks.\n\n"
                "**What this does:** We scraped thousands of expert discussions "
                "from Stack Exchange engineering communities, used LLMs to extract "
                "structured knowledge frames (risks, heuristics, workarounds, "
                "principles), and built a semantic knowledge graph connecting them. "
                "When you ask a question, the agent:\n\n"
                "1. **Retrieves** relevant frames via multi-vector embedding search\n"
                "2. **Traverses** the knowledge graph to find causal links, "
                "mitigations, and contradictions\n"
                "3. **Synthesizes** a structured report with design rules, "
                "reasoning chains, and engineering rationale\n"
                "4. **Backfills** knowledge gaps with textbook-level guidance\n\n"
                "Every claim cites its source frame -- click any frame ID to see "
                "the original quote, author, and thread URL.\n\n"
                "**Built by** [Douglas McGowan](https://www.linkedin.com/in/douglas-mcgowan/), "
                "Ph.D. Researcher at UC Berkeley's "
                "[Co-Design Lab](https://codesign.berkeley.edu/)."
            )
    st.caption(
        "Ask a design-for-manufacturability question. The agent retrieves practitioner "
        "knowledge from forum discussions, traverses the knowledge graph for related insights, "
        "and synthesizes a structured report."
    )

    # Inject scroll-dismiss behavior for popovers
    inject_scroll_dismiss(pin_popovers)

    # Query input
    with st.form("query_form", clear_on_submit=False):
        query_text = st.text_input(
            "Ask a DFM question",
            placeholder="e.g. What should I consider when designing thin-walled aluminum parts for CNC machining?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "\U0001f50d Analyze",
            use_container_width=True,
            type="primary",
        )

    # Check for auto-submit from example buttons
    auto_query = st.session_state.pop("_auto_query", None)
    query = None
    if submitted and query_text.strip():
        query = query_text.strip()
    elif auto_query:
        query = auto_query

    # ── Execute pipeline ──
    if query:
        import time as _time

        # Step icons -- unique animation hint per pipeline phase
        _STEP_ICONS = {
            "decompose":  "\U0001f9e9",  # puzzle piece
            "retrieve":   "\U0001f50d",  # magnifying glass
            "traverse":   "\U0001f578\ufe0f",  # spider web
            "rank":       "\U0001f3af",  # bullseye
            "synthesize": "\U0001f9e0",  # brain
            "backfill":   "\U0001f4da",  # books
        }

        # Progress bar + animated step display + elapsed clock
        progress_bar = st.progress(0, text="Starting DFM analysis...")
        step_container = st.empty()

        def _on_progress(step, total, label, step_name=""):
            """Called by query_dfm at each pipeline step."""
            frac = step / total
            icon = _STEP_ICONS.get(step_name, "\u2699\ufe0f")
            elapsed_so_far = _time.time() - _t0
            mins, secs = divmod(int(elapsed_so_far), 60)
            clock = f"{mins}:{secs:02d}" if mins else f"{secs}s"
            progress_bar.progress(frac, text=f"{icon} Step {step}/{total}  |  {clock} elapsed")
            step_container.markdown(
                f"### {icon} Step {step} of {total}\n{label}"
            )

        _t0 = _time.time()

        # Apply analysis mode: set model and patch reasoning_effort if needed
        _agent_module.MODEL = mode_cfg["model"]
        _reasoning_effort = mode_cfg["reasoning_effort"]
        _originals = {}

        if _reasoning_effort is not None:
            # Wrap LLM-calling functions to inject reasoning_effort
            from functools import wraps

            _orig_decompose = _agent_module.decompose_query
            _orig_synth = _agent_module.synthesize_report_v2
            _orig_backfill = _agent_module.backfill_knowledge_gaps
            _originals = {"d": _orig_decompose, "s": _orig_synth, "b": _orig_backfill}

            def _patched_decompose(client, question, tracker=None):
                if len(question.split()) <= 5:
                    return [question]
                try:
                    comp = client.beta.chat.completions.parse(
                        model=_agent_module.MODEL,
                        messages=[
                            {"role": "system", "content": DECOMPOSE_PROMPT},
                            {"role": "user", "content": question},
                        ],
                        response_format=QueryDecomposition,
                        reasoning_effort=_reasoning_effort,
                    )
                    if tracker:
                        tracker.record(comp.usage)
                    r = comp.choices[0].message.parsed
                    return r.sub_aspects if r and r.sub_aspects else [question]
                except Exception:
                    return [question]

            def _patched_synth(client, query, ranked_frames, sub_aspects=None, tracker=None):
                frame_blocks = "\n\n".join(
                    _agent_module.format_frame_for_prompt(fid, info)
                    for fid, _, info in ranked_frames
                )
                sub_line = ""
                if sub_aspects and len(sub_aspects) > 1:
                    sub_line = (
                        f"\nSub-aspects investigated: {', '.join(sub_aspects)}\n"
                        "Ensure the report covers findings for each sub-aspect "
                        "where evidence exists.\n"
                    )
                user_msg = (
                    f"Designer's question: {query}\n{sub_line}\n"
                    f"Relevant knowledge frames ({len(ranked_frames)} total):\n\n"
                    f"{frame_blocks}"
                )
                comp = client.beta.chat.completions.parse(
                    model=_agent_module.MODEL,
                    messages=[
                        {"role": "system", "content": SYNTHESIS_PROMPT_V2},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format=DFMReportV2,
                    reasoning_effort=_reasoning_effort,
                )
                if tracker:
                    tracker.record(comp.usage)
                return comp.choices[0].message.parsed

            def _patched_backfill(client, report, tracker=None):
                if not report.knowledge_gaps:
                    return None
                gaps_text = "\n".join(f"- {g}" for g in report.knowledge_gaps)
                user_msg = f"Knowledge gaps to backfill:\n{gaps_text}"
                try:
                    comp = client.beta.chat.completions.parse(
                        model=_agent_module.MODEL,
                        messages=[
                            {"role": "system", "content": BACKFILL_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        response_format=BackfillOutput,
                        reasoning_effort=_reasoning_effort,
                    )
                    if tracker:
                        tracker.record(comp.usage)
                    return comp.choices[0].message.parsed
                except Exception:
                    return None

            _agent_module.decompose_query = _patched_decompose
            _agent_module.synthesize_report_v2 = _patched_synth
            _agent_module.backfill_knowledge_gaps = _patched_backfill

        try:
            report, enrichment, ranked, sub_aspects, diagnostic, backfill_result, tracker = query_dfm(
                query,
                top_k=top_k,
                max_frames=max_frames,
                max_hops=max_hops,
                decompose=decompose,
                backfill=backfill,
                verbose=False,
                G=G,
                progress_callback=_on_progress,
            )

            elapsed = _time.time() - _t0
            progress_bar.progress(1.0, text="\u2705 Done!")
            step_container.markdown(f"### \u2705 Pipeline complete in {elapsed:.0f}s\nBuilding frame details...")
            frame_details = get_frame_details(report, enrichment, G)

            token_usage = tracker.as_dict() if tracker else {}

            result = {
                "query": query,
                "report": report,
                "enrichment": enrichment,
                "ranked": ranked,
                "sub_aspects": sub_aspects,
                "diagnostic": diagnostic,
                "backfill_result": backfill_result,
                "frame_details": frame_details,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "elapsed_s": round(elapsed, 1),
                "token_usage": token_usage,
            }

            # Update persistent token tally
            _update_token_tally(token_usage)

            st.session_state.query_history.append(result)
            st.session_state.active_idx = len(st.session_state.query_history) - 1

        except Exception as e:
            progress_bar.empty()
            step_container.empty()
            st.error(f"**{type(e).__name__}:** {e}")
            st.stop()
        finally:
            # Restore original functions if patched
            if _originals:
                _agent_module.decompose_query = _originals["d"]
                _agent_module.synthesize_report_v2 = _originals["s"]
                _agent_module.backfill_knowledge_gaps = _originals["b"]

        # Clean up progress elements and rerun to show the report
        progress_bar.empty()
        step_container.empty()
        st.toast("DFM analysis complete!", icon="\u2705")
        st.rerun()

    # Display active report
    if st.session_state.active_idx >= 0 and st.session_state.query_history:
        idx = st.session_state.active_idx
        if idx < len(st.session_state.query_history):
            result = st.session_state.query_history[idx]
            st.markdown(f"---")
            st.markdown(f"**Query:** {result['query']}")
            render_report(result, G, concise=concise)

            # Scroll to report if navigating from history
            if st.session_state.pop("_scroll_to_report", False):
                import streamlit.components.v1 as _comp
                _comp.html("""
                <script>
                (function() {
                    var main = window.parent.document.querySelector('[data-testid="stMain"]');
                    if (main) main.scrollTop = 0;
                })();
                </script>
                """, height=0)
    elif st.session_state.active_idx < 0:
        # Home / welcome state
        st.markdown("---")

        # Usage stats
        tally = _load_token_tally()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Queries", f"{tally.get('queries', 0):,}")
        col2.metric("Total Tokens", f"{tally.get('total_tokens', 0):,}")
        col3.metric("Unique Visitors", f"{len(tally.get('unique_ips', []))}")

        st.markdown("### Getting Started")
        st.markdown(
            "Type a design or manufacturing question above and click **Analyze**. "
            "The agent will:\n\n"
            "1. Retrieve relevant knowledge frames from the embedded vector space\n"
            "2. Traverse the semantic knowledge graph (CAUSAL_LINK, MITIGATES, SPECIFIES, CONTRADICTS)\n"
            "3. Rank frames by relevance, credibility, and edge confidence\n"
            "4. Synthesize a structured DFM report with design rules and reasoning chains\n"
            "5. Identify knowledge gaps and backfill with textbook engineering guidance\n\n"
            "Click any **frame reference** in the report to see full details including "
            "the original source quote, author credibility, and thread URL."
        )

        st.markdown("**Example questions:**")
        examples = [
            "What should I consider when designing thin-walled aluminum parts for CNC machining?",
            "What fixturing methods work best for thin flexible parts?",
            "What are the risks of deep pocket milling in steel?",
            "How do I choose between climb and conventional milling?",
        ]
        for ex in examples:
            cached = example_cache.get(ex)
            if cached:
                # Show the analysis mode used for this cached result
                c_model = cached.get("model", "")
                c_effort = cached.get("reasoning_effort")
                if c_model == "gpt-4.1":
                    mode_tag = "\u26a1 Quick Search"
                elif c_model == "gpt-5" and c_effort == "low":
                    mode_tag = "\U0001f9e0 Efficient Thinking"
                else:
                    mode_tag = "\U0001f52c Deep Reasoning"
                label = f"{ex}  |  {mode_tag}"
            else:
                label = ex
            if st.button(label, key=f"ex_{ex[:20]}", use_container_width=True):
                if cached:
                    # Serve pre-cached result instantly
                    report = DFMReportV2.model_validate(cached["report"])
                    backfill = (
                        BackfillOutput.model_validate(cached["backfill_result"])
                        if cached.get("backfill_result")
                        else None
                    )
                    result = {
                        "query": ex,
                        "report": report,
                        "enrichment": cached.get("enrichment", {}),
                        "ranked": cached.get("ranked", []),
                        "sub_aspects": cached.get("sub_aspects", []),
                        "diagnostic": cached.get("diagnostic", {}),
                        "backfill_result": backfill,
                        "frame_details": cached.get("frame_details", []),
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "elapsed_s": cached.get("elapsed_s", 0),
                        "token_usage": cached.get("token_usage", {}),
                        "from_cache": True,
                    }
                    st.session_state.query_history.append(result)
                    st.session_state.active_idx = len(st.session_state.query_history) - 1
                    st.rerun()
                else:
                    # Fall back to live query
                    st.session_state["_auto_query"] = ex
                    st.rerun()


if __name__ == "__main__":
    main()
