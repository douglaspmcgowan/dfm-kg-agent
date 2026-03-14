"""
DFM Knowledge Graph Agent -- Retrieval-Augmented Generation (V2)

Given a design question or CAD requirement, retrieves relevant knowledge frames
from the Qdrant vector index, traverses the provenance graph for semantically
related frames, and synthesizes a structured DFM report.

Pipeline:
  1. (Optional) Decompose query into engineering sub-aspects via LLM
  2. Embed query/sub-aspects with text-embedding-3-large
  3. Retrieve top-K frames from frames_knowledge + frames_language (Qdrant)
  4. Traverse semantic edges in provenance graph (configurable hops):
       MITIGATES -> find solutions to retrieved risks
       CONTRADICTS -> surface expert disagreements
       SPECIFIES -> find scoping conditions
       CAUSAL_LINK -> find root cause explanations
  5. Rank all collected frames by relevance * credibility * edge confidence
  6. Synthesize a structured DFM report via GPT

V2 report format: organized by engineering discipline, with reasoning chains,
first-principles inference tier, and edge-aware context. Use --format v1 for
the original flat report format.

Usage:
    python dfm_agent.py "what are the risks of thin walls in aluminum?"
    python dfm_agent.py --interactive
    python dfm_agent.py "1mm wall 6061-T6" --top-k 15 --verbose --save
    python dfm_agent.py "thin wall risks" --format v1 --no-decompose
    python dfm_agent.py "aluminum machining risks" --max-hops 3 --verbose

Environment variables:
    OPENAI_API_KEY   required
    OPENAI_MODEL     default: gpt-5
    TOP_K            default: 10 (frames to retrieve from Qdrant per sub-aspect)
    MAX_FRAMES       default: 20 (max frames passed to synthesis)
"""

import json
import os
import sys
import io
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

from openai import OpenAI
from qdrant_client import QdrantClient

# Fix Windows cp1252 terminal encoding (Greek letters, Unicode arrows, etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding=sys.stdout.encoding, errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding=sys.stderr.encoding, errors="replace"
    )

from dfm_report_schema import (
    # V1
    SourcedClaim, ContradictionEntry, DFMReportOutput,
    SYNTHESIS_PROMPT_V1,
    # V2/V3
    DFMReportV2, QueryDecomposition,
    SYNTHESIS_PROMPT_V2, DECOMPOSE_PROMPT,
    # V3: backfill
    BackfillOutput, BACKFILL_PROMPT,
)

BASE_DIR     = Path(__file__).parent
GRAPH_PICKLE = BASE_DIR / "provenance_graph.gpickle"
QDRANT_PATH  = str(BASE_DIR / "qdrant_data")
QDRANT_URL   = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
REPORTS_DIR  = BASE_DIR / "dfm_reports"

MODEL           = os.getenv("OPENAI_MODEL", "gpt-5")
EMBEDDING_MODEL = "text-embedding-3-large"
TOP_K_RETRIEVE  = int(os.getenv("TOP_K", "10"))
MAX_FRAMES_SYNTH = int(os.getenv("MAX_FRAMES", "20"))


# ─────────────────────────────────────────────
# TOKEN TRACKING
# ─────────────────────────────────────────────

class TokenTracker:
    """Accumulates prompt / completion token counts across API calls."""
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.embedding_tokens = 0
        self.calls = 0

    def record(self, usage):
        """Record token usage from an OpenAI ChatCompletion.usage object."""
        if usage is None:
            return
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.calls += 1

    def record_embedding(self, usage):
        """Record token usage from an OpenAI Embedding response."""
        if usage is None:
            return
        self.embedding_tokens += getattr(usage, "total_tokens", 0) or 0
        self.calls += 1

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens + self.embedding_tokens

    def as_dict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.calls,
        }

CREDIBILITY_WEIGHTS = {"high": 1.5, "medium": 1.1, "low": 1.0}
SEMANTIC_EDGE_TYPES = {"MITIGATES", "CONTRADICTS", "SPECIFIES", "CAUSAL_LINK"}

# V2 scoring weights
EDGE_CONFIDENCE_WEIGHTS = {"high": 1.3, "medium": 1.0, "low": 0.7}
FRAME_TYPE_BONUS = {"principle": 1.2, "risk": 1.1}
EPISTEMIC_WEIGHTS = {
    "absolute_authoritative": 1.3,
    "direct_neutral": 1.1,
    "hedged_emphatic": 1.0,
    "tentative_subjective": 0.9,
    "narrative_implied": 0.9,
}


# ─────────────────────────────────────────────
# GRAPH UTILITIES
# ─────────────────────────────────────────────

def get_frame_context(G, frame_id):
    """
    Walk the graph to get author, credibility tier, source URL, and post metadata
    for a frame node. Returns a dict with these fields.
    """
    author = "unknown"
    credibility = "low"
    reputation = -1
    source_url = ""
    is_accepted = False
    post_score = 0
    se_tags = ""

    for post_id in G.predecessors(frame_id):
        if G.nodes[post_id].get("node_type") != "Post":
            continue
        post_data = G.nodes[post_id]
        post_score = post_data.get("post_score", 0) or 0
        se_tags = post_data.get("se_tags", "") or ""
        is_accepted = post_data.get("is_accepted_answer", "") == "True"

        for auth_node in G.successors(post_id):
            if G.nodes[auth_node].get("node_type") != "Author":
                continue
            auth_data = G.nodes[auth_node]
            author = auth_data.get("author_username", "unknown")
            credibility = auth_data.get("credibility_tier", "low")
            reputation = auth_data.get("reputation", -1)
            break

        for thread_id in G.predecessors(post_id):
            if G.nodes[thread_id].get("node_type") == "Thread":
                source_url = thread_id
                break
        break

    return {
        "author":      author,
        "credibility": credibility,
        "reputation":  reputation,
        "source_url":  source_url,
        "is_accepted": is_accepted,
        "post_score":  post_score,
        "se_tags":     se_tags,
    }


# ─────────────────────────────────────────────
# QUERY DECOMPOSITION (Phase 5)
# ─────────────────────────────────────────────

def decompose_query(client, question, tracker=None):
    """
    Use the LLM to break a broad design question into engineering sub-aspects.
    Returns a list of sub-aspect strings. Falls back to [question] on failure.
    Skips decomposition for queries <= 5 words.
    """
    if len(question.split()) <= 5:
        return [question]

    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": DECOMPOSE_PROMPT},
                {"role": "user",   "content": question},
            ],
            response_format=QueryDecomposition,
        )
        if tracker:
            tracker.record(completion.usage)
        result = completion.choices[0].message.parsed
        if result and result.sub_aspects:
            return result.sub_aspects
    except Exception as e:
        print(f"  Decomposition failed ({type(e).__name__}), using original query")

    return [question]


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

def embed_query(client, text, tracker=None):
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    if tracker:
        tracker.record_embedding(response.usage)
    return response.data[0].embedding


def retrieve_frames(qdrant, query_vector, top_k):
    """
    Search frames_knowledge and frames_language, merge by max score.
    Returns list of (frame_id, score) sorted descending.
    """
    k_results = qdrant.query_points(
        collection_name="frames_knowledge",
        query=query_vector,
        limit=top_k,
        with_payload=["frame_id"],
    ).points

    l_results = qdrant.query_points(
        collection_name="frames_language",
        query=query_vector,
        limit=top_k,
        with_payload=["frame_id"],
    ).points

    scores = {}
    for point in k_results:
        fid = point.payload.get("frame_id", "")
        if fid:
            scores[fid] = point.score

    for point in l_results:
        fid = point.payload.get("frame_id", "")
        if fid:
            scores[fid] = max(scores.get(fid, 0.0), point.score)

    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve_with_decomposition(client, qdrant, question, sub_aspects, top_k, tracker=None):
    """
    Retrieve frames for each sub-aspect + original question, merge by max score.
    Returns: (merged dict {fid: score}, aspect_hits dict {fid: [sub_aspects]})
    """
    all_scores = {}
    aspect_hits = defaultdict(list)

    queries = [question] + [a for a in sub_aspects if a != question]

    for aspect in queries:
        query_vector = embed_query(client, aspect, tracker=tracker)
        retrieved = retrieve_frames(qdrant, query_vector, top_k)
        for fid, score in retrieved:
            if fid not in all_scores or score > all_scores[fid]:
                all_scores[fid] = score
            aspect_hits[fid].append(aspect)

    return all_scores, dict(aspect_hits)


# ─────────────────────────────────────────────
# GRAPH TRAVERSAL (Phase 2 -- enhanced)
# ─────────────────────────────────────────────

def traverse_graph(G, seed_frame_ids, max_hops=2):
    """
    Starting from seed frames, follow semantic edges to find related frames.

    Enhancements over v1:
    - Multi-hop (configurable, default 2)
    - Bidirectional: follows both successors and predecessors
    - Captures edge metadata: confidence, reasoning
    - Tracks full path chains
    - Collects all semantic edge types between a pair

    Returns dict: frame_id -> {
        distance: int,
        edges: [{edge_type, from_frame, confidence, reasoning}],
        path: [(frame_id, edge_type), ...]
    }
    """
    visited = {}
    for fid in seed_frame_ids:
        if G.has_node(fid):
            visited[fid] = {"distance": 0, "edges": [], "path": []}

    frontier = [fid for fid in seed_frame_ids if G.has_node(fid)]

    for hop in range(1, max_hops + 1):
        next_frontier = []
        for fid in frontier:
            # Collect candidates from both directions
            neighbor_edges = defaultdict(list)

            # Outgoing: fid -> neighbor
            for neighbor in G.successors(fid):
                if G.nodes.get(neighbor, {}).get("node_type") != "Frame":
                    continue
                if neighbor in visited:
                    continue
                for key in G[fid][neighbor]:
                    et = G.edges[fid, neighbor, key].get("edge_type", "")
                    if et in SEMANTIC_EDGE_TYPES:
                        neighbor_edges[neighbor].append({
                            "edge_type":  et,
                            "from_frame": fid,
                            "confidence": G.edges[fid, neighbor, key].get("confidence", ""),
                            "reasoning":  G.edges[fid, neighbor, key].get("reasoning", ""),
                        })

            # Incoming: neighbor -> fid (find frames that point TO fid)
            for neighbor in G.predecessors(fid):
                if G.nodes.get(neighbor, {}).get("node_type") != "Frame":
                    continue
                if neighbor in visited:
                    continue
                for key in G[neighbor][fid]:
                    et = G.edges[neighbor, fid, key].get("edge_type", "")
                    if et in SEMANTIC_EDGE_TYPES:
                        neighbor_edges[neighbor].append({
                            "edge_type":  et,
                            "from_frame": fid,
                            "confidence": G.edges[neighbor, fid, key].get("confidence", ""),
                            "reasoning":  G.edges[neighbor, fid, key].get("reasoning", ""),
                        })

            for neighbor, edge_list in neighbor_edges.items():
                if neighbor in visited:
                    continue
                parent_path = visited[fid]["path"]
                new_path = parent_path + [(fid, edge_list[0]["edge_type"])]
                visited[neighbor] = {
                    "distance":  hop,
                    "edges":     edge_list,
                    "path":      new_path,
                }
                next_frontier.append(neighbor)

        frontier = next_frontier

    return visited


# ─────────────────────────────────────────────
# RANKING (Phase 3 -- enhanced)
# ─────────────────────────────────────────────

def rank_frames(traversal, G, seed_scores, max_frames):
    """
    Score each frame with multiple factors:
      qdrant_score * credibility * distance * accepted * edge_confidence
      * frame_type * epistemic * post_score

    Returns list of (frame_id, final_score, context_dict) sorted descending.
    """
    scored = []

    for fid, trav_info in traversal.items():
        if not G.has_node(fid):
            continue

        node_data = G.nodes[fid]
        ctx = get_frame_context(G, fid)

        base_score = seed_scores.get(fid, 0.5)
        cred_weight = CREDIBILITY_WEIGHTS.get(ctx["credibility"], 1.0)
        distance_factor = 1.0 / (trav_info["distance"] + 1)
        accepted_bonus = 1.2 if ctx["is_accepted"] else 1.0

        # Edge confidence: max confidence from edges (seeds have no edges)
        edge_conf = 1.0
        if trav_info["edges"]:
            conf_values = [EDGE_CONFIDENCE_WEIGHTS.get(e.get("confidence", ""), 1.0)
                           for e in trav_info["edges"]]
            edge_conf = max(conf_values)

        # Frame type bonus (principles and risks weighted higher)
        ft_bonus = FRAME_TYPE_BONUS.get(node_data.get("frame_type", ""), 1.0)

        # Epistemic stance weight
        ep_weight = EPISTEMIC_WEIGHTS.get(node_data.get("epistemic_stance", ""), 1.0)

        # Post score bonus (capped at +0.2)
        ps = ctx.get("post_score", 0) or 0
        ps_bonus = 1.0 + min(max(ps, 0), 20) * 0.01

        final_score = (base_score * cred_weight * distance_factor * accepted_bonus
                       * edge_conf * ft_bonus * ep_weight * ps_bonus)

        scored.append((fid, final_score, {
            "node":      node_data,
            "context":   ctx,
            "traversal": trav_info,
        }))

    scored.sort(key=lambda x: -x[1])
    return scored[:max_frames]


# ─────────────────────────────────────────────
# CHAIN-PULL (V3 -- ensure reasoning chains fire)
# ─────────────────────────────────────────────

CHAIN_PULL_PRIORITY = {"CAUSAL_LINK": 3, "MITIGATES": 2, "SPECIFIES": 1, "CONTRADICTS": 0}
CONF_ORDER = {"high": 3, "medium": 2, "low": 1, "": 0}


def pull_chain_partners(ranked, traversal, G, seed_scores, max_pull=5):
    """
    After ranking selects top-N, scan their semantic edges for connected
    frames that exist in the traversal but didn't make the ranked cut.
    Pull those frames in as chain-completion evidence so reasoning chains fire.

    Priority: CAUSAL_LINK > MITIGATES > SPECIFIES > CONTRADICTS.
    Returns: (extended_ranked, pull_set) where pull_set is the set of
    frame_ids that were chain-pulled.
    """
    ranked_ids = {fid for fid, _, _ in ranked}
    traversal_ids = set(traversal.keys())
    candidates = []  # (priority, conf_order, neighbor, edge_type, ranked_fid)

    for fid, _, _ in ranked:
        if not G.has_node(fid):
            continue
        # Check outgoing edges: fid -> neighbor
        for neighbor in G.successors(fid):
            if neighbor in ranked_ids or neighbor not in traversal_ids:
                continue
            if G.nodes.get(neighbor, {}).get("node_type") != "Frame":
                continue
            for key in G[fid][neighbor]:
                et = G.edges[fid, neighbor, key].get("edge_type", "")
                if et in SEMANTIC_EDGE_TYPES:
                    conf = G.edges[fid, neighbor, key].get("confidence", "")
                    pri = CHAIN_PULL_PRIORITY.get(et, 0)
                    candidates.append((pri, CONF_ORDER.get(conf, 0), neighbor, et, fid, conf))

        # Check incoming edges: neighbor -> fid
        for neighbor in G.predecessors(fid):
            if neighbor in ranked_ids or neighbor not in traversal_ids:
                continue
            if G.nodes.get(neighbor, {}).get("node_type") != "Frame":
                continue
            for key in G[neighbor][fid]:
                et = G.edges[neighbor, fid, key].get("edge_type", "")
                if et in SEMANTIC_EDGE_TYPES:
                    conf = G.edges[neighbor, fid, key].get("confidence", "")
                    pri = CHAIN_PULL_PRIORITY.get(et, 0)
                    candidates.append((pri, CONF_ORDER.get(conf, 0), neighbor, et, fid, conf))

    # Sort by priority desc, confidence desc; deduplicate
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    seen = set()
    pulled = []
    for pri, conf_ord, neighbor, et, ranked_fid, conf_str in candidates:
        if neighbor in seen:
            continue
        seen.add(neighbor)
        pulled.append((neighbor, et, ranked_fid, conf_str))
        if len(pulled) >= max_pull:
            break

    # Build ranked entries for pulled frames
    pull_set = set()
    for neighbor, et, ranked_fid, conf_str in pulled:
        pull_set.add(neighbor)
        node_data = G.nodes[neighbor]
        ctx = get_frame_context(G, neighbor)
        trav_info = traversal.get(neighbor, {"distance": 1, "edges": [], "path": []})
        # Mark as chain-pull
        trav_info = dict(trav_info)  # copy
        trav_info["is_chain_pull"] = True
        trav_info["pull_reason"] = f"{et} partner of {ranked_fid} (confidence: {conf_str})"
        # Add the connecting edge info if not already present
        if not any(e.get("from_frame") == ranked_fid and e.get("edge_type") == et
                   for e in trav_info.get("edges", [])):
            trav_info.setdefault("edges", []).append({
                "edge_type": et, "from_frame": ranked_fid,
                "confidence": conf_str, "reasoning": "",
            })

        base_score = seed_scores.get(neighbor, 0.3)
        ranked.append((neighbor, base_score, {
            "node": node_data,
            "context": ctx,
            "traversal": trav_info,
        }))

    return ranked, pull_set


# ─────────────────────────────────────────────
# RELEVANCE DIAGNOSTIC (V3)
# ─────────────────────────────────────────────

def compute_relevance_diagnostic(seed_scores, top_k, threshold=0.7):
    """
    Compute a relevance diagnostic based on seed frame similarity scores.
    Returns a dict with grade, counts, and average similarity.
    """
    sorted_scores = sorted(seed_scores.values(), reverse=True)[:top_k]
    if not sorted_scores:
        return {"grade": "No data", "relevant": 0, "total": 0,
                "avg_sim": 0.0, "detail": "0/0 frames relevant"}

    relevant = sum(1 for s in sorted_scores if s >= threshold)
    total = len(sorted_scores)
    avg_sim = sum(sorted_scores) / total
    pct = relevant / total

    if pct >= 0.7:
        grade = "Strong coverage"
    elif pct >= 0.4:
        grade = "Moderate coverage"
    elif pct >= 0.2:
        grade = "Sparse coverage"
    else:
        grade = "Weak coverage"

    return {
        "grade":    grade,
        "relevant": relevant,
        "total":    total,
        "avg_sim":  round(avg_sim, 3),
        "detail":   f"{relevant}/{total} frames above {threshold} similarity (avg: {avg_sim:.3f})",
    }


# ─────────────────────────────────────────────
# FRAME FORMATTING (Phase 4 -- edge-aware)
# ─────────────────────────────────────────────

def format_frame_for_prompt(fid, score_info):
    """Format one frame with full metadata and edge context for the synthesis prompt."""
    node = score_info["node"]
    ctx  = score_info["context"]
    trav = score_info["traversal"]

    # Edge relationship lines
    edge_lines = []
    for edge in trav.get("edges", []):
        et = edge.get("edge_type", "")
        from_f = edge.get("from_frame", "")
        conf = edge.get("confidence", "")
        reasoning = edge.get("reasoning", "")
        if et and from_f:
            reason_short = reasoning[:150] + "..." if len(reasoning) > 150 else reasoning
            edge_lines.append(
                f"  Edge       : {et} from {from_f} (confidence: {conf})\n"
                f"  Edge reason: {reason_short}"
            )

    # Path line for multi-hop frames
    path = trav.get("path", [])
    path_line = ""
    if path:
        path_str = " -> ".join(f"{pfid} [{pet}]" for pfid, pet in path)
        path_line = f"\n  Path       : {path_str} -> {fid}"

    rep_str = f", rep: {ctx['reputation']}" if ctx['reputation'] != -1 else ""
    accepted_str = " [ACCEPTED ANSWER]" if ctx["is_accepted"] else ""
    ps = ctx.get("post_score", 0) or 0
    ps_str = f", score: {ps}" if ps else ""

    # Optional fields (only shown if non-empty)
    tags = ctx.get("se_tags", "") or ""
    tags_line = f"\n  Tags       : {tags}" if tags else ""

    ex_ctx = node.get("example_context", "") or ""
    ex_ctx_line = f"\n  Context    : {ex_ctx}" if ex_ctx else ""

    stance = node.get("epistemic_stance", "") or ""

    # Chain-pull marker
    is_chain_pull = trav.get("is_chain_pull", False)
    pull_reason = trav.get("pull_reason", "")
    chain_tag = " [CHAIN-COMPLETION]" if is_chain_pull else ""
    pull_line = f"\n  Pull reason: {pull_reason}" if pull_reason else ""

    edge_block = "\n".join(edge_lines) if edge_lines else ""
    if edge_block:
        edge_block = "\n" + edge_block

    return (
        f"[{fid}]{accepted_str}{chain_tag}\n"
        f"  Type       : {node.get('frame_type','')} | "
        f"Scope: {node.get('scope','')} | "
        f"Applicability: {node.get('applicability','')}\n"
        f"  Subject    : {node.get('subject','')}\n"
        f"  Stance     : {stance}\n"
        f"  Author     : {ctx['author']} (credibility: {ctx['credibility']}{rep_str}{ps_str})\n"
        f"  Quote      : \"{node.get('source_quote','')}\"\n"
        f"  Main point : {node.get('main_point','')}"
        f"{ex_ctx_line}"
        f"{tags_line}"
        f"{edge_block}"
        f"{path_line}"
        f"{pull_line}"
    )


# ─────────────────────────────────────────────
# SYNTHESIS
# ─────────────────────────────────────────────

def synthesize_report_v1(client, query, ranked_frames, tracker=None):
    """V1: flat structured report."""
    frame_blocks = "\n\n".join(
        format_frame_for_prompt(fid, info) for fid, _, info in ranked_frames
    )
    user_msg = (
        f"Designer's question: {query}\n\n"
        f"Relevant knowledge frames ({len(ranked_frames)} total):\n\n"
        f"{frame_blocks}"
    )
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_PROMPT_V1},
            {"role": "user",   "content": user_msg},
        ],
        response_format=DFMReportOutput,
    )
    if tracker:
        tracker.record(completion.usage)
    return completion.choices[0].message.parsed


def synthesize_report_v2(client, query, ranked_frames, sub_aspects=None, tracker=None):
    """V2: discipline-organized report with reasoning chains."""
    frame_blocks = "\n\n".join(
        format_frame_for_prompt(fid, info) for fid, _, info in ranked_frames
    )

    sub_aspect_line = ""
    if sub_aspects and len(sub_aspects) > 1:
        sub_aspect_line = (
            f"\nSub-aspects investigated: {', '.join(sub_aspects)}\n"
            "Ensure the report covers findings for each sub-aspect where evidence exists.\n"
        )

    user_msg = (
        f"Designer's question: {query}\n"
        f"{sub_aspect_line}\n"
        f"Relevant knowledge frames ({len(ranked_frames)} total):\n\n"
        f"{frame_blocks}"
    )

    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_PROMPT_V2},
            {"role": "user",   "content": user_msg},
        ],
        response_format=DFMReportV2,
    )
    if tracker:
        tracker.record(completion.usage)
    return completion.choices[0].message.parsed


# ─────────────────────────────────────────────
# KNOWLEDGE GAP BACKFILL (V3)
# ─────────────────────────────────────────────

def backfill_knowledge_gaps(client, report, tracker=None):
    """
    Optional second LLM call: for each knowledge gap in the report,
    generate a brief textbook-level answer. Returns BackfillOutput or None.
    """
    if not report.knowledge_gaps:
        return None

    gaps_text = "\n".join(f"- {gap}" for gap in report.knowledge_gaps)
    user_msg = f"Knowledge gaps to backfill:\n{gaps_text}"

    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": BACKFILL_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            response_format=BackfillOutput,
        )
        if tracker:
            tracker.record(completion.usage)
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"  Backfill failed ({type(e).__name__}: {e})")
        return None


# ─────────────────────────────────────────────
# ENRICHMENT (add author/URL to report claims)
# ─────────────────────────────────────────────

def enrich_claim(claim, G):
    """Add author, credibility, source_url to a SourcedClaim by looking up the frame_id."""
    fid = claim.frame_id
    if not G.has_node(fid):
        return {"author": "unknown", "credibility": "low", "source_url": ""}
    ctx = get_frame_context(G, fid)
    return {
        "author":      ctx["author"],
        "credibility": ctx["credibility"],
        "source_url":  ctx["source_url"],
    }


def build_enrichment_v1(report, G):
    """Build enrichment dict for V1 report."""
    enrichment = {}
    for claim in report.risks + report.recommended_practices:
        if claim.frame_id not in enrichment:
            enrichment[claim.frame_id] = enrich_claim(claim, G)
    for contra in report.contradictions:
        for fid in (contra.frame_id_a, contra.frame_id_b):
            if fid not in enrichment:
                enrichment[fid] = get_frame_context(G, fid) if G.has_node(fid) else {}
    return enrichment


def build_enrichment_v2(report, G):
    """Build enrichment dict for V2 report."""
    enrichment = {}
    for section in report.disciplines:
        for claim in section.evidence:
            if claim.frame_id not in enrichment:
                enrichment[claim.frame_id] = enrich_claim(claim, G)
        for chain in section.reasoning_chains:
            for cf in chain.frames:
                if cf.frame_id not in enrichment:
                    fid = cf.frame_id
                    if G.has_node(fid):
                        enrichment[fid] = get_frame_context(G, fid)
                    else:
                        enrichment[fid] = {"author": "unknown", "credibility": "low", "source_url": ""}
    for rule in report.design_rules:
        if rule.frame_id not in enrichment:
            enrichment[rule.frame_id] = enrich_claim(rule, G)
    for contra in report.contradictions:
        for fid in (contra.frame_id_a, contra.frame_id_b):
            if fid not in enrichment:
                enrichment[fid] = get_frame_context(G, fid) if G.has_node(fid) else {}
    return enrichment


# ─────────────────────────────────────────────
# MAIN QUERY FUNCTION
# ─────────────────────────────────────────────

def query_dfm(question, top_k=None, max_frames=None, max_hops=2,
              decompose=None, report_format="v2", backfill=False, verbose=False,
              G=None, progress_callback=None):
    """
    Main entry point. Returns (report, enrichment, ranked, sub_aspects, diagnostic, backfill_result, tracker).
    Loads graph and Qdrant fresh per call (suitable for CLI; cache G for server use).
    Pass G= to reuse a pre-loaded graph (e.g. from Streamlit @st.cache_resource).
    Pass progress_callback= to receive step updates: callback(step, total, label, step_name).
    tracker is a TokenTracker with prompt_tokens, completion_tokens, embedding_tokens, total_tokens.
    """
    top_k      = top_k or TOP_K_RETRIEVE
    max_frames = max_frames or MAX_FRAMES_SYNTH

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key, timeout=300.0)
    tracker = TokenTracker()

    # Load graph (skip if caller provided one)
    if G is None:
        if not GRAPH_PICKLE.exists():
            raise FileNotFoundError(f"{GRAPH_PICKLE} not found. Run build_graph.py first.")
        with open(GRAPH_PICKLE, "rb") as f:
            G = pickle.load(f)

    if verbose:
        n_frames = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "Frame")
        n_sem = sum(
            1 for _, _, d in G.edges(data=True)
            if d.get("edge_type") in SEMANTIC_EDGE_TYPES
        )
        print(f"Graph: {n_frames} frames, {n_sem} semantic edges")

    # Load Qdrant -- prefer cloud URL, fall back to local path
    _qurl = QDRANT_URL
    _qkey = QDRANT_API_KEY
    try:
        import streamlit as _st
        _qurl = _qurl or _st.secrets.get("QDRANT_URL", "")
        _qkey = _qkey or _st.secrets.get("QDRANT_API_KEY", "")
    except Exception:
        pass
    if _qurl:
        qdrant = QdrantClient(url=_qurl, api_key=_qkey or None)
    else:
        if not os.path.exists(QDRANT_PATH):
            raise FileNotFoundError(f"{QDRANT_PATH} not found. Run build_embeddings.py first.")
        qdrant = QdrantClient(path=QDRANT_PATH)

    # Build dynamic step list based on toggles
    should_decompose = decompose if decompose is not None else (len(question.split()) > 5)
    _steps = []
    if should_decompose:
        _steps.append("decompose")
    _steps.extend(["retrieve", "traverse", "rank", "synthesize"])
    if backfill:
        _steps.append("backfill")
    _total_steps = len(_steps)
    _step_idx = 0

    def _progress(step_name, label):
        nonlocal _step_idx
        try:
            _step_idx = _steps.index(step_name) + 1
        except ValueError:
            _step_idx += 1
        if progress_callback:
            progress_callback(_step_idx, _total_steps, label, step_name)

    # Step: Decompose (optional)
    sub_aspects = None
    if should_decompose:
        _progress("decompose", "Decomposing query into sub-aspects")
        if verbose:
            print(f"Decomposing query...")
        sub_aspects = decompose_query(client, question, tracker=tracker)
        if verbose:
            print(f"  Sub-aspects: {sub_aspects}")

    # Step: Retrieve
    _progress("retrieve", "Retrieving frames from vector index")
    if sub_aspects and len(sub_aspects) > 1:
        if verbose:
            print(f"Retrieving top-{top_k} frames per sub-aspect ({len(sub_aspects) + 1} queries)...")
        seed_scores, aspect_hits = retrieve_with_decomposition(
            client, qdrant, question, sub_aspects, top_k, tracker=tracker
        )
        seed_ids = sorted(seed_scores.keys(), key=lambda f: -seed_scores[f])
    else:
        if verbose:
            print(f"Embedding query...")
        query_vector = embed_query(client, question, tracker=tracker)
        if verbose:
            print(f"Retrieving top-{top_k} frames from Qdrant...")
        retrieved = retrieve_frames(qdrant, query_vector, top_k)
        seed_ids    = [fid for fid, _ in retrieved]
        seed_scores = {fid: score for fid, score in retrieved}

    qdrant.close()

    if verbose:
        print(f"Retrieved {len(seed_ids)} unique seed frames")

    # Step 2b: Relevance diagnostic
    diagnostic = compute_relevance_diagnostic(seed_scores, top_k)
    if verbose:
        print(f"Relevance: {diagnostic['grade']} -- {diagnostic['detail']}")

    # Step: Traverse
    _progress("traverse", f"Traversing semantic edges ({max_hops} hops)")
    if verbose:
        print(f"Traversing semantic edges ({max_hops} hops, bidirectional)...")
    traversal = traverse_graph(G, seed_ids, max_hops=max_hops)

    n_traversed = sum(1 for v in traversal.values() if v["distance"] > 0)
    if verbose:
        print(f"  Seeds: {len(seed_ids)}, Traversed: {n_traversed} additional frames")

    # Step: Rank
    _progress("rank", "Ranking frames by relevance and credibility")
    ranked = rank_frames(traversal, G, seed_scores, max_frames)

    # Step 4b: Chain-pull (V3) -- pull in connected partners for reasoning chains
    ranked, pull_set = pull_chain_partners(ranked, traversal, G, seed_scores)
    if verbose:
        print(f"Top {len(ranked)} frames for synthesis ({len(pull_set)} chain-pulled):")
        for fid, score, info in ranked[:5]:
            node = info["node"]
            trav = info["traversal"]
            edge_str = "seed"
            if trav.get("is_chain_pull"):
                edge_str = "CHAIN-PULL"
            elif trav["edges"]:
                edge_str = trav["edges"][0].get("edge_type", "?")
            print(f"  {fid} | {node.get('frame_type','')} | "
                  f"{node.get('subject','')[:40]} | "
                  f"score={score:.3f} | via={edge_str}")
        if len(ranked) > 5:
            print(f"  ... and {len(ranked)-5} more")

    # Step: Synthesize (this is the slow GPT call)
    _progress("synthesize", f"Synthesizing report with {MODEL}")
    if verbose:
        print(f"Synthesizing {report_format} report with {MODEL}...")

    if report_format == "v2":
        report = synthesize_report_v2(client, question, ranked, sub_aspects, tracker=tracker)
        enrichment = build_enrichment_v2(report, G)
    else:
        report = synthesize_report_v1(client, question, ranked, tracker=tracker)
        enrichment = build_enrichment_v1(report, G)

    # Step: Backfill knowledge gaps (optional, V3)
    if backfill:
        _progress("backfill", "Backfilling knowledge gaps")
    backfill_result = None
    if backfill and report_format == "v2" and report.knowledge_gaps:
        if verbose:
            print(f"Backfilling {len(report.knowledge_gaps)} knowledge gaps...")
        backfill_result = backfill_knowledge_gaps(client, report, tracker=tracker)

    return report, enrichment, ranked, sub_aspects, diagnostic, backfill_result, tracker


# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────

def print_report_v1(report, enrichment, query):
    """Display V1 flat report."""
    width = 70
    print(f"\n{'='*width}")
    print(f"DFM REPORT")
    print(f"{'='*width}")
    print(f"Query: {query}\n")

    print(f"SUMMARY\n{'-'*width}")
    print(f"{report.summary}\n")

    if report.risks:
        print(f"RISKS ({len(report.risks)})\n{'-'*width}")
        for i, risk in enumerate(report.risks, 1):
            enc = enrichment.get(risk.frame_id, {})
            author = enc.get("author", "unknown")
            cred   = enc.get("credibility", "")
            url    = enc.get("source_url", "")
            sq = risk.source_quote
            sq_display = sq[:120] + "..." if len(sq) > 120 else sq
            print(f"{i}. {risk.claim}")
            print(f"   Source : {author} ({cred}) | {risk.frame_id}")
            print(f"   Quote  : \"{sq_display}\"")
            if url:
                print(f"   URL    : {url}")
            print()

    if report.recommended_practices:
        print(f"RECOMMENDED PRACTICES ({len(report.recommended_practices)})\n{'-'*width}")
        for i, prac in enumerate(report.recommended_practices, 1):
            enc = enrichment.get(prac.frame_id, {})
            author = enc.get("author", "unknown")
            cred   = enc.get("credibility", "")
            url    = enc.get("source_url", "")
            sq = prac.source_quote
            sq_display = sq[:120] + "..." if len(sq) > 120 else sq
            print(f"{i}. {prac.claim}")
            print(f"   Source : {author} ({cred}) | {prac.frame_id}")
            print(f"   Quote  : \"{sq_display}\"")
            if url:
                print(f"   URL    : {url}")
            print()

    if report.contradictions:
        print(f"CONTRADICTIONS ({len(report.contradictions)})\n{'-'*width}")
        for i, contra in enumerate(report.contradictions, 1):
            print(f"{i}. {contra.summary}")
            print(f"   Position A [{contra.frame_id_a}]: {contra.position_a}")
            print(f"   Position B [{contra.frame_id_b}]: {contra.position_b}")
            print()

    if report.knowledge_gaps:
        print(f"KNOWLEDGE GAPS\n{'-'*width}")
        print(f"{report.knowledge_gaps}\n")

    print(f"{'='*width}")


def print_report_v2(report, enrichment, query, sub_aspects=None,
                    diagnostic=None, backfill_result=None):
    """Display V2/V3 discipline-organized report (full detail)."""
    width = 70
    print(f"\n{'='*width}")
    print(f"DFM REPORT (V2)")
    print(f"{'='*width}")
    print(f"Query: {query}")
    if sub_aspects and len(sub_aspects) > 1:
        print(f"Sub-aspects: {', '.join(sub_aspects)}")
    if diagnostic:
        print(f"Relevance: {diagnostic['grade']} -- {diagnostic['detail']}")
    print()

    print(f"SUMMARY\n{'-'*width}")
    print(f"{report.summary}\n")

    for section in report.disciplines:
        print(f"DISCIPLINE: {section.discipline.upper()}\n{'-'*width}")

        if section.first_principles:
            print(f"[Engineering Rationale]")
            print(f"{section.first_principles}\n")

        if hasattr(section, 'formulas') and section.formulas:
            print(f"Formulas ({len(section.formulas)}):")
            for fm in section.formulas:
                print(f"  {fm.formula_name}: {fm.expression}")
                print(f"    Variables: {fm.variables}")
                print(f"    Relevance: {fm.relevance}")
            print()

        if section.evidence:
            print(f"Evidence ({len(section.evidence)}):")
            for i, ev in enumerate(section.evidence, 1):
                enc = enrichment.get(ev.frame_id, {})
                author = enc.get("author", "unknown")
                cred   = enc.get("credibility", "")
                url    = enc.get("source_url", "")
                sq = ev.source_quote
                sq_display = sq[:120] + "..." if len(sq) > 120 else sq
                print(f"  {i}. {ev.claim}")
                print(f"     Source : {author} ({cred}) | {ev.frame_id}")
                print(f"     Quote  : \"{sq_display}\"")
                if url:
                    print(f"     URL    : {url}")
            print()

        if section.reasoning_chains:
            print(f"Reasoning Chains ({len(section.reasoning_chains)}):")
            for chain in section.reasoning_chains:
                print(f"  Chain: {chain.chain_label}")
                for cf in chain.frames:
                    role = cf.role_in_chain
                    edge_str = ""
                    if cf.edge_from and cf.edge_type:
                        edge_str = f" <-- {cf.edge_type} from {cf.edge_from}"
                        if cf.edge_confidence:
                            edge_str += f" ({cf.edge_confidence})"
                    enc = enrichment.get(cf.frame_id, {})
                    author = enc.get("author", "unknown")
                    print(f"    [{role}] {cf.frame_id} ({author}){edge_str}")
                    print(f"      {cf.claim}")
            print()

    if report.contradictions:
        print(f"CONTRADICTIONS ({len(report.contradictions)})\n{'-'*width}")
        for i, c in enumerate(report.contradictions, 1):
            print(f"{i}. {c.summary}")
            print(f"   Position A [{c.frame_id_a}]: {c.position_a}")
            print(f"   Position B [{c.frame_id_b}]: {c.position_b}")
            if c.edge_reasoning:
                print(f"   Edge reasoning: {c.edge_reasoning}")
            print()

    if report.design_rules:
        print(f"DESIGN RULES ({len(report.design_rules)})\n{'-'*width}")
        for i, rule in enumerate(report.design_rules, 1):
            enc = enrichment.get(rule.frame_id, {})
            author = enc.get("author", "unknown")
            print(f"{i}. {rule.claim}")
            print(f"   Source : {author} | {rule.frame_id}")
        print()

    if report.knowledge_gaps:
        print(f"KNOWLEDGE GAPS\n{'-'*width}")
        for gap in report.knowledge_gaps:
            print(f"  - {gap}")
        print()

    if backfill_result and backfill_result.backfills:
        print(f"TEXTBOOK BACKFILL (general engineering -- not sourced from KG)\n{'-'*width}")
        for bf in backfill_result.backfills:
            print(f"  Gap: {bf.gap}")
            print(f"  Answer: {bf.backfill_answer}")
            print(f"  Confidence: {bf.confidence_note}")
            if bf.relevant_standards:
                print(f"  Standards: {bf.relevant_standards}")
            print()

    print(f"{'='*width}")


def print_report_v2_concise(report, query, sub_aspects=None,
                            diagnostic=None, backfill_result=None):
    """Display concise V2 report with inline frame ID references.

    Keeps frame IDs as [frame_id] references after each claim so the user
    can trace sources, but hides quotes, URLs, and author details.
    Use 'unpack' in interactive mode to expand cited frames.
    """
    width = 70
    print(f"\n{'='*width}")
    print(f"DFM REPORT")
    print(f"{'='*width}")
    print(f"Query: {query}")
    if sub_aspects and len(sub_aspects) > 1:
        print(f"Sub-aspects: {', '.join(sub_aspects)}")
    if diagnostic:
        print(f"Relevance: {diagnostic['grade']} -- {diagnostic['detail']}")
    print()

    print(f"SUMMARY\n{'-'*width}")
    print(f"{report.summary}\n")

    for section in report.disciplines:
        label = section.discipline.replace("_", " ").upper()
        print(f"{label}\n{'-'*width}")

        if section.first_principles:
            print(f"{section.first_principles}\n")

        if hasattr(section, 'formulas') and section.formulas:
            for fm in section.formulas:
                print(f"  {fm.formula_name}: {fm.expression}")
                print(f"    {fm.relevance}")
            print()

        if section.evidence:
            print("Key findings:")
            for i, ev in enumerate(section.evidence, 1):
                print(f"  {i}. {ev.claim} [{ev.frame_id}]")
            print()

        if section.reasoning_chains:
            for chain in section.reasoning_chains:
                print(f"  Chain: {chain.chain_label}")
                for cf in chain.frames:
                    print(f"    [{cf.role_in_chain}] {cf.claim} [{cf.frame_id}]")
            print()

    if report.design_rules:
        print(f"DESIGN RULES\n{'-'*width}")
        for i, rule in enumerate(report.design_rules, 1):
            print(f"  {i}. {rule.claim} [{rule.frame_id}]")
        print()

    if report.contradictions:
        print(f"CONTRADICTIONS\n{'-'*width}")
        for i, c in enumerate(report.contradictions, 1):
            print(f"  {i}. {c.summary} [{c.frame_id_a} vs {c.frame_id_b}]")
        print()

    # --- Knowledge Graph sourced content ends here ---

    if report.knowledge_gaps:
        print(f"KNOWLEDGE GAPS (not covered by available frames)\n{'-'*width}")
        for gap in report.knowledge_gaps:
            print(f"  - {gap}")
        print()

    if backfill_result and backfill_result.backfills:
        print(f"TEXTBOOK BACKFILL (general engineering -- not from knowledge graph)\n{'-'*width}")
        for bf in backfill_result.backfills:
            print(f"  Gap: {bf.gap}")
            print(f"  -> {bf.backfill_answer}")
            if bf.relevant_standards:
                print(f"     Ref: {bf.relevant_standards}")
            print()

    print(f"{'='*width}")
    print(f"Type 'unpack' for full frame details, 'context' for source credibility.")


# ─────────────────────────────────────────────
# UNPACK -- show full frame details for cited frames
# ─────────────────────────────────────────────

def collect_cited_frame_ids(report):
    """Collect all frame_ids referenced in a V2 report."""
    cited = set()
    for section in report.disciplines:
        for ev in section.evidence:
            cited.add(ev.frame_id)
        for chain in section.reasoning_chains:
            for cf in chain.frames:
                cited.add(cf.frame_id)
    for rule in report.design_rules:
        cited.add(rule.frame_id)
    for c in report.contradictions:
        cited.add(c.frame_id_a)
        cited.add(c.frame_id_b)
    return cited


def unpack_frames(report, enrichment, G, frame_filter=None):
    """Show full details for frames cited in the report.

    If frame_filter is given, show only that frame. Otherwise show all cited frames.
    """
    width = 70
    cited = collect_cited_frame_ids(report)

    if frame_filter:
        # Allow partial match (user might type "123456" instead of "123456-0")
        matches = [fid for fid in cited if frame_filter in fid]
        if not matches:
            print(f"Frame '{frame_filter}' not found in last report.")
            print(f"Cited frames: {', '.join(sorted(cited))}")
            return
        targets = matches
    else:
        targets = sorted(cited)

    print(f"\n{'='*width}")
    print(f"FRAME DETAILS ({len(targets)} frame{'s' if len(targets) != 1 else ''})")
    print(f"{'='*width}\n")

    for fid in targets:
        enc = enrichment.get(fid, {})
        if not enc and G.has_node(fid):
            enc = get_frame_context(G, fid)

        node = G.nodes.get(fid, {}) if G.has_node(fid) else {}

        print(f"[{fid}]")
        print(f"  Type         : {node.get('frame_type', '?')} | "
              f"Scope: {node.get('scope', '?')} | "
              f"Applicability: {node.get('applicability', '?')}")
        print(f"  Subject      : {node.get('subject', '?')}")
        print(f"  Main point   : {node.get('main_point', '?')}")

        sq = node.get('source_quote', '')
        if sq:
            print(f"  Source quote  : \"{sq}\"")

        print(f"  Epistemic    : {node.get('epistemic_stance', '?')}")
        print(f"  Post role    : {node.get('post_role', '?')}")

        ex_val = node.get('example_value', '')
        if ex_val:
            print(f"  Example value: {ex_val}")
        ex_ctx = node.get('example_context', '')
        if ex_ctx:
            print(f"  Example ctx  : {ex_ctx}")

        author = enc.get('author', 'unknown')
        cred = enc.get('credibility', '?')
        rep = enc.get('reputation', -1)
        rep_str = f", reputation: {rep}" if rep != -1 else ""
        print(f"  Author       : {author} (credibility: {cred}{rep_str})")

        ps = enc.get('post_score', 0) or 0
        accepted = enc.get('is_accepted', False)
        acc_str = " [ACCEPTED ANSWER]" if accepted else ""
        print(f"  Post score   : {ps}{acc_str}")

        url = enc.get('source_url', '')
        if url:
            print(f"  Thread URL   : {url}")

        tags = enc.get('se_tags', '')
        if tags:
            print(f"  Tags         : {tags}")

        print()


def get_frame_details(report, enrichment, G, frame_filter=None):
    """Return structured frame details for cited frames (data version of unpack_frames).

    Returns list of dicts, each with keys matching the fields shown by unpack_frames().
    Used by the Streamlit UI to render frame popovers without printing to stdout.
    """
    cited = collect_cited_frame_ids(report)

    if frame_filter:
        targets = [fid for fid in cited if frame_filter in fid]
    else:
        targets = sorted(cited)

    details = []
    for fid in targets:
        enc = enrichment.get(fid, {})
        if not enc and G.has_node(fid):
            enc = get_frame_context(G, fid)

        node = G.nodes.get(fid, {}) if G.has_node(fid) else {}

        details.append({
            "frame_id":         fid,
            "frame_type":       node.get("frame_type", "?"),
            "scope":            node.get("scope", "?"),
            "applicability":    node.get("applicability", "?"),
            "subject":          node.get("subject", "?"),
            "main_point":       node.get("main_point", "?"),
            "source_quote":     node.get("source_quote", ""),
            "epistemic_stance": node.get("epistemic_stance", "?"),
            "post_role":        node.get("post_role", "?"),
            "example_value":    node.get("example_value", ""),
            "example_context":  node.get("example_context", ""),
            "author":           enc.get("author", "unknown"),
            "credibility":      enc.get("credibility", "?"),
            "reputation":       enc.get("reputation", -1),
            "post_score":       enc.get("post_score", 0) or 0,
            "is_accepted":      enc.get("is_accepted", False),
            "source_url":       enc.get("source_url", ""),
            "se_tags":          enc.get("se_tags", ""),
        })

    return details


# ─────────────────────────────────────────────
# CONTEXT -- provenance graph lookup + LLM summary
# ─────────────────────────────────────────────

def gather_provenance_context(G, frame_ids):
    """Walk the provenance graph to collect full context for a set of frames.

    Returns a list of dicts, one per frame, grouped by thread.
    """
    thread_groups = {}  # thread_url -> {thread_info, frames: [{frame_info, author_info, post_info}]}

    for fid in frame_ids:
        if not G.has_node(fid):
            continue
        node = G.nodes[fid]
        ctx = get_frame_context(G, fid)

        # Get thread info
        thread_url = ctx.get('source_url', '')
        thread_info = {}
        if thread_url and G.has_node(thread_url):
            td = G.nodes[thread_url]
            thread_info = {
                'thread_title': td.get('thread_title', ''),
                'se_site': td.get('se_site', ''),
                'post_count': td.get('post_count', 0),
                'url': thread_url,
            }

        # Get post info
        post_info = {}
        for post_id in G.predecessors(fid):
            if G.nodes.get(post_id, {}).get('node_type') == 'Post':
                pd = G.nodes[post_id]
                post_info = {
                    'post_type': pd.get('post_type', ''),
                    'post_score': pd.get('post_score', 0),
                    'post_date': pd.get('post_date', ''),
                    'is_accepted': pd.get('is_accepted_answer', '') == 'True',
                }
                break

        frame_entry = {
            'frame_id': fid,
            'frame_type': node.get('frame_type', ''),
            'subject': node.get('subject', ''),
            'main_point': node.get('main_point', ''),
            'epistemic_stance': node.get('epistemic_stance', ''),
            'author': ctx.get('author', 'unknown'),
            'credibility': ctx.get('credibility', 'low'),
            'reputation': ctx.get('reputation', -1),
            'post_info': post_info,
        }

        key = thread_url or fid
        if key not in thread_groups:
            thread_groups[key] = {
                'thread_info': thread_info,
                'frames': [],
            }
        thread_groups[key]['frames'].append(frame_entry)

    return list(thread_groups.values())


def summarize_context(client, provenance_data, query):
    """Use the LLM to summarize provenance context for the user."""
    from dfm_report_schema import CONTEXT_SUMMARY_PROMPT

    # Format provenance data as text for the LLM
    lines = [f"Original query: {query}\n"]
    for group in provenance_data:
        ti = group['thread_info']
        if ti:
            lines.append(f"Thread: \"{ti.get('thread_title', '?')}\"")
            lines.append(f"  Site: {ti.get('se_site', '?')} | "
                         f"Posts in thread: {ti.get('post_count', '?')} | "
                         f"URL: {ti.get('url', '?')}")
        lines.append(f"  Frames from this thread:")
        for fe in group['frames']:
            rep_str = f", reputation: {fe['reputation']}" if fe['reputation'] != -1 else ""
            acc_str = " [ACCEPTED]" if fe['post_info'].get('is_accepted') else ""
            ps = fe['post_info'].get('post_score', 0) or 0
            lines.append(f"    [{fe['frame_id']}] {fe['frame_type']} | "
                         f"subject: {fe['subject']}")
            lines.append(f"      Author: {fe['author']} "
                         f"(credibility: {fe['credibility']}{rep_str})")
            lines.append(f"      Post score: {ps}{acc_str} | "
                         f"Stance: {fe['epistemic_stance']}")
            lines.append(f"      Main point: {fe['main_point']}")
        lines.append("")

    provenance_text = "\n".join(lines)

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": CONTEXT_SUMMARY_PROMPT},
                {"role": "user", "content": provenance_text},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Context summary failed ({type(e).__name__}: {e})"


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def save_report(report, enrichment, query, report_format, sub_aspects=None,
                diagnostic=None, backfill_result=None):
    """Save report + enrichment to dfm_reports/ as JSON."""
    REPORTS_DIR.mkdir(exist_ok=True)
    safe = "".join(c if c.isalnum() or c == " " else "" for c in query)[:50].strip()
    filename = REPORTS_DIR / f"report_{safe.replace(' ', '_')}.json"

    if report_format == "v2":
        output = {
            "query":        query,
            "model":        MODEL,
            "format":       "v2",
            "sub_aspects":  sub_aspects,
            "diagnostic":   diagnostic,
            "summary":      report.summary,
            "disciplines": [
                {
                    "discipline":       s.discipline,
                    "first_principles": s.first_principles,
                    "formulas":         [f.model_dump() for f in s.formulas] if hasattr(s, 'formulas') and s.formulas else [],
                    "evidence": [
                        {**e.model_dump(), **enrichment.get(e.frame_id, {})}
                        for e in s.evidence
                    ],
                    "reasoning_chains": [c.model_dump() for c in s.reasoning_chains],
                }
                for s in report.disciplines
            ],
            "contradictions": [c.model_dump() for c in report.contradictions],
            "design_rules": [
                {**r.model_dump(), **enrichment.get(r.frame_id, {})}
                for r in report.design_rules
            ],
            "knowledge_gaps": report.knowledge_gaps,
        }
        if backfill_result and backfill_result.backfills:
            output["backfill"] = [b.model_dump() for b in backfill_result.backfills]
    else:
        output = {
            "query":   query,
            "model":   MODEL,
            "format":  "v1",
            "summary": report.summary,
            "risks": [
                {**r.model_dump(), **enrichment.get(r.frame_id, {})}
                for r in report.risks
            ],
            "recommended_practices": [
                {**p.model_dump(), **enrichment.get(p.frame_id, {})}
                for p in report.recommended_practices
            ],
            "contradictions": [c.model_dump() for c in report.contradictions],
            "knowledge_gaps": report.knowledge_gaps,
        }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return filename


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def _print_interactive_help():
    """Print help for interactive mode commands."""
    width = 60
    print(f"\n{'-'*width}")
    print("COMMANDS:")
    print("  <question>              Ask a design/manufacturing question")
    print("  unpack                  Show full details for all cited frames")
    print("  unpack <frame_id>       Show details for a specific frame")
    print("  context                 Summarize author credibility & thread context")
    print("  context <frame_id>      Context for a specific frame")
    print("  help                    Show this help message")
    print("  quit / q / exit         Exit")
    print(f"{'-'*width}\n")


def main():
    parser = argparse.ArgumentParser(
        description="DFM Knowledge Graph Agent -- retrieval + synthesis"
    )
    parser.add_argument(
        "query", nargs="?",
        help="Design question or CAD requirement"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Interactive query loop with unpack/context commands"
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K_RETRIEVE,
        help=f"Frames to retrieve from Qdrant (default: {TOP_K_RETRIEVE})"
    )
    parser.add_argument(
        "--max-frames", type=int, default=MAX_FRAMES_SYNTH,
        help=f"Max frames passed to synthesis (default: {MAX_FRAMES_SYNTH})"
    )
    parser.add_argument(
        "--max-hops", type=int, default=2,
        help="Max graph traversal hops (default: 2)"
    )
    parser.add_argument(
        "--decompose", action=argparse.BooleanOptionalAction, default=None,
        help="Decompose query into sub-aspects (default: auto for queries > 5 words)"
    )
    parser.add_argument(
        "--format", choices=["v1", "v2"], default="v2",
        dest="report_format",
        help="Report format: v1 (flat) or v2 (discipline-organized, default: v2)"
    )
    parser.add_argument(
        "--concise", action="store_true",
        help="Print concise report with frame ID references (default in interactive)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Print full report with quotes/URLs/authors (overrides concise default)"
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="Generate textbook answers for knowledge gaps (default in interactive)"
    )
    parser.add_argument(
        "--no-backfill", action="store_true",
        help="Disable textbook backfill for knowledge gaps"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print retrieval + traversal details"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save report JSON to dfm_reports/"
    )
    args = parser.parse_args()

    # Resolve concise/full: interactive defaults to concise, CLI defaults to full
    is_interactive = args.interactive or (not args.query)
    use_concise = args.concise or (is_interactive and not args.full)
    use_backfill = args.backfill or (is_interactive and not args.no_backfill)

    def _run_query(q, G=None):
        """Run a query and return the result dict for interactive state."""
        report, enrichment, ranked, sub_aspects, diagnostic, backfill_result, _tok = query_dfm(
            q,
            top_k=args.top_k,
            max_frames=args.max_frames,
            max_hops=args.max_hops,
            decompose=args.decompose,
            report_format=args.report_format,
            backfill=use_backfill,
            verbose=args.verbose,
        )
        if args.report_format == "v2" and use_concise:
            print_report_v2_concise(report, q, sub_aspects, diagnostic, backfill_result)
        elif args.report_format == "v2":
            print_report_v2(report, enrichment, q, sub_aspects, diagnostic, backfill_result)
        else:
            print_report_v1(report, enrichment, q)
        if args.save:
            path = save_report(report, enrichment, q, args.report_format,
                               sub_aspects, diagnostic, backfill_result)
            print(f"\nReport saved: {path}")

        return {
            "report": report,
            "enrichment": enrichment,
            "ranked": ranked,
            "sub_aspects": sub_aspects,
            "diagnostic": diagnostic,
            "backfill_result": backfill_result,
            "query": q,
        }

    if args.query and not args.interactive:
        _run_query(args.query)

    elif args.interactive or args.query:
        # Interactive mode (or single query that drops into interactive)

        # Load graph once for interactive session (unpack/context need it)
        if not GRAPH_PICKLE.exists():
            print(f"Error: {GRAPH_PICKLE} not found. Run build_graph.py first.")
            return
        with open(GRAPH_PICKLE, "rb") as f:
            G = pickle.load(f)

        # Count frames and semantic edges for welcome message
        n_frames = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "Frame")
        n_sem = sum(
            1 for _, _, d in G.edges(data=True)
            if d.get("edge_type") in SEMANTIC_EDGE_TYPES
        )

        print(f"\nDFM Knowledge Graph Agent (interactive)")
        print(f"Model: {MODEL} | Frames: {n_frames:,} | Semantic edges: {n_sem:,}")
        mode_parts = []
        if use_concise:
            mode_parts.append("concise")
        if use_backfill:
            mode_parts.append("backfill")
        if mode_parts:
            print(f"Defaults: {', '.join(mode_parts)} (use --full / --no-backfill to change)")
        print()
        print("Commands: <question>, unpack, context, help, quit")
        print()

        # Initialize OpenAI client for context command
        api_key = os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key, timeout=300.0) if api_key else None

        last_result = None

        # If a query was passed, run it first
        if args.query:
            try:
                last_result = _run_query(args.query, G)
            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}")

        while True:
            try:
                q = input("Question> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q.lower() in ("quit", "q", "exit"):
                break

            if q.lower() == "help":
                _print_interactive_help()
                continue

            # --- UNPACK command ---
            if q.lower().startswith("unpack"):
                if last_result is None:
                    print("No report yet. Ask a question first.")
                    continue
                parts = q.split(maxsplit=1)
                frame_filter = parts[1].strip() if len(parts) > 1 else None
                unpack_frames(
                    last_result["report"],
                    last_result["enrichment"],
                    G,
                    frame_filter=frame_filter,
                )
                continue

            # --- CONTEXT command ---
            if q.lower().startswith("context"):
                if last_result is None:
                    print("No report yet. Ask a question first.")
                    continue
                if not client:
                    print("OPENAI_API_KEY not set. Cannot generate context summary.")
                    continue

                report = last_result["report"]
                parts = q.split(maxsplit=1)
                frame_filter = parts[1].strip() if len(parts) > 1 else None

                # Determine which frame IDs to look up
                all_cited = collect_cited_frame_ids(report)
                if frame_filter:
                    target_ids = [fid for fid in all_cited if frame_filter in fid]
                    if not target_ids:
                        print(f"Frame '{frame_filter}' not found in last report.")
                        print(f"Cited frames: {', '.join(sorted(all_cited))}")
                        continue
                else:
                    target_ids = sorted(all_cited)

                print(f"Looking up provenance for {len(target_ids)} frame(s)...")
                provenance = gather_provenance_context(G, target_ids)
                print("Generating context summary...")
                summary = summarize_context(
                    client, provenance, last_result["query"]
                )
                width = 70
                print(f"\n{'='*width}")
                print(f"SOURCE CONTEXT & CREDIBILITY")
                print(f"{'='*width}")
                print(summary)
                print(f"{'='*width}\n")
                continue

            # --- Regular query ---
            try:
                last_result = _run_query(q, G)
            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}")

    else:
        parser.print_help()
        print("\nExamples:")
        print('  python dfm_agent.py "risks of thin walls in aluminum"')
        print('  python dfm_agent.py "carbide speeds on steel" --verbose --save')
        print('  python dfm_agent.py "thin walls" --format v1 --no-decompose')
        print('  python dfm_agent.py --interactive')
        print('  python dfm_agent.py --interactive --full --no-backfill')


if __name__ == "__main__":
    main()
