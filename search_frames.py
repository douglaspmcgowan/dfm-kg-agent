"""
Search the Qdrant frame embedding index interactively.

Supports three search modes:
  - Two-vector (knowledge + language) -- default
  - Three-vector (knowledge + language + metadata)
  - Single collection (knowledge only, language only, or metadata only)

Usage:
  python search_frames.py                  # two-vector mode (default)
  python search_frames.py --three-vector   # include metadata vector
  python search_frames.py --single knowledge  # search only knowledge collection

Query syntax:
  Just type a question:
    > what happens when you run carbide too slow

  Prefix with filters:
    > type:risk what can go wrong with tool wear
    > scope:tooling best practices
    > cred:high experienced advice on feeds and speeds
    > type:heuristic,risk scope:machining_process variable usage
    > accepted:true answers about aluminum machining
    > site:engineering tag:aluminum

  Commands:
    > :help     -- show this help
    > :mode 2   -- switch to two-vector mode
    > :mode 3   -- switch to three-vector mode
    > :top 10   -- change number of results
    > :weights 0.5 0.3 0.2  -- set merge weights (knowledge, language, metadata)
    > :detail   -- toggle detailed output
    > :stats    -- show collection stats
    > quit      -- exit

Requires: pip install qdrant-client openai
"""

import sys
import os
from openai import OpenAI
from qdrant_client import QdrantClient, models


# -----------------------------------------------
# CONFIG
# -----------------------------------------------

QDRANT_PATH = "./qdrant_data"
EMBEDDING_MODEL = "text-embedding-3-large"

openai_client = OpenAI()


# -----------------------------------------------
# EMBEDDING
# -----------------------------------------------

def embed_query(text):
    """Embed a single query string."""
    response = openai_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding


# -----------------------------------------------
# FILTER BUILDING
# -----------------------------------------------

def parse_filters(raw_query):
    """Parse prefix filters from query string.

    Supported prefixes:
      type:risk             -> filter on frame_type == "risk"
      type:risk,heuristic   -> filter on frame_type in ["risk", "heuristic"]
      scope:tooling         -> filter on scope == "tooling"
      cred:high             -> filter on author_credibility_tier == "high"
      role:tutorial         -> filter on post_role == "tutorial"
      stance:direct_neutral -> filter on epistemic_stance == "direct_neutral"
      applicability:universal -> filter on applicability == "universal"
      accepted:true         -> filter on is_accepted_answer == true
      posttype:answer       -> filter on post_type == "answer"
      tag:aluminum          -> filter on se_tags containing "aluminum"
      site:engineering      -> filter on se_site == "engineering"

    Multiple prefixes are combined with must (AND).
    Returns: (clean_query, qdrant_filter_or_None)
    """
    # Standard field filters (exact match or multi-value)
    filter_map = {
        "type": "frame_type",
        "scope": "scope",
        "cred": "author_credibility_tier",
        "role": "post_role",
        "stance": "epistemic_stance",
        "applicability": "applicability",
        "posttype": "post_type",
        "site": "se_site",
    }

    words = raw_query.split()
    conditions = []
    query_words = []

    for word in words:
        matched = False

        # Check standard field filters
        for prefix, field in filter_map.items():
            if word.startswith(f"{prefix}:"):
                value = word.split(":", 1)[1]
                if "," in value:
                    conditions.append(
                        models.FieldCondition(
                            key=field,
                            match=models.MatchAny(any=value.split(",")),
                        )
                    )
                else:
                    conditions.append(
                        models.FieldCondition(
                            key=field,
                            match=models.MatchValue(value=value),
                        )
                    )
                matched = True
                break

        # Boolean filter: accepted:true / accepted:false
        if not matched and word.startswith("accepted:"):
            val = word.split(":", 1)[1].lower()
            if val in ("true", "yes", "1"):
                conditions.append(
                    models.FieldCondition(
                        key="is_accepted_answer",
                        match=models.MatchValue(value=True),
                    )
                )
            elif val in ("false", "no", "0"):
                conditions.append(
                    models.FieldCondition(
                        key="is_accepted_answer",
                        match=models.MatchValue(value=False),
                    )
                )
            matched = True

        # Tag filter: tag:aluminum (substring match on the comma-separated se_tags string)
        if not matched and word.startswith("tag:"):
            tag_val = word.split(":", 1)[1]
            conditions.append(
                models.FieldCondition(
                    key="se_tags",
                    match=models.MatchText(text=tag_val),
                )
            )
            matched = True

        # Score filter: score>5 or score>=10
        if not matched and word.startswith("score>"):
            try:
                threshold = word.replace("score>=", "").replace("score>", "")
                threshold = int(threshold)
                conditions.append(
                    models.FieldCondition(
                        key="post_score",
                        range=models.Range(gte=threshold),
                    )
                )
                matched = True
            except ValueError:
                pass

        # Reputation filter: rep>1000
        if not matched and word.startswith("rep>"):
            try:
                threshold = word.replace("rep>=", "").replace("rep>", "")
                threshold = int(threshold)
                conditions.append(
                    models.FieldCondition(
                        key="author_reputation",
                        range=models.Range(gte=threshold),
                    )
                )
                matched = True
            except ValueError:
                pass

        if not matched:
            query_words.append(word)

    query = " ".join(query_words)
    if not query:
        query = raw_query  # if ONLY filters, use the whole thing as query too

    if not conditions:
        return query, None
    else:
        return query, models.Filter(must=conditions)


# -----------------------------------------------
# SEARCH
# -----------------------------------------------

def search_collection(qdrant, collection_name, query_embedding, limit=10, query_filter=None):
    """Search a single Qdrant collection. Returns list of ScoredPoint."""
    return qdrant.query_points(
        collection_name=collection_name,
        query=query_embedding,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    ).points


def merge_two(k_results, l_results):
    """Merge knowledge + language results using max-score strategy.

    Each frame gets the HIGHER of its two similarity scores.
    Returns: list of (frame_id, info_dict) sorted by score descending.
    """
    scores = {}

    for point in k_results:
        fid = point.payload["frame_id"]
        sim = point.score
        scores[fid] = {
            "score": sim,
            "k_score": sim,
            "l_score": 0.0,
            "m_score": 0.0,
            "best_match": "knowledge",
            "payload": point.payload,
        }

    for point in l_results:
        fid = point.payload["frame_id"]
        sim = point.score
        if fid in scores:
            scores[fid]["l_score"] = sim
            if sim > scores[fid]["k_score"]:
                scores[fid]["score"] = sim
                scores[fid]["best_match"] = "language"
        else:
            scores[fid] = {
                "score": sim,
                "k_score": 0.0,
                "l_score": sim,
                "m_score": 0.0,
                "best_match": "language",
                "payload": point.payload,
            }

    ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    return ranked


def merge_three(k_results, l_results, m_results, weights=(0.5, 0.3, 0.2)):
    """Merge all three result sets using weighted scores.

    Args:
        weights: (knowledge_weight, language_weight, metadata_weight)
    Returns: list of (frame_id, info_dict) sorted by combined score descending.
    """
    w_k, w_l, w_m = weights
    scores = {}

    for point in k_results:
        fid = point.payload["frame_id"]
        scores[fid] = {
            "k_score": point.score, "l_score": 0.0, "m_score": 0.0,
            "payload": point.payload,
        }

    for point in l_results:
        fid = point.payload["frame_id"]
        if fid in scores:
            scores[fid]["l_score"] = point.score
        else:
            scores[fid] = {
                "k_score": 0.0, "l_score": point.score, "m_score": 0.0,
                "payload": point.payload,
            }

    for point in m_results:
        fid = point.payload["frame_id"]
        if fid in scores:
            scores[fid]["m_score"] = point.score
        else:
            scores[fid] = {
                "k_score": 0.0, "l_score": 0.0, "m_score": point.score,
                "payload": point.payload,
            }

    # Compute weighted score and find best vector
    for fid, s in scores.items():
        s["score"] = w_k * s["k_score"] + w_l * s["l_score"] + w_m * s["m_score"]
        best_val = max(s["k_score"], s["l_score"], s["m_score"])
        if best_val == s["k_score"]:
            s["best_match"] = "knowledge"
        elif best_val == s["l_score"]:
            s["best_match"] = "language"
        else:
            s["best_match"] = "metadata"

    ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    return ranked


# -----------------------------------------------
# DISPLAY
# -----------------------------------------------

def display_results(ranked, top_k=5, detailed=False):
    """Pretty-print ranked results."""
    if not ranked:
        print("\n  No results found.\n")
        return

    count = min(top_k, len(ranked))
    print(f"\n{'='*80}")
    print(f"  Top {count} results:")
    print(f"{'='*80}\n")

    for rank, (frame_id, info) in enumerate(ranked[:count], 1):
        p = info["payload"]

        # Score breakdown
        score_parts = []
        if info.get("k_score", 0) > 0:
            score_parts.append(f"K:{info['k_score']:.3f}")
        if info.get("l_score", 0) > 0:
            score_parts.append(f"L:{info['l_score']:.3f}")
        if info.get("m_score", 0) > 0:
            score_parts.append(f"M:{info['m_score']:.3f}")
        score_str = "  ".join(score_parts)

        # SE info line
        se_parts = []
        if p.get("post_score", 0) != 0:
            se_parts.append(f"Score: {p['post_score']}")
        if p.get("is_accepted_answer"):
            se_parts.append("Accepted")
        if p.get("se_tags"):
            se_parts.append(f"Tags: {p['se_tags']}")
        se_str = "  |  ".join(se_parts) if se_parts else ""

        print(f"  #{rank}  [{info['score']:.3f}]  best match: {info['best_match']}")
        print(f"      Scores: {score_str}")
        print(f"      ID: {frame_id}")
        print(f"      Type: {p.get('frame_type','')}  |  Scope: {p.get('scope','')}  |  Applic: {p.get('applicability','')}")
        print(f"      Subject: {p.get('subject','')}")
        print(f"      Author: {p.get('author_username','')} (rep: {p.get('author_reputation',0)}, {p.get('author_credibility_tier','')})")
        print(f"      Stance: {p.get('epistemic_stance','')}  |  Role: {p.get('post_role','')}")
        if se_str:
            print(f"      SE: {se_str}")

        if detailed:
            k_doc = p.get("knowledge_text", "")
            l_doc = p.get("language_text", "")
            m_doc = p.get("metadata_text", "")
            if k_doc:
                print(f"      Knowledge: {k_doc[:200]}{'...' if len(k_doc) > 200 else ''}")
            if l_doc:
                print(f"      Quote: {l_doc[:200]}{'...' if len(l_doc) > 200 else ''}")
            if m_doc:
                print(f"      Meta text: {m_doc[:200]}{'...' if len(m_doc) > 200 else ''}")

        print()


def print_help():
    """Print usage help."""
    print("""
  Query syntax:
    Just type a question:
      > what happens when you run carbide too slow

    Prefix with filters:
      > type:risk what can go wrong with tool wear
      > scope:tooling best practices
      > cred:high experienced advice on feeds and speeds
      > type:heuristic,risk scope:machining_process tool wear
      > accepted:true answers about aluminum machining
      > tag:aluminum lathe operations
      > site:engineering score>5 best practices

    Filter prefixes:
      type:          frame_type (risk, heuristic, observation, principle, etc.)
      scope:         scope domain (programming, tooling, design_geometry, etc.)
      cred:          author credibility tier (high, medium, low)
      role:          post role (tutorial, solution, addition, etc.)
      stance:        epistemic stance (absolute_authoritative, direct_neutral, etc.)
      applicability: (universal, context_dependent, situational)
      accepted:      is_accepted_answer (true/false)
      posttype:      post_type (question, answer)
      tag:           SE tag (substring match, e.g. tag:aluminum)
      site:          SE site (e.g. site:engineering)
      score>N        minimum post score (e.g. score>5)
      rep>N          minimum author reputation (e.g. rep>1000)

    Comma-separate for multi-value: type:risk,heuristic

  Commands:
    :help              show this help
    :mode 2            two-vector mode (knowledge + language)
    :mode 3            three-vector mode (knowledge + language + metadata)
    :top N             show top N results (default 5)
    :weights K L M     set merge weights for three-vector mode
    :detail            toggle detailed output (shows full text)
    :stats             show collection stats
    quit / exit / q    exit
""")


# -----------------------------------------------
# MAIN
# -----------------------------------------------

def main():
    # Parse CLI args
    mode = 2  # default: two-vector
    if "--three-vector" in sys.argv:
        mode = 3
    if "--single" in sys.argv:
        idx = sys.argv.index("--single")
        if idx + 1 < len(sys.argv):
            mode = f"single_{sys.argv[idx + 1]}"

    top_k = 5
    weights = (0.5, 0.3, 0.2)
    detailed = False

    # Open Qdrant
    if not os.path.exists(QDRANT_PATH):
        print(f"ERROR: {QDRANT_PATH} not found. Run build_embeddings.py first.")
        sys.exit(1)

    qdrant = QdrantClient(path=QDRANT_PATH)

    try:
        info = qdrant.get_collection("frames_knowledge")
        frame_count = info.points_count
    except Exception as e:
        print(f"ERROR: Could not load collections: {e}")
        print("Run build_embeddings.py first.")
        sys.exit(1)

    mode_str = "two-vector (knowledge + language)" if mode == 2 else "three-vector" if mode == 3 else str(mode)
    print(f"Frame Search -- {frame_count} frames indexed")
    print(f"Mode: {mode_str}")
    print(f"Type :help for query syntax and commands\n")

    while True:
        try:
            raw = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            break

        # Handle commands
        if raw.startswith(":"):
            parts = raw.split()
            cmd = parts[0].lower()

            if cmd == ":help":
                print_help()
            elif cmd == ":mode" and len(parts) >= 2:
                if parts[1] == "2":
                    mode = 2
                    print("  Switched to two-vector mode (knowledge + language)")
                elif parts[1] == "3":
                    mode = 3
                    print("  Switched to three-vector mode (knowledge + language + metadata)")
                else:
                    print("  Usage: :mode 2 or :mode 3")
            elif cmd == ":top" and len(parts) >= 2:
                try:
                    top_k = int(parts[1])
                    print(f"  Showing top {top_k} results")
                except ValueError:
                    print("  Usage: :top N (integer)")
            elif cmd == ":weights" and len(parts) >= 4:
                try:
                    weights = (float(parts[1]), float(parts[2]), float(parts[3]))
                    print(f"  Weights set to knowledge={weights[0]}, language={weights[1]}, metadata={weights[2]}")
                except ValueError:
                    print("  Usage: :weights 0.5 0.3 0.2")
            elif cmd == ":detail":
                detailed = not detailed
                print(f"  Detailed output: {'ON' if detailed else 'OFF'}")
            elif cmd == ":stats":
                for name in ["frames_knowledge", "frames_language", "frames_metadata"]:
                    try:
                        info = qdrant.get_collection(name)
                        print(f"  {name}: {info.points_count} frames")
                    except Exception:
                        print(f"  {name}: not found")
                print(f"  Mode: {mode}")
                print(f"  Weights: K={weights[0]} L={weights[1]} M={weights[2]}")
                print(f"  Top K: {top_k}")
            else:
                print("  Unknown command. Type :help for options.")
            continue

        # Parse filters from query
        query, query_filter = parse_filters(raw)

        # Embed the query
        query_embedding = embed_query(query)

        # Search collections based on mode
        if mode == 2:
            k_results = search_collection(qdrant, "frames_knowledge", query_embedding, limit=top_k, query_filter=query_filter)
            l_results = search_collection(qdrant, "frames_language", query_embedding, limit=top_k, query_filter=query_filter)
            ranked = merge_two(k_results, l_results)
            display_results(ranked, top_k=top_k, detailed=detailed)

        elif mode == 3:
            k_results = search_collection(qdrant, "frames_knowledge", query_embedding, limit=top_k, query_filter=query_filter)
            l_results = search_collection(qdrant, "frames_language", query_embedding, limit=top_k, query_filter=query_filter)
            m_results = search_collection(qdrant, "frames_metadata", query_embedding, limit=top_k, query_filter=query_filter)
            ranked = merge_three(k_results, l_results, m_results, weights=weights)
            display_results(ranked, top_k=top_k, detailed=detailed)

        elif isinstance(mode, str) and mode.startswith("single_"):
            col_name = f"frames_{mode.split('_', 1)[1]}"
            if col_name not in ["frames_knowledge", "frames_language", "frames_metadata"]:
                print(f"  Unknown collection: {col_name}")
                continue
            results = search_collection(qdrant, col_name, query_embedding, limit=top_k, query_filter=query_filter)
            ranked = merge_two(results, [])  # empty second list
            display_results(ranked, top_k=top_k, detailed=detailed)


if __name__ == "__main__":
    main()
