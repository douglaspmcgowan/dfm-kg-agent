"""
Generate pre-cached DFM reports for the Streamlit app example questions.

Quick Search:       gpt-4.1, top_k=5,  max_hops=2, max_frames=15
Efficient Thinking: gpt-5 low,  top_k=10, max_hops=2, max_frames=20
Deep Reasoning:     gpt-5 medium, top_k=10, max_hops=3, max_frames=20

Usage:
    python generate_example_cache.py
"""

import os
import sys
import json
import time
import pickle
from pathlib import Path
from functools import wraps

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("OPENAI_MODEL", "gpt-5")

import dfm_agent as agent
from dfm_agent import query_dfm, collect_cited_frame_ids, get_frame_details, GRAPH_PICKLE
from dfm_report_schema import (
    QueryDecomposition, DFMReportV2, BackfillOutput,
    DECOMPOSE_PROMPT, SYNTHESIS_PROMPT_V2, BACKFILL_PROMPT,
)

EXAMPLES = [
    {   # Quick Search
        "question": "What should I consider when designing thin-walled aluminum parts for CNC machining?",
        "model": "gpt-4.1",
        "reasoning_effort": None,
        "top_k": 5, "max_hops": 2, "max_frames": 15,
    },
    {   # Quick Search
        "question": "What fixturing methods work best for thin flexible parts?",
        "model": "gpt-4.1",
        "reasoning_effort": None,
        "top_k": 5, "max_hops": 2, "max_frames": 15,
    },
    {   # Efficient Thinking
        "question": "What are the risks of deep pocket milling in steel?",
        "model": "gpt-5",
        "reasoning_effort": "low",
        "top_k": 10, "max_hops": 2, "max_frames": 20,
    },
    {   # Deep Reasoning
        "question": "How do I choose between climb and conventional milling?",
        "model": "gpt-5",
        "reasoning_effort": None,  # medium (default)
        "top_k": 10, "max_hops": 3, "max_frames": 20,
    },
]

OUT_FILE = Path(__file__).parent / "example_cache.json"


# Load graph once
print("Loading graph...")
with open(GRAPH_PICKLE, "rb") as f:
    G = pickle.load(f)
print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def run_example(question, model_name, reasoning_effort, top_k=10, max_hops=2, max_frames=20):
    """Run a single example query and return serializable result."""
    agent.MODEL = model_name

    original_decompose = agent.decompose_query
    original_synth = agent.synthesize_report_v2
    original_backfill = agent.backfill_knowledge_gaps

    if reasoning_effort is not None:
        def patched_decompose(client, question, tracker=None):
            if len(question.split()) <= 5:
                return [question]
            try:
                completion = client.beta.chat.completions.parse(
                    model=agent.MODEL,
                    messages=[
                        {"role": "system", "content": DECOMPOSE_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    response_format=QueryDecomposition,
                    reasoning_effort=reasoning_effort,
                )
                if tracker:
                    tracker.record(completion.usage)
                result = completion.choices[0].message.parsed
                if result and result.sub_aspects:
                    return result.sub_aspects
            except Exception as e:
                print(f"  Decompose failed: {e}")
            return [question]

        def patched_synth(client, query, ranked_frames, sub_aspects=None, tracker=None):
            frame_blocks = "\n\n".join(
                agent.format_frame_for_prompt(fid, info) for fid, _, info in ranked_frames
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
                model=agent.MODEL,
                messages=[
                    {"role": "system", "content": SYNTHESIS_PROMPT_V2},
                    {"role": "user", "content": user_msg},
                ],
                response_format=DFMReportV2,
                reasoning_effort=reasoning_effort,
            )
            if tracker:
                tracker.record(completion.usage)
            return completion.choices[0].message.parsed

        def patched_backfill(client, report, tracker=None):
            if not report.knowledge_gaps:
                return None
            gaps_text = "\n".join(f"- {gap}" for gap in report.knowledge_gaps)
            user_msg = f"Knowledge gaps to backfill:\n{gaps_text}"
            try:
                completion = client.beta.chat.completions.parse(
                    model=agent.MODEL,
                    messages=[
                        {"role": "system", "content": BACKFILL_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format=BackfillOutput,
                    reasoning_effort=reasoning_effort,
                )
                if tracker:
                    tracker.record(completion.usage)
                return completion.choices[0].message.parsed
            except Exception as e:
                print(f"  Backfill failed: {e}")
                return None

        agent.decompose_query = patched_decompose
        agent.synthesize_report_v2 = patched_synth
        agent.backfill_knowledge_gaps = patched_backfill

    effort_label = f" (reasoning_effort={reasoning_effort})" if reasoning_effort else " (medium)"
    print(f"\n{'='*60}")
    print(f"  {model_name}{effort_label}")
    print(f"  top_k={top_k}, max_hops={max_hops}, max_frames={max_frames}")
    print(f"  Q: {question[:80]}...")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        report, enrichment, ranked, sub_aspects, diagnostic, backfill_result, tracker = query_dfm(
            question, top_k=top_k, max_frames=max_frames, max_hops=max_hops,
            decompose=True, backfill=True, verbose=False, G=G,
        )
        elapsed = time.time() - t0

        frame_details = get_frame_details(report, enrichment, G)
        token_usage = tracker.as_dict() if tracker else {}

        # Serialize to JSON-safe dict
        result = {
            "model": model_name,
            "reasoning_effort": reasoning_effort,
            "report": report.model_dump(),
            "enrichment": enrichment,
            "ranked": ranked,
            "sub_aspects": sub_aspects,
            "diagnostic": diagnostic,
            "backfill_result": backfill_result.model_dump() if backfill_result else None,
            "frame_details": frame_details,
            "elapsed_s": round(elapsed, 1),
            "token_usage": token_usage,
        }

        print(f"  DONE in {elapsed:.1f}s")
        print(f"  Tokens: {token_usage.get('total_tokens', 0):,}")
        print(f"  Rules: {len(report.design_rules)}, Disciplines: {len(report.disciplines)}")
        return result

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        agent.decompose_query = original_decompose
        agent.synthesize_report_v2 = original_synth
        agent.backfill_knowledge_gaps = original_backfill


def main():
    cache = {}

    for i, ex in enumerate(EXAMPLES):
        print(f"\n--- Example {i+1}/{len(EXAMPLES)} ---")
        result = run_example(
            ex["question"], ex["model"], ex["reasoning_effort"],
            top_k=ex.get("top_k", 10),
            max_hops=ex.get("max_hops", 2),
            max_frames=ex.get("max_frames", 20),
        )
        if result:
            cache[ex["question"]] = result
        else:
            print(f"  Skipping failed example: {ex['question'][:60]}...")

    with open(OUT_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\nCache saved to {OUT_FILE} ({len(cache)} examples)")


if __name__ == "__main__":
    main()
