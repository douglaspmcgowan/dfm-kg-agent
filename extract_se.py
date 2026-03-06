"""
DFM Knowledge Graph -- Frame Extraction Pipeline (Stack Exchange variant)
Reference: DFM Knowledge Graph -- Project Reference Document v2

Reads se_posts_normalized.json (normalizer output), extracts structured knowledge
frames from each question and answer post, writes extractions_test_se.json or
extractions_se.json.

Key differences from extract.py (PracticalMachinist variant):
  - No BBCode quote handling (SE uses HTML, already converted to plain text by normalizer)
  - No back-propagation (Q&A structure is explicit; answer->question linking via question_id_ref)
  - Input is thread-structured: each thread has a question + answers
  - Each answer is extracted separately with the question text as context
  - Broader topic scope: machining + CNC + CAD/CAM + DFM + mechanical design

Usage:
    python extract_se.py                              # runs DEFAULT_TEST_N threads
    MAX_POSTS=0 python extract_se.py                  # all threads
    MAX_POSTS=20 python extract_se.py                 # 20 threads
    DEBUG_FILTERING=true python extract_se.py         # verbose filter output
    ENABLE_CONSOLIDATION=false python extract_se.py   # skip dedup step
    PARALLEL_REQUESTS=1 python extract_se.py          # sequential (no parallelism)
    OVERWRITE=true MAX_POSTS=100 python extract_se.py # overwrite existing output

Environment variables:
    OPENAI_API_KEY           required
    OPENAI_MODEL             default: gpt-5-nano
    INPUT_FILE               default: se_posts_normalized.json
    MAX_POSTS                default: 5 (0 = all) -- counts THREADS, not individual posts
    SLEEP_SECONDS            default: 0.2 (sequential mode only)
    PARALLEL_REQUESTS        default: 10 (concurrent threads; 1 = sequential)
    OUTPUT_TEST_FILE         default: extractions_test_se.json
    OUTPUT_FULL_FILE         default: extractions_se.json
    OVERWRITE                default: false (refuse to overwrite extractions_se.json)
    DEBUG_FILTERING          default: false
    ENABLE_CONSOLIDATION     default: true
    CONSOLIDATION_THRESHOLD  default: 0.90
    RESUME                   default: false (resume from checkpoint file)
"""

import os
import re
import json
import time
import threading
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple
from difflib import SequenceMatcher

from openai import OpenAI
from schema_se import ExtractionOutput

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL      = os.getenv("OPENAI_MODEL", "gpt-5-nano")
INPUT_FILE = os.getenv("INPUT_FILE", "se_posts_normalized.json")

DEFAULT_TEST_N = 5

_MAX_POSTS_ENV = os.getenv("MAX_POSTS")
if _MAX_POSTS_ENV is None or not _MAX_POSTS_ENV.strip():
    MAX_POSTS = DEFAULT_TEST_N
else:
    MAX_POSTS = int(_MAX_POSTS_ENV)

SLEEP_SECONDS           = float(os.getenv("SLEEP_SECONDS", "0.2"))
OUTPUT_TEST_FILE        = os.getenv("OUTPUT_TEST_FILE", "extractions_test_se.json")
OUTPUT_FULL_FILE        = os.getenv("OUTPUT_FULL_FILE", "extractions_se.json")
DEBUG_FILTERING         = os.getenv("DEBUG_FILTERING", "false").lower() == "true"
ENABLE_CONSOLIDATION    = os.getenv("ENABLE_CONSOLIDATION", "true").lower() == "true"
CONSOLIDATION_THRESHOLD = float(os.getenv("CONSOLIDATION_THRESHOLD", "0.90"))
PARALLEL_REQUESTS       = int(os.getenv("PARALLEL_REQUESTS", "10"))
OVERWRITE               = os.getenv("OVERWRITE", "false").lower() == "true"
RESUME                  = os.getenv("RESUME", "false").lower() == "true"


# ─────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Collapse whitespace and normalize common unicode punctuation for comparison."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2014', '-').replace('\u2013', '-')
    return text.strip()


# ─────────────────────────────────────────────
# QUOTE VERIFICATION
# Catches hallucinated source_quotes -- the model must quote verbatim from the post.
# ─────────────────────────────────────────────

def fuzzy_substring_match(quote: str, source: str, threshold: float = 0.85) -> bool:
    """
    Returns True if quote is found in (or closely matches a substring of) source.
    Exact match first. Falls back to sliding-window SequenceMatcher ratio.
    Quotes shorter than 20 chars must match exactly.
    """
    if quote in source:
        return True
    norm_q = normalize_text(quote).lower()
    norm_s = normalize_text(source).lower()
    if norm_q in norm_s:
        return True
    if len(norm_q) < 20:
        return False
    q_len = len(norm_q)
    best  = 0.0
    # Try exact-length sliding window
    for i in range(len(norm_s) - q_len + 1):
        r = SequenceMatcher(None, norm_q, norm_s[i:i + q_len]).ratio()
        if r > best:
            best = r
        if r >= threshold:
            return True
    # Try wider windows for long quotes (model may add/drop a few words)
    if q_len > 80:
        for stretch in [int(q_len * 1.1), int(q_len * 0.9)]:
            for i in range(max(0, len(norm_s) - stretch + 1)):
                window = norm_s[i:i + stretch]
                r = SequenceMatcher(None, norm_q, window).ratio()
                if r > best:
                    best = r
                if r >= threshold:
                    return True
    # Try sentence-level matching
    for sentence in re.split(r'[.!?]\s+', norm_s):
        if len(sentence) > 10:
            r = SequenceMatcher(None, norm_q, sentence).ratio()
            if r >= threshold:
                return True
    if DEBUG_FILTERING and best > 0.5:
        print(f"    [QUOTE MISMATCH] best={best:.2f} - {norm_q[:80]}")
    return False


# ─────────────────────────────────────────────
# EPISTEMIC STANCE OVERRIDE HEURISTIC
# ─────────────────────────────────────────────

HEDGE_PATTERNS = [
    "usually", "typically", "often", "generally", "tends to",
    "in my experience", "from what i've seen", "from what i have seen",
    "i think", "i believe", "seems", "might", "could be", "maybe",
    "probably", "a few", "some of", "around", "somewhere",
    "i like to", "i prefer", "i tend to",
    "most of the time", "more often than not",
    "i worked with a", "i worked with one", "i've worked with a",
    "i've seen", "i have seen",
    "not always", "not necessarily",
]

STRONG_PATTERNS = [
    "always", "never", "cannot", "can not", "can't",
    "must", "only", "required", "critical", "you will",
    "you must", "impossible", "not supported", "will not",
    "are not", "is not", "do not", "does not",
    "nothing but", "not standardized", "i hate", "i love",
    "my favorite",
]


def apply_epistemic_stance_overrides(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scan source_quote for hedge/uncertainty language and override epistemic_stance
    when the model got it wrong.

    Rules:
      - hedge + strong -> hedged_emphatic
      - hedge + no strong -> tentative_subjective (if model said absolute or direct)
      - no hedge + no strong -> direct_neutral (if model said absolute)
    """
    sq = (frame.get("source_quote") or "").lower()
    if not sq:
        return frame

    has_hedge = any(p in sq for p in HEDGE_PATTERNS)
    has_strong = any(p in sq for p in STRONG_PATTERNS)
    current = frame.get("epistemic_stance", "")

    overrides_applied = []

    if has_hedge and has_strong:
        if current != "hedged_emphatic":
            frame["epistemic_stance"] = "hedged_emphatic"
            overrides_applied.append(f"{current}->hedged_emphatic")
    elif has_hedge and not has_strong:
        if current in ("absolute_authoritative", "direct_neutral"):
            frame["epistemic_stance"] = "tentative_subjective"
            overrides_applied.append(f"{current}->tentative_subjective")
    elif not has_hedge and not has_strong:
        if current == "absolute_authoritative":
            frame["epistemic_stance"] = "direct_neutral"
            overrides_applied.append(f"{current}->direct_neutral (no strong keywords)")

    if DEBUG_FILTERING and overrides_applied:
        print(f"    [EPISTEMIC] {', '.join(overrides_applied)} -- {sq[:70]}...")

    return frame


# ─────────────────────────────────────────────
# SYSTEM PROMPT
#
# DESIGN INTENT:
# Frame type definitions and per-field guidance live in schema_se.py Field(description=...).
# The model receives those descriptions automatically via the structured output schema.
# This prompt covers only behavioral rules that span all fields at once.
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are extracting structured knowledge frames from technical Q&A posts about machining, CNC programming, CAD/CAM, design for manufacturability (DFM), tolerances, and mechanical design.

Each frame captures ONE piece of knowledge, evidenced by a verbatim quote from the post.
Frames feed into a DFM knowledge graph whose goal is to capture practical engineering
knowledge -- machining practice, design constraints, tooling wisdom, CAD/CAM techniques,
and manufacturing realities.

Frame type definitions and field instructions are in the schema. Follow them exactly.

VERBATIM QUOTING
source_quote must be copied CHARACTER FOR CHARACTER from the post text. Do not
paraphrase, truncate, rearrange, or combine sentences. Do not insert ellipses (...).
If the passage you want to quote is too long, pick a shorter contiguous substring --
but that substring must appear EXACTLY in the post text.

EXTRACTION PRIORITY
When a post contains BOTH definitions/explanations AND the author's personal practice
or opinion about those concepts, prioritize the personal practice:
  "I do/avoid/hate X because Y" -> HIGHER priority (heuristic)
  "X is defined as..." -> LOWER priority (observation)
Extract both when distinct, but if you must choose, choose the experiential claim.
Definitions are in textbooks; personal practice is tribal knowledge.

VALUE THRESHOLD
Skip frames that are purely tautological or content-free platitudes -- statements
that could apply to any situation and give no specific guidance. Examples of what
NOT to extract:
  "Use the right tool for the job" -- no specific guidance
  "Follow safety guidelines" -- no specific content
  "Use parameters within your machine's capabilities" -- restates the obvious
But DO extract frames that state a specific decision rule, concrete heuristic,
or actionable principle, even if the language is simple or brief. For example,
"whatever has the lowest limit is the limit to be followed" IS a specific,
transferable decision rule and SHOULD be extracted.
If the same short answer makes one specific point AND a generic restatement of it,
extract only the specific point.

QUESTION POSTS
Most question posts contain NO extractable knowledge. Set has_relevant_content=false
unless the question EMBEDS a genuine knowledge claim. Look for:
- Experiential observations ("we found that Y causes Z", "X is seen to be more durable")
- Empirical claims about process or material behavior ("copper cuts extremely slowly")
- Design constraints backed by reasoning ("it HAS to be milled because adapters won't work")
- Factual statements about how components work ("the ring is under tension once installed,
  hence the Free ID being smaller than the groove diameter")

Do NOT extract any of the following from question posts:
- The question itself ("how do I do X?" is not knowledge)
- Problem setup or given parameters ("n=1629 rpm, f=0.1 mm/rev" is homework, not knowledge)
- The asker's specific equipment specs ("my lathe goes up to 2000 rpm" is situational setup)
- Textbook calculations or unit conversions the asker is performing
- Lists of options the asker is considering ("should I do A, B, or C?")
- Requests for resources or references ("does anyone have a good book on X?")

Only extract from a question when the asker states something transferable that an
engineer would benefit from knowing independent of this specific question.

ANSWER POSTS
You will receive context about the original question. Extract knowledge from the ANSWER
text only -- do NOT extract frames from the question context. The question context helps
you understand what the answer is addressing and what post_role to assign.

THREAD_CONTEXT -- NOT USED
thread_context is not used for Stack Exchange posts. Always output null.

POST_ROLE SELECTION
post_role classifies THIS SPECIFIC POST, not the author's general role.
All frames from the same post share one post_role.
- For question posts, use "question" when in doubt.
- For answer posts, use "solution" when in doubt.
- But always look and decide: answers can be "tutorial", "war_story", "addition",
  "general_advice", etc. depending on their actual content.
- A short reply that confirms or supplements someone else's point is "addition".
- Reserve "war_story" for posts with a clear story arc: specific job/incident -> problem/challenge -> outcome.
- Reserve "tutorial" for posts that are primarily teaching/explaining concepts at length.
- "reaction" is for short replies (endorsement, thanks, "+1") that don't contain extractable knowledge.

TUTORIAL / EXAMPLE POSTS -- QUOTE SELECTION
Many valuable posts mix general claims with tutorial walkthroughs that include scaffold
lines ("So: X == 10", "for the sake of brevity, lets limit to..."). These posts often
contain BOTH throwaway example settings AND transferable knowledge.

Your job: extract the transferable knowledge, quote the non-scaffold sentences.

QUOTE SELECTION RULES:
1. Prefer quoting sentences that state general facts, principles, heuristics, or
   observations -- even if they appear inside a tutorial-style post.
2. If a claim is illustrated by a tutorial example, put the example detail in
   example_value -- not as the main_point.
3. Do NOT create a frame whose main_point is just a tutorial decision/setting
   (e.g. "limit to 10 holes", "macro does not call tool"). These are not knowledge.
4. Do NOT treat tutorial Yes/No decisions or author-chosen defaults as risks.
   A risk requires a real external limitation (cannot/not supported/only if) or failure mode.

CRITICAL: Do NOT skip an entire post because it contains scaffold lines. A tutorial post
that says "Macro programming provides a means of shortening code and doing repetitive
tasks easily and quickly" contains a perfectly good observation -- quote THAT sentence,
not the scaffold around it.

REFERENCE LISTS AND TABLES
Do NOT extract individual items from a reference list, variable table, parameter list,
or G-code table as separate frames. If a post lists multiple items in a table or list,
that is ONE observation about the list/table if one exists, not separate frames for each entry."""


def make_user_message(post: Dict[str, Any], text: str) -> str:
    """Build the user message for the LLM."""
    parts = [
        f"Post ID: {post.get('id')}",
        f"Thread: {post.get('thread_title', '')}",
        f"URL: {post.get('thread_url', '')}",
        f"Post Type: {post.get('post_type', 'unknown')}",
        f"Post Position: {post.get('post_position', 'unknown')}",
        f"Author: {post.get('author')}",
        f"Author Credibility: {post.get('author_credibility_tier', 'unknown')}"
        f" (reputation: {post.get('author_reputation', 'unknown')})",
        f"Date: {post.get('date')}",
        f"Score: {post.get('score', 0)}",
    ]
    se_tags = post.get("se_tags")
    if se_tags:
        parts.append(f"Tags: {', '.join(se_tags)}")
    is_acc = post.get("is_accepted")
    if is_acc is not None:
        parts.append(f"Accepted Answer: {'yes' if is_acc else 'no'}")
    parts.append("")
    parts.append("POST TEXT (extract frames ONLY from this content):")
    parts.append(text)
    return "\n".join(parts) + "\n"


# ─────────────────────────────────────────────
# CONSOLIDATION -- PHASE 1: RULES
# ─────────────────────────────────────────────

def rules_based_consolidation(
    frames: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, float]]]:
    """
    Flag near-duplicate source_quotes for LLM merge review.
    Near-duplicates mean the model extracted two frames from the same passage.
    """
    flagged: List[Tuple[int, int, float]] = []
    for i, f1 in enumerate(frames):
        for j, f2 in enumerate(frames[i + 1:], start=i + 1):
            q1 = normalize_text(f1.get('source_quote', '')).lower()
            q2 = normalize_text(f2.get('source_quote', '')).lower()
            if len(q1) > 20 and len(q2) > 20:
                ratio = SequenceMatcher(None, q1, q2).ratio()
                if ratio >= CONSOLIDATION_THRESHOLD:
                    flagged.append((i, j, ratio))
                    if DEBUG_FILTERING:
                        print(f"    [CONSOL R2] Flagged ({i},{j}) similarity={ratio:.2f}")

    return frames, flagged


# ─────────────────────────────────────────────
# CONSOLIDATION -- PHASE 2: LLM MERGE
# ─────────────────────────────────────────────

def get_merge_prompt(f1: Dict, f2: Dict) -> str:
    return f"""Two knowledge frames from the same post have very similar source quotes.
Decide whether to merge them or keep both.

Frame 1:
  frame_type: {f1.get('frame_type')}
  subject: {f1.get('subject')}
  main_point: {f1.get('main_point')}
  source_quote: {f1.get('source_quote')}

Frame 2:
  frame_type: {f2.get('frame_type')}
  subject: {f2.get('subject')}
  main_point: {f2.get('main_point')}
  source_quote: {f2.get('source_quote')}

Rules:
- MERGE if both capture the same knowledge nugget from overlapping text
- KEEP_BOTH if they capture genuinely distinct claims (e.g. one principle + one heuristic)
- KEEP_FIRST or KEEP_SECOND if one is clearly more complete

Respond ONLY with valid JSON:
{{
  "action": "merge" | "keep_both" | "keep_first" | "keep_second",
  "reasoning": "one sentence",
  "merged_frame": {{ ...complete frame object... }}
}}
merged_frame required only when action is "merge"."""


def llm_based_merge(
    client: OpenAI,
    frames: List[Dict[str, Any]],
    flagged: List[Tuple[int, int, float]]
) -> List[Dict[str, Any]]:
    if not flagged:
        return frames
    print(f"  [CONSOL] LLM reviewing {len(flagged)} flagged pair(s)...")
    to_remove: set = set()
    to_add: List[Dict] = []
    for idx1, idx2, _ in flagged:
        try:
            merge_kwargs = dict(
                model=MODEL,
                messages=[{"role": "user", "content": get_merge_prompt(frames[idx1], frames[idx2])}],
                response_format={"type": "json_object"},
            )
            response = client.chat.completions.create(**merge_kwargs)
            decision = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"    [WARNING] LLM merge failed ({idx1},{idx2}): {e} - keeping both")
            continue
        action = decision.get("action", "keep_both")
        if DEBUG_FILTERING:
            print(f"    Pair ({idx1},{idx2}): {action} - {decision.get('reasoning', '')[:70]}")
        if action == "merge" and decision.get("merged_frame"):
            to_add.append(decision["merged_frame"])
            to_remove.update([idx1, idx2])
        elif action == "keep_first":
            to_remove.add(idx2)
        elif action == "keep_second":
            to_remove.add(idx1)
    result = [f for i, f in enumerate(frames) if i not in to_remove]
    result.extend(to_add)
    if to_remove:
        print(f"  [CONSOL] Removed {len(to_remove)} frames after merge")
    return result


# ─────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────

def extract_one(
    client: OpenAI,
    post: Dict[str, Any],
    question_context: Optional[str] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Extract knowledge frames from one SE post (question or answer).

    Args:
        client: OpenAI client
        post: normalized post dict from normalize_se.py
        question_context: for answer posts, the question's plain text (for context only)
        quiet: suppress per-post detail printing (used in parallel mode)
    """
    text = post.get("text", "") or ""

    base_result = {
        "post_id":               post.get("id", ""),
        "post_type":             post.get("post_type"),
        "thread_url":            post.get("thread_url", ""),
        "thread_title":          post.get("thread_title", ""),
        "post_position":         post.get("post_position"),
        "post_date":             post.get("date", ""),
        "post_score":            post.get("score"),
        "author_username":       post.get("author", ""),
        "author_reputation":     post.get("author_reputation"),
        "author_credibility_tier": post.get("author_credibility_tier"),
        "se_tags":               post.get("se_tags"),
        "se_site":               post.get("se_site"),
        "is_accepted_answer":    post.get("is_accepted"),
        "has_relevant_content":  False,
        "frames":                [],
        "non_quoted_text":       text,
    }

    if not text.strip():
        return {**base_result, "skip_reason": "No text content."}

    # Build user message
    user_msg = make_user_message(post, text)

    # If this is an answer, prepend question context
    if question_context:
        user_msg = (
            "ORIGINAL QUESTION (for context only -- do NOT extract frames from this):\n"
            f"{question_context}\n\n"
            "---\n\n"
            f"{user_msg}"
        )

    try:
        parse_kwargs = dict(
            model=MODEL,
            seed=42,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            response_format=ExtractionOutput,
        )

        completion = client.beta.chat.completions.parse(**parse_kwargs)

        parsed: ExtractionOutput = completion.choices[0].message.parsed
        extraction = parsed.model_dump()
        model_has_relevant = extraction.get("has_relevant_content")

        # Build full result from normalizer ground-truth + LLM extraction
        data = {
            "post_id":               post.get("id", ""),
            "post_type":             post.get("post_type"),
            "thread_url":            post.get("thread_url", ""),
            "thread_title":          post.get("thread_title", ""),
            "post_position":         post.get("post_position"),
            "post_date":             post.get("date", ""),
            "post_score":            post.get("score"),
            "author_username":       post.get("author", ""),
            "author_reputation":     post.get("author_reputation"),
            "author_credibility_tier": post.get("author_credibility_tier"),
            "se_tags":               post.get("se_tags"),
            "se_site":               post.get("se_site"),
            "is_accepted_answer":    post.get("is_accepted"),
            "has_relevant_content":  model_has_relevant,
            "frames":                extraction.get("frames", []),
            "non_quoted_text":       text,
        }

        # Propagate post-level fields into each frame
        for frame in data.get("frames") or []:
            frame["source_url"] = data["thread_url"]
            frame["author_username"] = data["author_username"]
            frame["post_id"] = data["post_id"]
            frame["post_date"] = data["post_date"]
            frame["post_score"] = data["post_score"]
            frame["is_accepted_answer"] = data["is_accepted_answer"]
            # Not populated during extraction
            frame["thread_context"] = None
            frame["related_frames"] = None

        raw_frames = data.get("frames") or []
        if not quiet:
            print(f"  RAW frames: {len(raw_frames)}")

        # ── Filtering ─────────────────────────────
        cleaned: List[Dict[str, Any]] = []
        filtered_log: List[Dict[str, str]] = []
        stats = {
            "missing_quote": 0,
            "quote_mismatch": 0,
            "missing_main_point": 0,
            "missing_scope": 0,
            "missing_applicability": 0,
            "missing_example_context": 0,
        }

        for frame in raw_frames:
            sq = (frame.get("source_quote") or "").strip()

            if not sq:
                stats["missing_quote"] += 1
                filtered_log.append({
                    "reason": "missing_quote",
                    "frame_type": frame.get("frame_type", "?"),
                    "subject": frame.get("subject", "?"),
                })
                if DEBUG_FILTERING:
                    print(f"    [FILTER] No source_quote - {frame.get('frame_type')}")
                continue

            if not fuzzy_substring_match(sq, text):
                stats["quote_mismatch"] += 1
                filtered_log.append({
                    "reason": "quote_mismatch",
                    "frame_type": frame.get("frame_type", "?"),
                    "subject": frame.get("subject", "?"),
                    "source_quote_excerpt": sq[:100],
                })
                if DEBUG_FILTERING:
                    print(f"    [FILTER] Quote not in post - {sq[:80]}...")
                continue

            if not (frame.get("main_point") or "").strip():
                stats["missing_main_point"] += 1
                filtered_log.append({
                    "reason": "missing_main_point",
                    "frame_type": frame.get("frame_type", "?"),
                    "subject": frame.get("subject", "?"),
                    "source_quote_excerpt": sq[:100],
                })
                if DEBUG_FILTERING:
                    print(f"    [FILTER] No main_point - {sq[:60]}...")
                continue

            # Warn (but don't drop) if classification fields are missing
            if not frame.get("scope"):
                stats["missing_scope"] += 1
                if DEBUG_FILTERING:
                    print(f"    [FILTER] No scope - {sq[:60]}...")
            if not frame.get("applicability"):
                stats["missing_applicability"] += 1
                if DEBUG_FILTERING:
                    print(f"    [FILTER] No applicability - {sq[:60]}...")
            if frame.get("applicability") in ("context_dependent", "situational") and not frame.get("example_context"):
                stats["missing_example_context"] += 1
                if DEBUG_FILTERING:
                    print(f"    [FILTER] No example_context on {frame.get('applicability')} frame - {sq[:60]}...")

            # Epistemic stance override heuristic
            apply_epistemic_stance_overrides(frame)

            cleaned.append(frame)

        hard_dropped = stats["missing_quote"] + stats["quote_mismatch"] + stats["missing_main_point"]
        if hard_dropped and not quiet:
            print(f"  Filtered: {stats['missing_quote']} no_quote | "
                  f"{stats['quote_mismatch']} quote_mismatch | "
                  f"{stats['missing_main_point']} no_main_point")
        soft_gaps = stats["missing_scope"] + stats["missing_applicability"] + stats["missing_example_context"]
        if soft_gaps and not quiet:
            print(f"  Gaps (kept): {stats['missing_scope']} no_scope | "
                  f"{stats['missing_applicability']} no_applicability | "
                  f"{stats['missing_example_context']} no_example_context")

        if not quiet:
            print(f"  CLEANED frames: {len(cleaned)}")

        # ── Consolidation ──────────────────────────
        if ENABLE_CONSOLIDATION and cleaned:
            cleaned, flagged = rules_based_consolidation(cleaned)
            if flagged:
                cleaned = llm_based_merge(client, cleaned, flagged)
            if not quiet:
                print(f"  CONSOLIDATED frames: {len(cleaned)}")

        data["frames"] = cleaned
        data["has_relevant_content"] = bool(cleaned)
        if filtered_log:
            data["_filtered_frames_log"] = filtered_log
        if not cleaned:
            if model_has_relevant and hard_dropped > 0:
                if not quiet:
                    print(f"  [RETRY] model_has_relevant=true but 0 frames survived. Retrying...")
                retry_result = _retry_extraction(client, post, text, data, filtered_log, question_context, quiet=quiet)
                if retry_result is not None:
                    return retry_result
                data["skip_reason"] = "No valid frames after filtering (retry also failed)."
            elif model_has_relevant:
                data["skip_reason"] = "No valid frames after filtering."
            else:
                data["skip_reason"] = "Model indicated no relevant content."
        else:
            data.pop("skip_reason", None)

        return data

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        if not quiet:
            print(f"  ERROR: {err}")
        return {**base_result, "skip_reason": f"Exception: {err}"}


RETRY_SUPPLEMENT = """Your previous extraction attempt produced frames, but ALL were filtered out
because the source_quote values did not match the post text (hallucinated or paraphrased quotes).

This time:
- Copy source_quote EXACTLY from the post text -- character for character.
- Prefer quoting sentences that state the knowledge claim, not standalone scaffold lines.
- DO extract general claims from non-scaffold sentences. This post contains them.
- Quote normal sentences like observations, definitions, and heuristics.

Filter reasons from the previous attempt:
{filter_reasons}
"""


def _retry_extraction(
    client: OpenAI,
    post: Dict[str, Any],
    text: str,
    original_data: Dict[str, Any],
    filtered_log: List[Dict[str, str]],
    question_context: Optional[str] = None,
    quiet: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Re-run extraction with a supplemental instruction explaining what went wrong.
    Returns the updated data dict if retry succeeds (>0 frames), or None if it also fails.
    """
    reason_lines = []
    for entry in filtered_log[:10]:
        reason_lines.append(
            f"  - {entry['reason']}: {entry.get('frame_type', '?')} / "
            f"{entry.get('subject', '?')} / {entry.get('source_quote_excerpt', 'N/A')[:60]}"
        )
    filter_summary = "\n".join(reason_lines) if reason_lines else "  (no details available)"

    supplement = RETRY_SUPPLEMENT.format(filter_reasons=filter_summary)
    retry_system = SYSTEM_PROMPT + "\n\n" + supplement

    user_msg = make_user_message(post, text)
    if question_context:
        user_msg = (
            "ORIGINAL QUESTION (for context only -- do NOT extract frames from this):\n"
            f"{question_context}\n\n"
            "---\n\n"
            f"{user_msg}"
        )

    try:
        parse_kwargs = dict(
            model=MODEL,
            seed=42,
            messages=[
                {"role": "system", "content": retry_system},
                {"role": "user",   "content": user_msg},
            ],
            response_format=ExtractionOutput,
        )
        completion = client.beta.chat.completions.parse(**parse_kwargs)

        parsed: ExtractionOutput = completion.choices[0].message.parsed
        retry_data = parsed.model_dump()
        retry_frames = retry_data.get("frames") or []
        if not quiet:
            print(f"  [RETRY] RAW frames: {len(retry_frames)}")

        # Apply same filtering as main pass
        retry_cleaned: List[Dict[str, Any]] = []
        for frame in retry_frames:
            sq = (frame.get("source_quote") or "").strip()
            if not sq:
                continue
            if not fuzzy_substring_match(sq, text):
                if DEBUG_FILTERING:
                    print(f"    [RETRY FILTER] Quote not in post - {sq[:80]}...")
                continue
            if not (frame.get("main_point") or "").strip():
                continue
            # Epistemic stance overrides
            apply_epistemic_stance_overrides(frame)
            # Propagate post-level fields
            frame["source_url"] = original_data["thread_url"]
            frame["author_username"] = original_data["author_username"]
            frame["post_id"] = original_data["post_id"]
            frame["post_date"] = original_data["post_date"]
            frame["post_score"] = original_data["post_score"]
            frame["is_accepted_answer"] = original_data.get("is_accepted_answer")
            frame["thread_context"] = None
            frame["related_frames"] = None
            retry_cleaned.append(frame)

        if not quiet:
            print(f"  [RETRY] CLEANED frames: {len(retry_cleaned)}")

        if not retry_cleaned:
            return None

        # Consolidation on retry
        if ENABLE_CONSOLIDATION and retry_cleaned:
            retry_cleaned, flagged = rules_based_consolidation(retry_cleaned)
            if flagged:
                retry_cleaned = llm_based_merge(client, retry_cleaned, flagged)
            if not quiet:
                print(f"  [RETRY] CONSOLIDATED frames: {len(retry_cleaned)}")

        original_data["frames"] = retry_cleaned
        original_data["has_relevant_content"] = True
        original_data.pop("skip_reason", None)
        original_data["_retry_used"] = True
        return original_data

    except Exception as e:
        if not quiet:
            print(f"  [RETRY] ERROR: {type(e).__name__}: {e}")
        return None


# ─────────────────────────────────────────────
# THREAD-LEVEL EXTRACTION
# ─────────────────────────────────────────────

def extract_thread(
    client: OpenAI,
    thread: Dict[str, Any],
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract frames from a question and all its answers within one thread.

    Each answer is extracted in its own LLM call with the question text as context.
    """
    posts = thread.get("posts") or []
    if not posts:
        return []

    question_post = posts[0]  # post_position=1
    answer_posts = posts[1:]  # post_position=2+
    thread_id = thread.get("thread_id", "?")

    results = []

    # 1. Extract question
    if not quiet:
        print(f"  Q: {question_post.get('id', '?')}")
    q_result = extract_one(client, question_post, question_context=None, quiet=quiet)
    results.append(q_result)

    # 2. Extract each answer with question as context
    if answer_posts:
        # Truncate question text for context (avoid oversized prompts)
        question_text_for_context = (question_post.get("text") or "")[:2000]

        for answer_post in answer_posts:
            if not quiet:
                print(f"  A: {answer_post.get('id', '?')}")
            a_result = extract_one(
                client, answer_post,
                question_context=question_text_for_context,
                quiet=quiet,
            )
            # Link answer frames back to question
            question_id = question_post.get("id", "")
            a_result["question_id_ref"] = question_id
            for frame in a_result.get("frames") or []:
                frame["question_id_ref"] = question_id
            results.append(a_result)

    return results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found.")

    client = OpenAI(api_key=api_key)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        threads: List[Dict[str, Any]] = json.load(f)

    n_total  = len(threads)
    n_run    = n_total if MAX_POSTS == 0 else min(MAX_POSTS, n_total)
    out_file = OUTPUT_FULL_FILE if RESUME else (OUTPUT_TEST_FILE if n_run <= DEFAULT_TEST_N else OUTPUT_FULL_FILE)

    # Safety: refuse to overwrite existing output unless explicitly allowed.
    if os.path.exists(out_file) and out_file != OUTPUT_TEST_FILE and not OVERWRITE:
        size_kb = os.path.getsize(out_file) / 1024
        raise FileExistsError(
            f"{out_file} already exists ({size_kb:.0f} KB). "
            f"Set OVERWRITE=true to replace it, or use "
            f"OUTPUT_FULL_FILE=extractions_se_new.json to write elsewhere."
        )

    parallel = PARALLEL_REQUESTS

    # ── Resume from checkpoint ──
    resumed_results: List[Dict[str, Any]] = []
    if RESUME:
        checkpoint_file = out_file.replace(".json", "_checkpoint.json")
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r", encoding="utf-8") as cpf:
                cp_data = json.load(cpf)
            resumed_results = cp_data.get("results", [])
            # Build set of post_ids already extracted
            done_post_ids = {r["post_id"] for r in resumed_results}
            # Filter done threads from ALL threads, then apply MAX_POSTS as batch size
            remaining_threads = []
            n_skipped = 0
            for t in threads:
                thread_post_ids = {p["id"] for p in t.get("posts", [])}
                if thread_post_ids.issubset(done_post_ids):
                    n_skipped += 1
                    continue
                remaining_threads.append(t)
            threads = remaining_threads
            # MAX_POSTS acts as batch size when resuming
            n_run = len(threads) if MAX_POSTS == 0 else min(MAX_POSTS, len(threads))
            print(f"Resume       : loaded {len(resumed_results)} posts from {checkpoint_file}")
            print(f"               skipping {n_skipped} already-done threads, {len(threads)} remain, running {n_run}")
        else:
            print(f"Resume       : no checkpoint file found ({checkpoint_file}), starting fresh")

    # Count total posts across threads
    total_posts = sum(len(t.get("posts", [])) for t in threads[:n_run])

    print(f"Model        : {MODEL}")
    print(f"Input        : {INPUT_FILE} ({n_total} threads)")
    print(f"Running      : {n_run} threads ({total_posts} posts) -> {out_file}")
    print(f"Consolidation: {'ON' if ENABLE_CONSOLIDATION else 'OFF'}")
    print(f"Parallel     : {parallel}")
    print(f"Debug        : {'ON' if DEBUG_FILTERING else 'OFF'}")
    print()

    start_time = time.time()

    all_results: List[Dict[str, Any]] = []

    if parallel <= 1:
        # ── Sequential mode (verbose per-thread output) ──
        for i, thread in enumerate(threads[:n_run], start=1):
            tid = thread.get("thread_id", "?")
            title = str(thread.get("thread_title", ""))[:60]
            n_posts = len(thread.get("posts", []))
            print(f"[{i}/{n_run}] {tid} ({n_posts} posts) | {title}")
            thread_results = extract_thread(client, thread, quiet=False)
            all_results.extend(thread_results)
            time.sleep(SLEEP_SECONDS)
    else:
        # ── Parallel mode ──
        # Each thread is one work unit. Answers within a thread are sequential
        # (they need the question text as context).
        progress_lock = threading.Lock()
        completed_count = [0]
        total_posts_done = [0]
        total_posts_skipped = [0]
        total_frames_done = [0]

        def _thread_worker(index: int, thread: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
            thread_results = extract_thread(client, thread, quiet=True)
            with progress_lock:
                completed_count[0] += 1
                tid = thread.get("thread_id", "?")
                n_posts = len(thread.get("posts", []))
                n_frames = sum(len(r.get("frames", [])) for r in thread_results)
                n_skipped = sum(1 for r in thread_results if not r.get("has_relevant_content", False))
                total_posts_done[0] += n_posts
                total_posts_skipped[0] += n_skipped
                total_frames_done[0] += n_frames
                elapsed = time.time() - start_time
                skip_pct = (total_posts_skipped[0] / total_posts_done[0] * 100) if total_posts_done[0] else 0
                print(f"  [{completed_count[0]}/{n_run}] {tid} | {n_posts} posts, {n_frames} frames | "
                      f"total: {total_posts_done[0]}/{total_posts} posts, {total_frames_done[0]} frames, "
                      f"{skip_pct:.0f}% skipped | {elapsed:.0f}s", flush=True)
            return index, thread_results

        indexed_results: List[Optional[Tuple[int, List]]] = [None] * n_run
        checkpoint_interval = int(os.environ.get("CHECKPOINT_INTERVAL", "25"))
        checkpoint_file = out_file.replace(".json", "_checkpoint.json")
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(_thread_worker, i, thread): i
                for i, thread in enumerate(threads[:n_run])
            }
            for future in concurrent.futures.as_completed(futures):
                idx, thread_results = future.result()
                indexed_results[idx] = (idx, thread_results)

                # Incremental checkpoint every N threads
                if checkpoint_interval > 0 and completed_count[0] % checkpoint_interval == 0:
                    _cp_results = list(resumed_results)  # start with previously resumed
                    for item in indexed_results:
                        if item is not None:
                            _cp_results.extend(item[1])
                    _cp_frames = sum(len(r.get("frames", [])) for r in _cp_results)
                    with open(checkpoint_file, "w", encoding="utf-8") as cpf:
                        json.dump({"results": _cp_results}, cpf, indent=2, ensure_ascii=False)
                    _elapsed = time.time() - start_time
                    print(f"\n  ** CHECKPOINT {completed_count[0]}/{n_run} threads | "
                          f"{len(_cp_results)} posts | {_cp_frames} frames | "
                          f"{_elapsed:.0f}s elapsed | saved to {checkpoint_file} **\n",
                          flush=True)

        # Flatten in thread order
        for item in indexed_results:
            if item is not None:
                _, thread_results = item
                all_results.extend(thread_results)

    # Prepend resumed results
    if resumed_results:
        all_results = resumed_results + all_results
        print(f"\nMerged {len(resumed_results)} resumed + {len(all_results) - len(resumed_results)} new posts")

    elapsed = time.time() - start_time
    n_actual_posts = len(all_results)
    print(f"\nExtraction: {elapsed:.1f}s ({elapsed/max(n_actual_posts,1):.1f}s/post)")

    output = {
        "results": all_results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total_frames = sum(len(r.get("frames", [])) for r in all_results)

    # Compute distributions for the run summary
    scope_dist:      Dict[str, int] = {}
    apply_dist:      Dict[str, int] = {}
    epistemic_dist:  Dict[str, int] = {}
    frame_type_dist: Dict[str, int] = {}
    post_role_dist:  Dict[str, int] = {}
    for r in all_results:
        for fr in r.get("frames", []):
            s  = fr.get("scope") or "null"
            a  = fr.get("applicability") or "null"
            e  = fr.get("epistemic_stance") or "null"
            ft = fr.get("frame_type") or "null"
            pr = fr.get("post_role") or "null"
            scope_dist[s]       = scope_dist.get(s, 0) + 1
            apply_dist[a]       = apply_dist.get(a, 0) + 1
            epistemic_dist[e]   = epistemic_dist.get(e, 0) + 1
            frame_type_dist[ft] = frame_type_dist.get(ft, 0) + 1
            post_role_dist[pr]  = post_role_dist.get(pr, 0) + 1

    # Count example_context fill rate
    non_universal_frames = sum(
        1 for r in all_results
        for fr in r.get("frames", [])
        if fr.get("applicability") in ("context_dependent", "situational")
    )
    example_context_on_non_universal = sum(
        1 for r in all_results
        for fr in r.get("frames", [])
        if fr.get("applicability") in ("context_dependent", "situational")
        and fr.get("example_context")
    )
    ec_rate = (
        f"{example_context_on_non_universal}/{non_universal_frames}"
        f" ({100*example_context_on_non_universal/non_universal_frames:.0f}%)"
        if non_universal_frames else "n/a"
    )

    skipped = sum(1 for r in all_results if r.get("skip_reason"))
    with_frames = sum(1 for r in all_results if r.get("frames"))

    def _dist_str(d: Dict[str, int]) -> str:
        return ", ".join(f"{k}:{v}" for k, v in sorted(d.items(), key=lambda x: -x[1]))

    total_elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE  {n_run} threads, {n_actual_posts} posts, {total_frames} frames")
    print(f"  Posts with frames : {with_frames}")
    print(f"  Posts skipped     : {skipped}")
    print(f"  EC fill (non-univ): {ec_rate}")
    print(f"  Frame types       : {_dist_str(frame_type_dist)}")
    print(f"  Post roles        : {_dist_str(post_role_dist)}")
    print(f"  Scopes            : {_dist_str(scope_dist)}")
    print(f"  Applicability     : {_dist_str(apply_dist)}")
    print(f"  Epistemic stance  : {_dist_str(epistemic_dist)}")
    print(f"  Time              : {total_elapsed:.1f}s")
    print(f"  Output            : {out_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
