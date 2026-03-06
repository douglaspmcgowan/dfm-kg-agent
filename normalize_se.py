"""
Stack Exchange -> extraction-ready format normalizer.

Reads stackexchange_posts_filtered.json (or full SE JSON) and produces
se_posts_normalized.json -- a list of thread objects where each thread
contains a question post and its answer posts, with HTML converted to
plain text and fields mapped to the extraction pipeline's expected format.

Usage:
    python normalize_se.py
    python normalize_se.py --input path/to/se_posts.json
    python normalize_se.py --output se_posts_custom.json
"""

import argparse
import html
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup


# ─────────────────────────────────────────────
# HTML -> PLAIN TEXT
# ─────────────────────────────────────────────

def html_to_text(html_str: str) -> str:
    """
    Convert SE HTML body to plain text for extraction.

    - <blockquote> content is KEPT as regular text (SE blockquotes are
      formatting/emphasis, not inter-user quoting like PM's BBCode).
    - <img> tags are removed (images can't be extracted as text).
    - <pre><code> blocks become plain text code.
    - Lists become bullet/numbered text.
    - Links become just their text content.
    - HTML entities are decoded.
    """
    if not html_str:
        return ""

    soup = BeautifulSoup(html_str, "html.parser")

    # Remove <img> tags entirely
    for img in soup.find_all("img"):
        img.decompose()

    # Convert <pre><code> blocks to plain text
    for pre in soup.find_all("pre"):
        code = pre.find("code")
        code_text = code.get_text() if code else pre.get_text()
        pre.replace_with("\n" + code_text.strip() + "\n")

    # Convert ordered/unordered lists
    for ul in soup.find_all(["ul", "ol"]):
        items = []
        for i, li in enumerate(ul.find_all("li", recursive=False), 1):
            prefix = f"{i}. " if ul.name == "ol" else "- "
            items.append(prefix + li.get_text().strip())
        ul.replace_with("\n".join(items) + "\n")

    # Get text -- BeautifulSoup handles <p>, <br>, <a>, <strong>, <em>, etc.
    text = soup.get_text(separator="\n")

    # Decode any remaining HTML entities
    text = html.unescape(text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ─────────────────────────────────────────────
# CREDIBILITY TIER
# ─────────────────────────────────────────────

def compute_se_credibility(reputation: Optional[int]) -> str:
    """
    Map SE reputation to credibility tiers.

    Thresholds aligned with SE privilege levels:
      high:   >= 10,000 (close/reopen privileges, access to mod tools)
      medium: >= 1,000  (established user, can edit/review)
      low:    < 1,000
    """
    if reputation is None:
        return "low"
    if reputation >= 10_000:
        return "high"
    if reputation >= 1_000:
        return "medium"
    return "low"


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────

def normalize_post(
    se_post: Dict[str, Any],
    post_data: Dict[str, Any],
    post_type: str,
    position: int,
    thread_title: str,
    thread_url: str,
    se_tags: List[str],
    se_site: str,
    question_owner_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normalize one SE post (question or answer) into extraction-ready format.

    post_data is the raw SE object:
      - For questions: the top-level SE question object (body in question_body)
      - For answers: an answer object from the answers array (body in body)
    """
    owner = (
        post_data.get("question_owner") or post_data.get("owner") or {}
    )
    display_name = owner.get("display_name") or "unknown"
    reputation = owner.get("reputation")

    # Body field differs between questions and answers
    if post_type == "question":
        body_html = post_data.get("question_body") or ""
        creation_epoch = post_data.get("question_creation_date")
        score = post_data.get("question_score") or 0
        post_id = f"se-{se_site}-{se_post.get('question_id')}"
        is_accepted = None
    else:
        body_html = post_data.get("body") or ""
        creation_epoch = post_data.get("creation_date")
        score = post_data.get("score") or 0
        answer_id = post_data.get("answer_id")
        post_id = f"se-{se_site}-{se_post.get('question_id')}-a{answer_id}"
        is_accepted = post_data.get("is_accepted", False)

    text = html_to_text(body_html)

    # Convert epoch to ISO 8601
    date_iso = None
    if creation_epoch:
        date_iso = datetime.fromtimestamp(
            creation_epoch, tz=timezone.utc
        ).isoformat()

    # Is this the OP following up in an answer?
    is_op_followup = (
        post_type == "answer"
        and question_owner_name
        and display_name == question_owner_name
    )

    return {
        "id": post_id,
        "post_type": post_type,
        "post_position": position,
        "thread_title": thread_title,
        "thread_url": thread_url,
        "author": display_name,
        "author_reputation": reputation,
        "author_credibility_tier": compute_se_credibility(reputation),
        "date": date_iso,
        "text": text,
        "text_length": len(text),
        "score": score,
        "is_accepted": is_accepted,
        "is_op_followup": is_op_followup,
        "se_tags": se_tags,
        "se_site": se_site,
    }


def normalize_thread(se_post: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one SE question (+ its answers) into a thread object.
    """
    site = se_post.get("site") or "engineering"
    qid = se_post.get("question_id")
    title = html.unescape(se_post.get("question_title") or "")
    url = se_post.get("question_link") or ""
    tags = se_post.get("question_tags") or []
    question_owner = se_post.get("question_owner") or {}
    question_owner_name = question_owner.get("display_name")

    posts = []

    # Question post
    q_post = normalize_post(
        se_post=se_post,
        post_data=se_post,
        post_type="question",
        position=1,
        thread_title=title,
        thread_url=url,
        se_tags=tags,
        se_site=site,
        question_owner_name=question_owner_name,
    )
    posts.append(q_post)

    # Answer posts (sorted by creation_date for stable ordering)
    answers = sorted(
        se_post.get("answers") or [],
        key=lambda a: a.get("creation_date") or 0,
    )
    for i, answer in enumerate(answers, start=2):
        a_post = normalize_post(
            se_post=se_post,
            post_data=answer,
            post_type="answer",
            position=i,
            thread_title=title,
            thread_url=url,
            se_tags=tags,
            se_site=site,
            question_owner_name=question_owner_name,
        )
        posts.append(a_post)

    return {
        "thread_id": f"se-{site}-{qid}",
        "thread_url": url,
        "thread_title": title,
        "se_tags": tags,
        "se_site": site,
        "posts": posts,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

DEFAULT_INPUT = os.path.join(
    "src", "scraping_logic", "stackexchange_posts_filtered.json"
)
DEFAULT_OUTPUT = "se_posts_normalized.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize SE posts to extraction-ready format"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input SE JSON file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output normalized JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        se_posts = json.load(f)

    threads = []
    total_posts = 0
    for se_post in se_posts:
        thread = normalize_thread(se_post)
        threads.append(thread)
        total_posts += len(thread["posts"])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(threads, f, indent=2, ensure_ascii=False)

    n_questions = len(se_posts)
    n_answers = total_posts - n_questions
    print(f"Normalized {n_questions} SE questions + {n_answers} answers")
    print(f"  -> {len(threads)} threads, {total_posts} total posts")
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
