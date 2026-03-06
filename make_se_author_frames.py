"""
Build SEAuthorFrame records for every unique author in stackexchange_posts_filtered.json.

No LLM calls needed -- all data comes from the SE API owner objects embedded in the posts.
Writes se_author_frames.json.

Usage:
    python make_se_author_frames.py
    INPUT_FILE=src/scraping_logic/stackexchange_posts_filtered.json python make_se_author_frames.py
    OUTPUT_FILE=se_author_frames.json python make_se_author_frames.py
"""

import json
import os
from typing import Dict, Any, Optional

INPUT_FILE  = os.getenv("INPUT_FILE",  "src/scraping_logic/stackexchange_posts_filtered.json")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "se_author_frames.json")


def credibility_tier(reputation: Optional[int]) -> str:
    """
    Derive author_credibility_tier from SE reputation score.
    high   = reputation >= 10000  (close/reopen vote privileges)
    medium = reputation >= 1000   (established user)
    low    = everything else
    """
    if reputation is None:
        return "low"
    if reputation >= 10000:
        return "high"
    if reputation >= 1000:
        return "medium"
    return "low"


def build_author_frame(owner: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw SE owner dict to an SEAuthorFrame dict."""
    rep = owner.get("reputation")
    return {
        "author_username":        owner.get("display_name", ""),
        "user_id":                owner.get("user_id"),
        "account_id":             owner.get("account_id"),
        "reputation":             rep,
        "user_type":              owner.get("user_type"),
        "accept_rate":            owner.get("accept_rate"),   # sparse; present only if SE returned it
        "profile_link":           owner.get("link"),
        "author_credibility_tier": credibility_tier(rep),
    }


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found.")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        posts = json.load(f)

    # Collect best (highest-reputation) owner record per display_name
    # In practice SE reputation is stable, but some authors appear in both question
    # and answer owner fields -- take the one with the highest reputation seen.
    best_by_name: Dict[str, Dict[str, Any]] = {}

    def _update(owner: Optional[Dict[str, Any]]) -> None:
        if not owner:
            return
        name = owner.get("display_name") or ""
        if not name:
            return
        existing = best_by_name.get(name)
        if existing is None:
            best_by_name[name] = owner
        else:
            # Prefer the record with higher reputation (most complete)
            existing_rep = existing.get("reputation") or 0
            new_rep = owner.get("reputation") or 0
            if new_rep > existing_rep:
                best_by_name[name] = owner

    for post in posts:
        _update(post.get("question_owner"))
        for answer in post.get("answers") or []:
            _update(answer.get("owner"))

    author_frames = [build_author_frame(owner) for owner in best_by_name.values()]

    # Sort by descending reputation for readability
    author_frames.sort(key=lambda a: -(a.get("reputation") or 0))

    output = {
        "source_file": INPUT_FILE,
        "total_authors": len(author_frames),
        "author_frames": author_frames,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    tiers = {"high": 0, "medium": 0, "low": 0}
    for af in author_frames:
        tiers[af["author_credibility_tier"]] += 1

    user_types: Dict[str, int] = {}
    for af in author_frames:
        ut = af.get("user_type") or "unknown"
        user_types[ut] = user_types.get(ut, 0) + 1

    print(f"Input  : {INPUT_FILE}")
    print(f"Output : {OUTPUT_FILE}")
    print(f"Total unique authors : {len(author_frames)}")
    print(f"Credibility tiers    : high={tiers['high']} medium={tiers['medium']} low={tiers['low']}")
    print(f"User types           : {', '.join(f'{k}:{v}' for k,v in sorted(user_types.items(), key=lambda x: -x[1]))}")


if __name__ == "__main__":
    main()
