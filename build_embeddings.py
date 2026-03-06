"""
Build Qdrant embedding index from extracted SE frames.

Creates three collections (each with its own vector space):
  frames_knowledge  -- subject + main_point + example_context + example_value + thread_context
  frames_language   -- source_quote (raw language)
  frames_metadata   -- structured metadata fields as natural language text

All three collections store the same payload (metadata) per frame, but embed
different text. This enables multi-vector search with merged scoring.

Data source: StackExchange format (extractions_se*.json)

Run once after extraction, or whenever the extractions file changes.

Usage:
  python build_embeddings.py                          # default: extractions_se_all.json
  python build_embeddings.py extractions_se.json      # specify input file

Requires: pip install qdrant-client openai
"""

import json
import sys
import os
import time
from openai import OpenAI
from qdrant_client import QdrantClient, models


# -----------------------------------------------
# CONFIG
# -----------------------------------------------

QDRANT_PATH = "./qdrant_data"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072  # dimensions for text-embedding-3-large
BATCH_SIZE = 2048  # OpenAI max per request

COLLECTION_NAMES = ["frames_knowledge", "frames_language", "frames_metadata"]

openai_client = OpenAI()  # reads OPENAI_API_KEY from environment


# -----------------------------------------------
# LOAD FRAMES
# -----------------------------------------------

def load_frames(path):
    """Load frames from SE extractions JSON, flatten into a list with unique IDs."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    frames = []
    for result in data["results"]:
        if not result.get("has_relevant_content"):
            continue
        for i, frame in enumerate(result["frames"]):
            frame_id = f"{result['post_id']}-{i}"
            frames.append({
                "id": frame_id,
                "frame": frame,
                "post_meta": result,
            })
    return frames


# -----------------------------------------------
# TEXT BUILDERS
# -----------------------------------------------

def build_knowledge_text(frame):
    """subject + main_point + example_context + example_value + thread_context
    -> distilled knowledge claim."""
    parts = []

    subject = frame.get("subject") or ""
    main_point = frame.get("main_point", "")

    if subject:
        parts.append(f"{subject}: {main_point}")
    else:
        parts.append(main_point)

    example_context = frame.get("example_context") or ""
    if example_context:
        parts.append(example_context)

    example_value = frame.get("example_value") or ""
    if example_value:
        parts.append(example_value)

    thread_context = frame.get("thread_context") or ""
    if thread_context:
        parts.append(thread_context)

    return " ".join(parts)


def build_language_text(frame):
    """Source quote as-is -- raw language."""
    return frame.get("source_quote", "")


def build_metadata_text(frame, post_meta=None):
    """Structured metadata fields as readable natural language text.

    Includes: frame_type, post_role, scope, subject, applicability,
    epistemic_stance, extraction_rationale, validation status,
    validation_notes, related frame count, contradiction status,
    se_tags, is_accepted_answer, se_site, user_type.
    """
    parts = []

    # Core classification
    frame_type = (frame.get("frame_type") or "").replace("_", " ")
    parts.append(f"frame type: {frame_type}")

    post_role = (frame.get("post_role") or "").replace("_", " ")
    parts.append(f"post role: {post_role}")

    scope = (frame.get("scope") or "").replace("_", " ")
    parts.append(f"scope: {scope}")

    subject = frame.get("subject") or "unspecified"
    parts.append(f"subject: {subject}")

    applicability = (frame.get("applicability") or "").replace("_", " ")
    parts.append(f"applicability: {applicability}")

    epistemic = (frame.get("epistemic_stance") or "").replace("_", " ")
    parts.append(f"epistemic stance: {epistemic}")

    # Extraction rationale (often null -- include when present)
    rationale = frame.get("extraction_rationale")
    if rationale:
        parts.append(f"extraction rationale: {rationale}")
    else:
        parts.append("extraction rationale: none")

    # Validation status
    validated = frame.get("validated")
    if validated is True:
        parts.append("validated: yes")
    elif validated is False:
        parts.append("validated: no, rejected")
    else:
        parts.append("validated: not yet reviewed")

    # Validation notes
    val_notes = frame.get("validation_notes")
    if val_notes:
        parts.append(f"validation notes: {val_notes}")

    # Related frames -- count only (raw IDs are noise for embeddings)
    related = frame.get("related_frames") or []
    parts.append(f"related frames: {len(related)}")

    # Contradictions -- count + flag
    contradicted = frame.get("contradicted_by") or []
    if contradicted:
        parts.append(f"contradicted by {len(contradicted)} other frames")
    else:
        parts.append("no known contradictions")

    # SE-specific fields
    if post_meta:
        se_tags = post_meta.get("se_tags")
        if se_tags:
            parts.append(f"tags: {', '.join(se_tags)}")

        is_accepted = frame.get("is_accepted_answer")
        if is_accepted is None:
            is_accepted = post_meta.get("is_accepted_answer")
        if is_accepted is True:
            parts.append("accepted answer: yes")
        elif is_accepted is False:
            parts.append("accepted answer: no")

        se_site = post_meta.get("se_site")
        if se_site:
            parts.append(f"site: {se_site}")

    # user_type (from author data if present -- not currently in SE extractions)
    user_type = post_meta.get("user_type") if post_meta else None
    if user_type:
        parts.append(f"user type: {user_type}")

    return ". ".join(parts)


# -----------------------------------------------
# PAYLOAD BUILDER (structured metadata for filtering)
# -----------------------------------------------

def build_payload(frame, post_meta, knowledge_text, language_text, metadata_text):
    """Build payload dict for Qdrant points.

    Includes all filterable metadata fields plus the document texts
    (so we can display them in search results without a separate lookup).
    """
    related = frame.get("related_frames") or []
    contradicted = frame.get("contradicted_by") or []

    # Post score (upvotes - downvotes)
    post_score = (
        frame.get("post_score")
        or post_meta.get("post_score")
        or 0
    )

    # Author reputation
    author_reputation = post_meta.get("author_reputation") or 0

    payload = {
        # Core frame classification
        "frame_type": frame.get("frame_type") or "",
        "scope": frame.get("scope") or "",
        "applicability": frame.get("applicability") or "",
        "epistemic_stance": frame.get("epistemic_stance") or "",
        "post_role": frame.get("post_role") or "",
        "subject": frame.get("subject") or "",

        # Post/author identity
        "author_username": frame.get("author_username") or post_meta.get("author_username") or "",
        "post_id": frame.get("post_id") or post_meta.get("post_id") or "",
        "thread_title": post_meta.get("thread_title") or "",

        # Date and URL
        "post_date": frame.get("post_date") or post_meta.get("post_date") or "",
        "source_url": frame.get("source_url") or post_meta.get("thread_url") or "",

        # Numeric scores
        "post_score": post_score,
        "author_reputation": author_reputation,
        "author_credibility_tier": post_meta.get("author_credibility_tier") or "",

        # SE-specific
        "se_tags": ", ".join(post_meta.get("se_tags") or []),
        "se_site": post_meta.get("se_site") or "",
        "is_accepted_answer": frame.get("is_accepted_answer") if frame.get("is_accepted_answer") is not None else (post_meta.get("is_accepted_answer") or False),
        "post_type": post_meta.get("post_type") or "",
        "question_id_ref": frame.get("question_id_ref") or post_meta.get("question_id_ref") or "",
        "post_position": post_meta.get("post_position") or 0,

        # Author extras (may be null in current data)
        "accept_rate": post_meta.get("accept_rate") or 0,

        # Review status
        "validated": str(frame.get("validated")) if frame.get("validated") is not None else "unreviewed",
        "related_frame_count": len(related),
        "has_contradictions": len(contradicted) > 0,

        # Document texts for display in search results
        "knowledge_text": knowledge_text,
        "language_text": language_text,
        "metadata_text": metadata_text,

        # Frame ID for cross-collection merging
        "frame_id": "",  # set by caller
    }

    return payload


# -----------------------------------------------
# EMBEDDING
# -----------------------------------------------

def embed_texts(texts):
    """Embed a list of texts via OpenAI. Returns list of vectors."""
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        response = openai_client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        vectors.extend([item.embedding for item in response.data])
    return vectors


# -----------------------------------------------
# BUILD INDEX
# -----------------------------------------------

def build_index(frames, append=False):
    """Create/rebuild all three Qdrant collections, or append to existing."""
    # Create persistent Qdrant client (stores to local folder)
    qdrant = QdrantClient(path=QDRANT_PATH)

    if append:
        # Append mode: keep existing collections, offset IDs
        info = qdrant.get_collection(COLLECTION_NAMES[0])
        id_offset = info.points_count
        print(f"Append mode: existing {id_offset} points, appending {len(frames)} new")
    else:
        id_offset = 0
        # Delete existing collections (makes script re-runnable)
        for name in COLLECTION_NAMES:
            if qdrant.collection_exists(name):
                qdrant.delete_collection(name)

        # Create all three collections with same vector config
        for name in COLLECTION_NAMES:
            qdrant.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
            )

    # Build texts and payloads for all frames
    ids = []
    knowledge_texts = []
    language_texts = []
    metadata_texts = []
    payloads = []

    for idx, entry in enumerate(frames):
        ids.append(id_offset + idx)  # Qdrant uses integer IDs
        k_text = build_knowledge_text(entry["frame"])
        l_text = build_language_text(entry["frame"])
        m_text = build_metadata_text(entry["frame"], entry["post_meta"])
        knowledge_texts.append(k_text)
        language_texts.append(l_text)
        metadata_texts.append(m_text)

        payload = build_payload(entry["frame"], entry["post_meta"], k_text, l_text, m_text)
        payload["frame_id"] = entry["id"]
        payloads.append(payload)

    print(f"Embedding {len(ids)} frames across 3 collections...")
    print(f"  Knowledge texts: avg {sum(len(t) for t in knowledge_texts) // len(knowledge_texts)} chars")
    print(f"  Language texts:  avg {sum(len(t) for t in language_texts) // len(language_texts)} chars")
    print(f"  Metadata texts:  avg {sum(len(t) for t in metadata_texts) // len(metadata_texts)} chars")

    t0 = time.time()

    knowledge_embeddings = embed_texts(knowledge_texts)
    print(f"  Knowledge embeddings done ({time.time() - t0:.1f}s)")

    language_embeddings = embed_texts(language_texts)
    print(f"  Language embeddings done ({time.time() - t0:.1f}s)")

    metadata_embeddings = embed_texts(metadata_texts)
    print(f"  Metadata embeddings done ({time.time() - t0:.1f}s)")

    # Upsert into each collection
    # All three collections get the SAME payloads, but different vectors
    qdrant.upsert(
        collection_name="frames_knowledge",
        points=[
            models.PointStruct(id=ids[i], vector=knowledge_embeddings[i], payload=payloads[i])
            for i in range(len(ids))
        ],
    )
    print(f"  Knowledge collection stored")

    qdrant.upsert(
        collection_name="frames_language",
        points=[
            models.PointStruct(id=ids[i], vector=language_embeddings[i], payload=payloads[i])
            for i in range(len(ids))
        ],
    )
    print(f"  Language collection stored")

    qdrant.upsert(
        collection_name="frames_metadata",
        points=[
            models.PointStruct(id=ids[i], vector=metadata_embeddings[i], payload=payloads[i])
            for i in range(len(ids))
        ],
    )
    print(f"  Metadata collection stored")

    elapsed = time.time() - t0
    print(f"\nStored {len(ids)} frames in 3 collections ({elapsed:.1f}s total)")
    print(f"Database saved to {QDRANT_PATH}/")
    print(f"  Total vectors: {len(ids) * 3}")

    qdrant.close()


# -----------------------------------------------
# MAIN
# -----------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Qdrant embedding index from SE extractions")
    parser.add_argument("input_file", nargs="?", default="extractions_se_all.json",
                        help="Extractions JSON file (default: extractions_se_all.json)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing collections instead of rebuilding")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"ERROR: {args.input_file} not found")
        sys.exit(1)

    frames = load_frames(args.input_file)
    print(f"Loaded {len(frames)} frames from {args.input_file}")

    if not frames:
        print("No frames to embed.")
        sys.exit(0)

    build_index(frames, append=args.append)
