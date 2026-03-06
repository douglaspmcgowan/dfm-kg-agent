"""
Shared Pydantic schemas and prompts for the LLM review + revise pipeline.

Contains:
  - FrameReview: structured review output (moved from llm_reviewer.py)
  - FrameRevision / FieldRevision: structured revision output
  - REVIEW_SYSTEM_PROMPT: reviewer rubric with human-calibrated guidance
  - REVISION_SYSTEM_PROMPT: revision rubric with calibration notes
  - build_user_message(): enhanced with optional post_text
  - build_revision_message(): revision prompt with flagged fields + review feedback
  - CALIBRATION: per-field thresholds from human-LLM gap analysis
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

from schema_se import FrameType, ScopeDomain, Applicability, EpistemicStance


# ---------------------------------------------------------------------------
# Review schema (moved from llm_reviewer.py)
# ---------------------------------------------------------------------------
class FrameReview(BaseModel):
    """Review of a single extracted knowledge frame."""

    extraction_accuracy: Literal[1, 2, 3] = Field(
        description=(
            "Does the main point accurately capture the knowledge in the source quote? "
            "1 = quote does not support the main point or main point misrepresents the quote. "
            "2 = main point mostly captures the quote but has minor omissions or overstatements. "
            "3 = main point accurately and completely captures the knowledge in the quote."
        )
    )

    tribal_knowledge: Literal[1, 2, 3] = Field(
        description=(
            "How specialized is this knowledge? "
            "1 = generic textbook knowledge, easily found in any reference. "
            "2 = moderately specialized -- a practitioner might know this but it is not trivial. "
            "3 = genuine tribal/experiential knowledge unlikely to be found in textbooks."
        )
    )

    clarity: Literal[1, 2, 3] = Field(
        description=(
            "How clear and usable is the main point? "
            "1 = poorly written, ambiguous, or hard to understand. "
            "2 = understandable but could be clearer or more precise. "
            "3 = clear, precise, and immediately actionable."
        )
    )

    frame_type_rating: Literal[1, 2, 3, 4, 5] = Field(
        description=(
            "How appropriate is the system's frame_type label? "
            "1 = completely wrong. 2 = poor, a different label would clearly be better. "
            "3 = acceptable, reasonable but not ideal. 4 = good, correct and appropriate. "
            "5 = excellent, could not be improved."
        )
    )

    post_role_rating: Literal[1, 2, 3, 4, 5] = Field(
        description=(
            "How appropriate is the system's post_role label? "
            "1 = completely wrong. 2 = poor. 3 = acceptable. 4 = good. 5 = excellent."
        )
    )

    scope_rating: Literal[1, 2, 3, 4, 5] = Field(
        description=(
            "How appropriate is the system's scope label? "
            "1 = completely wrong. 2 = poor. 3 = acceptable. 4 = good. 5 = excellent."
        )
    )

    applicability_rating: Literal[1, 2, 3, 4, 5] = Field(
        description=(
            "How appropriate is the system's applicability label? "
            "1 = completely wrong. 2 = poor. 3 = acceptable. 4 = good. 5 = excellent."
        )
    )

    main_point_quality: Literal[1, 2, 3, 4, 5] = Field(
        description=(
            "How well does the main point capture the transferable knowledge? "
            "1 = completely wrong. 2 = poor. 3 = acceptable. 4 = good. 5 = excellent."
        )
    )

    epistemic_stance_rating: Literal[1, 2, 3, 4, 5] = Field(
        description=(
            "How appropriate is the system's epistemic_stance label? "
            "1 = completely wrong. 2 = poor. 3 = acceptable. 4 = good. 5 = excellent."
        )
    )

    overall: Literal["Keep", "Revise", "Remove"] = Field(
        description=(
            "Overall judgment: "
            "Keep = frame is accurate, useful, and correctly classified. "
            "Revise = frame captures real knowledge but needs corrections to labels or main point. "
            "Remove = frame is wrong, trivial, or does not represent meaningful extracted knowledge."
        )
    )

    comments: str = Field(
        description=(
            "Brief explanation of your ratings. Note any label corrections you would suggest, "
            "concerns about the quote, or observations about the extraction quality."
        )
    )


# ---------------------------------------------------------------------------
# Revision schemas
# ---------------------------------------------------------------------------
class FieldRevision(BaseModel):
    """Record of one field that was changed during revision."""

    field_name: str = Field(
        description="Name of the revised field (e.g., 'scope', 'frame_type', 'main_point')"
    )
    original_value: str = Field(
        description="The original value before revision"
    )
    revised_value: str = Field(
        description="The corrected value"
    )
    justification: str = Field(
        description="Brief explanation of why this correction is needed"
    )


class FrameRevision(BaseModel):
    """Structured revision output for one frame."""

    action: Literal["keep", "revise", "remove"] = Field(
        description=(
            "What to do with this frame based on the review feedback. "
            "'keep' = no changes needed, review concerns are not actionable. "
            "'revise' = apply the field corrections below. "
            "'remove' = frame should be dropped (wrong, trivial, or duplicate)."
        )
    )

    # Revisable categorical fields -- output the correct value
    frame_type: FrameType = Field(
        description="Corrected frame_type label. Keep original if no change needed."
    )
    scope: ScopeDomain = Field(
        description="Corrected scope label. Keep original if no change needed."
    )
    applicability: Applicability = Field(
        description="Corrected applicability label. Keep original if no change needed."
    )
    epistemic_stance: EpistemicStance = Field(
        description="Corrected epistemic_stance label. Keep original if no change needed."
    )

    # Revisable text fields
    main_point: str = Field(
        description=(
            "Corrected main_point. Must be fully supported by the source_quote. "
            "Keep original text if no change needed."
        )
    )
    example_context: Optional[str] = Field(
        default=None,
        description=(
            "Corrected or newly-added example_context. Required when applicability "
            "is context_dependent or situational. Keep original if no change needed."
        )
    )

    changes: list[FieldRevision] = Field(
        default_factory=list,
        description="List of fields that were changed. Empty if action is 'keep'."
    )

    revision_rationale: str = Field(
        description=(
            "Overall explanation of the revision decision. "
            "Reference specific review comments and explain corrections."
        )
    )


# ---------------------------------------------------------------------------
# Review system prompt -- rubric with human-calibrated guidance
# ---------------------------------------------------------------------------
REVIEW_SYSTEM_PROMPT = """\
You are an expert reviewer evaluating an AI-powered knowledge extraction pipeline \
for manufacturing/machining domain knowledge.

The pipeline reads Q&A posts from engineering Stack Exchange sites (topics include \
CNC machining, materials, tooling, and shop practices) and uses a large language model \
to extract structured knowledge frames from each post.

A knowledge frame is one discrete, self-contained piece of knowledge extracted from a \
forum post. Each frame captures a single claim, rule, observation, or practice described \
by the post's author.

For each frame you will be shown:
- Source Quote: the verbatim text from the original forum post that the system identified \
as supporting this frame.
- Main Point: a 2-3 sentence summary the system wrote to capture the transferable \
knowledge claim from the quote.
- System-assigned labels: frame_type, post_role, scope, applicability, epistemic_stance.
- Additional context: subject, example_value, example_context (when present).
- Post context: thread title, post score (upvotes minus downvotes), and topic tags.
- Original post text (when available): the full post for verifying source_quote accuracy.

Frame type definitions:
- risk: a condition that leads to a failure mode or operational limitation
- heuristic: a personal practice, preference, or directed advice with rationale
- principle: a causal explanation of WHY something happens
- observation: an empirical, factual, or definitional claim (no action, no causal mechanism)
- comparison: explicitly weighs two or more options against each other
- workaround: a non-standard fix done because of a stated limitation
- case: a specific real-world job or incident narrative
- other: does not fit the above

Applicability levels:
- universal: true across virtually all instances of the subject
- context_dependent: true under specific conditions (material, control brand, operation)
- situational: specific to a particular shop/machine configuration

Epistemic stance levels:
- absolute_authoritative: objective fact, strong language (must, always, never), no hedges
- direct_neutral: stated as fact, technical/tutorial tone, no emphasis or hedges
- hedged_emphatic: personal hedges (in my experience) BUT strong language (hate, required)
- tentative_subjective: weak/uncertain language (might, could, maybe, I think, usually)
- narrative_implied: not directly stated, inferred from a story or described outcome

Post role definitions:
- question: OP asking for help or information
- solution: direct answer to a problem posed in the thread
- addition: reply that adds supplementary info, tips, or alternatives
- general_advice: unprompted guidance not tied to a specific problem
- tutorial: teaching-oriented post explaining concepts or techniques
- war_story: narrative of a specific past job with struggle/challenge/surprise
- debate: disagreement between users
- reaction: reply that endorses, acknowledges, or thanks without substantive knowledge

RATING GUIDANCE:

1. EXTRACTION ACCURACY: Only rate 1 or 2 if the main point clearly misrepresents or \
substantially overstates the source quote. Minor inferential additions that make the \
claim more actionable are acceptable and should not be penalized.

2. TRIBAL KNOWLEDGE: Knowledge that reflects practitioner experience, shop-floor wisdom, \
or hard-won lessons should be rated 2 or 3, even if the underlying concept appears in \
textbooks. The presentation from personal experience matters.

3. SCOPE: The fixed scope taxonomy (design_geometry, machining_process, material, tooling, \
programming, shop_operations, quality, cost, machine_capability, other) sometimes does \
not perfectly fit cross-domain knowledge. Rate scope 3 or higher if the assigned label is \
in the right general area, even if a more nuanced label might be better. Only rate 1-2 \
if the label is clearly wrong (e.g., "cost" for a frame about material behavior).

4. POST ROLE: Be attentive to whether a post is truly a "solution" vs "addition" vs \
"general_advice." Consider the post's function in the thread context.

Rate each frame using the structured output schema. Be calibrated and consistent."""


# ---------------------------------------------------------------------------
# Revision system prompt
# ---------------------------------------------------------------------------
REVISION_SYSTEM_PROMPT = """\
You are revising a knowledge frame extracted from a technical forum post. A previous \
review step identified issues with this frame's metadata labels and/or main point. \
Your job is to correct the specified fields while preserving the frame's source_quote \
anchor and overall knowledge content.

REVISION RULES:
1. Do NOT change source_quote -- it is the verbatim provenance anchor.
2. Do NOT change subject -- it is the graph node anchor.
3. Do NOT change post_role -- it is shared across all frames in this post.
4. main_point must be FULLY supported by the source_quote. Remove any claims not \
present in the quote. Do not add information beyond what the quote states.
5. When changing applicability to context_dependent or situational, you MUST provide \
example_context describing the specific conditions.
6. If the review comments suggest the frame should be removed, set action="remove" \
and explain in revision_rationale.
7. If you disagree with the review and the original labels are correct, set \
action="keep" and explain why in revision_rationale.

SCOPE LABELS (choose from):
design_geometry, machining_process, material, tooling, programming, shop_operations, \
quality, cost, machine_capability, other

FRAME TYPES (choose from):
risk, heuristic, principle, case, workaround, observation, comparison, other

APPLICABILITY (choose from):
universal, context_dependent, situational

EPISTEMIC STANCE (choose from):
absolute_authoritative, direct_neutral, hedged_emphatic, tentative_subjective, narrative_implied

CALIBRATION NOTES (from human expert validation):
- The review system tends to rate scope more harshly than human experts (mean diff = -0.86 \
on a 5-point scale). Only change the scope label if it is clearly wrong (e.g., "cost" for \
a frame about material behavior). Borderline cases should keep the original label.
- The review system tends to flag main points for adding "unstated information." Minor \
inferential additions are acceptable if they make the claim more actionable. Only revise \
if the main point fundamentally misrepresents the source quote.
- Human experts accepted 83% of frames as Keep. The review system accepted only 51%. \
Err on the side of keeping the original when the review feedback is borderline."""


# ---------------------------------------------------------------------------
# Calibration thresholds (from human-LLM gap analysis)
# ---------------------------------------------------------------------------
CALIBRATION = {
    "scope_rating": {
        "llm_bias": -0.86,
        "revise_threshold": 2,
        "rationale": "LLM very harsh on scope; only fix clearly wrong labels",
    },
    "main_point_quality": {
        "llm_bias": -0.49,
        "revise_threshold": 3,
        "rationale": "LLM penalizes minor additions; revise only substantive issues",
    },
    "extraction_accuracy": {
        "llm_bias": -0.42,
        "revise_threshold": 1,
        "rationale": "Ceiling effect in human data; only fix true misrepresentations",
    },
    "frame_type_rating": {
        "llm_bias": +0.12,
        "revise_threshold": 2,
        "rationale": "No significant bias; standard threshold",
    },
    "applicability_rating": {
        "llm_bias": -0.19,
        "revise_threshold": 2,
        "rationale": "Moderate bias; only fix clear over-generalizations",
    },
    "epistemic_stance_rating": {
        "llm_bias": +0.24,
        "revise_threshold": 2,
        "rationale": "LLM lenient here; trust its low ratings",
    },
}


# ---------------------------------------------------------------------------
# Build user message for reviewing one frame
# ---------------------------------------------------------------------------
def build_user_message(
    frame: dict,
    post: dict,
    frame_num: int,
    post_text: str | None = None,
) -> str:
    """Build the user message for reviewing a single frame.

    Args:
        frame: the extracted frame dict
        post: the post-level dict (thread_title, post_score, se_tags, etc.)
        frame_num: display number for the frame
        post_text: optional full post text (non_quoted_text) for source_quote verification
    """
    parts = [f"Frame {frame_num}"]

    # Post context
    if post.get("thread_title"):
        parts.append(f"Thread: {post['thread_title']}")
    post_role = frame.get("post_role", "unknown")
    parts.append(f"Post Role: {post_role}")
    if post.get("post_score") is not None:
        parts.append(f"Post Score: {post['post_score']}")
    if post.get("se_tags"):
        parts.append(f"Tags: {', '.join(post['se_tags'])}")

    parts.append("")

    # Original post text (for source_quote verification)
    if post_text:
        parts.append("Original Post Text:")
        parts.append(post_text)
        parts.append("")

    # Source quote
    parts.append("Source Quote:")
    parts.append(frame.get("source_quote", "(no quote)"))
    parts.append("")

    # Main point
    parts.append(f"Main Point: {frame.get('main_point', '(none)')}")
    parts.append("")

    # System-assigned labels
    parts.append("System-assigned labels:")
    parts.append(f"  Frame Type: {frame.get('frame_type', '?')}")
    parts.append(f"  Post Role: {frame.get('post_role', '?')}")
    parts.append(f"  Scope: {frame.get('scope', '?')}")
    parts.append(f"  Applicability: {frame.get('applicability', '?')}")
    parts.append(f"  Epistemic Stance: {frame.get('epistemic_stance', '?')}")
    if frame.get("subject"):
        parts.append(f"  Subject: {frame['subject']}")
    if frame.get("example_value"):
        parts.append(f"  Example Value: {frame['example_value']}")
    if frame.get("example_context"):
        parts.append(f"  Example Context: {frame['example_context']}")
    if frame.get("extraction_rationale"):
        parts.append(f"  Extraction Rationale: {frame['extraction_rationale']}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Build revision prompt for one frame
# ---------------------------------------------------------------------------
def build_revision_message(
    frame: dict,
    post: dict,
    review: dict,
    flagged_fields: list[str],
    post_text: str | None = None,
) -> str:
    """Build the user message for revising a single frame.

    Args:
        frame: the extracted frame dict
        post: the post-level dict
        review: the review dict (ratings + comments)
        flagged_fields: list of field names that triggered revision
        post_text: optional full post text for ground truth
    """
    parts = []

    # Flagged fields
    flag_details = []
    for f in flagged_fields:
        rating = review.get(f, "?")
        flag_details.append(f"{f}={rating}")
    parts.append(f"FIELDS FLAGGED FOR REVISION: {', '.join(flag_details)}")
    parts.append("")

    # Review feedback
    parts.append("REVIEW FEEDBACK:")
    parts.append(f"  Overall: {review.get('overall', '?')}")
    parts.append(f"  Comments: {review.get('comments', '(none)')}")
    parts.append(f"  Ratings: "
                 f"extraction_accuracy={review.get('extraction_accuracy', '?')}, "
                 f"tribal_knowledge={review.get('tribal_knowledge', '?')}, "
                 f"clarity={review.get('clarity', '?')}, "
                 f"frame_type={review.get('frame_type_rating', '?')}, "
                 f"post_role={review.get('post_role_rating', '?')}, "
                 f"scope={review.get('scope_rating', '?')}, "
                 f"applicability={review.get('applicability_rating', '?')}, "
                 f"main_point_quality={review.get('main_point_quality', '?')}, "
                 f"epistemic_stance={review.get('epistemic_stance_rating', '?')}")
    parts.append("")

    # Original frame
    parts.append("ORIGINAL FRAME:")
    parts.append(f"  Frame Type: {frame.get('frame_type', '?')}")
    parts.append(f"  Scope: {frame.get('scope', '?')}")
    parts.append(f"  Applicability: {frame.get('applicability', '?')}")
    parts.append(f"  Epistemic Stance: {frame.get('epistemic_stance', '?')}")
    parts.append(f"  Source Quote: {frame.get('source_quote', '(none)')}")
    parts.append(f"  Main Point: {frame.get('main_point', '(none)')}")
    if frame.get("subject"):
        parts.append(f"  Subject: {frame['subject']}")
    if frame.get("example_value"):
        parts.append(f"  Example Value: {frame['example_value']}")
    if frame.get("example_context"):
        parts.append(f"  Example Context: {frame['example_context']}")
    parts.append("")

    # Post context
    if post.get("thread_title"):
        parts.append(f"Thread: {post['thread_title']}")
    if post.get("se_tags"):
        parts.append(f"Tags: {', '.join(post['se_tags'])}")
    parts.append("")

    # Original post text
    if post_text:
        parts.append("ORIGINAL POST TEXT:")
        parts.append(post_text)

    return "\n".join(parts)
