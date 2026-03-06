"""
Schema for DFM Knowledge Graph frame extraction -- Stack Exchange variant.
Reference: DFM Knowledge Graph -- Project Reference Document v2

Based on schema.py with SE-specific adaptations:
  - SEAuthorFrame replaces AuthorFrame (reputation-based, no rank/post_count)
  - KnowledgeFrame adds post_score, is_accepted_answer, question_id_ref
  - SEPostExtractionResult replaces PostExtractionResult (SE metadata)
  - Removes post_reaction_count (replaced by post_score)

All controlled vocabularies (FrameType, ScopeDomain, etc.) are identical
to schema.py -- they are source-agnostic.

DESIGN NOTES:
  - All model guidance lives in Field(description=...) here, not in extract_se.py's system prompt.
    The system prompt covers behavioral rules only (priority, completeness, example handling).
    Frame type definitions and per-field filling guidance live entirely in the schema.
  - ExtractionOutput is the schema sent to the LLM via response_format. It contains ONLY
    fields the LLM should populate (has_relevant_content + frames). Post/author metadata
    is NOT in ExtractionOutput -- extract_se.py adds it from normalizer data afterward.
  - SEPostExtractionResult is the output data structure written to JSON files. It is NOT
    sent to the LLM.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


# -----------------------------------------------
# CONTROLLED VOCABULARIES (identical to schema.py)
# -----------------------------------------------

FrameType = Literal[
    "risk",           # condition(s) -> failure mode OR operational limitation/constraint
    "heuristic",      # personal practice/preference with rationale, OR directed advice with rationale
    "principle",      # causal explanation WHY something happens
    "case",           # specific real-world job or incident narrative
    "workaround",     # non-standard fix for a stated limitation
    "observation",    # empirical/factual/definitional claim -- no action, no causal mechanism
    "comparison",     # weighs two or more options/approaches against each other
    "other",          # explain in extraction_rationale
]

ScopeDomain = Literal[
    "design_geometry",     # part features, tolerances, wall thickness, radii, hole sizing
    "machining_process",   # toolpaths, speeds/feeds, fixturing, operations, sequencing
    "material",            # material behavior, machinability, selection, stock availability
    "tooling",             # tool selection, geometry, wear, availability, breakage
    "programming",         # CNC code, macros, CAM strategy, control-specific behavior
    "shop_operations",     # scheduling, machine availability, personnel, capacity
    "quality",             # inspection, surface finish, tolerances, measurement
    "cost",                # cost drivers, quoting, setup time, efficiency, batch decisions
    "machine_capability",  # specific machine/control features, limits, configurations
    "other",               # explain in extraction_rationale
]

Applicability = Literal[
    "universal",         # true across virtually all instances of the subject
    "context_dependent", # DEFAULT -- true within the subject under specific contexts (material, brand, operation)
    "situational",       # specific to a particular shop/machine configuration or installed option
]

PostRole = Literal[
    "question",       # OP asking for help or information
    "solution",       # direct answer to a problem posed in the thread
    "addition",       # reply that adds supplementary info, tips, alternatives (no struggle narrative)
    "general_advice", # unprompted guidance not tied to a specific problem
    "tutorial",       # teaching-oriented post explaining concepts, processes, or techniques
    "war_story",      # narrative of a SPECIFIC past job with struggle/challenge/surprise -- requires story arc
    "debate",         # disagreement between users; strong contradiction signal
    "reaction",       # reply that endorses, acknowledges, thanks, or briefly responds without substantive knowledge
    "other",
]

CredibilityTier = Literal["high", "medium", "low"]

EpistemicStance = Literal[
    "absolute_authoritative",  # objective fact, no hedges, strong language (must, always, never)
    "direct_neutral",          # stated as fact, technical/tutorial tone, no emphasis or hedges
    "hedged_emphatic",         # personal hedges (in my experience) BUT strong language (hate, required)
    "tentative_subjective",    # weak/uncertain language (might, some, usually, I think)
    "narrative_implied",       # not directly stated, inferred from story or results
]


# -----------------------------------------------
# SE AUTHOR FRAME
# -----------------------------------------------

class SEAuthorFrame(BaseModel):
    """
    One record per unique SE author.
    SE provides reputation-based metrics instead of PM's rank/post_count system.
    """
    author_username: str = Field(description="Display name on Stack Exchange. Primary key.")
    reputation: Optional[int] = Field(default=None, description="SE reputation score.")
    user_type: Optional[str] = Field(
        default=None,
        description="SE user type: 'registered', 'unregistered', 'moderator', 'does_not_exist'."
    )
    accept_rate: Optional[int] = Field(
        default=None,
        description="Percentage of user's questions with accepted answers (0-100). Sparse -- only present for some users."
    )
    author_credibility_tier: CredibilityTier = Field(
        description=(
            "Derived from reputation. "
            "high = reputation >= 10000 (close/reopen privileges); "
            "medium = reputation >= 1000 (established user); "
            "low = everything else."
        )
    )


# -----------------------------------------------
# KNOWLEDGE FRAME (SE variant)
# -----------------------------------------------

class KnowledgeFrame(BaseModel):
    """
    One extracted piece of knowledge from a Stack Exchange post.

    Sections:
      1. Source and context     -- traceability back to the post
      2. Knowledge content      -- scope, subject, applicability, main claim
      3. Example illustration   -- optional example + required context for non-universal
      4. Epistemic metadata     -- how the claim is expressed
      5. Extraction metadata    -- LLM reasoning
      6. Review fields          -- populated post-extraction, not by LLM
    """

    # -- 1. Source and Context --

    frame_type: FrameType = Field(
        description=(
        """Pick the MOST specific type that fits the source_quote.

- risk: condition(s) -> failure mode OR operational limitation. "X will get you in trouble if Y", "cannot do X", "not supported", "not available on all controls". Includes machine/control/tooling/physics limitations that impose boundaries.
- heuristic: author's personal rule, practice, or preference with rationale ("I do/avoid X because Y"), OR directed advice ("you should do X because Y"). If the author says "I do/use/avoid/hate/prefer X", that is a heuristic even if it reads as advice.
- principle: causal explanation WHY something happens. Physical, mechanical, or logical cause -> effect.
- observation: empirical/factual/definitional claim about how something IS or BEHAVES. No personal action, no causal mechanism.
- comparison: explicitly weighs two or more options against each other. Must contain contrast between alternatives.
- workaround: non-standard fix done BECAUSE of a stated limitation.
- case: specific real-world job or incident with context and resolution.
- other: genuine value that doesn't fit above -- explain in extraction_rationale.

When a quote contains an example/illustration, extract the underlying knowledge -- do not label the example itself as a type."""
    ))

    author_username: Optional[str] = Field(
        default=None,
        description="Post author display name. Foreign key to SEAuthorFrame. Populated by extract_se.py."
    )

    source_url: Optional[str] = Field(
        default=None,
        description="Question URL from SE. Populated by extract_se.py."
    )

    post_id: Optional[str] = Field(
        default=None,
        description="Post ID from normalizer. Populated by extract_se.py."
    )

    post_date: Optional[str] = Field(
        default=None,
        description="Post date ISO string. Populated by extract_se.py."
    )

    post_role: PostRole = Field(
        description=(
            "Role of this post in the thread. Choose the best fit:\n"
            "- question: OP asking for help or information\n"
            "- solution: direct answer to a problem posed in the thread\n"
            "- addition: reply that adds supplementary information, tips, alternatives, or related experience WITHOUT a narrative of personal struggle. Most replies that share facts, techniques, or 'another option is...' are additions\n"
            "- general_advice: unprompted guidance not tied to a specific problem in the thread\n"
            "- tutorial: teaching-oriented post explaining concepts or techniques -- whether formally structured or conversational\n"
            "- war_story: narrative of a SPECIFIC past job or incident where the author faced a struggle, challenge, or surprising outcome. REQUIRES a concrete story arc (situation -> problem -> resolution). A reply that merely mentions experience or shares tips is NOT a war_story\n"
            "- debate: disagreement between users\n"
            "- reaction: reply that endorses, acknowledges, thanks, or briefly responds without adding substantive knowledge\n"
            "- other: explain in extraction_rationale"
        )
    )

    post_score: Optional[int] = Field(
        default=None,
        description="Platform score (upvotes - downvotes) for this post. Populated by extract_se.py."
    )

    is_accepted_answer: Optional[bool] = Field(
        default=None,
        description="Whether this post was the accepted answer. Null for questions. Populated by extract_se.py."
    )

    question_id_ref: Optional[str] = Field(
        default=None,
        description="For answer-sourced frames: the question post ID this answers. Populated by extract_se.py."
    )

    source_quote: str = Field(
        ...,
        description=(
        """Verbatim substring from the post text supporting this ONE claim. Must include the key limitation/cause/advice words. Prefer quoting the sentence that states the actual knowledge claim rather than standalone scaffold lines, but include surrounding context if it supports the claim."""
    ))

    # -- 2. Knowledge Content --

    scope: ScopeDomain = Field(
        description=(
            "Primary knowledge domain this frame belongs to.\n"
            "- design_geometry: part features, tolerances, wall thickness, hole sizing, radii\n"
            "- machining_process: toolpaths, speeds/feeds, fixturing, operations, sequencing\n"
            "- material: material behavior, machinability, selection, stock\n"
            "- tooling: tool selection, geometry, wear, availability, breakage\n"
            "- programming: CNC code, macros, CAM strategy, control-specific behavior\n"
            "- shop_operations: scheduling, machine availability, personnel, capacity\n"
            "- quality: inspection, surface finish, tolerances, measurement\n"
            "- cost: cost drivers, quoting, setup efficiency, batch decisions\n"
            "- machine_capability: specific machine/control features, limits, configurations\n"
            "- other: ONLY if genuinely none fit -- explain in extraction_rationale"
        )
    )

    subject: Optional[str] = Field(
        default=None,
        description=(
            "Concise noun phrase for the primary concept. This becomes the graph node anchor.\n"
            "Examples: 'local variable volatility', 'system variable non-standardization',\n"
            "'thin wall deflection', 'tool overhang', 'carbide vs HSS end mills',\n"
            "'machine capability gap', 'canned cycle support'."
        )
    )

    applicability: Applicability = Field(
        description=(
        """How broadly the claim applies WITHIN the subject.

applicability narrows scope and subject further:
  scope (broadest) -> subject (specific concept) -> applicability (how widely it holds within that subject).

- universal: the claim is a fundamental fact that holds regardless of specific equipment,
  material, or configuration. Basic physics, geometry, universally accepted practice,
  or definitional truths. Use universal when the claim would be true in ANY shop with
  ANY equipment for this subject. If the subject itself is narrow (e.g. "Fanuc macro
  variables"), universal means true for ALL Fanuc macro variables, not all CNC programming.
- context_dependent: the claim mentions or implies a specific context that limits where
  it applies -- a material type, control brand/family, operation category, or technique.
  Use when the source_quote references particular equipment, materials, or conditions.
- situational: the claim depends on a specific shop configuration, installed option,
  or one-off scenario that most shops would not share.

Deciding between universal and context_dependent:
  "Canned cycles are macros" -> universal (definitional truth, all controls)
  "Local variables reset to null on program end" -> universal (true for all controls with local vars)
  "I use global variables because they are retained" -> context_dependent (personal practice, implies specific control context)
  "Not all controls support custom macro B" -> context_dependent (control-specific)
  "Our shop's Haas with the WIPS option..." -> situational (specific shop config)

Control/brand/option mentions (Fanuc/Haas/"not all controls") -> not universal.
Hedges (usually/typically/in my experience/a few) -> not universal."""
    ))

    main_point: str = Field(
        ...,
        description=(
        """State ONE transferable knowledge claim (2-3 sentences) fully supported by source_quote. No meta-framing. No verbatim copying. Do not merge multiple independent claims; split into multiple frames if needed. The main_point must state a general lesson or fact -- not a tutorial setting or worked-example decision. Include any important threshold, value, or boundary condition that makes the claim actionable. For risk frames, include both the condition and the failure mode/limitation."""
    ))

    # -- 3. Example Illustration --

    example_value: Optional[str] = Field(
        default=None,
        description=(
        """Fill if the post includes a concrete scenario or worked example that contextualizes the claim. Summarize the example in 1-2 sentences. Leave null if no example is present."""
    ))

    example_context: Optional[str] = Field(
        default=None,
        description=(
        """FILL THIS FIELD when applicability is context_dependent or situational. Describe the real-world context that scopes this claim: what machine, material, operation, control, or shop situation makes it relevant.

Examples: "Fanuc controls with custom macro B", "thin-walled aluminum parts under 2mm", "high-volume production shops", "older Haas machines without WIPS option".

Leave null ONLY when applicability is universal. If you set applicability to context_dependent or situational, you MUST fill example_context."""
    ))

    # -- 4. Epistemic Metadata --

    epistemic_stance: EpistemicStance = Field(
        description=(
        """How the claim is expressed in the source_quote.

- absolute_authoritative: stated as objective fact with strong language (must, always, never, cannot, only, required, critical). No hedges.
- direct_neutral: DEFAULT -- stated as fact, technical or tutorial tone, no emphasis and no hedges. When in doubt, use direct_neutral.
- hedged_emphatic: personal hedges (I feel, in my experience, I find) BUT strong internal language (hate, required, critical). Emphatic about a subjective view.
- tentative_subjective: weak or uncertain language (might, could, maybe, I think, seems, usually, typically, some, a few, most of the time).
- narrative_implied: the claim is not directly stated but clearly inferred from a story, example, or described outcome."""
    ))

    # -- 5. Extraction Metadata --

    extraction_rationale: Optional[str] = Field(
        default=None,
        description=(
            "LLM reasoning for extraction decisions. REQUIRED if frame_type is 'other' "
            "or scope is 'other' -- explain why none of the standard options fit. "
            "Optional otherwise."
        )
    )

    # -- 6. Review Fields (populated post-extraction, not by LLM) --

    contradicted_by: Optional[List[str]] = Field(
        default=None,
        description=(
            "Frame IDs of frames that contradict this one. "
            "Populated during human review -- do not fill during extraction."
        )
    )

    related_frames: Optional[List[str]] = Field(
        default=None,
        description=(
            "Post IDs or frame identifiers of related frames. "
            "Populated post-extraction. Do not fill during extraction."
        )
    )

    thread_context: Optional[str] = Field(
        default=None,
        description=(
        """Thread context summary. Populated after extraction; always output null during extraction."""
    ))

    validated: Optional[bool] = Field(
        default=None,
        description="Human reviewer confirmed? Null until reviewed."
    )

    validation_notes: Optional[str] = Field(
        default=None,
        description="Reviewer notes: edits, contradiction context, follow-up flags."
    )


# -----------------------------------------------
# EXTRACTION OUTPUT (LLM-facing schema)
# -----------------------------------------------

class ExtractionOutput(BaseModel):
    """
    Schema passed as response_format to client.beta.chat.completions.parse().
    Contains ONLY fields the LLM should populate. Post/author metadata is NOT
    included -- extract_se.py adds it from normalizer data afterward.
    """
    has_relevant_content: bool = Field(
        description=(
            "True if you output at least one extractable knowledge frame.\n"
            "If you output zero frames, set false."
        )
    )
    frames: List[KnowledgeFrame] = Field(default_factory=list)


# -----------------------------------------------
# SE POST-LEVEL RESULT (output data structure)
# -----------------------------------------------

class SEPostExtractionResult(BaseModel):
    """
    Full output record for one SE post. Combines LLM extraction with normalizer
    metadata. Written to extractions JSON files. NOT passed to the LLM.
    """

    # Post identity -- from normalizer
    post_id: str
    post_type: Optional[str] = None          # "question" or "answer"
    thread_url: Optional[str] = None
    thread_title: Optional[str] = None
    post_position: Optional[int] = None
    post_date: Optional[str] = None
    post_score: Optional[int] = None

    # Author identity -- from SE API via normalizer
    author_username: Optional[str] = None
    author_reputation: Optional[int] = None
    author_credibility_tier: Optional[str] = None

    # SE metadata
    se_tags: Optional[List[str]] = None
    se_site: Optional[str] = None
    is_accepted_answer: Optional[bool] = None
    question_id_ref: Optional[str] = None    # for answers: the parent question's post_id

    has_relevant_content: bool
    frames: List[KnowledgeFrame] = Field(default_factory=list)
    skip_reason: Optional[str] = None
