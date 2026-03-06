"""
DFM Report schemas and prompts for V1, V2, and V3 report formats.

V1: Flat lists (risks, recommended_practices, contradictions, knowledge_gaps)
V2: Discipline-organized with reasoning chains, first-principles inference, edge context
V3: V2 + formulas, chain-pull support, gap backfill, relevance diagnostics
"""

from typing import List, Optional
from pydantic import BaseModel


# ─────────────────────────────────────────────
# V1 SCHEMAS (backward compat)
# ─────────────────────────────────────────────

class SourcedClaim(BaseModel):
    claim: str
    frame_id: str


class ContradictionEntry(BaseModel):
    summary: str
    position_a: str
    frame_id_a: str
    position_b: str
    frame_id_b: str


class DFMReportOutput(BaseModel):
    """V1 flat report. Author/URL are enriched post-synthesis from the graph."""
    summary: str
    risks: List[SourcedClaim]
    recommended_practices: List[SourcedClaim]
    contradictions: List[ContradictionEntry]
    knowledge_gaps: Optional[str] = None


# ─────────────────────────────────────────────
# V2/V3 SCHEMAS (discipline-organized + reasoning chains + formulas)
# ─────────────────────────────────────────────

class ChainedFrame(BaseModel):
    """One frame within a reasoning chain, with its role and connecting edge."""
    frame_id: str
    role_in_chain: str  # root_cause, effect, mitigation, condition, origin
    edge_from: Optional[str] = None  # frame_id this connects from
    edge_type: Optional[str] = None  # CAUSAL_LINK, MITIGATES, SPECIFIES, CONTRADICTS
    edge_confidence: Optional[str] = None  # high, medium, low
    claim: str


class ReasoningChain(BaseModel):
    """A connected set of frames with explicit causal/mitigation relationships."""
    chain_label: str  # e.g. "Tool deflection causes chatter in thin walls"
    frames: List[ChainedFrame]


class FormulaEntry(BaseModel):
    """A governing equation or scaling rule relevant to this discipline."""
    formula_name: str       # e.g., "Euler-Bernoulli cantilever deflection"
    expression: str         # e.g., "delta = F*L^3 / (3*E*I)"
    variables: str          # e.g., "F=cutting force, L=wall height, E=modulus, I=moment of inertia"
    relevance: str          # 1-sentence explanation of how it applies to the DFM question


class DisciplineSection(BaseModel):
    """One engineering discipline's contribution to the DFM report."""
    discipline: str  # design_geometry, machining_process, material, tooling, etc.
    first_principles: str  # LLM inference about the physics (labeled clearly)
    formulas: List[FormulaEntry] = []  # governing equations and scaling rules
    evidence: List[SourcedClaim]
    reasoning_chains: List[ReasoningChain]


class ContradictionEntryV2(BaseModel):
    """Enhanced contradiction with edge reasoning."""
    summary: str
    position_a: str
    frame_id_a: str
    position_b: str
    frame_id_b: str
    edge_reasoning: Optional[str] = None  # reasoning from CONTRADICTS edge


class DFMReportV2(BaseModel):
    """V2/V3 structured DFM report organized by engineering discipline."""
    summary: str
    disciplines: List[DisciplineSection]
    contradictions: List[ContradictionEntryV2]
    design_rules: List[SourcedClaim]  # actionable rules across all disciplines
    knowledge_gaps: List[str]  # specific gap descriptions


class QueryDecomposition(BaseModel):
    """LLM response for query decomposition."""
    sub_aspects: List[str]
    reasoning: str


# ─────────────────────────────────────────────
# GAP BACKFILL SCHEMAS (Phase 5)
# ─────────────────────────────────────────────

class GapBackfill(BaseModel):
    """Textbook-level answer for a knowledge gap, clearly labeled as non-sourced."""
    gap: str                    # the original gap description
    backfill_answer: str        # 2-4 sentence textbook-level answer
    confidence_note: str        # e.g., "General mechanical engineering knowledge"
    relevant_standards: Optional[str] = None  # e.g., "See ASME Y14.5, AWS D1.2"


class BackfillOutput(BaseModel):
    """Structured output for gap backfill LLM call."""
    backfills: List[GapBackfill]


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

SYNTHESIS_PROMPT_V1 = """You are a Design for Manufacturability (DFM) expert assistant.

Given a designer's question and a set of knowledge frames extracted from machinist forums
and engineering Q&A, produce a structured DFM report.

Rules:
- Only make claims that are directly supported by the provided frames. Do not add
  outside knowledge.
- Every risk and recommended_practice MUST cite the frame_id it came from.
- If two frames give opposing guidance on the same point, add a contradiction entry
  with both frame_ids. Do not pick a side.
- If the frames do not adequately cover the question, state this in knowledge_gaps.
- Prefer frames from high-credibility authors and accepted answers when claims conflict.
- Be concise: each claim should be 1-2 sentences."""


SYNTHESIS_PROMPT_V2 = """You are a Design for Manufacturability (DFM) expert assistant.

Given a designer's question and a set of knowledge frames extracted from machinist forums
and engineering Q&A sites, produce a structured DFM report organized by engineering discipline.

== EVIDENCE TIER (strict traceability) ==
- Claims in the 'evidence' and 'reasoning_chains' fields MUST cite the frame_id they came from.
- Each claim should synthesize the frame's insight in 1-2 sentences. Do not copy the source quote.
- reasoning_chains should follow the edge relationships provided between frames (CAUSAL_LINK,
  MITIGATES, SPECIFIES). Use the edge_type and edge reasoning to explain how frames connect.
- For each ChainedFrame, set role_in_chain to one of: root_cause, effect, mitigation,
  condition, origin. Set edge_from and edge_type to match the edges provided in the input.

== CHAIN-COMPLETION FRAMES ==
- Frames marked [CHAIN-COMPLETION] were included specifically because they have semantic
  edges connecting them to higher-ranked frames. Use these to build reasoning_chains.
- When you see a CAUSAL_LINK or MITIGATES edge between a ranked frame and a chain-completion
  frame, construct a ReasoningChain showing both frames with their roles and the connecting edge.
- Chain-completion frames should appear in reasoning_chains, not in standalone evidence
  (unless they also provide independent evidence).
- If no chain-completion frames are present, build reasoning chains from any frames that have
  edges between them.

== INFERENCE TIER (first-principles physics and engineering) ==
The 'first_principles' field should explain the GOVERNING PHYSICS for this discipline.
You MAY draw on general engineering knowledge. Start with "Engineering rationale:".

Include the following where applicable:

1. GOVERNING EQUATIONS with variable definitions. Put formal expressions in the 'formulas'
   list (not inline in first_principles text). Examples:
   - Cantilever deflection: delta = F*H^3 / (3*E*I) for wall-as-beam under cutting force
   - Thin-shell buckling: p_cr = 0.807*E*(t/R)^2 / sqrt(1-nu^2) for cylinders
   - Natural frequency scaling: f1 ~ sqrt(E*I / (rho*A*H^4)) for chatter prediction
   - Thermal tolerance budgeting: delta_T <= 0.25*T / (alpha*L) for expansion limits
   Only include equations that are directly relevant to the designer's question.

2. GEOMETRIC SCALING RULES and practical thresholds. Examples:
   - Height-to-thickness aspect ratio limits: H/t <= 10 for precision, H/t <= 20 general
   - Minimum inside radius: R_inside >= 0.5*t (ensures tool access, reduces stress concentration)
   - Thread engagement: >= 1.5D in aluminum, >= 1.0D in steel
   Only include rules that are relevant to the question.

3. LIFECYCLE CONSIDERATIONS beyond initial manufacturing:
   - Metrology deformation: thin parts deflecting under CMM probe or caliper contact
   - Handling and assembly damage: denting during shipping, racking for anodization
   - Burr management: difficulty deburring thin edges without rolling or plastic deformation
   - Post-processing distortion: stress relief, heat treatment, anodizing effects

4. COUPLED COMPLIANCE effects (when relevant to machining questions):
   - Total deflection = tool deflection + workpiece deflection (delta_total = delta_tool + delta_wall)
   - Long-reach endmill + thin wall = compounding flexibility
   - Internal radii affect minimum tool diameter, which affects tool stiffness

Keep first_principles to 3-5 sentences of physics explanation. Put equations and scaling
rules in the formulas list with variable definitions and relevance notes.

== ORGANIZATION ==
- Group frames by their scope field into discipline sections. Use the scope values from the
  frames (design_geometry, machining_process, material, tooling, programming, etc.).
- Only create discipline sections that have actual evidence from the frames.
- design_rules should be actionable rules distilled ACROSS all disciplines. Each must cite
  a frame_id.

== CONTRADICTIONS ==
- If two frames give opposing guidance on the same point, add a contradiction entry with
  both frame_ids. Do not pick a side. Include the edge_reasoning if a CONTRADICTS edge
  was provided between them.

== KNOWLEDGE GAPS ==
- If the frames do not cover an aspect that the question logically requires, add it as a
  specific knowledge gap. Be precise about what is missing (e.g., "No data on thermal
  distortion effects for this geometry/material combination").

== GENERAL RULES ==
- Prefer frames from high-credibility authors and accepted answers when claims conflict.
- Be concise: each claim should be 1-2 sentences.
- Pay attention to epistemic_stance: absolute_authoritative claims carry more weight than
  tentative_subjective ones."""


DECOMPOSE_PROMPT = """You are a DFM (Design for Manufacturability) engineering analyst.

Given a designer's question about manufacturing, decompose it into 3-6 specific engineering
sub-aspects that should be investigated. Each sub-aspect should target a different physical
phenomenon, failure mode, or engineering discipline.

Rules:
- Each sub-aspect should be a short phrase (3-8 words) suitable as a search query
- Cover different engineering disciplines: structural, thermal, machining dynamics, material,
  tooling, surface finish, etc.
- Focus on concrete failure modes, physical phenomena, and practical constraints
- Do not repeat the original question; break it into complementary angles"""


CONTEXT_SUMMARY_PROMPT = """You are helping a designer understand the credibility and context
of machining advice they received from a DFM knowledge graph.

Given provenance data about the sources behind DFM claims, produce a concise summary covering:
1. Who provided this advice -- their reputation level, credibility indicators, and expertise signals
2. The original discussion context -- what problem was being discussed, on which Stack Exchange site
3. Reliability signals -- accepted answers, high-score posts, authoritative epistemic stance

Group your summary by thread when multiple frames come from the same discussion.
Keep it to 3-5 sentences per thread cluster. Use plain language suitable for a designer
who may not be a machining expert."""


BACKFILL_PROMPT = """You are a Design for Manufacturability (DFM) textbook author.

Given a list of knowledge gaps identified in a DFM report, provide brief textbook-level
answers for each gap. These answers fill in what a machinist knowledge graph does NOT
contain, using general engineering knowledge.

Rules:
- Each backfill_answer should be 2-4 sentences of actionable engineering guidance.
- Include quantitative thresholds, rules of thumb, or standard references where possible
  (e.g., "minimum wall thickness for CNC milled aluminum is typically 0.8-1.0 mm for
  unsupported walls with H/t < 10").
- Set confidence_note to explain the source level (e.g., "General mechanical engineering
  knowledge", "Common industry practice", "Standard handbook value -- verify for specific alloy").
- If a relevant standard exists (ASME, ISO, AWS, Machinery's Handbook, etc.), cite it in
  relevant_standards.
- Do NOT fabricate specific numeric values if you are uncertain; instead say
  "consult [standard/handbook] for alloy-specific data".
- These are clearly labeled as general engineering inference, NOT sourced from the
  knowledge graph. The reader understands this is textbook-level guidance."""
