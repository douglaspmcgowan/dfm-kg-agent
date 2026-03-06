# DFM Knowledge Graph – Quick Context for Code Editing

Goal: Extract structured machining / CNC / DFM knowledge (heuristics, risks, principles, observations etc.) from forum posts on PracticalMachinist, CNCzone, Reddit r/CNC etc. → feed into a knowledge graph for design-for-manufacturability feedback.

Main files involved right now:
- schema.py          → Pydantic models (KnowledgeFrame, AuthorFrame, PostExtractionResult)
- extract.py         → main extraction pipeline (quote stripping → OpenAI structured outputs → filtering → consolidation)
- iterate.py         → automated prompt/schema improvement loop (runs extract.py repeatedly, assesses with strong model, proposes surgical edits)

Core extraction target = KnowledgeFrame

Key fields (must be correct):
- frame_type          one of: risk / heuristic / principle / case / workaround / recommendation / observation / other
- source_quote        NON-NEGOTIABLE verbatim substring from the **non-quoted** part of the post text
- main_point          1–2 sentence core claim — tailored to frame_type
- condition_value     real asserted threshold/parameter (do NOT put toy/example numbers here)
- example_value       illustrative/toy values only (e.g. "for brevity, 10 holes")
- directness          explicit / hedged / implied
- expression_strength strong / moderate / weak
- scope               short noun phrase: what the rule applies to (e.g. "thin-wall aluminum milling")
- subject             concise noun phrase (potential KG node label)
- post_role           question / solution / war_story / debate / general_advice / tutorial / other

Critical distinctions (frequent misclassifications):
heuristic     ← personal practice / rule of thumb + rationale ("I never use X because Y")
observation   ← empirical fact, no personal action ("X behaves this way")
principle     ← causal explanation WHY something happens (not plain definitions)
recommendation ← explicit advice what should be done
risk          ← condition(s) → failure mode

Social noise to NEVER extract:
greetings, thanks, sign-offs, "yep exactly", "let me know if I missed anything", "that's all I can think of right now", narrating own writing process, pure example statements without underlying rule

Quote stripping: removes [quote]...[/quote], > quoted lines, "Originally posted by", "X said: ... Click to expand...", etc.

Filtering rules in extract.py:
- source_quote must be missing → drop
- fuzzy substring match < ~85% → drop (anti-hallucination)
- main_point empty → drop
- if source_quote contains "for brevity", "let's say", "for example" etc. → move any condition_value → example_value

Current biggest pain points (early 2026):
- missing high-value experiential content hidden inside tutorial posts
- over-extracting social noise or author meta-commentary
- misclassifying hedged statements as explicit
- toy values leaking into condition_value
- hallucinated main_points on near-empty posts after quote stripping

Models in play:
extraction → usually gpt-4o-mini, gpt-5-nano, gpt-5-mini, claude-sonnet-4.5, gemini-flash variants
assessment/iteration loop → stronger model (o3, gpt-5.2, claude-sonnet-4.5 / opus-4.6, gemini-3-pro)

When suggesting changes please:
• prefer small, targeted edits to Field(description=…) in schema.py or specific sections in SYSTEM_PROMPT
• give complete replacement text (not diffs)
• explain why the change addresses a concrete failure pattern
• preserve the overall structure & logic of the current pipeline

Thanks for any help improving extraction quality / prompt precision / filtering logic.